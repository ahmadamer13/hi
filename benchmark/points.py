"""Point-based crowd localisation models (one point per head), for high/extreme density.

These models output head POINTS, not boxes. To reuse the same tracker and direction logic, each
point gets a pseudo head box whose size comes from the local crowd spacing (half the distance to the
nearest neighbour, clamped). Detection quality is scored with the point-in-box metric
(metrics_det.PointAccumulator), which is applied to box models too (box centre), so both families are
compared on the same footing.

P2PNet (Song et al., ICCV 2021) code is NOT vendored: its license limits use to academic research.
`benchmark.download` clones it into models/ext/ and this adapter imports it from there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .detectors import CLS_ID, Detections, Detector


def pseudo_boxes(points: np.ndarray, min_px: float = 6.0, max_px: float = 60.0, k: float = 0.5) -> np.ndarray:
    """Square head boxes sized from nearest-neighbour spacing (dense crowd => small heads)."""
    if not len(points):
        return np.zeros((0, 4), np.float32)
    if len(points) == 1:
        s = np.array([max_px / 2])
    else:
        d = np.linalg.norm(points[:, None] - points[None], axis=2)
        np.fill_diagonal(d, np.inf)
        s = np.clip(d.min(1) * k, min_px / 2, max_px / 2)
    return np.stack([points[:, 0] - s, points[:, 1] - s, points[:, 0] + s, points[:, 1] + s], 1).astype(np.float32)


class P2PNetDetector(Detector):
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, repo_dir: str | Path, weights: str | Path, imgsz: int = 1280, conf: float = 0.3,
                 device: str | None = None, half: bool = False):
        import torch
        import torchvision
        self.torch = torch
        repo_dir = str(Path(repo_dir).resolve())
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
        # the repo builds VGG with pretrained=True (ImageNet download); the checkpoint overwrites it anyway
        orig = torchvision.models.vgg16_bn
        torchvision.models.vgg16_bn = lambda *a, **k: orig(weights=None)
        try:
            from models.backbone import Backbone_VGG  # noqa: E402  (from the P2PNet clone)
            from models.p2pnet import P2PNet
        finally:
            torchvision.models.vgg16_bn = orig
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device not in ("cpu",) and not str(self.device).startswith("cuda"):
            self.device = f"cuda:{self.device}"
        model = P2PNet(Backbone_VGG("vgg16_bn", True), row=2, line=2)
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt.get("model", ckpt))
        self.model = model.to(self.device).eval()
        self.half = half and self.device.startswith("cuda")
        if self.half:
            self.model = self.model.half()
        self.imgsz, self.conf = imgsz, conf
        self.classes = {"head"}
        self.name = "p2pnet"
        self.point_model = True

    def points(self, img) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        h0, w0 = img.shape[:2]
        s = self.imgsz / max(h0, w0)
        nw, nh = max(128, int(w0 * s) // 128 * 128), max(128, int(h0 * s) // 128 * 128)
        x = cv2.resize(img, (nw, nh))[..., ::-1].astype(np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        t = torch.from_numpy(np.ascontiguousarray(x)).to(self.device)
        if self.half:
            t = t.half()
        with torch.inference_mode():
            out = self.model(t)
        sc = torch.softmax(out["pred_logits"].float(), -1)[0, :, 1].cpu().numpy()
        pts = out["pred_points"][0].float().cpu().numpy()
        m = sc >= self.conf
        pts = pts[m] * np.array([w0 / nw, h0 / nh], np.float32)
        return pts.astype(np.float32), sc[m].astype(np.float32)

    def __call__(self, img) -> Detections:
        pts, sc = self.points(img)
        d = Detections(pseudo_boxes(pts), sc, np.full(len(sc), CLS_ID["head"], np.int32))
        d.points = pts  # type: ignore[attr-defined]
        return d
