"""Uniform detector interface over every candidate format.

All adapters return `Detections` in ORIGINAL image pixel coordinates with canonical class ids
(see CLASSES). Classes a model was not trained for are simply absent from its output, and the
evaluator only scores a model on the classes it declares.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("head", "person", "bus", "car", "truck")
CLS_ID = {c: i for i, c in enumerate(CLASSES)}


@dataclass
class Detections:
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))  # xyxy
    scores: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float32))
    classes: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int32))  # canonical ids

    def __len__(self) -> int:
        return len(self.scores)

    def select(self, mask: np.ndarray) -> "Detections":
        return Detections(self.boxes[mask], self.scores[mask], self.classes[mask])

    @staticmethod
    def concat(items: list["Detections"]) -> "Detections":
        items = [d for d in items if len(d)]
        if not items:
            return Detections()
        return Detections(np.concatenate([d.boxes for d in items]).astype(np.float32),
                          np.concatenate([d.scores for d in items]).astype(np.float32),
                          np.concatenate([d.classes for d in items]).astype(np.int32))


def _map_classes(raw_cls: np.ndarray, class_map: dict[int, str]):
    keep = np.array([int(c) in class_map for c in raw_cls], dtype=bool)
    mapped = np.array([CLS_ID[class_map[int(c)]] for c in raw_cls[keep]], dtype=np.int32)
    return keep, mapped


def nms_per_class(d: Detections, iou: float) -> Detections:
    if not len(d):
        return d
    keep_all = []
    for c in np.unique(d.classes):
        idx = np.where(d.classes == c)[0]
        b = d.boxes[idx]
        k = cv2.dnn.NMSBoxes(
            [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in b],
            d.scores[idx].tolist(), 0.0, iou)
        keep_all.extend(idx[np.array(k, dtype=int).reshape(-1)])
    return d.select(np.array(sorted(keep_all), dtype=int))


def letterbox(img: np.ndarray, size: tuple[int, int]):
    """size=(w,h). Returns padded image, scale, (pad_x, pad_y)."""
    w, h = size
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = int(round(iw * s)), int(round(ih * s))
    canvas = np.full((h, w, 3), 114, np.uint8)
    px, py = (w - nw) // 2, (h - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, s, (px, py)


class Detector:
    name: str = "base"
    classes: set[str] = set()

    def __call__(self, img_bgr: np.ndarray) -> Detections:  # pragma: no cover - interface
        raise NotImplementedError

    def warmup(self, shape=(1080, 1920, 3), n: int = 3) -> None:
        dummy = np.zeros(shape, np.uint8)
        for _ in range(n):
            self(dummy)


class UltralyticsDetector(Detector):
    """.pt / .onnx / .engine through the ultralytics runtime (YOLOv8/11, ultralytics RT-DETR)."""

    def __init__(self, weights: str | Path, class_map: dict[int, str], imgsz: int = 640,
                 conf: float = 0.05, iou: float = 0.6, half: bool = False, device: str | None = None,
                 max_det: int = 1000):
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="detect")
        self.class_map, self.imgsz, self.conf, self.iou = class_map, imgsz, conf, iou
        self.half, self.device, self.max_det = half, device, max_det
        self.classes = set(class_map.values())
        self.name = Path(weights).stem

    def __call__(self, img):
        kw = {"quantize": 16} if self.half else {}  # ultralytics>=8.4 name for FP16
        r = self.model.predict(img, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                               device=self.device, max_det=self.max_det, verbose=False,
                               classes=sorted(self.class_map), **kw)[0]
        b = r.boxes
        raw = b.cls.cpu().numpy().astype(int)
        keep, mapped = _map_classes(raw, self.class_map)
        return Detections(b.xyxy.cpu().numpy()[keep].astype(np.float32),
                          b.conf.cpu().numpy()[keep].astype(np.float32), mapped)


class OnnxYoloDetector(Detector):
    """Raw ONNX with an Ultralytics-style head: output (1, 4+nc, N) cx,cy,w,h + class scores.

    providers: e.g. ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"].
    On the target GPU pass trt_fp16=True to get FP16 TensorRT through onnxruntime.
    """

    def __init__(self, path: str | Path, class_map: dict[int, str], imgsz: int | None = None,
                 conf: float = 0.05, iou: float = 0.6, input_format: str = "rgb",
                 providers: list[str] | None = None, trt_fp16: bool = True):
        import onnxruntime as ort
        providers = providers or ort.get_available_providers()
        opts = []
        for p in providers:
            if p == "TensorrtExecutionProvider":
                opts.append({"trt_fp16_enable": trt_fp16, "trt_engine_cache_enable": True,
                             "trt_engine_cache_path": str(Path(path).parent / "trt_cache")})
            else:
                opts.append({})
        self.sess = ort.InferenceSession(str(path), providers=providers, provider_options=opts)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        h, w = inp.shape[2], inp.shape[3]
        self.static_size = h if isinstance(h, int) else None  # fixed-shape export ignores imgsz
        if not isinstance(h, int):  # dynamic axes
            h = w = imgsz or 640
        self.size = (w, h)
        self.class_map, self.conf, self.iou = class_map, conf, iou
        self.bgr = input_format.lower() == "bgr"
        self.classes = set(class_map.values())
        self.name = Path(path).stem
        self.fp16_input = "float16" in inp.type

    def __call__(self, img):
        lb, s, (px, py) = letterbox(img, self.size)
        x = lb if self.bgr else lb[..., ::-1]
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        if self.fp16_input:
            x = x.astype(np.float16)
        outs = self.sess.run(None, {self.input_name: x})
        if len(outs) == 3:
            # DeepStream layout: boxes (1,N,4) cx,cy,w,h in input px, scores (1,N,1), classes (1,N,1)
            # (verified empirically; the model's bundled inference.py assumes a different shape)
            cx, cy, w, h = outs[0][0].astype(np.float32).T
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
            sc = outs[1][0].reshape(-1).astype(np.float32)
            cls = outs[2][0].reshape(-1).astype(int)
        else:
            # Ultralytics layout: (1, 4+nc, N) cx,cy,w,h + per-class scores
            out = outs[0][0].astype(np.float32).T
            scores_all = out[:, 4:]
            cls = scores_all.argmax(1)
            sc = scores_all[np.arange(len(cls)), cls]
            cx, cy, w, h = out[:, 0], out[:, 1], out[:, 2], out[:, 3]
            xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        m = sc >= self.conf
        xyxy, cls, sc = xyxy[m], cls[m], sc[m]
        keep, mapped = _map_classes(cls, self.class_map)
        xyxy, sc = xyxy[keep], sc[keep]
        boxes = ((xyxy - np.array([px, py, px, py], np.float32)) / s).astype(np.float32)
        return nms_per_class(Detections(boxes, sc.astype(np.float32), mapped), self.iou)


class OnnxRTDETRDetector(Detector):
    """RT-DETR exported by benchmark.trt.export: outputs logits (1,Q,C) and boxes (1,Q,4) cxcywh in [0,1]."""

    def __init__(self, path, class_map, imgsz=640, conf=0.05, providers=None, trt_fp16=True):
        import onnxruntime as ort
        providers = providers or ort.get_available_providers()
        opts = [{"trt_fp16_enable": trt_fp16, "trt_engine_cache_enable": True,
                 "trt_engine_cache_path": str(Path(path).parent / "trt_cache")}
                if p == "TensorrtExecutionProvider" else {} for p in providers]
        self.sess = ort.InferenceSession(str(path), providers=providers, provider_options=opts)
        self.imgsz, self.conf, self.class_map = imgsz, conf, class_map
        self.classes = set(class_map.values())
        self.name = Path(path).stem

    def __call__(self, img):
        h0, w0 = img.shape[:2]
        x = cv2.resize(img, (self.imgsz, self.imgsz))[..., ::-1].transpose(2, 0, 1)[None]
        logits, boxes = self.sess.run(None, {"images": np.ascontiguousarray(x, np.float32) / 255.0})
        prob = 1 / (1 + np.exp(-logits[0]))
        cls = prob.argmax(1)
        sc = prob[np.arange(len(cls)), cls]
        m = sc >= self.conf
        keep, mapped = _map_classes(cls[m], self.class_map)
        b = boxes[0][m][keep]
        sc = sc[m][keep]
        xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * w0, (b[:, 1] - b[:, 3] / 2) * h0,
                         (b[:, 0] + b[:, 2] / 2) * w0, (b[:, 1] + b[:, 3] / 2) * h0], 1)
        return Detections(xyxy.astype(np.float32), sc.astype(np.float32), mapped)


class HFRTDETRDetector(Detector):
    """transformers RT-DETR / RT-DETRv2 (PyTorch). No NMS needed (set-based detector)."""

    def __init__(self, model_dir: str | Path, class_map: dict[int, str], imgsz: int = 640,
                 conf: float = 0.05, device: str | None = None, half: bool = False):
        import torch
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.proc = AutoImageProcessor.from_pretrained(str(model_dir))
        self.model = AutoModelForObjectDetection.from_pretrained(str(model_dir)).to(self.device).eval()
        self.half = half and self.device.startswith("cuda")
        if self.half:
            self.model = self.model.half()
        self.imgsz, self.conf, self.class_map = imgsz, conf, class_map
        self.classes = set(class_map.values())
        self.name = Path(model_dir).name

    def __call__(self, img):
        torch = self.torch
        h0, w0 = img.shape[:2]
        rgb = cv2.resize(img, (self.imgsz, self.imgsz))[..., ::-1]
        x = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255)[None]
        x = x.to(self.device, dtype=torch.half if self.half else torch.float)
        with torch.inference_mode():
            out = self.model(pixel_values=x)
        res = self.proc.post_process_object_detection(out, threshold=self.conf,
                                                      target_sizes=[(h0, w0)])[0]
        raw = res["labels"].cpu().numpy().astype(int)
        keep, mapped = _map_classes(raw, self.class_map)
        return Detections(res["boxes"].float().cpu().numpy()[keep].astype(np.float32),
                          res["scores"].float().cpu().numpy()[keep].astype(np.float32), mapped)


class DarknetDetector(Detector):
    """Darknet cfg/weights via OpenCV DNN (CUDA backend if OpenCV is built with it)."""

    def __init__(self, cfg: str | Path, weights: str | Path, class_map: dict[int, str],
                 input_size=(640, 640), conf: float = 0.05, iou: float = 0.6, cuda: bool = False,
                 fp16: bool = True):
        self.net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
        if cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16 if fp16 else cv2.dnn.DNN_TARGET_CUDA)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.size = tuple(input_size)
        self.class_map, self.conf, self.iou = class_map, conf, iou
        self.classes = set(class_map.values())
        self.name = Path(weights).stem

    def __call__(self, img):
        h0, w0 = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, self.size, swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = np.concatenate([o.reshape(-1, o.shape[-1]) for o in self.net.forward(self.out_names)])
        obj = outs[:, 4]
        cls_scores = outs[:, 5:] * obj[:, None]
        cls = cls_scores.argmax(1)
        sc = cls_scores[np.arange(len(cls)), cls]
        m = sc >= self.conf
        outs, cls, sc = outs[m], cls[m], sc[m]
        keep, mapped = _map_classes(cls, self.class_map)
        outs, sc = outs[keep], sc[keep]
        cx, cy, w, h = outs[:, 0] * w0, outs[:, 1] * h0, outs[:, 2] * w0, outs[:, 3] * h0
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).astype(np.float32)
        return nms_per_class(Detections(boxes, sc.astype(np.float32), mapped), self.iou)


class TiledDetector(Detector):
    """Full-frame pass + high-res tiles over the far region only, merged with class-wise NMS.

    far_roi: (x1, y1, x2, y2) in image pixels, typically the top band of the frame where heads are
    small. Tiles of `tile` px with `overlap` fraction are cut from that ROI and run at native res.
    Boxes touching an internal tile border are dropped (the overlapping tile owns them).
    """

    def __init__(self, base: Detector, far_roi: tuple[int, int, int, int], tile: int = 640,
                 overlap: float = 0.2, iou: float = 0.6, far_only_classes=("head", "person")):
        self.base, self.far_roi, self.tile, self.overlap, self.iou = base, far_roi, tile, overlap, iou
        self.classes = base.classes
        self.far_cls = {CLS_ID[c] for c in far_only_classes}
        self.name = f"{base.name}+tiled{tile}"

    def tiles(self, shape):
        x1, y1, x2, y2 = self.far_roi
        x2, y2 = min(x2, shape[1]), min(y2, shape[0])
        step = int(self.tile * (1 - self.overlap))
        xs = list(range(x1, max(x1 + 1, x2 - self.tile + 1), step)) or [x1]
        ys = list(range(y1, max(y1 + 1, y2 - self.tile + 1), step)) or [y1]
        if xs[-1] + self.tile < x2:
            xs.append(max(x1, x2 - self.tile))
        if ys[-1] + self.tile < y2:
            ys.append(max(y1, y2 - self.tile))
        return [(x, y, min(x + self.tile, shape[1]), min(y + self.tile, shape[0])) for y in ys for x in xs]

    def __call__(self, img):
        parts = [self.base(img)]
        margin = 2.0
        for tx1, ty1, tx2, ty2 in self.tiles(img.shape):
            d = self.base(img[ty1:ty2, tx1:tx2])
            if not len(d):
                continue
            d = d.select(np.isin(d.classes, list(self.far_cls)))
            b = d.boxes
            touch = np.zeros(len(d), bool)
            if tx1 > 0:
                touch |= b[:, 0] <= margin
            if ty1 > 0:
                touch |= b[:, 1] <= margin
            if tx2 < img.shape[1]:
                touch |= b[:, 2] >= (tx2 - tx1) - margin
            if ty2 < img.shape[0]:
                touch |= b[:, 3] >= (ty2 - ty1) - margin
            d = d.select(~touch)
            d.boxes = d.boxes + np.array([tx1, ty1, tx1, ty1], np.float32)
            parts.append(d)
        return nms_per_class(Detections.concat(parts), self.iou)


class EnsembleDetector(Detector):
    """Combine a people model and a bus model (e.g. head model + COCO model for buses)."""

    def __init__(self, members: list[tuple[Detector, tuple[str, ...]]]):
        self.members = members
        self.classes = set().union(*[set(c) for _, c in members])
        self.name = "+".join(m.name for m, _ in members)

    def __call__(self, img):
        out = []
        for det, cls in self.members:
            d = det(img)
            out.append(d.select(np.isin(d.classes, [CLS_ID[c] for c in cls])))
        return Detections.concat(out)


def build(cand: dict, imgsz: int | None = None, device: str | None = None, half: bool = False,
          conf: float = 0.05, trt: bool = False) -> Detector:
    """Instantiate a candidate from models.yaml."""
    from .download import local_path
    cmap = {int(k): v for k, v in cand["classes"].items()}
    path = local_path(cand["source"], cand)
    a = cand["adapter"]
    trt_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    if a == "ultralytics":
        if trt:  # engine built by benchmark.trt.export for this input size
            eng = path.with_name(f"{path.stem}_{imgsz or 640}.engine")
            if not eng.exists():
                raise FileNotFoundError(f"{eng} missing: run python -m benchmark.trt.export on the target GPU")
            path = eng
        return UltralyticsDetector(path, cmap, imgsz=imgsz or 640, conf=conf, half=half, device=device)
    if a == "onnx_yolo":
        providers = trt_providers if trt else None
        return OnnxYoloDetector(path, cmap, imgsz=imgsz, conf=conf,
                                input_format=cand.get("input_format", "rgb"), providers=providers,
                                trt_fp16=half)
    if a == "hf_rtdetr":
        if trt:
            onnx = path.parent / f"{path.name}_{imgsz or 640}.onnx"
            if not onnx.exists():
                raise FileNotFoundError(f"{onnx} missing: run python -m benchmark.trt.export")
            return OnnxRTDETRDetector(onnx, cmap, imgsz=imgsz or 640, conf=conf, providers=trt_providers,
                                      trt_fp16=half)
        return HFRTDETRDetector(path, cmap, imgsz=imgsz or 640, conf=conf, device=device, half=half)
    if a == "darknet":
        size = tuple(cand.get("input_size", (640, 640)))
        if imgsz:
            size = (imgsz, int(round(imgsz * size[1] / size[0] / 32) * 32))
        return DarknetDetector(local_path(cand["cfg"]), path, cmap, input_size=size, conf=conf,
                               cuda=device not in (None, "cpu"), fp16=half)
    raise ValueError(f"adapter {a!r} not available")
