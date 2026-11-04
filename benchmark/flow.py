"""Zone-level dense-crowd fallback: optical flow -> per grid cell % of motion against allowed direction.

Estimators: OpenCV Farneback (CPU/CUDA) and torchvision RAFT-small. Bus boxes are masked out.
Ground truth per cell comes from labelled tracks: share of moving GT tracks in the cell whose
direction label is 'against'.
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np

from .direction import CameraConfig, to_ground


class Farneback:
    name = "farneback"

    def __init__(self, scale: float = 0.5):
        self.scale = scale

    def __call__(self, prev_bgr, cur_bgr):
        s = self.scale
        a = cv2.cvtColor(cv2.resize(prev_bgr, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(cv2.resize(cur_bgr, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return f / s, s  # flow in full-res px, sampled on the downscaled grid


class RaftSmall:
    name = "raft-small"

    def __init__(self, scale: float = 0.5, device: str | None = None, iters: int = 6):
        import torch
        from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.device).eval()
        self.scale, self.iters = scale, iters

    def _prep(self, img):
        s = self.scale
        h, w = img.shape[:2]
        nh, nw = int(h * s) // 8 * 8, int(w * s) // 8 * 8
        x = cv2.resize(img, (nw, nh))[..., ::-1].copy()
        t = self.torch.from_numpy(x).permute(2, 0, 1).float()[None] / 127.5 - 1.0
        return t.to(self.device), (w / nw)

    def __call__(self, prev_bgr, cur_bgr):
        a, k = self._prep(prev_bgr)
        b, _ = self._prep(cur_bgr)
        with self.torch.inference_mode():
            f = self.model(a, b, num_flow_updates=self.iters)[-1][0].permute(1, 2, 0).cpu().numpy()
        return f * k, 1.0 / k


def cell_against_fraction(flow, sample_scale, cam: CameraConfig, frame_shape, dt: float,
                          bus_boxes=(), cell_px: int = 64, min_speed_mps: float = 0.4,
                          min_moving_frac: float = 0.15, cls_name: str = "person"):
    """Returns dict[(cx, cy)] -> (against_frac, moving_px_frac). Cells with little motion omitted."""
    H, W = frame_shape[:2]
    fh, fw = flow.shape[:2]
    step = max(1, int(round(8 * sample_scale)))  # subsample flow grid for speed
    ys, xs = np.mgrid[0:fh:step, 0:fw:step]
    px = np.stack([xs.ravel() / sample_scale, ys.ravel() / sample_scale], 1)
    fv = flow[ys.ravel(), xs.ravel()]
    mask = np.ones(len(px), bool)
    for x1, y1, x2, y2 in bus_boxes:  # mask buses (dilated 10%)
        dx, dy = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
        mask &= ~((px[:, 0] >= x1 - dx) & (px[:, 0] <= x2 + dx) & (px[:, 1] >= y1 - dy) & (px[:, 1] <= y2 + dy))
    px, fv = px[mask], fv[mask]
    g0 = to_ground(cam.H, px)
    g1 = to_ground(cam.H, px + fv)
    v = (g1 - g0) / max(dt, 1e-3)
    speed = np.linalg.norm(v, axis=1)
    out = {}
    cells = (px // cell_px).astype(int)
    keys, inv = np.unique(cells, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    moving = speed >= min_speed_mps
    n_all = np.bincount(inv, minlength=len(keys))
    n_mov = np.bincount(inv, weights=moving, minlength=len(keys))
    for k_idx, key in enumerate(map(tuple, keys)):
        frac_moving = n_mov[k_idx] / max(n_all[k_idx], 1)
        if frac_moving < min_moving_frac:
            continue
        centre = (np.array(key) + 0.5) * cell_px
        zone = cam.zone_at(centre, cls_name)
        if zone is None or zone.mode in ("exclude", "both"):
            continue
        a = zone.allowed(to_ground(cam.H, centre[None])[0])
        if a is None:
            continue
        mv = (inv == k_idx) & moving
        cosang = (v[mv] @ a) / np.maximum(speed[mv], 1e-9)
        out[key] = (float((cosang < math.cos(math.radians(120))).mean()), float(frac_moving))
    return out


def gt_cell_against(clip, frame_lo: int, frame_hi: int, cam: CameraConfig, cell_px: int = 64):
    """Per cell: share of 'against' among moving GT tracks centred in the cell over [lo, hi]."""
    tracks = {}
    for fr in range(frame_lo, frame_hi + 1):
        for row in clip.gt_frame(fr):
            tid = int(row[0])
            if clip.directions.get(tid) in ("with", "against"):
                b = row[2:6]
                tracks.setdefault(tid, []).append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    cells: dict = {}
    for tid, pts in tracks.items():
        key = tuple((np.mean(pts, 0) // cell_px).astype(int))
        cells.setdefault(key, []).append(clip.directions[tid] == "against")
    return {k: float(np.mean(v)) for k, v in cells.items() if len(v) >= 2}


def score_cells(pred: dict, gt: dict, flag_thr: float = 0.3) -> dict:
    keys = set(pred) & set(gt)
    if not keys:
        return {"cells": 0}
    err = [abs(pred[k][0] - gt[k]) for k in keys]
    tp = sum(pred[k][0] >= flag_thr and gt[k] >= flag_thr for k in keys)
    fp = sum(pred[k][0] >= flag_thr and gt[k] < flag_thr for k in keys)
    fn = sum(pred[k][0] < flag_thr and gt[k] >= flag_thr for k in keys)
    return {"cells": len(keys), "mae_against_frac": float(np.mean(err)), "tp": tp, "fp": fp, "fn": fn,
            "coverage": len(keys) / max(len(gt), 1)}


def time_flow(est, frame, n: int = 10) -> float:
    est(frame, frame)
    t = time.perf_counter()
    for _ in range(n):
        est(frame, frame)
    return (time.perf_counter() - t) / n
