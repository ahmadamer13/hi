"""COCO-style detection metrics with crowd-aware settings.

- AP@0.5, AP@[.5:.95] (101-point interpolation), AR@maxDets (maxDets=1000 for dense crowds)
- predictions whose centre falls in an ignore polygon / matches an ignore GT box are not FPs
- false positives per frame at an operating threshold
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

IOU_THRS = np.linspace(0.5, 0.95, 10)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    aa = (a[:, 2:] - a[:, :2]).clip(0).prod(1)
    ab = (b[:, 2:] - b[:, :2]).clip(0).prod(1)
    return inter / np.maximum(aa[:, None] + ab[None] - inter, 1e-9)


def ioa(pred: np.ndarray, ign: np.ndarray) -> np.ndarray:
    """Intersection over prediction area (for ignore boxes)."""
    if not len(pred) or not len(ign):
        return np.zeros((len(pred), len(ign)), np.float32)
    tl = np.maximum(pred[:, None, :2], ign[None, :, :2])
    br = np.minimum(pred[:, None, 2:], ign[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    ap = (pred[:, 2:] - pred[:, :2]).clip(0).prod(1)
    return inter / np.maximum(ap[:, None], 1e-9)


@dataclass
class ClassAccumulator:
    """Accumulates per-frame matches for one class over many frames."""
    max_dets: int = 1000
    scores: list = field(default_factory=list)             # per pred score
    tp: list = field(default_factory=list)                 # per pred: (10,) bool per IoU thr
    n_gt: int = 0
    n_frames: int = 0

    def add(self, pred_boxes, pred_scores, gt_boxes, gt_ignore=None, ignore_mask=None):
        self.n_frames += 1
        gt_ignore = np.zeros(len(gt_boxes), bool) if gt_ignore is None else gt_ignore.astype(bool)
        order = np.argsort(-pred_scores)[: self.max_dets]
        pb, ps = pred_boxes[order], pred_scores[order]
        if ignore_mask is not None:
            keep = ~ignore_mask[order]
            pb, ps = pb[keep], ps[keep]
        g_valid, g_ign = gt_boxes[~gt_ignore], gt_boxes[gt_ignore]
        self.n_gt += len(g_valid)
        ious = box_iou(pb, g_valid)
        ign_hit = (ioa(pb, g_ign).max(1) > 0.5) if len(g_ign) else np.zeros(len(pb), bool)
        tps = np.zeros((len(pb), len(IOU_THRS)), bool)
        drop = np.zeros((len(pb), len(IOU_THRS)), bool)
        for t, thr in enumerate(IOU_THRS):
            used = np.zeros(len(g_valid), bool)
            for i in range(len(pb)):  # greedy by score, as in COCO
                if len(g_valid):
                    cand = np.where(~used & (ious[i] >= thr))[0]
                    if len(cand):
                        j = cand[np.argmax(ious[i, cand])]
                        used[j] = True
                        tps[i, t] = True
                        continue
                if ign_hit[i]:
                    drop[i, t] = True
        # predictions matched to ignore regions at IoU 0.5 are dropped from all thresholds
        keep = ~drop[:, 0]
        self.scores.extend(ps[keep].tolist())
        self.tp.extend(tps[keep].tolist())

    def summary(self, op_thr: float = 0.4) -> dict:
        if not self.scores:
            return {"AP50": 0.0, "AP": 0.0, "AR": 0.0, "n_gt": self.n_gt, "fp_per_frame": 0.0,
                    "precision@op": 0.0, "recall@op": 0.0, "best_f1": 0.0, "best_f1_thr": None}
        s = np.array(self.scores)
        tp = np.array(self.tp)
        o = np.argsort(-s, kind="mergesort")
        s, tp = s[o], tp[o]
        aps, ars = [], []
        for t in range(len(IOU_THRS)):
            ctp = np.cumsum(tp[:, t])
            cfp = np.cumsum(~tp[:, t])
            rec = ctp / max(self.n_gt, 1)
            prec = ctp / np.maximum(ctp + cfp, 1)
            prec = np.maximum.accumulate(prec[::-1])[::-1]
            rs = np.linspace(0, 1, 101)
            idx = np.searchsorted(rec, rs, side="left")
            aps.append(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))
            ars.append(rec[-1] if len(rec) else 0.0)
        ctp = np.cumsum(tp[:, 0])
        cfp = np.cumsum(~tp[:, 0])
        rec = ctp / max(self.n_gt, 1)
        prec = ctp / np.maximum(ctp + cfp, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        b = int(np.argmax(f1))
        k = int((s >= op_thr).sum())
        return {
            "AP50": float(aps[0]), "AP": float(np.mean(aps)), "AR": float(np.mean(ars)),
            "n_gt": int(self.n_gt),
            "fp_per_frame": float(cfp[k - 1] / max(self.n_frames, 1)) if k else 0.0,
            "precision@op": float(prec[k - 1]) if k else 0.0,
            "recall@op": float(rec[k - 1]) if k else 0.0,
            "best_f1": float(f1[b]), "best_f1_thr": float(s[b]),
        }


def match_frame(pred_boxes, gt_boxes, thr=0.5):
    """Hungarian matching on IoU (for tracking metrics). Returns list of (pi, gi)."""
    if not len(pred_boxes) or not len(gt_boxes):
        return []
    iou = box_iou(pred_boxes, gt_boxes)
    cost = 1 - iou
    cost[iou < thr] = 1e6
    r, c = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(r, c) if iou[i, j] >= thr]
