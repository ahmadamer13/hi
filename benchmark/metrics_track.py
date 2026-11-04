"""Tracking (CLEAR-MOT, IDF1) and direction metrics.

Direction scoring (the metric that matters):
  * Each GT track labelled with/against is associated with the predicted track that covers it
    on the most frames (IoU >= 0.5).
  * The system's verdict for that GT track is the predicted track's dominant non-gray state
    (red if it was ever confirmed red for >= `min_red_frames`, else green if confirmed green,
    else gray). Unmatched GT tracks are gray (missed, not wrong).
  * Wrong-way positives: GT 'against'. A red on a GT 'with'/'static'/'cross' track, or on a predicted
    track matching no GT at all (a detector FP), is a false alarm.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics_det import box_iou, match_frame


def _point_sim(pred_boxes, gt_boxes):
    """1 - normalised centre distance if the predicted centre lies inside the GT box, else 0."""
    c = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2
    inside = ((c[:, None, 0] >= gt_boxes[None, :, 0]) & (c[:, None, 0] <= gt_boxes[None, :, 2]) &
              (c[:, None, 1] >= gt_boxes[None, :, 1]) & (c[:, None, 1] <= gt_boxes[None, :, 3]))
    gc = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
    diag = np.linalg.norm(gt_boxes[:, 2:] - gt_boxes[:, :2], axis=1)
    d = np.linalg.norm(c[:, None] - gc[None], axis=2) / np.maximum(diag[None], 1e-6)
    return np.where(inside, np.clip(1 - d, 0.05, 1.0), 0.0)


def _point_match(pred_boxes, gt_boxes):
    sim = _point_sim(pred_boxes, gt_boxes)
    r, c = linear_sum_assignment(-sim)
    return [(int(i), int(j)) for i, j in zip(r, c) if sim[i, j] > 0]


class TrackEval:
    def __init__(self, iou_thr=0.5, match="iou"):
        """match='iou' (box models, standard MOT) or 'point' (point models: track centre inside GT box)."""
        self.iou_thr, self.match = iou_thr, match
        self.tp = self.fp = self.fn = self.idsw = 0
        self.last_match: dict[int, int] = {}
        self.pair_frames: Counter = Counter()      # (gt, pred) -> frames matched
        self.gt_frames: Counter = Counter()
        self.pred_frames: Counter = Counter()
        self.pred_states: dict[int, Counter] = defaultdict(Counter)
        self.n_frames = 0
        # HOTA accumulators per localisation threshold alpha
        self.alphas = np.arange(0.05, 0.96, 0.05)
        self.h_tp = np.zeros(len(self.alphas))
        self.h_pairs = [Counter() for _ in self.alphas]
        self.n_gt_dets = self.n_pred_dets = 0

    def add(self, gt_ids, gt_boxes, pred_ids, pred_boxes, pred_states=None):
        self.n_frames += 1
        for g in gt_ids:
            self.gt_frames[int(g)] += 1
        for p in pred_ids:
            self.pred_frames[int(p)] += 1
        if pred_states is not None:
            for p, s in zip(pred_ids, pred_states):
                self.pred_states[int(p)][s] += 1
        pb, gb = np.asarray(pred_boxes, np.float32).reshape(-1, 4), np.asarray(gt_boxes, np.float32).reshape(-1, 4)
        self.n_gt_dets += len(gb)
        self.n_pred_dets += len(pb)
        if len(pb) and len(gb):
            iou = box_iou(pb, gb) if self.match == "iou" else _point_sim(pb, gb)
            r, c = linear_sum_assignment(-iou)
            for k, a in enumerate(self.alphas):
                ok = iou[r, c] >= a
                self.h_tp[k] += ok.sum()
                for pi, gi in zip(r[ok], c[ok]):
                    self.h_pairs[k][(int(gt_ids[gi]), int(pred_ids[pi]))] += 1
        m = match_frame(pb, gb, self.iou_thr) if self.match == "iou" else _point_match(pb, gb)
        self.tp += len(m)
        self.fp += len(pred_ids) - len(m)
        self.fn += len(gt_ids) - len(m)
        for pi, gi in m:
            g, p = int(gt_ids[gi]), int(pred_ids[pi])
            if g in self.last_match and self.last_match[g] != p:
                self.idsw += 1
            self.last_match[g] = p
            self.pair_frames[(g, p)] += 1

    def summary(self) -> dict:
        n_gt = self.tp + self.fn
        gts, preds = sorted(self.gt_frames), sorted(self.pred_frames)
        idtp = 0
        if gts and preds:
            gi = {g: i for i, g in enumerate(gts)}
            pi = {p: i for i, p in enumerate(preds)}
            w = np.zeros((len(gts), len(preds)))
            for (g, p), n in self.pair_frames.items():
                w[gi[g], pi[p]] = n
            r, c = linear_sum_assignment(-w)
            idtp = int(w[r, c].sum())
        idfp = sum(self.pred_frames.values()) - idtp
        idfn = sum(self.gt_frames.values()) - idtp
        return {
            **self.hota(),
            "MOTA": 1 - (self.fn + self.fp + self.idsw) / max(n_gt, 1),
            "IDF1": 2 * idtp / max(2 * idtp + idfp + idfn, 1),
            "IDSW": self.idsw,
            "IDSW_per_100_gt_tracks": 100 * self.idsw / max(len(gts), 1),
            "n_gt_tracks": len(gts),
        }

    def hota(self) -> dict:
        """HOTA (Luiten et al. 2021). Approximation: one IoU-maximising match per frame, thresholded
        at each alpha (TrackEval weights matches by association score). Use TrackEval for official numbers."""
        hs, das, aas = [], [], []
        for k in range(len(self.alphas)):
            tp = self.h_tp[k]
            fn, fp = self.n_gt_dets - tp, self.n_pred_dets - tp
            det_a = tp / max(tp + fn + fp, 1)
            if tp:
                ass = [n * n / (self.gt_frames[g] + self.pred_frames[p] - n) for (g, p), n in self.h_pairs[k].items()]
                ass_a = sum(ass) / tp
            else:
                ass_a = 0.0
            hs.append(np.sqrt(det_a * ass_a)), das.append(det_a), aas.append(ass_a)
        return {"HOTA": float(np.mean(hs)), "DetA": float(np.mean(das)), "AssA": float(np.mean(aas))}

    # ------------------------------------------------------------------ direction
    def direction(self, gt_dirs: dict[int, str], min_red_frames: int = 1) -> dict:
        best_pred: dict[int, int] = {}
        for (g, p), n in self.pair_frames.items():
            if n > self.pair_frames.get((g, best_pred.get(g, -1)), 0):
                best_pred[g] = p
        matched_preds = {p for (_, p) in self.pair_frames}

        def verdict(p):
            s = self.pred_states.get(p, Counter())
            if s.get("red", 0) >= min_red_frames:
                return "red"
            if s.get("green", 0) > 0:
                return "green"
            return "gray"

        rows = []
        for g, lab in gt_dirs.items():
            if g not in self.gt_frames:
                continue
            v = verdict(best_pred[g]) if g in best_pred else "gray"
            rows.append((lab, v))
        moving = [(l, v) for l, v in rows if l in ("with", "against")]
        decided = [(l, v) for l, v in moving if v != "gray"]
        correct = sum((l == "with" and v == "green") or (l == "against" and v == "red") for l, v in decided)
        tp = sum(l == "against" and v == "red" for l, v in rows)
        fn = sum(l == "against" and v != "red" for l, v in rows)
        fp_gt = sum(l != "against" and v == "red" for l, v in rows)
        fp_orphan = sum(1 for p in self.pred_states if p not in matched_preds and verdict(p) == "red")
        fp = fp_gt + fp_orphan
        prec = tp / (tp + fp) if tp + fp else None   # None = undefined (nothing flagged)
        rec = tp / (tp + fn) if tp + fn else None    # None = no wrong-way GT in this set
        f1 = (2 * prec * rec / max(prec + rec, 1e-9)) if prec is not None and rec is not None else None
        return {
            "direction_accuracy": correct / len(decided) if decided else None,  # among decided moving tracks
            "decided_fraction": len(decided) / max(len(moving), 1),  # coverage (1 - gray rate)
            "ww_precision": prec, "ww_recall": rec,
            "ww_f1": f1,
            "ww_tp": tp, "ww_fp": fp, "ww_fp_unmatched_track": fp_orphan, "ww_fn": fn,
            "n_moving_gt_tracks": len(moving),
        }
