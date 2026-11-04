"""ByteTrack (Zhang et al. 2022) re-implemented, MIT-compatible, no external deps beyond scipy.

Tuned for dense crowds: two-stage association (high then low score), longer track buffer to
survive short occlusions, and a per-track quality score used later to decide tracking vs flow
fallback. Each class is tracked independently (a head can never become a bus).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics_det import box_iou


class KalmanXYAH:
    """Constant-velocity Kalman on (cx, cy, aspect, h) as in SORT/ByteTrack."""
    _std_pos, _std_vel = 1 / 20, 1 / 160

    def __init__(self):
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def initiate(self, z):
        mean = np.r_[z, np.zeros(4)]
        h = z[3]
        std = [2 * self._std_pos * h, 2 * self._std_pos * h, 1e-2, 2 * self._std_pos * h,
               10 * self._std_vel * h, 10 * self._std_vel * h, 1e-5, 10 * self._std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        q = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-2, self._std_pos * h,
                               self._std_vel * h, self._std_vel * h, 1e-5, self._std_vel * h]))
        return self.F @ mean, self.F @ cov @ self.F.T + q

    def update(self, mean, cov, z):
        h = mean[3]
        r = np.diag(np.square([self._std_pos * h, self._std_pos * h, 1e-1, self._std_pos * h]))
        s = self.H @ cov @ self.H.T + r
        k = cov @ self.H.T @ np.linalg.inv(s)
        mean = mean + k @ (z - self.H @ mean)
        cov = (np.eye(8) - k @ self.H) @ cov
        return mean, cov


def xyxy_to_xyah(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w / max(h, 1e-6), h])


def xyah_to_xyxy(m):
    cx, cy, a, h = m[:4]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


@dataclass
class Track:
    tid: int
    cls: int
    mean: np.ndarray
    cov: np.ndarray
    score: float
    start_frame: int
    last_frame: int
    hits: int = 1
    state: str = "tentative"  # tentative | confirmed | lost | removed
    score_hist: list = field(default_factory=list)

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)

    def quality(self, window: int = 30) -> float:
        """Mean detection score x fraction of recent frames with a matched detection."""
        age = self.last_frame - self.start_frame + 1
        recent = self.score_hist[-window:]
        hit_ratio = len(recent) / max(min(age, window), 1)
        return float(np.mean(recent) * min(hit_ratio, 1.0)) if recent else 0.0


class ByteTrack:
    def __init__(self, high_thr=0.5, low_thr=0.1, new_thr=0.6, match_thr=0.8, low_match_thr=0.5,
                 buffer_frames=60, min_hits=3):
        self.high_thr, self.low_thr, self.new_thr = high_thr, low_thr, new_thr
        self.match_thr, self.low_match_thr = match_thr, low_match_thr
        self.buffer, self.min_hits = buffer_frames, min_hits
        self.kf = KalmanXYAH()
        self.tracks: list[Track] = []
        self._next = 1

    def _associate(self, tracks, boxes, thr):
        if not tracks or not len(boxes):
            return [], list(range(len(tracks))), list(range(len(boxes)))
        tb = np.array([t.box for t in tracks])
        cost = 1 - box_iou(tb, boxes)
        r, c = linear_sum_assignment(cost)
        matches = [(i, j) for i, j in zip(r, c) if cost[i, j] <= thr]
        mt = {i for i, _ in matches}
        md = {j for _, j in matches}
        return matches, [i for i in range(len(tracks)) if i not in mt], [j for j in range(len(boxes)) if j not in md]

    def update(self, frame: int, boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray):
        """Returns list of (tid, cls, box_xyxy, score, quality) for confirmed tracks updated this frame."""
        for t in self.tracks:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
        out = []
        for c in np.unique(np.r_[classes, [t.cls for t in self.tracks]].astype(int)) if (len(classes) or self.tracks) else []:
            cm = classes == c
            b, s = boxes[cm], scores[cm]
            hi, lo = s >= self.high_thr, (s >= self.low_thr) & (s < self.high_thr)
            pool = [t for t in self.tracks if t.cls == c and t.state in ("confirmed", "tentative", "lost")]
            m1, um_t, um_d = self._associate(pool, b[hi], self.match_thr)
            hb, hs = b[hi], s[hi]
            for i, j in m1:
                self._hit(pool[i], frame, hb[j], hs[j])
            rem = [pool[i] for i in um_t if pool[i].state != "lost"]
            lost_rem = [pool[i] for i in um_t if pool[i].state == "lost"]
            m2, um_t2, _ = self._associate(rem, b[lo], self.low_match_thr)
            lb_, ls_ = b[lo], s[lo]
            for i, j in m2:
                self._hit(rem[i], frame, lb_[j], ls_[j])
            for i in um_t2:
                t = rem[i]
                t.state = "removed" if t.state == "tentative" else "lost"
            for t in lost_rem:
                if frame - t.last_frame > self.buffer:
                    t.state = "removed"
            for j in um_d:
                if hs[j] >= self.new_thr:
                    m, cv = self.kf.initiate(xyxy_to_xyah(hb[j]))
                    self.tracks.append(Track(self._next, int(c), m, cv, float(hs[j]), frame, frame,
                                             score_hist=[float(hs[j])]))
                    self._next += 1
        self.tracks = [t for t in self.tracks if t.state != "removed"]
        for t in self.tracks:
            if t.last_frame == frame and t.state == "confirmed":
                out.append((t.tid, t.cls, t.box.copy(), t.score, t.quality()))
        return out

    def _hit(self, t: Track, frame: int, box, score):
        t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(box))
        t.score, t.last_frame = float(score), frame
        t.hits += 1
        t.score_hist.append(float(score))
        if len(t.score_hist) > 120:
            t.score_hist = t.score_hist[-120:]
        if t.state == "tentative" and t.hits >= self.min_hits:
            t.state = "confirmed"
        elif t.state == "lost":
            t.state = "confirmed"
