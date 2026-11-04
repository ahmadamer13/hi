"""OC-SORT (Cao et al., "Observation-Centric SORT", CVPR 2023), written from the paper.

Three additions over SORT, all kept here:
  ORU  observation-centric re-update: after an occlusion, the Kalman state is re-run along a
       virtual straight trajectory between the last and the new observation (kills drift).
  OCM  observation-centric momentum: association cost adds a direction-consistency term
       (track velocity from its own past observations vs. direction to the candidate detection).
       Useful here, because pilgrims in a stream move in coherent directions.
  OCR  observation-centric recovery: unmatched tracks get a second chance against their LAST
       OBSERVATION box, not the Kalman prediction.
Optional BYTE pass associates low-score detections, as in the authors' release.

Same output interface as tracker.ByteTrack: list of (tid, cls, box_xyxy, score, quality).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics_det import box_iou


def _z(b):  # xyxy -> [cx, cy, s(area), r(aspect)]
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)])


def _xyxy(x):
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / max(w, 1e-6)
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class _KF:
    """SORT constant-velocity Kalman on [cx, cy, s, r, vcx, vcy, vs]."""

    def __init__(self, z):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1
        self.H = np.eye(4, 7)
        self.R = np.diag([1, 1, 10, 10.0])
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 1e-4])
        self.x = np.r_[z, 0, 0, 0].astype(float)
        self.P = np.diag([10, 10, 10, 10, 1e4, 1e4, 1e4]).astype(float)

    def predict(self):
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def state(self):
        return self.x.copy(), self.P.copy()

    def restore(self, st):
        self.x, self.P = st[0].copy(), st[1].copy()


class _Track:
    def __init__(self, tid, cls, box, score, frame, delta_t):
        self.tid, self.cls, self.delta_t = tid, cls, delta_t
        self.kf = _KF(_z(box))
        self.obs: dict[int, np.ndarray] = {frame: box.copy()}   # frame -> observed box
        self.last_obs = box.copy()
        self.last_obs_frame = frame
        self.frozen = None            # KF state at last observation (for ORU)
        self.time_since_update = 0
        self.hits, self.score, self.start = 1, float(score), frame
        self.score_hist = [float(score)]
        self.velocity = np.zeros(2)   # unit direction from observations (OCM)

    @property
    def box(self):
        return _xyxy(self.kf.x)

    def predict(self):
        if self.time_since_update == 0:
            self.frozen = self.kf.state()
        self.kf.predict()
        self.time_since_update += 1

    def update(self, box, score, frame):
        if self.time_since_update > 1 and self.frozen is not None:  # ORU: re-run along a virtual path
            self.kf.restore(self.frozen)
            gap = frame - self.last_obs_frame
            for k in range(1, gap):
                virt = self.last_obs + (box - self.last_obs) * (k / gap)
                self.kf.predict()
                self.kf.update(_z(virt))
            self.kf.predict()
        # velocity direction from the observation ~delta_t frames back
        prev = None
        for dt in range(self.delta_t, 0, -1):
            if frame - dt in self.obs:
                prev = self.obs[frame - dt]
                break
        prev = self.last_obs if prev is None else prev
        c0, c1 = (prev[:2] + prev[2:]) / 2, (box[:2] + box[2:]) / 2
        d = c1 - c0
        n = np.linalg.norm(d)
        if n > 1e-6:
            self.velocity = d / n
        self.kf.update(_z(box))
        self.obs[frame] = box.copy()
        for f in [f for f in self.obs if f < frame - 2 * self.delta_t]:
            del self.obs[f]
        self.last_obs, self.last_obs_frame = box.copy(), frame
        self.time_since_update = 0
        self.hits += 1
        self.score = float(score)
        self.score_hist = (self.score_hist + [float(score)])[-120:]

    def quality(self, window=30):
        age = self.last_obs_frame - self.start + 1
        recent = self.score_hist[-window:]
        return float(np.mean(recent) * min(len(recent) / max(min(age, window), 1), 1.0))


def _direction_cost(tracks, dets, inertia):
    """OCM term: cosine between each track's velocity and the direction (last obs -> detection)."""
    if not tracks or not len(dets):
        return np.zeros((len(tracks), len(dets)))
    vel = np.array([t.velocity for t in tracks])                         # (T,2)
    lo = np.array([(t.last_obs[:2] + t.last_obs[2:]) / 2 for t in tracks])
    dc = (dets[:, :2] + dets[:, 2:]) / 2                                 # (D,2)
    d = dc[None] - lo[:, None]                                           # (T,D,2)
    d /= np.maximum(np.linalg.norm(d, axis=2, keepdims=True), 1e-6)
    cos = np.clip((d * vel[:, None]).sum(2), -1, 1)
    diff = (np.pi / 2 - np.abs(np.arccos(cos))) / np.pi                  # in [-0.5, 0.5]
    valid = (np.linalg.norm(vel, axis=1) > 0)[:, None]
    return inertia * diff * valid


def _assign(cost_iou, extra, thr):
    if cost_iou.size == 0:
        return [], list(range(cost_iou.shape[0])), list(range(cost_iou.shape[1]))
    r, c = linear_sum_assignment(-(cost_iou + extra))
    m = [(i, j) for i, j in zip(r, c) if cost_iou[i, j] >= thr]
    mi, mj = {i for i, _ in m}, {j for _, j in m}
    return m, [i for i in range(cost_iou.shape[0]) if i not in mi], [j for j in range(cost_iou.shape[1]) if j not in mj]


class OCSort:
    def __init__(self, det_thr=0.5, low_thr=0.1, iou_thr=0.3, max_age=60, min_hits=3, delta_t=3,
                 inertia=0.2, use_byte=True, new_thr=0.6):
        self.det_thr, self.low_thr, self.iou_thr = det_thr, low_thr, iou_thr
        self.max_age, self.min_hits, self.delta_t = max_age, min_hits, delta_t
        self.inertia, self.use_byte, self.new_thr = inertia, use_byte, new_thr
        self.tracks: list[_Track] = []
        self._next = 1

    def update(self, frame, boxes, scores, classes):
        for t in self.tracks:
            t.predict()
        out = []
        cls_ids = set(np.asarray(classes, int).tolist()) | {t.cls for t in self.tracks}
        for c in cls_ids:
            cm = classes == c
            b, s = boxes[cm], scores[cm]
            hi, lo = s >= self.det_thr, (s >= self.low_thr) & (s < self.det_thr)
            hb, hs, lb, ls = b[hi], s[hi], b[lo], s[lo]
            trk = [t for t in self.tracks if t.cls == c]
            tb = np.array([t.box for t in trk]).reshape(-1, 4)
            # 1) IoU + OCM
            iou = box_iou(tb, hb)
            m, ut, ud = _assign(iou, _direction_cost(trk, hb, self.inertia) * hs[None] if len(hb) else iou, self.iou_thr)
            for i, j in m:
                trk[i].update(hb[j], hs[j], frame)
            # 2) BYTE: remaining tracks vs low-score detections
            if self.use_byte and ut and len(lb):
                rem = [trk[i] for i in ut]
                iou2 = box_iou(np.array([t.box for t in rem]), lb)
                m2, ut2, _ = _assign(iou2, np.zeros_like(iou2), self.iou_thr)
                for i, j in m2:
                    rem[i].update(lb[j], ls[j], frame)
                ut = [ut[i] for i in ut2]
            # 3) OCR: last observations vs remaining high-score detections
            if ut and ud:
                rem_t = [trk[i] for i in ut]
                rem_d = hb[ud]
                iou3 = box_iou(np.array([t.last_obs for t in rem_t]), rem_d)
                m3, _, ud3 = _assign(iou3, np.zeros_like(iou3), self.iou_thr)
                for i, j in m3:
                    rem_t[i].update(rem_d[j], hs[ud[j]], frame)
                ud = [ud[j] for j in ud3]
            for j in ud:
                if hs[j] >= self.new_thr:
                    self.tracks.append(_Track(self._next, int(c), hb[j], hs[j], frame, self.delta_t))
                    self._next += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        for t in self.tracks:
            if t.time_since_update == 0 and t.hits >= self.min_hits:
                out.append((t.tid, t.cls, t.box.copy(), t.score, t.quality()))
        return out
