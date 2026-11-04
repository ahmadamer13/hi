"""Camera-motion estimation and cut detection, so walking is not confused with panning/zooming.

Motion is estimated only from background pixels: feature points inside detected people are masked
out, because a walking crowd would otherwise be mistaken for the camera moving. Top corners are
masked too, because broadcast logos never move and would vote for "no camera motion".

Rule (uncertain, never wrong way): when the estimate is unreliable, `ok` is False and the caller
must discard motion history rather than guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionStep:
    T: np.ndarray        # 3x3: maps points in the current frame to the shot's reference frame
    ok: bool             # estimate is trustworthy for this frame
    cut: bool            # a new shot started on this frame (T was reset to identity)
    inliers: int = 0


def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return pts @ T[:2, :2].T + T[:2, 2]


def transform_boxes(T: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Box corners through a similarity transform (no rotation of the box itself)."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    a, b = transform_points(T, boxes[:, :2]), transform_points(T, boxes[:, 2:])
    return np.concatenate([np.minimum(a, b), np.maximum(a, b)], 1)


class CameraMotion:
    def __init__(self, work_w: int = 640, min_inliers: int = 20, min_inlier_ratio: float = 0.5,
                 ransac_px: float = 1.5, cut_corr: float = 0.55, max_step_zoom: float = 0.15):
        self.work_w, self.min_inliers, self.min_ratio = work_w, min_inliers, min_inlier_ratio
        self.ransac_px, self.cut_corr, self.max_step_zoom = ransac_px, cut_corr, max_step_zoom
        self.reset()

    def reset(self) -> None:
        self.prev_gray = self.prev_hist = self.prev_mask = None
        self.T = np.eye(3)
        self.inlier_avg = 0.0

    def _mask(self, shape, boxes, k):
        h, w = shape
        m = np.full((h, w), 255, np.uint8)
        cw, ch = int(w * 0.14), int(h * 0.16)     # broadcast logos sit in the top corners
        m[:ch, :cw] = 0
        m[:ch, w - cw:] = 0
        for x1, y1, x2, y2 in np.asarray(boxes, np.float64).reshape(-1, 4) * k:
            dx, dy = 0.15 * (x2 - x1), 0.15 * (y2 - y1)
            m[max(0, int(y1 - dy)):max(0, int(y2 + dy)), max(0, int(x1 - dx)):max(0, int(x2 + dx))] = 0
        return m

    def update(self, frame: np.ndarray, boxes=()) -> MotionStep:
        H, W = frame.shape[:2]
        k = self.work_w / W
        small = cv2.resize(frame, (self.work_w, max(1, int(round(H * k)))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        mask = self._mask(gray.shape, boxes, k)

        if self.prev_gray is None:
            self.prev_gray, self.prev_hist, self.prev_mask = gray, hist, mask
            self.T = np.eye(3)
            return MotionStep(self.T.copy(), True, True)

        corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
        A, inl, tracked = self._estimate(self.prev_gray, gray, self.prev_mask)
        n_in = int(inl.sum()) if inl is not None else 0
        ratio = n_in / max(tracked, 1)
        # a colour jump, or points that cannot be followed at all, means a different view
        cut = corr < self.cut_corr or (tracked < 8 and corr < 0.9)
        # a cut between similar-looking crowds keeps the colours; background points then stop matching
        if self.inlier_avg > 80 and n_in < 0.25 * self.inlier_avg:
            cut = True
        self.prev_gray, self.prev_hist, self.prev_mask = gray, hist, mask
        if cut:
            self.inlier_avg = 0.0
            self.T = np.eye(3)
            return MotionStep(self.T.copy(), True, True, n_in)
        self.inlier_avg = n_in if not self.inlier_avg else 0.9 * self.inlier_avg + 0.1 * n_in
        ok = A is not None and n_in >= self.min_inliers and ratio >= self.min_ratio
        if ok:
            zoom = float(np.hypot(A[0, 0], A[1, 0]))
            ok = abs(zoom - 1) <= self.max_step_zoom
        if ok:
            step = np.eye(3)
            step[:2, :2] = A[:, :2]
            step[:2, 2] = A[:, 2] / k             # translation back to full-resolution pixels
            self.T = self.T @ np.linalg.inv(step)
        return MotionStep(self.T.copy(), ok, False, n_in)

    def _estimate(self, prev, cur, mask):
        p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 7, mask=mask)
        if p0 is None or len(p0) < 8:
            return None, None, 0
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(cur, prev, p1, None, winSize=(21, 21), maxLevel=3)
        good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.abs(p0 - p0r).reshape(-1, 2).max(1) < 1.0)
        if good.sum() < 8:
            return None, None, int(good.sum())
        A, inl = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC,
                                             ransacReprojThreshold=self.ransac_px, maxIters=500)
        return A, inl, int(good.sum())
