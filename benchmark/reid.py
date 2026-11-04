"""Light appearance features: clothing colours of the upper and lower body.

No faces and no identity: only colour histograms, kept in memory for as long as a track lives. They let
the tracker refuse to hand a track to a differently dressed person when two people cross.
"""
from __future__ import annotations

import cv2
import numpy as np

CROP = (24, 48)   # width, height


def _hist(hsv: np.ndarray) -> np.ndarray:
    return cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).ravel()


def box_features(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """(N, 64) unit vectors; the dot product of two is a similarity in [0, 1]."""
    boxes = np.asarray(boxes, np.float64).reshape(-1, 4)
    out = np.zeros((len(boxes), 64), np.float32)
    H, W = frame.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        # centre of the box: less background, and the same part of the body in every frame
        dx = 0.2 * (x2 - x1)
        a, b = max(0, int(x1 + dx)), min(W, int(x2 - dx))
        c, d = max(0, int(y1)), min(H, int(y2))
        if b - a < 2 or d - c < 2:
            continue
        hsv = cv2.cvtColor(cv2.resize(frame[c:d, a:b], CROP, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        v = np.r_[_hist(hsv[:CROP[1] // 2]), _hist(hsv[CROP[1] // 2:])]
        n = np.linalg.norm(np.sqrt(v))
        if n > 0:
            out[i] = np.sqrt(v) / n
    return out
