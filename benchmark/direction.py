"""Ground-plane geometry and the reference direction classifier used to score detectors.

This is the minimum needed to measure "direction accuracy" in Phase 0. Phase 1 extends it
(multi-plane, schedules, group vote, bus/person interaction, registration).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detectors import CLS_ID

GREEN, RED, GRAY = "green", "red", "gray"


# ----------------------------------------------------------------------------- calibration
def fit_homography(image_pts, ground_pts) -> np.ndarray:
    ip, gp = np.asarray(image_pts, np.float64), np.asarray(ground_pts, np.float64)
    if len(ip) < 4:
        raise ValueError("need >= 4 point correspondences")
    H, _ = cv2.findHomography(ip, gp, cv2.RANSAC if len(ip) > 4 else 0, 0.5)
    if H is None:
        raise ValueError("degenerate calibration points")
    return H


def to_ground(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if not len(pts):  # cv2 returns None for empty input (e.g. everything masked by a bus)
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def reprojection_error_m(H, image_pts, ground_pts) -> float:
    return float(np.linalg.norm(to_ground(H, image_pts) - np.asarray(ground_pts), axis=1).mean())


def anchor_points(boxes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Feet (bottom-centre) for person and vehicles; box centre for heads (head-height plane)."""
    a = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]], 1)
    head = classes == CLS_ID["head"]
    a[head, 1] = (boxes[head, 1] + boxes[head, 3]) / 2
    return a


# ----------------------------------------------------------------------------- direction field
@dataclass
class Zone:
    name: str
    polygon: np.ndarray                  # image px
    mode: str                            # arrow | polyline | circular | both | exclude
    classes: tuple[str, ...] = ("head", "person", "bus", "car", "truck")
    direction_deg: float = 0.0           # arrow: ground-plane angle, 0 = +X, CCW positive
    polyline: np.ndarray | None = None   # ground m, ordered in the allowed direction
    center: np.ndarray | None = None     # circular: ground m
    clockwise: bool = False

    def contains(self, xy) -> bool:
        return cv2.pointPolygonTest(self.polygon.reshape(-1, 1, 2), (float(xy[0]), float(xy[1])), False) >= 0

    def allowed(self, g: np.ndarray) -> np.ndarray | None:
        if self.mode == "arrow":
            r = math.radians(self.direction_deg)
            return np.array([math.cos(r), math.sin(r)])
        if self.mode == "polyline":
            p = self.polyline
            seg = p[1:] - p[:-1]
            t = np.clip(((g - p[:-1]) * seg).sum(1) / np.maximum((seg ** 2).sum(1), 1e-9), 0, 1)
            proj = p[:-1] + seg * t[:, None]
            k = int(np.argmin(np.linalg.norm(proj - g, axis=1)))
            return seg[k] / max(np.linalg.norm(seg[k]), 1e-9)
        if self.mode == "circular":
            r = g - self.center
            n = np.linalg.norm(r)
            if n < 1e-6:
                return None
            tan = np.array([-r[1], r[0]]) / n  # CCW tangent in a right-handed ground frame
            return -tan if self.clockwise else tan
        return None  # both / exclude


@dataclass
class CameraConfig:
    H: np.ndarray
    H_head: np.ndarray | None
    zones: list[Zone]
    far_roi: tuple[int, int, int, int] | None = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "CameraConfig":
        d = json.loads(Path(path).read_text())
        cal = d["calibration"]
        H = fit_homography(cal["image_points"], cal["ground_points"])
        H_head = None
        if cal.get("head_plane"):
            hp = cal["head_plane"]
            H_head = fit_homography(hp["image_points"], hp["ground_points"])
        zones = []
        for z in d.get("zones", []):
            zones.append(Zone(
                name=z["name"], polygon=np.array(z["polygon"], np.float32), mode=z["mode"],
                classes=tuple(z.get("classes", Zone.classes)), direction_deg=z.get("direction_deg", 0.0),
                polyline=np.array(z["polyline_ground"], np.float64) if z.get("polyline_ground") else None,
                center=np.array(z["center_ground"], np.float64) if z.get("center_ground") else None,
                clockwise=z.get("clockwise", False)))
        far = tuple(d["far_roi"]) if d.get("far_roi") else None
        return cls(H, H_head, zones, far, d)

    def zone_at(self, img_xy, cls_name: str) -> Zone | None:
        for z in self.zones:  # first match wins; put exclusions first
            if cls_name in z.classes and z.contains(img_xy):
                return z
        return None

    def ground(self, img_pts: np.ndarray, is_head: np.ndarray) -> np.ndarray:
        g = to_ground(self.H, img_pts) if len(img_pts) else np.zeros((0, 2))
        if self.H_head is not None and is_head.any():
            g[is_head] = to_ground(self.H_head, img_pts[is_head])
        return g


# ----------------------------------------------------------------------------- classifier
@dataclass
class ClassParams:
    window_s: float
    min_disp_m: float
    min_age_s: float
    green_deg: float = 60.0
    red_deg: float = 120.0
    confirm_red_s: float = 0.6     # candidate must persist this long before RED
    confirm_green_s: float = 0.3
    release_s: float = 0.5         # time of contrary evidence to leave a state
    min_straightness: float = 0.0  # net distance / path walked in the window (jitter scores low)


PEOPLE = ClassParams(window_s=1.5, min_disp_m=1.5, min_age_s=1.0)
BUS = ClassParams(window_s=4.0, min_disp_m=5.0, min_age_s=3.0, confirm_red_s=1.5)


class GroundKalman:
    """2-D constant-velocity Kalman in metres (smooths detector jitter)."""

    def __init__(self, xy, q=2.0, r=0.15):
        self.x = np.r_[xy, 0.0, 0.0]
        self.P = np.diag([r, r, 4.0, 4.0])
        self.q, self.r = q, r

    def step(self, xy, dt):
        dt = max(dt, 1e-3)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q
        Hm = np.eye(2, 4)
        S = Hm @ self.P @ Hm.T + np.eye(2) * self.r ** 2
        K = self.P @ Hm.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(xy) - Hm @ self.x)
        self.P = (np.eye(4) - K @ Hm) @ self.P
        return self.x[:2].copy()


@dataclass
class TrackDirState:
    params: ClassParams
    first_t: float
    hist: list = field(default_factory=list)  # (t, smoothed xy)
    kf: GroundKalman | None = None
    state: str = GRAY
    cand: str = GRAY
    cand_since: float = 0.0
    first_red_t: float | None = None


class DirectionClassifier:
    def __init__(self, cam: CameraConfig, people: ClassParams = PEOPLE, bus: ClassParams = BUS):
        self.cam, self.people, self.bus = cam, people, bus
        self.tracks: dict[int, TrackDirState] = {}

    def _params(self, cls_name):
        return self.bus if cls_name in ("bus", "car", "truck") else self.people

    def update(self, t: float, tid: int, cls_name: str, img_anchor, ground_xy) -> str:
        st = self.tracks.get(tid)
        if st is None:
            st = self.tracks[tid] = TrackDirState(self._params(cls_name), t, kf=GroundKalman(ground_xy))
            st.cand_since = t
        dt = t - st.hist[-1][0] if st.hist else 0.0
        st.hist.append((t, st.kf.step(ground_xy, dt) if st.hist else np.asarray(ground_xy, float)))
        p = st.params
        while st.hist and st.hist[0][0] < t - p.window_s - 0.5:
            st.hist.pop(0)
        cand = self._candidate(st, t, cls_name, img_anchor)
        if cand != st.cand:
            st.cand, st.cand_since = cand, t
        held = t - st.cand_since
        if st.state != st.cand:
            need = {RED: p.confirm_red_s, GREEN: p.confirm_green_s, GRAY: p.release_s}[st.cand]
            if held >= need:
                st.state = st.cand
                if st.state == RED and st.first_red_t is None:
                    st.first_red_t = t
        return st.state

    def _candidate(self, st, t, cls_name, img_anchor) -> str:
        p = st.params
        if t - st.first_t < p.min_age_s:
            return GRAY
        zone = self.cam.zone_at(img_anchor, cls_name)
        if zone is None or zone.mode == "exclude":
            return GRAY
        past = [h for h in st.hist if h[0] <= t - p.window_s]
        if not past:
            return GRAY
        p0, p1 = past[-1][1], st.hist[-1][1]
        d = p1 - p0
        dist = float(np.linalg.norm(d))
        if dist < p.min_disp_m:
            return GRAY
        if p.min_straightness > 0:
            pts = [h[1] for h in st.hist if h[0] >= past[-1][0]]
            walked = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))
            if dist / max(walked, 1e-6) < p.min_straightness:
                return GRAY
        if zone.mode == "both":
            return GREEN
        a = zone.allowed(p1)
        if a is None:
            return GRAY
        ang = math.degrees(math.acos(float(np.clip(np.dot(d / dist, a), -1, 1))))
        if ang < p.green_deg:
            return GREEN
        if ang > p.red_deg:
            return RED
        return GRAY
