"""Frame-by-frame wrong-way engine used by the web app.

Differences from the file-based preview (demo.py), all aimed at "uncertain, never wrong way":
  * camera motion is removed before direction is measured (stabilize.py);
  * motion history is thrown away whenever the camera estimate is unreliable or the view changes;
  * without calibration, people are measured at the top of their box, because crowds hide feet and
    legs (a box bottom jumping up would look like walking against the flow), and their scale comes
    from a high percentile of recent box heights, because occlusion only ever shortens a box.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detectors import CLASSES, CLS_ID
from .direction import GRAY, GREEN, RED, ClassParams, DirectionClassifier, Zone
from .ocsort import OCSort
from .stabilize import CameraMotion, transform_boxes
from .tracker import ByteTrack

REAL_SIZE_M = {"head": 0.23, "person": 1.7, "bus": 3.2}
TRACKED = tuple(CLS_ID[c] for c in REAL_SIZE_M)
COLORS = {GREEN: (119, 206, 63), RED: (66, 62, 239), GRAY: (186, 178, 169)}   # BGR
ARROW_COLOR = (28, 184, 245)


class ArrowField:
    """Allowed direction from one arrow drawn in image pixels, applied to the whole view."""

    def __init__(self, arrow, frame_shape):
        self.zones = []
        self.set_arrow(arrow, frame_shape)

    def set_arrow(self, arrow, frame_shape):
        self.arrow = arrow
        if arrow is None:
            self.zones = []
            return
        x1, y1, x2, y2 = arrow
        if math.hypot(x2 - x1, y2 - y1) < 1:
            raise ValueError("the arrow has no length")
        h, w = frame_shape[:2]
        poly = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        # image y points down, ground y points up (metres below are built with -y)
        self.zones = [Zone("view", poly, "arrow", direction_deg=math.degrees(math.atan2(-(y2 - y1), x2 - x1)))]

    def zone_at(self, img_xy, cls_name):
        return self.zones[0] if self.zones else None


@dataclass
class FrameResult:
    items: list = field(default_factory=list)     # (track_id, class_name, box_xyxy, state)
    counts: dict = field(default_factory=dict)
    new_reds: list = field(default_factory=list)  # (track_id, class_name, box_xyxy)
    cut: bool = False
    camera_ok: bool = True


class LivePipeline:
    def __init__(self, detector, arrow, frame_shape, camera_moves=True, min_disp_m=1.5, tracker="ocsort",
                 hold_mark_s=2.0):
        self.det, self.camera_moves, self.tracker_name = detector, camera_moves, tracker
        self.hold_mark_s = hold_mark_s   # a confirmed mark stays on screen long enough to be seen
        self.marks: dict[int, tuple[str, float]] = {}
        self.frame_shape = frame_shape
        self.field = ArrowField(arrow, frame_shape)
        k = min_disp_m / 1.5
        self.people = ClassParams(window_s=1.5, min_disp_m=min_disp_m, min_age_s=1.0)
        self.bus = ClassParams(window_s=4.0, min_disp_m=5.0 * k, min_age_s=3.0, confirm_red_s=1.5)
        self.motion = CameraMotion()
        self.n = 0
        self.red_ids: set[int] = set()
        self.next_id_base = 0            # keeps track ids unique across camera views
        self._new_tracker()
        self._forget_motion()

    def _new_tracker(self):
        self.tracker = OCSort() if self.tracker_name == "ocsort" else ByteTrack()

    def _forget_motion(self):
        self.dirs = DirectionClassifier(self.field, people=self.people, bus=self.bus)
        self.heights: dict[int, list] = {}

    def set_arrow(self, arrow) -> None:
        self.field.set_arrow(arrow, self.frame_shape)
        self._forget_motion()
        self.marks.clear()

    def _ground(self, tid, name, box_ref):
        hist = self.heights.setdefault(tid, [])
        size = box_ref[2] - box_ref[0] if name == "head" else box_ref[3] - box_ref[1]
        hist.append(float(size))
        del hist[:-45]
        mpp = REAL_SIZE_M[name] / max(float(np.percentile(hist, 80)), 1.0)
        x = (box_ref[0] + box_ref[2]) / 2
        if name == "person":
            y = box_ref[1] + 0.1 * (box_ref[3] - box_ref[1])   # head and shoulders stay visible
        elif name == "head":
            y = (box_ref[1] + box_ref[3]) / 2
        else:
            y = box_ref[3]                                     # bus: bottom centre
        return np.array([x * mpp, -y * mpp])

    def _shown(self, uid, state, ts):
        """Gray right after a confirmed decision keeps showing it briefly; a new decision replaces it."""
        if state != GRAY:
            self.marks[uid] = (state, ts)
            return state
        held = self.marks.get(uid)
        if held and ts - held[1] <= self.hold_mark_s:
            return held[0]
        return GRAY

    def process(self, frame: np.ndarray, ts: float) -> FrameResult:
        d = self.det(frame)
        d = d.select(np.isin(d.classes, TRACKED))
        step = self.motion.update(frame, d.boxes)
        res = FrameResult(cut=step.cut and self.n > 0)
        if res.cut:
            self.next_id_base += 100000
            self._new_tracker()
            self._forget_motion()
            self.marks.clear()
        T, ok = (step.T, step.ok) if self.camera_moves else (np.eye(3), True)
        res.camera_ok = ok
        if not ok:
            self._forget_motion()
        tracks = self.tracker.update(self.n, d.boxes, d.scores, d.classes)
        self.n += 1
        counts = {"people": Counter(), "bus": Counter()}
        for tid, c, box, _score, _quality in tracks:
            name = CLASSES[c]
            uid = self.next_id_base + tid
            state = GRAY
            if self.field.arrow is not None:
                ref = transform_boxes(T, box[None])[0]
                ax, ay = (box[0] + box[2]) / 2, box[3]
                state = self.dirs.update(ts, uid, name, (ax, ay), self._ground(uid, name, ref))
            state = self._shown(uid, state, ts)
            res.items.append((uid, name, box, state))
            counts["bus" if name == "bus" else "people"][state] += 1
            if state == RED and uid not in self.red_ids:
                self.red_ids.add(uid)
                res.new_reds.append((uid, name, box))
        res.counts = {k: {s: v[s] for s in (GREEN, RED, GRAY)} for k, v in counts.items()}
        return res


def draw_marks(frame: np.ndarray, res: FrameResult, arrow=None) -> np.ndarray:
    scale = max(frame.shape[1] / 1920, 0.5)
    th = max(1, round(2 * scale))
    # gray first so red is never hidden underneath
    for uid, name, box, state in sorted(res.items, key=lambda it: {GRAY: 0, GREEN: 1, RED: 2}[it[3]]):
        x1, y1, x2, y2 = (int(v) for v in box)
        col = COLORS[state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, th + (1 if state == RED or name == "bus" else 0), cv2.LINE_AA)
        label = f"BUS {uid % 100000}" if name == "bus" else str(uid % 100000)
        fs = 0.55 * scale
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        top = max(tht + 4, y1)
        cv2.rectangle(frame, (x1, top - tht - 4), (x1 + tw + 4, top), col, -1)
        cv2.putText(frame, label, (x1 + 2, top - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (20, 20, 20), 1, cv2.LINE_AA)
    if arrow is not None:
        a, b = (int(arrow[0]), int(arrow[1])), (int(arrow[2]), int(arrow[3]))
        L = max(math.hypot(b[0] - a[0], b[1] - a[1]), 1)
        cv2.arrowedLine(frame, a, b, (40, 40, 40), th + 5, cv2.LINE_AA, tipLength=min(0.3, 60 * scale / L))
        cv2.arrowedLine(frame, a, b, ARROW_COLOR, th + 2, cv2.LINE_AA, tipLength=min(0.3, 60 * scale / L))
    return frame
