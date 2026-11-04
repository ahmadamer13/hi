"""Benchmark dataset format (see data/README.md).

manifest.csv columns:
    clip_id, video, date, time_block, density, lighting, weather, has_bus, camera_config, split
labels/<clip_id>/gt.csv:
    frame, track_id, class, x1, y1, x2, y2, ignore      (class in head|person|bus|car|truck)
labels/<clip_id>/frames.txt:  one frame index per line = frames that are FULLY box-annotated
labels/<clip_id>/directions.csv:
    track_id, direction      (with | against | static | cross | mixed | unsure)
labels/<clip_id>/ignore.json (optional): {"polygons": [[[x,y],...], ...]}  unlabeled crowd regions
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .detectors import CLS_ID

DENSITY_LEVELS = ("low", "medium", "high", "extreme")
DIRECTIONS = ("with", "against", "static", "cross", "mixed", "unsure")


@dataclass
class Clip:
    clip_id: str
    video: Path
    date: str
    time_block: str
    density: str
    lighting: str
    weather: str
    has_bus: bool
    camera_config: Path
    split: str
    label_dir: Path
    # loaded lazily
    gt: dict[int, np.ndarray] = field(default_factory=dict)   # frame -> (N, 7) [tid, cls, x1,y1,x2,y2, ignore]
    labeled_frames: list[int] = field(default_factory=list)
    directions: dict[int, str] = field(default_factory=dict)
    ignore_polys: list[np.ndarray] = field(default_factory=list)

    def load_labels(self) -> "Clip":
        gt_path = self.label_dir / "gt.csv"
        rows: dict[int, list] = {}
        if gt_path.exists():
            with open(gt_path) as f:
                for r in csv.DictReader(f):
                    rows.setdefault(int(r["frame"]), []).append([
                        int(r["track_id"]), CLS_ID[r["class"].strip()],
                        float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]),
                        int(r.get("ignore", 0) or 0)])
        self.gt = {k: np.array(v, np.float32) for k, v in rows.items()}
        fr = self.label_dir / "frames.txt"
        self.labeled_frames = (sorted(int(x) for x in fr.read_text().split()) if fr.exists()
                               else sorted(self.gt))
        dp = self.label_dir / "directions.csv"
        if dp.exists():
            with open(dp) as f:
                self.directions = {int(r["track_id"]): r["direction"].strip() for r in csv.DictReader(f)}
        ip = self.label_dir / "ignore.json"
        if ip.exists():
            self.ignore_polys = [np.array(p, np.float32) for p in json.loads(ip.read_text())["polygons"]]
        return self

    def gt_frame(self, frame: int) -> np.ndarray:
        return self.gt.get(frame, np.zeros((0, 7), np.float32))

    def gt_tracks(self) -> dict[int, list[tuple[int, int, np.ndarray]]]:
        """track_id -> [(frame, cls, box)] sorted by frame."""
        out: dict[int, list] = {}
        for fr in sorted(self.gt):
            for row in self.gt[fr]:
                out.setdefault(int(row[0]), []).append((fr, int(row[1]), row[2:6].copy()))
        return out


def load_manifest(path: str | Path, split: str | None = None) -> list[Clip]:
    path = Path(path)
    base = path.parent
    clips = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if split and r["split"] != split:
                continue
            clips.append(Clip(
                clip_id=r["clip_id"], video=(base / r["video"]).resolve(), date=r["date"],
                time_block=r["time_block"], density=r["density"], lighting=r["lighting"],
                weather=r["weather"], has_bus=r["has_bus"].strip().lower() in ("1", "true", "yes"),
                camera_config=(base / r["camera_config"]).resolve(), split=r["split"],
                label_dir=(base / "labels" / r["clip_id"]).resolve()))
    return clips


def check_split_leakage(clips: list[Clip]) -> list[str]:
    """Splits must be by DAY or TIME BLOCK, never random frames. Returns violations."""
    owner: dict[tuple[str, str], str] = {}
    errs = []
    for c in clips:
        key = (c.date, c.time_block)
        if key in owner and owner[key] != c.split:
            errs.append(f"{c.clip_id}: day/time-block {key} appears in both {owner[key]} and {c.split}")
        owner.setdefault(key, c.split)
    return errs


def iter_frames(video: Path, stride: int = 1):
    """Yields (frame_idx, timestamp_s, bgr). Uses container timestamps for reproducible replay."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if ts <= 0 and i > 0:
                ts = i / fps
            if i % stride == 0:
                yield i, ts, frame
            i += 1
    finally:
        cap.release()


def point_in_polys(pts: np.ndarray, polys: list[np.ndarray]) -> np.ndarray:
    if not polys or not len(pts):
        return np.zeros(len(pts), bool)
    m = np.zeros(len(pts), bool)
    for poly in polys:
        p = poly.reshape(-1, 1, 2)
        m |= np.array([cv2.pointPolygonTest(p, (float(x), float(y)), False) >= 0 for x, y in pts])
    return m
