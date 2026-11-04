"""Synthetic clip generator + oracle detector for harness self-tests (no real footage needed).

Scene: top-down-ish camera, 1 px = 0.05 m on the ground (homography is a pure scale), allowed
direction +X. People walk at ~1.3 m/s with/against; some stand; one bus drives with the flow.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from benchmark.detectors import CLS_ID, Detections, Detector

W, H, FPS, M_PER_PX = 960, 540, 25, 0.05


def make_clip(root: Path, clip_id="synth01", n_frames=125, seed=0, density="medium", date="2026-01-01",
              time_block="am", split="test"):
    rng = np.random.default_rng(seed)
    agents = []
    tid = 1
    for k in range(6):   # with-flow walkers
        agents.append(dict(tid=tid, cls="person", x=60 + 40 * k, y=80 + 60 * k, vx=+26, w=24, h=60, dir="with")); tid += 1
    for k in range(3):   # against-flow walkers
        agents.append(dict(tid=tid, cls="person", x=880 - 40 * k, y=110 + 90 * k, vx=-26, w=24, h=60, dir="against")); tid += 1
    for k in range(2):   # standing
        agents.append(dict(tid=tid, cls="person", x=480 + 60 * k, y=440, vx=0, w=24, h=60, dir="static")); tid += 1
    agents.append(dict(tid=tid, cls="bus", x=40, y=360, vx=+60, w=200, h=90, dir="with"))

    vid_dir = root / "clips"
    lab = root / "labels" / clip_id
    vid_dir.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(vid_dir / f"{clip_id}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    rows = []
    for f in range(n_frames):
        img = np.full((H, W, 3), 90, np.uint8)
        for a in agents:
            cx = a["x"] + a["vx"] * f / FPS + rng.normal(0, 0.3)
            x1, y1 = cx - a["w"] / 2, a["y"] - a["h"] / 2
            x2, y2 = x1 + a["w"], y1 + a["h"]
            if x2 < 0 or x1 > W:
                continue
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (200, 200, 200), -1)
            rows.append([f, a["tid"], a["cls"], round(max(x1, 0), 1), round(y1, 1), round(min(x2, W - 1), 1), round(y2, 1), 0])
        vw.write(img)
    vw.release()
    with open(lab / "gt.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "ignore"])
        w.writerows(rows)
    (lab / "frames.txt").write_text("\n".join(map(str, range(n_frames))))
    with open(lab / "directions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "direction"])
        for a in agents:
            w.writerow([a["tid"], a["dir"]])
    cam = {
        "calibration": {"image_points": [[0, 0], [W, 0], [W, H], [0, H]],
                        "ground_points": [[0, 0], [W * M_PER_PX, 0], [W * M_PER_PX, H * M_PER_PX], [0, H * M_PER_PX]]},
        "zones": [{"name": "all", "polygon": [[0, 0], [W, 0], [W, H], [0, H]], "mode": "arrow", "direction_deg": 0}],
        "far_roi": [0, 0, W, 200],
    }
    (root / "camera.json").write_text(json.dumps(cam))
    man = root / "manifest.csv"
    new = not man.exists()
    with open(man, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["clip_id", "video", "date", "time_block", "density", "lighting", "weather", "has_bus", "camera_config", "split"])
        w.writerow([clip_id, f"clips/{clip_id}.avi", date, time_block, density, "day", "clear", 1, "camera.json", split])
    return man


class OracleDetector(Detector):
    """Returns GT boxes with jitter / misses / false positives — exercises everything downstream."""

    def __init__(self, gt_csv: Path, jitter_px=1.0, miss=0.0, fp_per_frame=0.0, seed=0,
                 classes=("person", "bus")):
        self.rng = np.random.default_rng(seed)
        self.by_frame: dict[int, list] = {}
        with open(gt_csv) as f:
            for r in csv.DictReader(f):
                self.by_frame.setdefault(int(r["frame"]), []).append(
                    (CLS_ID[r["class"]], [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
        self.jitter, self.miss, self.fp = jitter_px, miss, fp_per_frame
        self.frame = 0
        self.classes = set(classes)
        self.name = "oracle"

    def __call__(self, img):
        items = self.by_frame.get(self.frame, [])
        self.frame += 1
        b, s, c = [], [], []
        for cls, box in items:
            if self.rng.random() < self.miss:
                continue
            b.append(np.array(box) + self.rng.normal(0, self.jitter, 4))
            s.append(0.9)
            c.append(cls)
        for _ in range(self.rng.poisson(self.fp)):
            x, y = self.rng.uniform(0, W - 30), self.rng.uniform(0, H - 60)
            b.append([x, y, x + 24, y + 60]); s.append(0.55); c.append(CLS_ID["person"])
        if not b:
            return Detections()
        return Detections(np.array(b, np.float32), np.array(s, np.float32), np.array(c, np.int32))
