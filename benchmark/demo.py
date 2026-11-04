"""PREVIEW: run wrong-way detection on your own video and get an annotated video back.

    python -m benchmark.demo --video my_clip.mp4 --direction 960,900,960,300
    python -m benchmark.demo --video my_clip.mp4 --camera configs/my_camera.json   # calibrated

--direction x1,y1,x2,y2  = an arrow in IMAGE pixels pointing the allowed way (from x1,y1 to x2,y2).
                           Open the saved first frame (<out>_frame0.jpg) to read pixel coordinates.
Without a calibration file, distances are estimated from each person's own size in the image
(head ~0.23 m wide, person ~1.7 m tall, bus ~3.2 m tall), so the 1.5 m / 5 m rules still apply
roughly with perspective. A calibrated camera.json (homography) is more accurate.

This is a PREVIEW for looking at behaviour on your footage. It is not the Phase 1 system
(no registration, alerts, dashboard) and its accuracy is not measured until Phase 0 runs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .data import iter_frames
from .detectors import CLASSES, CLS_ID, EnsembleDetector, build
from .direction import GRAY, GREEN, RED, CameraConfig, DirectionClassifier, Zone, anchor_points
from .download import load_registry
from .ocsort import OCSort
from .tracker import ByteTrack

COLORS = {GREEN: (60, 200, 60), RED: (40, 40, 230), GRAY: (170, 170, 170)}
REAL_SIZE_M = {"head": 0.23, "person": 1.7, "bus": 3.2, "car": 1.5, "truck": 3.0}  # size used for scale
VEHICLES = ("bus", "car", "truck")


class SelfScaleCamera:
    """No calibration: metres-per-pixel per track from the object's own image size (smoothed)."""

    def __init__(self, arrow, frame_shape):
        x1, y1, x2, y2 = arrow
        # image y points down; flip so the angle is a normal math angle like a ground plane
        ang = math.degrees(math.atan2(-(y2 - y1), x2 - x1))
        h, w = frame_shape[:2]
        poly = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
        self.zones = [Zone("all", poly, "arrow", direction_deg=ang)]
        self.scale: dict[int, float] = {}

    def zone_at(self, img_xy, cls_name):
        return self.zones[0]

    def to_metres(self, tid, cls_name, box, anchor):
        size_px = (box[2] - box[0]) if cls_name == "head" else (box[3] - box[1])
        mpp = REAL_SIZE_M[cls_name] / max(size_px, 1.0)
        old = self.scale.get(tid)
        mpp = mpp if old is None else 0.95 * old + 0.05 * mpp  # slow update: box jitter != motion
        self.scale[tid] = mpp
        return np.array([anchor[0] * mpp, -anchor[1] * mpp])


def parse_arrow(s):
    v = [float(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("--direction needs x1,y1,x2,y2")
    return v


def draw(frame, items, counts, arrow, fps, mode_note):
    for tid, cls_name, box, state in items:
        x1, y1, x2, y2 = map(int, box)
        c = COLORS[state]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2 if cls_name != "bus" else 3)
        label = f"BUS {tid}" if cls_name in VEHICLES else str(tid)
        cv2.putText(frame, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1, cv2.LINE_AA)
    if arrow is not None:
        a = tuple(map(int, arrow[:2])), tuple(map(int, arrow[2:]))
        cv2.arrowedLine(frame, a[0], a[1], (255, 255, 0), 4, tipLength=0.08)
    p = counts["people"]
    tot = p[GREEN] + p[RED] + p[GRAY]
    wrong_pct = 100 * p[RED] / max(p[GREEN] + p[RED], 1)
    lines = [f"PEOPLE {tot}  correct {p[GREEN]}  WRONG {p[RED]} ({wrong_pct:.0f}%)  unsure {p[GRAY]}",
             f"BUSES  correct {counts['bus'][GREEN]}  WRONG {counts['bus'][RED]}  unsure {counts['bus'][GRAY]}",
             f"{fps:.1f} FPS  |  PREVIEW - not validated  |  {mode_note}"]
    cv2.rectangle(frame, (0, 0), (min(frame.shape[1], 760), 70), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(frame, t, (8, 20 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def run(video, det, cam, tracker, out_path, arrow=None, stride=1, max_seconds=None, log=print):
    """Returns summary dict. `cam` is CameraConfig or SelfScaleCamera."""
    dircls = DirectionClassifier(cam)
    writer, events, t_start = None, {}, time.perf_counter()
    first_red, n = {}, 0
    for fi, ts, frame in iter_frames(video, stride=stride):
        if max_seconds and ts > max_seconds:
            break
        t0 = time.perf_counter()
        d = det(frame)
        tracks = tracker.update(fi, d.boxes, d.scores, d.classes)
        items = []
        if tracks:
            tb = np.array([t[2] for t in tracks])
            tc = np.array([t[1] for t in tracks])
            anchors = anchor_points(tb, tc)
            if isinstance(cam, CameraConfig):
                ground = cam.ground(anchors, tc == CLS_ID["head"])
            for k, (tid, c, box, _s, _q) in enumerate(tracks):
                name = CLASSES[c]
                g = ground[k] if isinstance(cam, CameraConfig) else cam.to_metres(tid, name, box, anchors[k])
                st = dircls.update(ts, tid, name, anchors[k], g)
                items.append((tid, name, box, st))
                if st == RED and tid not in first_red:
                    first_red[tid] = ts
                    events[tid] = {"track_id": tid, "class": name, "first_red_s": round(ts, 2),
                                   "x": int(anchors[k][0]), "y": int(anchors[k][1])}
        counts = {"people": Counter(), "bus": Counter()}
        for _, name, _, st in items:
            counts["bus" if name in VEHICLES else "people"][st] += 1
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        draw(frame, items, counts, arrow, fps, "calibrated" if isinstance(cam, CameraConfig) else "no calibration (self-scale)")
        if writer is None:
            src_fps = cv2.VideoCapture(str(video)).get(cv2.CAP_PROP_FPS) or 25
            writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), src_fps / stride,
                                     (frame.shape[1], frame.shape[0]))
        writer.write(frame)
        n += 1
        if n % 50 == 0:
            log(f"  {ts:6.1f} s  {fps:4.1f} FPS  tracks={len(items)}  wrong-way so far={len(first_red)}")
    if writer:
        writer.release()
    return {"frames": n, "seconds_processed": round(time.perf_counter() - t_start, 1),
            "wrong_way_tracks": len(first_red), "events": list(events.values())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--direction", type=parse_arrow, help="allowed-direction arrow x1,y1,x2,y2 in image px")
    g.add_argument("--camera", type=Path, help="calibrated camera config JSON (see configs/camera.example.json)")
    g.add_argument("--first-frame-only", action="store_true", help="just save frame 0 to read coordinates")
    ap.add_argument("--people-model", default="wesjos-11n-head", help="id from benchmark/models.yaml")
    ap.add_argument("--bus-model", default="yolo11s-coco", help="id, or 'none'")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--tracker", choices=["bytetrack", "ocsort"], default="ocsort")
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (2 = twice as fast)")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--device", default=None, help="e.g. 0 for the first GPU; default CPU/auto")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    if not a.video.exists():
        raise SystemExit(f"video not found: {a.video}")
    out = a.out or a.video.with_name(a.video.stem + "_wrongway.mp4")

    _, _, f0 = next(iter_frames(a.video))
    frame0 = out.with_name(out.stem + "_frame0.jpg")
    cv2.imwrite(str(frame0), f0)
    print(f"first frame saved: {frame0}  ({f0.shape[1]}x{f0.shape[0]} px)")
    if a.first_frame_only:
        return

    reg = {c["id"]: c for c in load_registry()["candidates"]}
    people = build(reg[a.people_model], imgsz=a.imgsz, device=a.device, conf=0.1)
    keep = ("head", "person") if a.bus_model == "none" else ("head", "person", "bus")
    members = [(people, tuple(c for c in people.classes if c in keep))]
    if a.bus_model != "none" and "bus" not in people.classes:
        members.append((build(reg[a.bus_model], imgsz=a.imgsz, device=a.device, conf=0.1), ("bus",)))
    det = EnsembleDetector(members)
    cam = CameraConfig.load(a.camera) if a.camera else SelfScaleCamera(a.direction, f0.shape)
    tracker = OCSort() if a.tracker == "ocsort" else ByteTrack()
    print(f"models: {det.name} | tracker: {a.tracker} | license note: {reg[a.people_model]['license']}")
    summary = run(a.video, det, cam, tracker, out, arrow=a.direction, stride=a.stride, max_seconds=a.max_seconds)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["track_id", "class", "first_red_s", "x", "y"])
        w.writeheader()
        w.writerows(summary["events"])
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\ndone: {summary['frames']} frames in {summary['seconds_processed']} s")
    print(f"wrong-way tracks: {summary['wrong_way_tracks']}  (list: {out.with_suffix('.csv')})")
    print(f"annotated video:  {out}")


if __name__ == "__main__":
    main()
