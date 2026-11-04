"""Cut a stratified set of 10-20 s benchmark clips from long recordings.

    python -m benchmark.finetune.sample_clips --recordings /footage/*.mp4 --n 120 --len 15 \
        --out data/ --counter wesjos-v8s-head

Recording filenames must start with an ISO date-time, e.g. 2026-06-14T05-30-00_cam1.mp4, so every
clip carries its day and time block (splits are made by day / time block, never by frame).
A quick head count from `--counter` pre-assigns a density level; annotators confirm or correct it
in manifest.csv. Lighting/weather columns are left for annotators ("?").
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
from pathlib import Path

import cv2

DENSITY_BINS = [(0, 50, "low"), (50, 200, "medium"), (200, 600, "high"), (600, 10 ** 9, "extreme")]
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})T(\d{2})")


def time_block(hour: int) -> str:
    return {0: "night", 1: "dawn", 2: "morning", 3: "midday", 4: "afternoon", 5: "evening"}[hour // 4]


def density_of(count: int) -> str:
    return next(n for lo, hi, n in DENSITY_BINS if lo <= count < hi)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--recordings", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--len", type=float, default=15.0)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--counter", default=None, help="model id from models.yaml for density pre-labels")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    counter = None
    if a.counter:
        from ..detectors import build
        from ..download import load_registry
        counter = build(next(c for c in load_registry()["candidates"] if c["id"] == a.counter), imgsz=1280, conf=0.3)

    recs = []
    for r in a.recordings:
        m = TS_RE.search(Path(r).name)
        if not m:
            raise SystemExit(f"{r}: filename must contain YYYY-MM-DDTHH")
        cap = cv2.VideoCapture(r)
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        cap.release()
        recs.append((r, m.group(1), int(m.group(2)), dur))
    # spread clips evenly over (day, time block) groups
    groups: dict = {}
    for r in recs:
        groups.setdefault((r[1], time_block(r[2])), []).append(r)
    per_group = max(1, a.n // max(len(groups), 1))
    (a.out / "clips").mkdir(parents=True, exist_ok=True)
    man = a.out / "manifest.csv"
    new = not man.exists()
    with open(man, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["clip_id", "video", "date", "time_block", "density", "lighting", "weather",
                        "has_bus", "camera_config", "split"])
        k = 0
        for (day, tb), rs in sorted(groups.items()):
            for _ in range(per_group):
                path, _, hour, dur = rng.choice(rs)
                start = rng.uniform(0, max(dur - a.len, 0))
                cid = f"{day}_{tb}_{k:04d}"
                dst = a.out / "clips" / f"{cid}.mp4"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start:.2f}", "-i", path,
                                "-t", str(a.len), "-c", "copy", str(dst)], check=True)
                dens = "?"
                if counter is not None:
                    cap = cv2.VideoCapture(str(dst))
                    ok, fr = cap.read()
                    cap.release()
                    if ok:
                        dens = density_of(len(counter(fr)))
                w.writerow([cid, f"clips/{cid}.mp4", day, tb, dens, "?", "?", "?", "camera.json", "?"])
                k += 1
    print(f"wrote {k} clips; now fill lighting/weather/has_bus, then run benchmark.finetune.split")


if __name__ == "__main__":
    main()
