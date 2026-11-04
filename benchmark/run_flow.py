"""Benchmark the dense-crowd flow fallback (Farneback vs RAFT-small).

    python -m benchmark.run_flow --manifest data/manifest.csv --split test --density high extreme \
        --estimators farneback raft-small --window-s 1.0 --out results/flow.json

Scores per-cell "% against" (from flow) against labelled tracks over 1 s windows. Bus GT boxes
are masked out (the live system masks detected buses).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import iter_frames, load_manifest
from .detectors import CLS_ID
from .direction import CameraConfig
from .flow import Farneback, RaftSmall, cell_against_fraction, gt_cell_against, score_cells
from .speed import device_info


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--density", nargs="*", default=None)
    ap.add_argument("--estimators", nargs="+", default=["farneback", "raft-small"])
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--window-s", type=float, default=1.0)
    ap.add_argument("--cell-px", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/flow.json")
    a = ap.parse_args(argv)
    clips = [c for c in load_manifest(a.manifest, a.split) if not a.density or c.density in a.density]
    results = {"device": device_info(), "runs": {}}
    for name in a.estimators:
        est = Farneback(a.scale) if name == "farneback" else RaftSmall(a.scale, a.device)
        agg = defaultdict(lambda: defaultdict(float))
        times = []
        for clip in clips:
            clip.load_labels()
            cam = CameraConfig.load(clip.camera_config)
            prev, prev_t, win_start, win_pred = None, None, None, defaultdict(list)
            for fi, ts, frame in iter_frames(clip.video):
                if prev is not None:
                    t0 = time.perf_counter()
                    flow, sc = est(prev, frame)
                    times.append(time.perf_counter() - t0)
                    gt = clip.gt_frame(fi)
                    bus = gt[gt[:, 1] == CLS_ID["bus"], 2:6] if len(gt) else []
                    cells = cell_against_fraction(flow, sc, cam, frame.shape, ts - prev_t, bus, a.cell_px)
                    for k, v in cells.items():
                        win_pred[k].append(v)
                if win_start is None:
                    win_start = (fi, ts)
                if ts - win_start[1] >= a.window_s:
                    pred = {k: (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v])))
                            for k, v in win_pred.items()}
                    s = score_cells(pred, gt_cell_against(clip, win_start[0], fi, cam, a.cell_px))
                    for bucket in ("all", f"density={clip.density}"):
                        for k, v in s.items():
                            agg[bucket][k] += v if k != "mae_against_frac" else v * s["cells"]
                    win_start, win_pred = (fi, ts), defaultdict(list)
                prev, prev_t = frame, ts
        out = {}
        for b, s in agg.items():
            n = max(s["cells"], 1)
            p = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else None
            r = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else None
            out[b] = {"cells": int(s["cells"]), "mae_against_frac": s["mae_against_frac"] / n,
                      "cell_precision": p, "cell_recall": r}
        out["ms_per_frame"] = float(np.mean(times) * 1000) if times else None
        results["runs"][name] = out
        print(name, json.dumps(out, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
