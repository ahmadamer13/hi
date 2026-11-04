"""Phase 0 benchmark runner.

    python -m benchmark.run --manifest data/manifest.csv --split test \
        --models all --sizes 640 960 1280 --tiled --device cuda:0 --half --trt \
        --out results/

For every (model, input size, tiled?) it runs detection on every frame of every clip, scores
detection on the labelled frames, runs ByteTrack + the reference direction classifier, scores
tracking and direction, measures speed, and writes results/<run_id>.json. `benchmark.report`
turns those files into the comparison table.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import load_manifest, check_split_leakage, iter_frames, point_in_polys
from .detectors import CLASSES, CLS_ID, TiledDetector, EnsembleDetector, build
from .direction import CameraConfig, DirectionClassifier, anchor_points
from .download import load_registry
from .metrics_det import ClassAccumulator, PointAccumulator
from .metrics_track import TrackEval
from .speed import device_info, gpu_mem_used_mb, measure
from .ocsort import OCSort
from .tracker import ByteTrack

TRACKERS = {
    "bytetrack": lambda: ByteTrack(high_thr=0.5, low_thr=0.1, new_thr=0.6, buffer_frames=60),
    "ocsort": lambda: OCSort(det_thr=0.5, low_thr=0.1, new_thr=0.6, max_age=60, delta_t=3, inertia=0.2),
}

PEOPLE_CLS = (CLS_ID["head"], CLS_ID["person"])
VEH_CLS = (CLS_ID["bus"],)


def evaluate_detector(det, clips, args, run_meta) -> dict:
    by_bucket: dict[str, dict[int, ClassAccumulator]] = defaultdict(
        lambda: {c: ClassAccumulator() for c in range(len(CLASSES))})
    loc_bucket: dict[str, dict[int, PointAccumulator]] = defaultdict(
        lambda: {c: PointAccumulator() for c in range(len(CLASSES))})
    track_people: dict[str, list] = defaultdict(list)
    track_bus: dict[str, list] = defaultdict(list)
    dir_people, dir_bus = defaultdict(list), defaultdict(list)
    latencies = defaultdict(list)  # bucket -> first-red latency in s for GT 'against'
    det_times = []
    first_frames = []

    for clip in clips:
        clip.load_labels()
        cam = CameraConfig.load(clip.camera_config)
        tracker = TRACKERS[getattr(args, "tracker_name", "bytetrack")]()
        dircls = DirectionClassifier(cam)
        # point models (P2PNet/PET) are matched by point-in-box; IoU would unfairly reject their small boxes
        te_p, te_b = TrackEval(match="point" if getattr(det, "point_model", False) else "iou"), TrackEval()
        labeled = set(clip.labeled_frames)
        gt_first_t: dict[int, float] = {}
        pred_first_red: dict[int, float] = {}
        frame_ts = {}
        for fi, ts, frame in iter_frames(clip.video):
            frame_ts[fi] = ts
            if len(first_frames) < 20:
                first_frames.append(frame)
            t0 = time.perf_counter()
            d = det(frame)
            det_times.append(time.perf_counter() - t0)

            # ---------------------------------------------------------- detection metrics
            if fi in labeled:
                gt = clip.gt_frame(fi)
                centres = (d.boxes[:, :2] + d.boxes[:, 2:]) / 2 if len(d) else np.zeros((0, 2))
                ign_mask = point_in_polys(centres, clip.ignore_polys)
                for c in det.classes:
                    ci = CLS_ID[c]
                    pm = d.classes == ci
                    gm = gt[:, 1] == ci if len(gt) else np.zeros(0, bool)
                    for bucket in ("all", f"density={clip.density}", f"light={clip.lighting}",
                                   f"weather={clip.weather}"):
                        g_boxes = gt[gm, 2:6] if len(gt) else np.zeros((0, 4))
                        g_ign = gt[gm, 6] if len(gt) else None
                        by_bucket[bucket][ci].add(d.boxes[pm], d.scores[pm], g_boxes, g_ign, ign_mask[pm])
                        # localisation: native points for point models, box centres otherwise
                        pts = getattr(d, "points", centres)
                        keep = pm & ~ign_mask
                        loc_bucket[bucket][ci].add(pts[keep], d.scores[keep], g_boxes, g_ign)

            # ---------------------------------------------------------- tracking + direction
            tracks = tracker.update(fi, d.boxes, d.scores, d.classes)
            ids, boxes, clss, states = [], [], [], []
            if tracks:
                tb = np.array([t[2] for t in tracks])
                tc = np.array([t[1] for t in tracks])
                anchors = anchor_points(tb, tc)
                ground = cam.ground(anchors, tc == CLS_ID["head"])
                for (tid, c, box, _s, _q), a, g in zip(tracks, anchors, ground):
                    st = dircls.update(ts, tid, CLASSES[c], a, g)
                    if st == "red" and tid not in pred_first_red:
                        pred_first_red[tid] = ts
                    ids.append(tid), boxes.append(box), clss.append(c), states.append(st)
            if fi in labeled:
                gt = clip.gt_frame(fi)
                for te, cls_set in ((te_p, PEOPLE_CLS), (te_b, VEH_CLS)):
                    gm = np.isin(gt[:, 1], cls_set) & (gt[:, 6] == 0) if len(gt) else np.zeros(0, bool)
                    pm = np.isin(np.array(clss, int), cls_set) if clss else np.zeros(0, bool)
                    # score each class family only against the GT class this model predicts
                    pred_cls_present = {CLASSES[c] for c in cls_set} & det.classes
                    if not pred_cls_present:
                        continue
                    gm &= np.isin(gt[:, 1], [CLS_ID[c] for c in pred_cls_present])
                    for row in gt[gm]:
                        gt_first_t.setdefault(int(row[0]), ts)
                    te.add(gt[gm, 0].astype(int), gt[gm, 2:6], np.array(ids)[pm] if len(ids) else [],
                           np.array(boxes)[pm] if len(boxes) else np.zeros((0, 4)),
                           [s for s, k in zip(states, pm) if k])

        dp = te_p.direction(clip.directions)
        db = te_b.direction(clip.directions)
        for bucket in ("all", f"density={clip.density}", f"light={clip.lighting}", f"weather={clip.weather}"):
            dir_people[bucket].append(dp)
            dir_bus[bucket].append(db)
            track_people[bucket].append(te_p)
            track_bus[bucket].append(te_b)
        # decision latency: GT 'against' track first seen -> its best pred track first confirmed red
        best = {}
        for (g, p), n in te_p.pair_frames.items():
            if n > best.get(g, (None, 0))[1]:
                best[g] = (p, n)
        for g, (p, _) in best.items():
            if clip.directions.get(g) == "against" and p in pred_first_red and g in gt_first_t:
                latencies[f"density={clip.density}"].append(pred_first_red[p] - gt_first_t[g])
        run_meta.setdefault("clips", []).append({"clip": clip.clip_id, "people": {**te_p.summary(), **dp},
                                                 "bus": {**te_b.summary(), **db}})

    def agg_dir(rows):
        keys = ("ww_tp", "ww_fp", "ww_fn", "ww_fp_unmatched_track", "n_moving_gt_tracks")
        s = {k: sum(r[k] for r in rows) for k in keys}
        n_dec = sum(r["decided_fraction"] * r["n_moving_gt_tracks"] for r in rows)
        n_cor = sum((r["direction_accuracy"] or 0) * r["decided_fraction"] * r["n_moving_gt_tracks"] for r in rows)
        p = s["ww_tp"] / (s["ww_tp"] + s["ww_fp"]) if s["ww_tp"] + s["ww_fp"] else None
        rc = s["ww_tp"] / (s["ww_tp"] + s["ww_fn"]) if s["ww_tp"] + s["ww_fn"] else None
        f1 = 2 * p * rc / max(p + rc, 1e-9) if p is not None and rc is not None else None
        return {**s, "direction_accuracy": n_cor / n_dec if n_dec else None,
                "decided_fraction": n_dec / max(s["n_moving_gt_tracks"], 1),
                "ww_precision": p, "ww_recall": rc, "ww_f1": f1}

    def agg_track(tes):
        tp, fp, fn, sw = (sum(getattr(t, k) for t in tes) for k in ("tp", "fp", "fn", "idsw"))
        n_tracks = sum(len(t.gt_frames) for t in tes)
        idf1 = [t.summary()["IDF1"] for t in tes if t.gt_frames]
        return {"MOTA": 1 - (fn + fp + sw) / max(tp + fn, 1), "IDSW": sw, "n_gt_tracks": n_tracks,
                "IDSW_per_100_gt_tracks": 100 * sw / max(n_tracks, 1),
                "IDF1_mean_over_clips": float(np.mean(idf1)) if idf1 else None,
                "HOTA_mean_over_clips": float(np.mean([t.hota()["HOTA"] for t in tes if t.gt_frames]))
                if any(t.gt_frames for t in tes) else None}

    det_summary = {b: {CLASSES[c]: {**acc.summary(args.op_thr), **loc_bucket[b][c].summary(args.op_thr)}
                       for c, acc in accs.items() if CLASSES[c] in det.classes} for b, accs in by_bucket.items()}
    return {
        "detection": det_summary,
        "tracking_people": {b: agg_track(t) for b, t in track_people.items()},
        "tracking_bus": {b: agg_track(t) for b, t in track_bus.items()},
        "direction_people": {b: agg_dir(r) for b, r in dir_people.items()},
        "direction_bus": {b: agg_dir(r) for b, r in dir_bus.items()},
        "decision_latency_s": {b: {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "n": len(v)}
                               for b, v in latencies.items() if v},
        "in_loop_det_ms_mean": float(np.mean(det_times) * 1000) if det_times else None,
        "_frames": first_frames,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", default=["all"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 960, 1280])
    ap.add_argument("--tiled", action="store_true", help="also run full-frame@640 + far-ROI tiles")
    ap.add_argument("--tile", type=int, default=640)
    ap.add_argument("--bus-model", default="yolo11s-coco",
                    help="COCO model used for buses when the candidate has no bus class")
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--trt", action="store_true", help="ONNX models via TensorRT EP (FP16 with --half)")
    ap.add_argument("--conf", type=float, default=0.05, help="detector output threshold (low, for AP)")
    ap.add_argument("--op-thr", type=float, default=0.4, help="operating threshold for FP/frame")
    ap.add_argument("--trackers", nargs="+", default=["bytetrack", "ocsort"], choices=sorted(TRACKERS))
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)

    clips = load_manifest(args.manifest, args.split)
    if not clips:
        raise SystemExit("no clips for split " + args.split)
    leaks = check_split_leakage(load_manifest(args.manifest))
    if leaks:
        raise SystemExit("split leakage (split by day/time block):\n" + "\n".join(leaks))
    reg = {c["id"]: c for c in load_registry()["candidates"] if c.get("enabled", True) is not False}
    ids = list(reg) if args.models == ["all"] else args.models
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = device_info()

    for mid in ids:
        cand = reg[mid]
        configs = [(s, False) for s in args.sizes] + ([(640, True)] if args.tiled else [])
        for size, tiled in configs:
            run_id = f"{mid}@{size}{'+tiled' if tiled else ''}"
            mem0 = gpu_mem_used_mb()
            try:
                det = build(cand, imgsz=size, device=args.device, half=args.half, conf=args.conf, trt=args.trt)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {run_id}: {e}")
                continue
            if getattr(det, "static_size", None) and det.static_size != size and not tiled:
                print(f"[skip] {run_id}: model has a static {det.static_size}px input")
                continue
            if tiled:
                cam = CameraConfig.load(clips[0].camera_config)
                if not cam.far_roi:
                    print(f"[skip] {run_id}: camera config has no far_roi")
                    continue
                det = TiledDetector(det, cam.far_roi, tile=args.tile)
            if "bus" not in det.classes and args.bus_model in reg:
                bus = build(reg[args.bus_model], imgsz=size, device=args.device, half=args.half, conf=args.conf, trt=args.trt)
                det = EnsembleDetector([(det, tuple(det.classes)), (bus, ("bus",))])
            meta = {"run_id": run_id, "model": mid, "imgsz": size, "tiled": tiled, "license": cand["license"],
                    "license_risk": cand["risk"], "device": dev, "half": args.half, "trt": args.trt,
                    "split": args.split, "n_clips": len(clips), "bus_from": None if "bus" in cand["classes"].values() else args.bus_model}
            speed = None
            for trk in args.trackers:
                args.tracker_name = trk
                rid = f"{run_id} [{trk}]"
                m = {**meta, "run_id": rid, "tracker": trk}
                print(f"[run] {rid} on {len(clips)} clips")
                res = evaluate_detector(det, clips, args, m)
                frames = res.pop("_frames")
                speed = speed or measure(det, frames, base_mem_mb=mem0)  # detector speed, tracker-independent
                res["speed"] = speed
                m.update(res)
                (out / f"{run_id.replace('/', '_')}__{trk}.json").write_text(json.dumps(m, indent=2, default=float))
                print(f"[done] {rid}: {speed['fps']:.1f} FPS, "
                      f"dir_acc={res['direction_people']['all']['direction_accuracy']}")


if __name__ == "__main__":
    main()
