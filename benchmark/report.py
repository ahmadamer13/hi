"""Aggregate results/*.json into the Phase 0 comparison table + recommendation.

    python -m benchmark.report results/ --min-fps 15 --out docs/phase0_results.md

Selection rule (from the spec):
  1. Only runs measured on a GPU with FP16 TensorRT count toward the >= min-fps gate.
  2. Primary metric: people direction accuracy at moderate density ("density=medium"), with
     wrong-way F1 as tie-break; bus direction accuracy must be reported separately.
  3. Pareto front over (direction accuracy, FPS, license risk).
  4. A permissive (low-risk) model is preferred unless an AGPL/unclear model is > `--license-margin`
     points better on the primary metric; then the report states the Ultralytics Enterprise
     licence requirement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RISK = {"low": 0, "medium": 1, "high": 2}


def _g(d, *path, default=None):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return default
        d = d[p]
    return d


def load(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]


def row(r: dict, bucket: str) -> dict:
    det = _g(r, "detection", bucket, default={}) or {}
    people_cls = "head" if "head" in det else "person"
    on_target = bool(_g(r, "device", "gpu")) and r.get("half") and (r.get("trt") or "engine" in r.get("model", ""))
    return {
        "run": r["run_id"], "license": r["license"], "risk": r["license_risk"],
        "people_cls": people_cls,
        "AP50_people": _g(det, people_cls, "AP50"), "AR_people": _g(det, people_cls, "AR"),
        "AP50_bus": _g(det, "bus", "AP50"),
        "fp_per_frame": _g(det, people_cls, "fp_per_frame"),
        "idsw_per_100": _g(r, "tracking_people", bucket, "IDSW_per_100_gt_tracks"),
        "dir_acc": _g(r, "direction_people", bucket, "direction_accuracy"),
        "decided": _g(r, "direction_people", bucket, "decided_fraction"),
        "ww_f1": _g(r, "direction_people", bucket, "ww_f1"),
        "bus_dir_acc": _g(r, "direction_bus", bucket, "direction_accuracy"),
        "fps": _g(r, "speed", "fps"), "lat_p95": _g(r, "speed", "latency_ms_p95"),
        "mem": _g(r, "speed", "gpu_mem_mb"), "on_target": bool(on_target),
        "gpu": _g(r, "device", "gpu") or "CPU (not target)",
    }


def pareto(rows):
    front = []
    for a in rows:
        dominated = any(
            (b["dir_acc"] or 0) >= (a["dir_acc"] or 0) and (b["fps"] or 0) >= (a["fps"] or 0)
            and RISK[b["risk"]] <= RISK[a["risk"]] and b is not a
            and ((b["dir_acc"] or 0) > (a["dir_acc"] or 0) or (b["fps"] or 0) > (a["fps"] or 0)
                 or RISK[b["risk"]] < RISK[a["risk"]])
            for b in rows)
        if not dominated:
            front.append(a["run"])
    return front


def recommend(rows, min_fps: float, margin: float) -> str:
    ok = [r for r in rows if r["on_target"] and (r["fps"] or 0) >= min_fps and r["dir_acc"] is not None]
    if not ok:
        return ("NO RECOMMENDATION: no run was measured on the target GPU with FP16 TensorRT at "
                f">= {min_fps} FPS. Run the benchmark on the RTX 3060/4060 box with --half --trt.")
    key = lambda r: (r["dir_acc"], r["ww_f1"] or 0, r["AP50_people"] or 0)  # noqa: E731
    best = max(ok, key=key)
    best_perm = max([r for r in ok if r["risk"] == "low"], key=key, default=None)
    if best_perm and best["risk"] != "low" and (best["dir_acc"] - best_perm["dir_acc"]) * 100 <= margin:
        choice, why = best_perm, (f"permissive model within {margin:.1f} pt of the best "
                                  f"({best['run']}, {best['license']})")
    else:
        choice, why = best, "best direction accuracy at the FPS gate"
    msg = f"Recommended: **{choice['run']}** ({choice['license']}) — {why}."
    if "AGPL" in choice["license"]:
        msg += (" This model is AGPL-3.0: a closed/commercial deployment requires an Ultralytics "
                "Enterprise license (or releasing the full system source under AGPL).")
    if choice["risk"] != "low" and "AGPL" not in choice["license"]:
        msg += " License status is unclear; obtain written terms from the author before deployment."
    return msg


def fmt(v, p=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{p}f}"
    return str(v)


def render(results: list[dict], min_fps: float, margin: float) -> str:
    out = ["# Phase 0 — measured results", ""]
    if not results:
        return "\n".join(out + ["_No result files found._"])
    for bucket in ("all", "density=low", "density=medium", "density=high", "density=extreme"):
        rows = [row(r, bucket) for r in results]
        if all(r["dir_acc"] is None and r["AP50_people"] is None for r in rows):
            continue
        out += [f"## {bucket}", "",
                "| run | license (risk) | people AP50 | people AR | bus AP50 | FP/frame | IDSW/100 trk | "
                "dir acc | decided | WW F1 | bus dir acc | FPS | p95 ms | GPU MB | device |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: -(r["dir_acc"] or 0)):
            out.append(
                f"| {r['run']} | {r['license']} ({r['risk']}) | {fmt(r['AP50_people'])} | {fmt(r['AR_people'])} | "
                f"{fmt(r['AP50_bus'])} | {fmt(r['fp_per_frame'], 2)} | {fmt(r['idsw_per_100'], 1)} | "
                f"{fmt(r['dir_acc'])} | {fmt(r['decided'], 2)} | {fmt(r['ww_f1'])} | {fmt(r['bus_dir_acc'])} | "
                f"{fmt(r['fps'], 1)} | {fmt(r['lat_p95'], 1)} | {fmt(r['mem'], 0)} | {r['gpu']} |")
        out += ["", f"Pareto front (dir acc × FPS × license): {', '.join(pareto(rows))}", ""]
    out += ["## Recommendation", "", recommend([row(r, "density=medium") for r in results], min_fps, margin)]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--min-fps", type=float, default=15)
    ap.add_argument("--license-margin", type=float, default=1.0, help="points of dir. accuracy")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    md = render(load(a.results), a.min_fps, a.license_margin)
    if a.out:
        a.out.write_text(md)
    print(md)


if __name__ == "__main__":
    main()
