# Wrong-Way Detection — pilgrimage walkway & road

Single-camera wrong-way detection for dense crowds and buses. Green = with the allowed direction,
Red = against, Gray = standing / uncertain. When unsure the system says *uncertain*, never *wrong way*.

## Status

| phase | state |
|---|---|
| **Phase 0 — model selection benchmark** | Harness, license review and shortlist delivered. **Measured results pending** our labelled footage and the target GPU. See [`docs/phase0_benchmark_report.md`](docs/phase0_benchmark_report.md). |
| Phase 1 — the system | **Not started.** Gated on approval of the measured Phase 0 report. |

## Layout

```
benchmark/          Phase 0 harness
  models.yaml       candidate registry (source, class map, license, risk)
  download.py       fetch candidates + SHA-256 lock file
  detectors.py      adapters: ultralytics, ONNX (YOLO / DeepStream / RT-DETR), HF RT-DETR, Darknet; tiling; ensembles
  tracker.py        ByteTrack (per class, track-quality score)
  ocsort.py         OC-SORT (ORU + OCM + OCR + BYTE pass)
  points.py         point-based head models (P2PNet) -> pseudo boxes
  stabilize.py      camera-motion estimate and cut detection (background pixels only)
  live.py           frame-by-frame engine used by the web app
  direction.py      homography, direction fields (arrow/polyline/circular/both), reference classifier
  metrics_det.py    AP / AR / FP-per-frame with ignore regions
  metrics_track.py  MOTA / IDF1 / IDSW, direction accuracy, wrong-way P/R/F1
  flow.py, run_flow.py   dense-crowd optical-flow fallback benchmark (Farneback, RAFT-small)
  run.py            benchmark runner        report.py   comparison table + Pareto + recommendation
  trt/export.py     FP16 TensorRT engine build (run on the target GPU)
  finetune/         clip sampling, label conversion, day/time-block split, fine-tuning
app/                          local web app (upload, draw arrow, watch live)
configs/camera.example.json   calibration + zones example
data/README.md                dataset format       docs/labeling_guide.md   annotation rules
tests/                        unit + synthetic end-to-end tests
```

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-benchmark.txt
python -m pytest -q tests            # CPU, about 15 s
python -m benchmark.download         # ~550 MB of candidate weights into models/
```

On the target GPU box use `Dockerfile.benchmark`; the exact commands are in §7 of the report.

## Watch a video live (web app)

```bash
python -m app.server        # then open the address it prints, usually http://localhost:8765
```
Add a video, drag an arrow in the allowed direction, and press Start watching. Green, red and gray boxes
appear live, with a picture of everyone marked wrong way. It runs only on this computer. When the camera view
changes it pauses and asks for a new arrow. Accuracy is not measured yet: check each red mark yourself.

## Try it on your own video (PREVIEW)

```bash
python -m benchmark.demo --video my_clip.mp4 --first-frame-only        # saves my_clip_wrongway_frame0.jpg
python -m benchmark.demo --video my_clip.mp4 --direction 960,900,960,300 --max-seconds 30
```
`--direction x1,y1,x2,y2` is an arrow in image pixels pointing the allowed way (read the numbers off
the saved first frame). Output: `my_clip_wrongway.mp4` (green/red/gray boxes + counters) and
`my_clip_wrongway.csv` (every wrong-way track and when it turned red).
This preview is not validated: it is for looking at behaviour, not for accuracy claims.
