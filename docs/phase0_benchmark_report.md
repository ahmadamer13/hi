# Phase 0 — Model selection benchmark report

**Status (2026-11-04): harness delivered and verified; no accuracy numbers measured yet. Approval is NOT requested yet.**

The protocol requires our own footage (≥ 100 labelled clips) and the target GPU (RTX 3060/4060 class,
FP16 TensorRT). Neither was available where this was built (a CPU-only cloud sandbox with no access
to camera footage). So this report contains:

1. what was checked directly: licenses, downloads, model formats, adapter correctness, CPU cost;
2. the benchmark harness that produces every metric in the protocol, tested on synthetic clips with known answers;
3. a shortlist with reasons, written as **hypotheses for the benchmark to confirm or reject**;
4. the steps that turn this into the final report with measured results.

Per the spec, **Phase 1 has not been started** and should not start until the measured version of this report is approved.

---

## 1. What was measured vs. not

| item | status |
|---|---|
| License of every candidate, checked at source | ✅ done (§2) |
| Download and load every candidate | ✅ 10 of 11 checkpoints; Owen718 is on Google Drive, which was unreachable from the sandbox (`benchmark.download` prints the manual path) |
| Adapter correctness (boxes land on objects, class maps, ONNX ↔ PyTorch parity) | ✅ checked on a public test image; RT-DETR ONNX export matches PyTorch to < 0.001 px |
| Metric code (AP/AR, FP/frame, MOTA/IDF1/IDSW, direction acc., WW-F1, decision latency, flow cell accuracy) | ✅ 21 unit/E2E tests with known answers (`pytest tests`) |
| AP/AR per density, bus AP, FP/frame on **our** footage | ⛔ pending: needs labelled clips |
| ID-switch rate and direction accuracy on **our** footage | ⛔ pending: needs labelled clips |
| FPS / latency / GPU memory at FP16 TensorRT on the **target GPU** | ⛔ pending: needs the target box |
| Fine-tuning gain | ⛔ pending: needs labelled train split |

## 2. Candidates and license risk (checked 2026-11-04)

| id | source | classes | format | license (as published) | commercial risk | finding |
|---|---|---|---|---|---|---|
| wesjos-11n/11m/v8s-head | HF `wesjos/YOLO-head-detection` | head | Ultralytics .pt | AGPL-3.0 | **high** | Commercial closed-source use requires an **Ultralytics Enterprise license**. Training data "~15k images", source not disclosed. Card says YOLOv8 variants give fewer false positives than YOLO11 on head-free images. |
| sharath33-phd-s | HF `Sharath33/Person` | person, head | ONNX (static 640×640) | OpenRAIL++ *(declared)* | **high / unclear** | `base_model` is `pyronear/yolov11s`, which is **AGPL-3.0**, and the architecture is Ultralytics YOLO11. The OpenRAIL++ label probably cannot override AGPL upstream terms. Trained on CrowdHuman. Shipped ONNX uses a DeepStream 3-output layout (cx,cy,w,h / score / class) that **does not match its own `inference.py`**. 0 downloads, no metrics published. Input fixed at 640. |
| owen718-v8-head | GitHub `Owen718/Head-Detection-Yolov8` (weights on Google Drive) | head | Ultralytics .pt | **none (no LICENSE file)** | **high** | No license means all rights reserved. Also YOLOv8 (AGPL) and CrowdHuman. Benchmark-only unless the author grants terms in writing. |
| lightnet-headbody-640/1024 | HF `yunkai/yolo-lightnet` | head, body | Darknet cfg/weights | Apache-2.0 | medium | Permissive weights but trained on CrowdHuman (non-commercial terms, see §3). Older (2023). Needs OpenCV **< 5**, because OpenCV 5 removed the Darknet importer. |
| rtdetr-r18vd / r50vd | HF `PekingU/rtdetr_*` | COCO (person, bus, car, truck) | transformers | Apache-2.0 | **low** | COCO-trained, no head class: must be fine-tuned. **Default 300 queries caps detections at 300 per image**, which is below extreme-density head counts. Fine-tune with 900+ queries (`--queries`) or use tiling. |
| rtdetrv2-r18vd | HF `PekingU/rtdetr_v2_r18vd` | COCO | transformers | Apache-2.0 | **low** | Same as above; the newer variant. |
| yolo11s/m-coco | Ultralytics assets | COCO | .pt | AGPL-3.0 | high | Bus and person reference baseline. |
| rfdetr-base *(g)* | HF `Roboflow/rf-detr-base`, PyPI `rfdetr` 1.11.2 | COCO | rfdetr | Apache-2.0 (nano–medium sizes; larger sizes use a different Roboflow license, so check per size) | low | Additional permissive candidate. Adapter not yet written (disabled in `models.yaml`); add on the target box if RT-DETR underperforms. |
| Farneback | OpenCV | flow | — | Apache-2.0 | low | Flow fallback candidate. |
| RAFT-small | torchvision | flow | — | BSD-3-Clause | low | Flow fallback candidate. |

The registry with these fields is `benchmark/models.yaml`. `models/manifest.lock.json` records the
SHA-256 of every file downloaded, so results are tied to exact weights.

## 3. Dataset license warning: affects the fine-tuning plan

The spec suggests fine-tuning on **CrowdHuman** and **JHU-CROWD++**. Both are published for
**non-commercial research use only**. JHU-CROWD++ also provides point labels, not boxes.
Weights fine-tuned on them carry that restriction, regardless of the model's own license. This also
applies to the "CrowdHuman-trained" candidates (b, c, d).

For a commercial deployment:
- fine-tune on **our own labelled footage only**, starting from a permissively licensed checkpoint
  (RT-DETR / RT-DETRv2 / RF-DETR, which are COCO-pretrained under Apache-2.0), **or**
- get legal sign-off or commercial terms from the dataset owners.

The harness supports both (`convert crowdhuman` exists), but the default plan assumes our data only.

## 4. CPU cost (relative only, NOT the target)

Measured end to end (pre-processing + inference + NMS) on a 1080p frame. Sandbox: 4 vCPU, FP32,
no GPU (`results/cpu_smoke_latency.json`). **This tells you the relative cost of each model, not whether
it reaches 15 FPS on an RTX 3060/4060.** TensorRT FP16 on such a GPU is typically 10–50× faster.

| model | 640 | 960 | 1280 |
|---|---|---|---|
| wesjos-11n-head | 36 ms | 54 ms | 106 ms |
| wesjos-v8s-head | 78 ms | 130 ms | 235 ms |
| wesjos-11m-head | 172 ms | 328 ms | 587 ms |
| sharath33-phd-s | 73 ms | *(static 640)* | *(static 640)* |
| lightnet-headbody-1024 | 217 ms | 453 ms | 855 ms |
| yolo11s-coco | 66 ms | 108 ms | 222 ms |
| rtdetr-r18vd | 303 ms | 702 ms | 1383 ms |
| rtdetrv2-r18vd | 338 ms | 729 ms | 1366 ms |
| rtdetr-r50vd | 632 ms | 1782 ms | 3755 ms |
| wesjos-v8s-head, 640 + 4 far-band tiles | 445 ms | | |

Takeaways for planning, still to be confirmed on the GPU:
- Cost grows roughly with pixel count (~2× from 640 to 960, ~4× to 1280). Tiling only the far band
  usually beats running the whole frame at 1280.
- RT-DETR-R18 costs about 4× a YOLO-s on CPU. On TensorRT FP16 the gap is much smaller (both are
  usually > 15 FPS at 640–960 on a 3060). This is the main thing the GPU run must settle.

## 5. Benchmark protocol as implemented

| spec requirement | implementation |
|---|---|
| ≥ 100 clips, all densities, day/night, rain/glare, buses | `finetune/sample_clips.py` (stratified by day × time block, density pre-label from a head count) and `docs/labeling_guide.md` (minimum stratum counts) |
| hand labels: heads/persons, buses, direction per track | `data/README.md` format; CVAT MOT 1.1 import (`finetune/convert.py cvat-mot`) |
| split by day / time block, never random frames | `convert.py split --by day`; `run.py` refuses to start if a day/time block leaks across splits |
| AP/AR per density, bus AP, FP/frame | `metrics_det.py`: COCO 101-pt AP@.5 and AP@[.5:.95], AR with maxDets = 1000, ignore regions, FP/frame at an operating threshold; bucketed by density, lighting and weather |
| sizes 640/960/1280 + tiled far region | `run.py --sizes 640 960 1280 --tiled` (far ROI from the camera config) |
| FP16 TensorRT FPS / latency / GPU memory on target | `trt/export.py` (Ultralytics → `.engine`; RT-DETR → ONNX → TensorRT EP; trtexec cross-check); `speed.py` (p50/p95 latency, FPS, GPU MB incl. weights) |
| full tracker, ID-switch rate | `tracker.py` (ByteTrack, per class, track-quality score); `metrics_track.py` (MOTA, IDF1, IDSW per 100 GT tracks) |
| **direction accuracy (final metric)** | `direction.py`: homography to metres, feet point for persons, head-height plane for heads, bottom-centre for buses; arrow / polyline / circular / both zones; 2-D Kalman; window 1.5 s / 4 s, min displacement 1.5 m / 5 m, angle 60°/120°, hysteresis. Reports direction accuracy, **decided fraction (1 − gray rate)**, wrong-way P/R/F1, false reds on detector-FP tracks, and decision latency |
| flow fallback benchmark | `run_flow.py`: Farneback vs RAFT-small, per-cell % against vs. labelled tracks, buses masked |
| comparison table, Pareto, recommendation, license risk | `report.py`. Only GPU + FP16-TRT runs can pass the ≥ 15 FPS gate. Pareto over (direction accuracy, FPS, license risk). **A permissive model wins unless an AGPL/unclear model is > 1 point better**; if AGPL is chosen, the report states that an Ultralytics Enterprise license is required |
| fine-tune top 1–2, re-benchmark | `finetune/convert.py to-yolo` (excludes test clips) and `finetune/train.py yolo|rtdetr`; then add the new weights to `models.yaml` and re-run on the same test split |

Two design choices in the scoring that you should check:
- **Direction accuracy is computed only over decided tracks** and is always reported next to
  "decided fraction". A model that says "gray" for everything cannot score well; one that guesses
  cannot hide its false reds.
- **A red on a track that matches no ground-truth object (a detector false positive) counts as a false
  alarm.** This is what turns detector FPs into the false-alarm rate that matters.

## 6. Shortlist (hypotheses for the GPU run)

1. **Permissive path (preferred):** RT-DETRv2-R18, fine-tuned on our frames with 3 classes
   (head, person, bus), 900 queries, 960 input plus far-band tiles. One model covers both people and
   buses. No AGPL.
2. **Accuracy reference (AGPL):** wesjos-v8s-head (fewest false positives per its card) and
   wesjos-11n-head (cheapest), each with yolo11s-coco or the RT-DETR model for buses. If one of
   these beats (1) by more than 1 point of direction accuracy, deploying it needs an Ultralytics
   Enterprise license.
3. **Baselines:** lightnet (Apache weights, CrowdHuman-trained) and sharath33 (license unclear).
   These are kept so the comparison is complete, not as deployment options.
4. **Flow:** Farneback is expected to be fast enough on CPU and may be enough for zone-level %.
   RAFT-small only if Farneback cell accuracy is poor at extreme density.

## 7. To produce the measured report (target box)

```bash
docker build -f Dockerfile.benchmark -t wrongway-bench .
docker run --gpus all --rm -v $PWD/data:/app/data -v $PWD/models:/app/models -v $PWD/results:/app/results \
  wrongway-bench bash -c '
    python -m pytest -q tests &&
    python -m benchmark.download &&
    python -m benchmark.trt.export --sizes 640 960 1280 &&
    python -m benchmark.run --manifest data/manifest.csv --split test --models all \
        --sizes 640 960 1280 --tiled --device 0 --half --trt --out results/ &&
    python -m benchmark.run_flow --manifest data/manifest.csv --split test --density high extreme &&
    python -m benchmark.report results/ --min-fps 15 --out docs/phase0_results.md'
# fine-tune top 1-2, add to models.yaml, re-run benchmark.run on the same test split, re-run report
```

## 8. Inputs needed from you before the measured report

1. **Footage access:** raw recordings covering the strata in `docs/labeling_guide.md`, plus
   any incident log of known wrong-way events.
2. **Labelling capacity:** about 100–120 clips × 15 s at 5 fps keyframes. Heads in extreme density
   dominate the effort.
3. **Target box:** RTX 3060 or 4060, driver ≥ 535, Docker with NVIDIA runtime.
4. **Commercial or non-commercial deployment?** This decides whether AGPL models and
   CrowdHuman-trained weights can be used at all, and therefore whether candidates (a)–(d) stay in
   the running or remain reference baselines only.
5. Owen718 weights: download manually from the Google Drive link in `models.yaml` into
   `models/owen718_yolov8_head.pt`, or drop the candidate (it has no license).

## 9. Known limitations of the harness

- Decision latency is measured from a GT track's first labelled frame to its first confirmed red. That
  overstates latency for people who start moving against the flow after they appear.
- Camera-shake compensation and multi-plane ramps are Phase 1 features and are not yet in the
  benchmark's direction classifier. For Phase 0, use clips without heavy shake, or mark them.
- The ByteTrack used here has no ReID. If ID switches dominate in high density, add a light ReID
  (Phase 1), and re-check whether the detector ranking changes.
- The density thresholds (50/200/600 heads) are assumptions. Calibrate them after the first labelling pass.
