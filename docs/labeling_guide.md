# Labeling guide — Phase 0 benchmark set

Tool: CVAT (self-hosted, so footage never leaves our network). Project labels: `head`, `person`, `bus`,
`car`, `truck`, `ignore`. Export format: **MOT 1.1**. Convert with
`python -m benchmark.finetune.convert cvat-mot --src <export> --clip-id <id> --out data/labels`.

## 1. Clip selection (≥ 100 clips, 10–20 s)
`python -m benchmark.finetune.sample_clips --recordings /footage/*.mp4 --n 120 --len 15 --counter wesjos-v8s-head`
then fill `lighting`, `weather` and `has_bus` by eye. Minimum coverage targets:

| stratum | min clips |
|---|---|
| density low / medium / high / extreme | 20 / 30 / 30 / 20 |
| night or IR | 20 |
| rain, glare or strong shadow | 15 |
| bus present (moving, stopping, reversing, parking, boarding) | 25 |
| known wrong-way events (operators' incident log) | as many as exist; **all go to val/test** |

Real wrong-way walkers are rare. If fewer than ~50 against-flow tracks exist in the footage, say so in
the report. Staged walks by staff are acceptable for test clips if flagged `staged` in a note.

## 2. Boxes
- **Every head** visible in the labelled frames, including partly occluded ones (box the visible extent
  plus an estimate of the hidden part when ≥ 50 % is visible). Heads are the unit in dense crowds.
- **Person** full-body boxes only in low/medium density clips, or for people standing fully visible. In
  high/extreme density do not box persons; draw an `ignore` polygon over areas where heads cannot be
  resolved (e.g. very far region) and put it in `ignore.json`.
- **Bus/car/truck**: full extent including occluded part; bottom edge on the road.
- Wheelchairs and strollers: box the person; the device is not a separate class.
- No faces are labelled or stored. Exports contain boxes only.

Annotate keyframes at **5 fps** with CVAT interpolation between keyframes, then check each interpolated
frame where tracks cross. `frames.txt` must list only frames that were checked by a human.

## 3. Track identity
Each head/person/bus keeps one ID while visible. If occluded < 2 s and it reappears, keep the ID. Longer
occlusions get a new ID (the tracker is not penalised for those).

## 4. Direction per track (`directions.csv`)
Label against the *allowed direction of the zone the track is in*, for the active direction profile:

| label | meaning |
|---|---|
| `with` | net movement ≥ 1.5 m (bus ≥ 5 m) within 60° of allowed |
| `against` | net movement ≥ 1.5 m (bus ≥ 5 m) more than 120° from allowed |
| `static` | standing / waiting / loading / parked (moves < 1.5 m, bus < 5 m) |
| `cross` | moving 60–120° from allowed (sideways) |
| `mixed` | changes between with/against within the clip (e.g. turns back) |
| `unsure` | annotator cannot tell; excluded from direction scoring |

Two annotators label direction independently on the test split; disagreements go to a third. Report
inter-annotator agreement: it is the ceiling for any measured "direction accuracy".
