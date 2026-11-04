# Benchmark data layout

```
data/
  manifest.csv               one row per clip (10-20 s)
  camera.json                camera config used by the clips (see configs/camera.example.json)
  clips/<clip_id>.mp4        NOT committed (privacy); keep on the secured storage
  labels/<clip_id>/
      gt.csv                 frame,track_id,class,x1,y1,x2,y2,ignore
      frames.txt             frame indices that are FULLY box-annotated (one per line)
      directions.csv         track_id,direction   (with|against|static|cross|mixed|unsure)
      ignore.json            optional {"polygons": [[[x,y],...]]} — unlabelable crowd areas
```

`manifest.csv` columns: `clip_id, video, date (YYYY-MM-DD), time_block (night|dawn|morning|midday|afternoon|evening),
density (low|medium|high|extreme), lighting (day|night|ir|mixed), weather (clear|rain|glare|haze),
has_bus (0/1), camera_config, split (train|val|test)`.

Splits are assigned with `python -m benchmark.finetune.convert split --by day` (or `--by time_block`);
`benchmark.run` refuses to run if the same day/time-block appears in two splits.

Density levels (heads visible in the frame; calibrate once on our camera and keep fixed):
low < 50, medium 50-200, high 200-600, extreme > 600. These are working assumptions; tune them
after the first labelling pass and record the final values here.

See `docs/labeling_guide.md` for annotation rules.
