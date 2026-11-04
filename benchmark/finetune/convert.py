"""Label conversion + split tools.

  # CVAT "MOT 1.1" export (gt/gt.txt + gt/labels.txt) -> our labels/<clip>/gt.csv
  python -m benchmark.finetune.convert cvat-mot  --src export_dir --clip-id CLIP --out data/labels
  # assign train/val/test by DAY (default) or TIME BLOCK — never by random frame
  python -m benchmark.finetune.convert split --manifest data/manifest.csv --by day --val 0.15 --test 0.25
  # our labels -> Ultralytics YOLO dataset (head, person, bus) for fine-tuning
  python -m benchmark.finetune.convert to-yolo --manifest data/manifest.csv --out datasets/ours --every 5
  # CrowdHuman .odgt -> YOLO (head = hbox, person = vbox). NOTE: CrowdHuman is non-commercial.
  python -m benchmark.finetune.convert crowdhuman --odgt annotation_train.odgt --images Images/ --out datasets/ch
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2

YOLO_NAMES = ["head", "person", "bus"]


def cvat_mot(src: Path, clip_id: str, out: Path):
    names = (src / "gt" / "labels.txt").read_text().split()
    dst = out / clip_id
    dst.mkdir(parents=True, exist_ok=True)
    frames = set()
    with open(src / "gt" / "gt.txt") as f, open(dst / "gt.csv", "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["frame", "track_id", "class", "x1", "y1", "x2", "y2", "ignore"])
        for line in f:
            fr, tid, x, y, bw, bh, not_ignored, cls, _vis = line.strip().split(",")[:9]
            fr = int(fr) - 1  # MOT is 1-based
            frames.add(fr)
            name = names[int(cls) - 1]
            if name == "ignore":
                name, ign = "person", 1
            else:
                ign = 0 if int(not_ignored) else 1
            x, y, bw, bh = map(float, (x, y, bw, bh))
            w.writerow([fr, tid, name, x, y, x + bw, y + bh, ign])
    # CVAT exports only keyframes+interpolated frames that contain boxes: annotators must confirm
    (dst / "frames.txt").write_text("\n".join(map(str, sorted(frames))))
    print(f"{clip_id}: {len(frames)} frames; add directions.csv (track_id,direction)")


def split(manifest: Path, by: str, val: float, test: float, seed: int):
    rows = list(csv.DictReader(open(manifest)))
    key = (lambda r: r["date"]) if by == "day" else (lambda r: (r["date"], r["time_block"]))
    groups = sorted({key(r) for r in rows}, key=str)
    random.Random(seed).shuffle(groups)
    n = len(groups)
    n_test, n_val = max(1, round(n * test)), max(1, round(n * val))
    assign = {g: "test" for g in groups[:n_test]}
    assign.update({g: "val" for g in groups[n_test:n_test + n_val]})
    for r in rows:
        r["split"] = assign.get(key(r), "train")
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    # density coverage check: every split should see every density level
    for s in ("train", "val", "test"):
        d = sorted({r["density"] for r in rows if r["split"] == s})
        print(f"{s}: {sum(r['split'] == s for r in rows)} clips, densities={d}")


def to_yolo(manifest: Path, out: Path, every: int):
    from ..data import iter_frames, load_manifest
    from ..detectors import CLASSES
    for clip in load_manifest(manifest):
        if clip.split not in ("train", "val"):
            continue  # test clips never enter training
        clip.load_labels()
        lab = set(clip.labeled_frames[::every])
        (out / "images" / clip.split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / clip.split).mkdir(parents=True, exist_ok=True)
        for fi, _, img in iter_frames(clip.video):
            if fi not in lab:
                continue
            h, w = img.shape[:2]
            stem = f"{clip.clip_id}_{fi:06d}"
            cv2.imwrite(str(out / "images" / clip.split / f"{stem}.jpg"), img)
            lines = []
            for row in clip.gt_frame(fi):
                name = CLASSES[int(row[1])]
                if name not in YOLO_NAMES or row[6]:
                    continue
                x1, y1, x2, y2 = row[2:6]
                lines.append(f"{YOLO_NAMES.index(name)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (out / "labels" / clip.split / f"{stem}.txt").write_text("\n".join(lines))
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnames: {dict(enumerate(YOLO_NAMES))}\n")


def crowdhuman(odgt: Path, images: Path, out: Path, split_name="train"):
    (out / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split_name).mkdir(parents=True, exist_ok=True)
    for line in open(odgt):
        rec = json.loads(line)
        src = images / f"{rec['ID']}.jpg"
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        dst = out / "images" / split_name / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())
        lines = []
        for g in rec["gtboxes"]:
            if g.get("tag") != "person" or g.get("extra", {}).get("ignore", 0):
                continue
            for cls, key in ((0, "hbox"), (1, "vbox")):
                if cls == 0 and g.get("head_attr", {}).get("ignore", 0):
                    continue
                x, y, bw, bh = g[key]
                x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + bw, w), min(y + bh, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                lines.append(f"{cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (out / "labels" / split_name / f"{rec['ID']}.txt").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cvat-mot"); p.add_argument("--src", type=Path); p.add_argument("--clip-id"); p.add_argument("--out", type=Path)
    p = sub.add_parser("split"); p.add_argument("--manifest", type=Path); p.add_argument("--by", choices=["day", "time_block"], default="day")
    p.add_argument("--val", type=float, default=0.15); p.add_argument("--test", type=float, default=0.25); p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("to-yolo"); p.add_argument("--manifest", type=Path); p.add_argument("--out", type=Path); p.add_argument("--every", type=int, default=5)
    p = sub.add_parser("crowdhuman"); p.add_argument("--odgt", type=Path); p.add_argument("--images", type=Path); p.add_argument("--out", type=Path)
    p.add_argument("--split-name", default="train")
    a = ap.parse_args(argv)
    if a.cmd == "cvat-mot":
        cvat_mot(a.src, a.clip_id, a.out)
    elif a.cmd == "split":
        split(a.manifest, a.by, a.val, a.test, a.seed)
    elif a.cmd == "to-yolo":
        to_yolo(a.manifest, a.out, a.every)
    else:
        crowdhuman(a.odgt, a.images, a.out, a.split_name)


if __name__ == "__main__":
    main()
