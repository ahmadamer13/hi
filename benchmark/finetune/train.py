"""Fine-tune the top Phase-0 candidates on our frames (+ optional public data).

  # Ultralytics family (AGPL): YOLO head models / COCO YOLO
  python -m benchmark.finetune.train yolo --weights models/wesjos__YOLO-head-detection/v8s-head.pt \
      --data datasets/ours/data.yaml --extra datasets/ch/data.yaml --imgsz 960 --epochs 60
  # RT-DETR (Apache-2.0) via transformers, YOLO-format dataset
  python -m benchmark.finetune.train rtdetr --model models/PekingU__rtdetr_v2_r18vd \
      --data datasets/ours --imgsz 960 --epochs 40

Afterwards: add the fine-tuned weights to models.yaml (new id, e.g. rtdetrv2-r18vd-ft) and re-run
benchmark.run on the SAME test split to report the gain. Test clips are never used for training
(convert.to_yolo skips them).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

NAMES = ["head", "person", "bus"]


def train_yolo(a):
    from ultralytics import YOLO
    data = a.data
    if a.extra:  # merge: ultralytics accepts lists of train dirs in one yaml
        import yaml
        d0, d1 = yaml.safe_load(open(a.data)), yaml.safe_load(open(a.extra))
        merged = {"path": "/", "names": d0["names"],
                  "train": [str(Path(d0["path"]) / d0["train"]), str(Path(d1["path"]) / d1["train"])],
                  "val": str(Path(d0["path"]) / d0["val"])}  # validate on OUR data only
        data = str(Path(a.data).with_name("merged.yaml"))
        Path(data).write_text(yaml.safe_dump(merged))
    m = YOLO(a.weights)
    m.train(data=data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device, project="runs/ft",
            name=Path(a.weights).stem, close_mosaic=10, degrees=0, perspective=0.0005, fliplr=0.5,
            hsv_v=0.5, patience=15, max_det=1000)


def train_rtdetr(a):
    import cv2
    import numpy as np
    import torch
    from transformers import AutoModelForObjectDetection

    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForObjectDetection.from_pretrained(
        a.model, num_labels=len(NAMES), id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)},
        ignore_mismatched_sizes=True, num_queries=a.queries).to(dev)
    root = Path(a.data)
    imgs = sorted((root / "images" / "train").glob("*.jpg"))

    def load(p):
        img = cv2.resize(cv2.imread(str(p)), (a.imgsz, a.imgsz))[..., ::-1]
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
            flip = True
        else:
            flip = False
        lab = root / "labels" / "train" / (p.stem + ".txt")
        rows = np.loadtxt(lab, ndmin=2) if lab.exists() and lab.stat().st_size else np.zeros((0, 5))
        if flip and len(rows):
            rows[:, 1] = 1 - rows[:, 1]
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float() / 255
        return x, {"class_labels": torch.tensor(rows[:, 0], dtype=torch.long),
                   "boxes": torch.tensor(rows[:, 1:5], dtype=torch.float)}

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * math.ceil(len(imgs) / a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    model.train()
    for ep in range(a.epochs):
        perm = np.random.permutation(len(imgs))
        for i in range(0, len(perm), a.batch):
            batch = [load(imgs[j]) for j in perm[i:i + a.batch]]
            x = torch.stack([b[0] for b in batch]).to(dev)
            labels = [{k: v.to(dev) for k, v in b[1].items()} for b in batch]
            loss = model(pixel_values=x, labels=labels).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            opt.step()
            sched.step()
        print(f"epoch {ep + 1}/{a.epochs} loss {loss.item():.3f}")
        dst = Path(f"runs/ft/{Path(a.model).name}-ft")
        model.save_pretrained(dst)
        pre = Path(a.model) / "preprocessor_config.json"
        if pre.exists():  # benchmark adapter loads the processor from the same dir
            (dst / pre.name).write_text(pre.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("yolo")
    p.add_argument("--weights", required=True); p.add_argument("--data", required=True); p.add_argument("--extra")
    p.add_argument("--imgsz", type=int, default=960); p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=8); p.add_argument("--device", default=None)
    p = sub.add_parser("rtdetr")
    p.add_argument("--model", required=True); p.add_argument("--data", required=True)
    p.add_argument("--imgsz", type=int, default=960); p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--queries", type=int, default=900, help="raise for dense crowds (COCO default 300)")
    p.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    train_yolo(a) if a.kind == "yolo" else train_rtdetr(a)


if __name__ == "__main__":
    main()
