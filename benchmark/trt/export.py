"""Build FP16 TensorRT engines for every candidate on the TARGET machine.

    python -m benchmark.trt.export --sizes 640 960 1280 [--int8-calib data/calib/]

- Ultralytics .pt  -> .engine via ultralytics export (format=engine, half=True)
- HF RT-DETR       -> ONNX (opset 18) -> trtexec --fp16
- raw ONNX         -> trtexec --fp16
- darknet          -> run through OpenCV DNN CUDA FP16 (no TRT engine; reported as such)

Engines are GPU/TensorRT-version specific: build them on the deployment GPU, not here.
INT8 is only allowed if the re-run benchmark shows < 1 % loss (spec); calibrate on our frames.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from ..download import load_registry, local_path


def trtexec(onnx: Path, engine: Path, fp16=True, int8_calib: Path | None = None, shape: str | None = None):
    exe = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    cmd = [exe, f"--onnx={onnx}", f"--saveEngine={engine}", "--memPoolSize=workspace:2048"]
    if fp16:
        cmd.append("--fp16")
    if int8_calib:
        cmd += ["--int8", f"--calib={int8_calib}"]
    if shape:
        cmd += [f"--shapes={shape}"]
    print(" ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def export_hf_rtdetr(model_dir: Path, size: int, out: Path):
    import torch
    from transformers import AutoModelForObjectDetection
    m = AutoModelForObjectDetection.from_pretrained(str(model_dir)).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            o = self.m(pixel_values=x)
            return o.logits, o.pred_boxes

    torch.onnx.export(Wrap(m), torch.zeros(1, 3, size, size), str(out), opset_version=18,
                      input_names=["images"], output_names=["logits", "boxes"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[640, 960, 1280])
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--int8-calib", type=Path)
    a = ap.parse_args(argv)
    for c in load_registry()["candidates"]:
        if c.get("enabled", True) is False or (a.only and c["id"] not in a.only):
            continue
        src = local_path(c["source"], c)
        for s in a.sizes:
            try:
                if c["adapter"] == "ultralytics":
                    from ultralytics import YOLO
                    # ultralytics >= 8.4 uses `quantize` (replaces half/int8)
                    eng = YOLO(str(src)).export(format="engine", imgsz=s, device=0,
                                                quantize=8 if a.int8_calib else 16)
                    Path(eng).rename(src.with_name(f"{src.stem}_{s}.engine"))  # build() looks for this
                elif c["adapter"] == "hf_rtdetr":
                    onnx = src.parent / f"{src.name}_{s}.onnx"
                    export_hf_rtdetr(src, s, onnx)  # run via onnxruntime TensorRT EP (FP16)
                    trtexec(onnx, onnx.with_suffix(".engine"), int8_calib=a.int8_calib)  # latency cross-check
                elif c["adapter"] == "onnx_yolo":
                    trtexec(src, src.with_name(f"{src.stem}_{s}.engine"), int8_calib=a.int8_calib,
                            shape=f"input:1x3x{s}x{s}")
                else:
                    print(f"[info] {c['id']}: no TRT path ({c['adapter']}); benchmarked via OpenCV CUDA FP16")
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {c['id']}@{s}: {e}")


if __name__ == "__main__":
    main()
