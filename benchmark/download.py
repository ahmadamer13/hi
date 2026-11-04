"""Download every enabled candidate in models.yaml into models/ and write a SHA-256 lock file.

    python -m benchmark.download                 # all enabled candidates
    python -m benchmark.download --only rtdetr-r18vd wesjos-v8s-head
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY = Path(__file__).with_name("models.yaml")


def load_registry(path: Path = REGISTRY) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _http(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.rename(dst)


def _hf_file(spec: str) -> tuple[str, str]:
    """hf://owner/repo/path/in/repo -> (repo_id, filename)."""
    parts = spec[len("hf://"):].split("/")
    return "/".join(parts[:2]), "/".join(parts[2:])


def local_path(spec: str, cand: dict | None = None) -> Path:
    """Where a source spec lives on disk after download."""
    if cand and cand.get("filename"):
        return MODELS_DIR / cand["filename"]
    if spec.startswith("hf://"):
        repo, fname = _hf_file(spec)
        if not fname:  # whole repo (transformers format)
            return MODELS_DIR / repo.replace("/", "__")
        return MODELS_DIR / repo.replace("/", "__") / fname
    if spec.startswith("ultralytics://"):
        return MODELS_DIR / spec[len("ultralytics://"):]
    raise ValueError(spec)


def fetch(spec: str, cand: dict) -> list[Path]:
    dst = local_path(spec, cand)
    if spec.startswith("hf://"):
        repo, fname = _hf_file(spec)
        if fname:
            if not dst.exists():
                _http(f"https://huggingface.co/{repo}/resolve/main/{fname}", dst)
            return [dst]
        # full transformers repo
        from huggingface_hub import snapshot_download
        snapshot_download(repo, local_dir=dst)
        return sorted(p for p in dst.rglob("*") if p.is_file() and ".cache" not in p.parts)
    if spec.startswith("ultralytics://"):
        if not dst.exists():
            name = spec[len("ultralytics://"):]
            _http(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{name}", dst)
        return [dst]
    if spec.startswith("gdrive://"):
        if not dst.exists():
            file_id = spec[len("gdrive://"):]
            try:
                subprocess.run([sys.executable, "-m", "gdown", file_id, "-O", str(dst)], check=True)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Google Drive download failed ({e}); fetch manually into {dst}") from e
        return [dst]
    raise ValueError(f"unknown source scheme: {spec}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    reg = load_registry()
    lock_path = MODELS_DIR / "manifest.lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    failures = []
    for cand in reg["candidates"]:
        if cand.get("enabled", True) is False:
            continue
        if args.only and cand["id"] not in args.only:
            continue
        try:
            files = fetch(cand["source"], cand)
            if cand.get("cfg"):
                files += fetch(cand["cfg"], {})
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {cand['id']}: {e}")
            failures.append(cand["id"])
            continue
        lock[cand["id"]] = {
            "source": cand["source"],
            "license": cand["license"],
            "files": {str(p.relative_to(ROOT)): sha256(p) for p in files if p.suffix != ".part"},
        }
        print(f"[ok]   {cand['id']}: {len(files)} file(s)")
    lock_path.write_text(json.dumps(lock, indent=2))
    if failures:
        print("failed:", ", ".join(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
