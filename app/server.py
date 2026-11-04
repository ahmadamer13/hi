"""Local web app: upload a video, draw the allowed direction, watch the marks live.

    python -m app.server        then open http://localhost:8765

Runs only on this computer (bound to 127.0.0.1). Uploaded videos stay in runs/app/uploads/.
Wrong-way thumbnails are kept in memory only and disappear when the app stops.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from benchmark.direction import GRAY, GREEN, RED
from benchmark.live import LivePipeline, draw_marks

ROOT = Path(__file__).resolve().parent.parent
INDEX = Path(__file__).with_name("static") / "index.html"
UPLOADS = ROOT / "runs" / "app" / "uploads"
VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".ts"}
MAX_UPLOAD = 20 * 1024 ** 3
STREAM_WIDTH = 1280

MODELS = {
    "dfine-m-obj2coco": {"label": "Most people found (D-FINE M)", "license": "Apache-2.0"},
    "dfine-s-obj2coco": {"label": "Faster, fewer people (D-FINE S)", "license": "Apache-2.0"},
    "wesjos-v8s-head": {"label": "Heads only, no buses (AGPL)", "license": "AGPL-3.0"},
}


def best_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DetectorPool:
    """Keeps one model in memory; loads (and downloads if needed) on first use."""

    def __init__(self):
        self.lock = threading.Lock()
        self.current = (None, None)

    def get(self, model_id: str):
        from benchmark.detectors import build
        from benchmark.download import fetch, load_registry, local_path
        with self.lock:
            if self.current[0] != model_id:
                cand = next(c for c in load_registry()["candidates"] if c["id"] == model_id)
                path = local_path(cand["source"], cand)
                if not path.exists():
                    fetch(cand["source"], cand)
                self.current = (None, None)
                self.current = (model_id, build(cand, imgsz=640, device=best_device(), conf=0.3))
            return self.current[1]


def encode_jpeg(frame, width=STREAM_WIDTH, quality=80) -> bytes:
    if frame.shape[1] > width:
        frame = cv2.resize(frame, (width, round(frame.shape[0] * width / frame.shape[1])), interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def probe(path: Path) -> dict | None:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not ok:
            return None
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return {"width": frame.shape[1], "height": frame.shape[0], "fps": fps, "duration": n / fps if n > 0 else None}
    finally:
        cap.release()


def clean_arrow(value, width, height):
    """None means no arrow (everything stays gray). Raises ValueError on anything malformed."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("arrow must be [x1, y1, x2, y2]")
    arrow = [float(v) for v in value]
    if not all(v == v for v in arrow):
        raise ValueError("arrow has an invalid number")
    if not (0 <= arrow[0] <= width and 0 <= arrow[2] <= width and 0 <= arrow[1] <= height and 0 <= arrow[3] <= height):
        raise ValueError("arrow is outside the video")
    if ((arrow[2] - arrow[0]) ** 2 + (arrow[3] - arrow[1]) ** 2) ** 0.5 < 5:
        raise ValueError("arrow is too short")
    return arrow


class Run(threading.Thread):
    """One playback. Waits for the operator whenever the camera view changes."""

    def __init__(self, path, info, detector_factory, opts):
        super().__init__(daemon=True)
        self.path, self.info, self.factory, self.opts = path, info, detector_factory, opts
        self.state = "loading"     # loading | running | needs_arrow | finished | stopped | error
        self.error = None
        self.t = float(opts.get("start") or 0.0)
        self.proc_fps = 0.0
        self.counts = {"people": {GREEN: 0, RED: 0, GRAY: 0}, "bus": {GREEN: 0, RED: 0, GRAY: 0}}
        self.events: list[dict] = []
        self.thumbs: dict[str, bytes] = {}
        self.views = 1
        self.camera_ok = True
        self.frames_done = 0
        self.frames_skipped = 0
        self.arrow = opts["arrow"]
        self.jpeg = None
        self.raw_jpeg = None
        self.seq = 0
        self.cond = threading.Condition()
        self._stop_flag = threading.Event()
        self._answer = threading.Event()
        self._new_arrow = self.arrow

    # ---------------------------------------------------------------- operator actions
    def stop(self):
        self._stop_flag.set()
        self._answer.set()

    def answer_arrow(self, arrow):
        """Arrow for the new view. None = no direction for this view: everything stays gray."""
        self._new_arrow = arrow
        self._answer.set()

    def status(self) -> dict:
        return {"state": self.state, "error": self.error, "t": round(self.t, 2),
                "duration": self.info["duration"], "fps": round(self.proc_fps, 1),
                "video_fps": self.info["fps"], "width": self.info["width"], "height": self.info["height"],
                "counts": self.counts, "events": self.events, "views": self.views,
                "camera_ok": self.camera_ok, "arrow": self.arrow, "frames_done": self.frames_done,
                "frames_skipped": self.frames_skipped, "seq": self.seq, "live": not self.opts.get("every_frame")}

    # ---------------------------------------------------------------- playback
    def _publish(self, jpeg, raw=None):
        with self.cond:
            self.jpeg, self.seq = jpeg, self.seq + 1
            if raw is not None:
                self.raw_jpeg = raw
            self.cond.notify_all()

    def _finish(self, state):
        with self.cond:
            self.state = state
            self.cond.notify_all()

    def run(self):
        try:
            self._run()
        except Exception as e:  # noqa: BLE001  shown to the operator
            self.error = f"{type(e).__name__}: {e}"
            self._finish("error")

    def _run(self):
        o, fps = self.opts, self.info["fps"]
        det = self.factory(o["model"])
        cap = cv2.VideoCapture(str(self.path))
        try:
            start = float(o.get("start") or 0.0)
            end = o.get("end")
            idx = int(round(start * fps))
            if idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            pipe = None
            wall0, media0 = None, start
            ema = None
            self.state = "running"
            while not self._stop_flag.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                ts = idx / fps
                if end is not None and ts > float(end):
                    break
                t0 = time.perf_counter()
                if pipe is None:
                    pipe = LivePipeline(det, self.arrow, frame.shape, camera_moves=o["camera_moves"],
                                        min_disp_m=o["min_disp_m"])
                    wall0 = t0
                res = pipe.process(frame, ts)
                self.t = ts
                self.camera_ok = res.camera_ok
                self.frames_done += 1

                if res.cut:
                    self.views += 1
                    if o.get("stop_at_cuts", True):
                        if not self._wait_for_arrow(pipe, frame):
                            return
                        wall0, media0 = time.perf_counter(), ts

                self.counts = res.counts or self.counts
                for uid, name, box in res.new_reds:
                    self._add_event(uid, name, box, frame, ts)
                out = draw_marks(frame.copy(), res, self.arrow)
                self._publish(encode_jpeg(out))
                dt = time.perf_counter() - t0
                ema = dt if ema is None else 0.9 * ema + 0.1 * dt
                self.proc_fps = 1.0 / max(ema, 1e-6)
                idx += 1
                if not o.get("every_frame"):
                    # always show the latest moment: skip frames we are too slow to process
                    target = media0 + (time.perf_counter() - wall0)
                    while idx / fps < target and not self._stop_flag.is_set():
                        if not cap.grab():
                            break
                        idx += 1
                        self.frames_skipped += 1
                    wait = idx / fps - target
                    if wait > 0:
                        self._stop_flag.wait(wait)
        finally:
            cap.release()
        self._finish("stopped" if self._stop_flag.is_set() else "finished")

    def _wait_for_arrow(self, pipe, frame) -> bool:
        self._answer.clear()
        with self.cond:
            self.raw_jpeg = encode_jpeg(frame)
            self.state = "needs_arrow"
            self.cond.notify_all()
        self._answer.wait()
        if self._stop_flag.is_set():
            return False
        self.arrow = self._new_arrow
        pipe.set_arrow(self.arrow)
        self.state = "running"
        return True

    def _add_event(self, uid, name, box, frame, ts):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        px, py = 0.25 * (x2 - x1) + 10, 0.1 * (y2 - y1) + 10
        crop = frame[max(0, int(y1 - py)):min(h, int(y2 + py)), max(0, int(x1 - px)):min(w, int(x2 + px))]
        eid = uuid.uuid4().hex
        if crop.size:
            self.thumbs[eid] = encode_jpeg(crop, width=240)
        self.events.append({"id": eid, "track": uid % 100000, "class": name, "t": round(ts, 2)})


class App:
    def __init__(self, detector_factory=None, uploads: Path = UPLOADS):
        self.uploads = uploads
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.videos: dict[str, dict] = {}
        self.run: Run | None = None
        self.factory = detector_factory or DetectorPool().get
        self.lock = threading.Lock()

    def start(self, body: dict) -> dict:
        v = self.videos.get(str(body.get("video")))
        if v is None:
            raise ValueError("upload a video first")
        info = v["info"]
        model = str(body.get("model", "dfine-m-obj2coco"))
        if self.factory.__qualname__.startswith("DetectorPool") and model not in MODELS:
            raise ValueError("unknown model")
        min_disp = float(body.get("min_disp_m", 1.5))
        if not 0.25 <= min_disp <= 5:
            raise ValueError("minimum movement must be between 0.25 and 5 m")
        start = max(0.0, float(body.get("start") or 0.0))
        end = body.get("end")
        end = None if end in (None, "") else float(end)
        if end is not None and end <= start:
            raise ValueError("the end time must be after the start time")
        opts = {"model": model, "arrow": clean_arrow(body.get("arrow"), info["width"], info["height"]),
                "camera_moves": bool(body.get("camera_moves", True)), "min_disp_m": min_disp,
                "every_frame": bool(body.get("every_frame", False)), "start": start, "end": end,
                "stop_at_cuts": bool(body.get("stop_at_cuts", True))}
        if opts["arrow"] is None:
            raise ValueError("draw the allowed direction on the video first")
        with self.lock:
            if self.run is not None and self.run.is_alive():
                self.run.stop()
                self.run.join(timeout=10)
            self.run = Run(v["path"], info, self.factory, opts)
            self.run.start()
        return {"started": True}


def make_handler(app: App, port: int):
    allowed = {f"localhost:{port}", f"127.0.0.1:{port}"}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        # ------------------------------------------------------------ helpers
        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, val in (extra or {}).items():
                self.send_header(k, val)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode())

        def _error(self, message, code=400):
            self._json({"error": message}, code)

        def _allowed(self) -> bool:
            # This server can read local video and use the GPU: refuse other websites.
            if self.headers.get("Host") not in allowed:
                self._error("this app only answers on localhost", 403)
                return False
            origin = self.headers.get("Origin")
            if origin and origin.split("://", 1)[-1] not in allowed:
                self._error("requests from other websites are not allowed", 403)
                return False
            return True

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > 1 << 20:
                raise ValueError("request too large")
            return json.loads(self.rfile.read(n) or b"{}")

        # ------------------------------------------------------------ routes
        def do_GET(self):
            if not self._allowed():
                return
            u = urlparse(self.path)
            q = parse_qs(u.query)
            try:
                if u.path == "/":
                    return self._send(200, INDEX.read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/models":
                    return self._json({"models": MODELS})
                if u.path == "/api/frame":
                    return self._frame(q)
                if u.path == "/api/status":
                    return self._json(app.run.status() if app.run else {"state": "idle"})
                if u.path == "/api/new_view_frame":
                    if app.run and app.run.raw_jpeg:
                        return self._send(200, app.run.raw_jpeg, "image/jpeg")
                    return self._error("no frame waiting", 404)
                if u.path == "/api/thumb":
                    data = app.run.thumbs.get(q.get("id", [""])[0]) if app.run else None
                    return self._send(200, data, "image/jpeg") if data else self._error("not found", 404)
                if u.path == "/api/events.csv":
                    return self._events_csv()
                if u.path == "/stream":
                    return self._stream()
                return self._error("not found", 404)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
            if not self._allowed():
                return
            u = urlparse(self.path)
            try:
                body = self._read_json()
                if u.path == "/api/start":
                    return self._json(app.start(body))
                if u.path == "/api/stop":
                    if app.run:
                        app.run.stop()
                    return self._json({"stopped": True})
                if u.path == "/api/answer":
                    run = app.run
                    if run is None or run.state != "needs_arrow":
                        return self._error("the video is not waiting for an arrow", 409)
                    if body.get("keep"):
                        run.answer_arrow(run.arrow)
                    else:
                        run.answer_arrow(clean_arrow(body.get("arrow"), run.info["width"], run.info["height"]))
                    return self._json({"ok": True})
                return self._error("not found", 404)
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return self._error(str(e))

        def do_PUT(self):
            if not self._allowed():
                return
            if urlparse(self.path).path != "/api/upload":
                return self._error("not found", 404)
            name = parse_qs(urlparse(self.path).query).get("name", ["video.mp4"])[0]
            ext = Path(name).suffix.lower()
            if ext not in VIDEO_EXT:
                return self._error(f"{ext or 'this file'} is not a video type this app reads. Use mp4, mov, avi or mkv.")
            try:
                remaining = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._error("missing file size")
            if remaining <= 0 or remaining > MAX_UPLOAD:
                return self._error("the file is empty or larger than 20 GB")
            vid = uuid.uuid4().hex
            path = app.uploads / f"{vid}{ext}"     # never built from the uploaded name
            with open(path, "wb") as f:
                while remaining:
                    chunk = self.rfile.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)
            info = probe(path) if not remaining else None
            if info is None:
                path.unlink(missing_ok=True)
                return self._error("this file could not be opened as a video")
            app.videos[vid] = {"path": path, "info": info, "name": Path(name).name}
            self._json({"video": vid, **info})

        # ------------------------------------------------------------ helpers with streams
        def _frame(self, q):
            v = app.videos.get(q.get("video", [""])[0])
            if v is None:
                return self._error("unknown video", 404)
            try:
                t = max(0.0, float(q.get("t", ["0"])[0]))
            except ValueError:
                return self._error("bad time")
            cap = cv2.VideoCapture(str(v["path"]))
            try:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ok, frame = cap.read()
            finally:
                cap.release()
            if not ok:
                return self._error("no frame at that time", 404)
            self._send(200, encode_jpeg(frame), "image/jpeg")

        def _stream(self):
            run = app.run
            if run is None:
                return self._error("nothing is playing", 404)
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seen = -1
            while True:
                with run.cond:
                    # nothing new yet, including before the first frame exists: sleep, never spin
                    if (run.jpeg is None or run.seq == seen) and run.state in ("loading", "running", "needs_arrow"):
                        run.cond.wait(1.0)
                    jpeg, seq, state = run.jpeg, run.seq, run.state
                if jpeg is not None and seq != seen:
                    seen = seq
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                     + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                    self.wfile.flush()
                elif state not in ("loading", "running", "needs_arrow"):
                    return
                if app.run is not run:
                    return

        def _events_csv(self):
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["time_s", "class", "track"])
            for e in (app.run.events if app.run else []):
                w.writerow([e["t"], e["class"], e["track"]])
            self._send(200, buf.getvalue().encode(), "text/csv",
                       {"Content-Disposition": 'attachment; filename="wrong_way_events.csv"'})

    return Handler


def serve(port: int = 8765, detector_factory=None, uploads: Path = UPLOADS, tries: int = 20):
    """Binds to localhost only. If the port is taken, uses the next free one."""
    app = App(detector_factory, uploads)
    last = None
    for candidate in ([0] if port == 0 else range(port, port + tries)):
        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", candidate), BaseHTTPRequestHandler)
            break
        except OSError as e:
            last = e
    else:
        raise SystemExit(f"no free port between {port} and {port + tries - 1}: {last}")
    httpd.RequestHandlerClass = make_handler(app, httpd.server_address[1])
    httpd.daemon_threads = True
    return httpd, app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    httpd, _ = serve(a.port)
    print(f"Open http://localhost:{httpd.server_address[1]}   (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
