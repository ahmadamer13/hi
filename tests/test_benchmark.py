import argparse
import math

import numpy as np
import pytest

from benchmark.data import check_split_leakage, load_manifest
from benchmark.detectors import CLS_ID, Detections, Detector, TiledDetector
from benchmark.direction import (GRAY, GREEN, RED, CameraConfig, ClassParams, DirectionClassifier,
                                 Zone, fit_homography, reprojection_error_m, to_ground)
from benchmark.metrics_det import ClassAccumulator, box_iou
from benchmark.metrics_track import TrackEval
from benchmark.report import recommend
from benchmark.run import evaluate_detector
from benchmark.tracker import ByteTrack
from tests.synth import OracleDetector, make_clip


# ------------------------------------------------------------------ detection metrics
def test_ap_perfect_and_fp():
    acc = ClassAccumulator()
    gt = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)
    acc.add(gt.copy(), np.array([0.9, 0.8]), gt)
    s = acc.summary(op_thr=0.5)
    assert s["AP50"] == pytest.approx(1.0) and s["AP"] == pytest.approx(1.0)
    assert s["fp_per_frame"] == 0
    acc2 = ClassAccumulator()
    acc2.add(np.r_[gt, [[50, 50, 60, 60]]].astype(np.float32), np.array([0.9, 0.8, 0.7]), gt)
    s2 = acc2.summary(op_thr=0.5)
    assert s2["fp_per_frame"] == 1 and s2["AP50"] == pytest.approx(1.0)


def test_ignore_regions_not_fp():
    acc = ClassAccumulator()
    gt = np.array([[0, 0, 10, 10], [100, 100, 200, 200]], np.float32)
    acc.add(np.array([[0, 0, 10, 10], [120, 120, 130, 130]], np.float32), np.array([0.9, 0.9]), gt,
            gt_ignore=np.array([0, 1]))
    s = acc.summary(op_thr=0.5)
    assert s["fp_per_frame"] == 0 and s["n_gt"] == 1


def test_iou():
    a = np.array([[0, 0, 10, 10]], np.float32)
    assert box_iou(a, a)[0, 0] == pytest.approx(1)
    assert box_iou(a, np.array([[5, 0, 15, 10]], np.float32))[0, 0] == pytest.approx(1 / 3)


# ------------------------------------------------------------------ geometry / direction field
def test_homography_roundtrip():
    img = [[100, 400], [800, 420], [700, 100], [200, 90], [450, 250]]
    H_true = np.array([[0.02, 0.001, -1], [0.0005, 0.05, -3], [0.0, 0.0008, 1]])
    g = to_ground(H_true, np.array(img, float))
    H = fit_homography(img, g)
    assert reprojection_error_m(H, img, g) < 1e-3


def test_direction_fields():
    poly = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], np.float32)
    z = Zone("p", poly, "polyline", polyline=np.array([[0, 0], [10, 0], [10, 10]]))
    assert np.allclose(z.allowed(np.array([5, 0.5])), [1, 0])
    assert np.allclose(z.allowed(np.array([10.2, 6])), [0, 1])
    c = Zone("c", poly, "circular", center=np.array([0, 0]))
    assert np.allclose(c.allowed(np.array([5, 0])), [0, 1])   # CCW
    c.clockwise = True
    assert np.allclose(c.allowed(np.array([5, 0])), [0, -1])
    assert Zone("b", poly, "both").allowed(np.array([1, 1])) is None


def _cam():
    poly = np.array([[0, 0], [1000, 0], [1000, 1000], [0, 1000]], np.float32)
    return CameraConfig(np.eye(3), None, [Zone("z", poly, "arrow", direction_deg=0)])


@pytest.mark.parametrize("v,expect", [((1.3, 0), GREEN), ((-1.3, 0), RED), ((0, 1.3), GRAY),
                                      ((0.3, 0), GRAY), ((0, 0), GRAY)])
def test_classifier_states(v, expect):
    dc = DirectionClassifier(_cam())
    st = None
    for i in range(100):  # 4 s at 25 fps
        t = i / 25
        g = np.array([100 + v[0] * t, 100 + v[1] * t])
        st = dc.update(t, 1, "person", (10, 10), g)
    assert st == expect


def test_red_needs_confirmation_and_min_age():
    dc = DirectionClassifier(_cam())
    states = [dc.update(i / 25, 1, "person", (10, 10), np.array([100 - 1.3 * i / 25, 100])) for i in range(100)]
    first_red = states.index(RED) / 25
    p = dc.people
    assert first_red >= p.window_s + p.confirm_red_s - 0.05
    assert first_red <= 3.0  # spec: people decision latency ~2-3 s


def test_bus_slow_maneuver_never_red():
    dc = DirectionClassifier(_cam())
    states = [dc.update(i / 25, 7, "bus", (10, 10), np.array([100 - 0.8 * i / 25, 100])) for i in range(250)]
    assert RED not in states   # 0.8 m/s * 4 s = 3.2 m < 5 m


def test_turning_back_releases_red():
    dc = DirectionClassifier(_cam())
    x, states = 100.0, []
    for i in range(250):
        x += (-1.3 if i < 100 else 1.3) / 25
        states.append(dc.update(i / 25, 1, "person", (10, 10), np.array([x, 100])))
    assert RED in states and states[-1] == GREEN


# ------------------------------------------------------------------ tracker / tracking metrics
def test_bytetrack_keeps_ids():
    tr = ByteTrack()
    te = TrackEval()
    for f in range(60):
        gt = np.array([[10 + 2 * f, 10, 30 + 2 * f, 60], [400 - 2 * f, 100, 420 - 2 * f, 150]], np.float32)
        out = tr.update(f, gt, np.array([0.9, 0.9]), np.array([1, 1]))
        te.add([1, 2], gt, [o[0] for o in out], np.array([o[2] for o in out]).reshape(-1, 4))
    s = te.summary()
    assert s["IDSW"] == 0 and s["IDF1"] > 0.9


def test_idsw_counted():
    te = TrackEval()
    b = np.array([[0, 0, 10, 10]], np.float32)
    te.add([1], b, [5], b)
    te.add([1], b, [6], b)
    assert te.summary()["IDSW"] == 1


# ------------------------------------------------------------------ tiling
class _Fixed(Detector):
    classes = {"head"}
    name = "fixed"

    def __call__(self, img):
        h, w = img.shape[:2]
        return Detections(np.array([[w / 2 - 5, h / 2 - 5, w / 2 + 5, h / 2 + 5]], np.float32),
                          np.array([0.9], np.float32), np.array([CLS_ID["head"]], np.int32))


def test_tiled_offsets_and_coverage():
    t = TiledDetector(_Fixed(), far_roi=(0, 0, 1920, 400), tile=640, overlap=0.2)
    tiles = t.tiles((1080, 1920, 3))
    assert tiles[0][:2] == (0, 0) and max(x2 for _, _, x2, _ in tiles) == 1920
    d = t(np.zeros((1080, 1920, 3), np.uint8))
    assert len(d) >= 4 and (d.boxes[:, 2] <= 1920).all()


# ------------------------------------------------------------------ splits / report
def test_split_leakage(tmp_path):
    make_clip(tmp_path, "a", n_frames=5, split="train", time_block="am")
    make_clip(tmp_path, "b", n_frames=5, split="test", time_block="am")
    assert check_split_leakage(load_manifest(tmp_path / "manifest.csv"))


def test_recommend_requires_target_gpu():
    rows = [dict(run="x", license="Apache-2.0", risk="low", on_target=False, fps=50, dir_acc=0.9,
                 ww_f1=0.8, AP50_people=0.8)]
    assert recommend(rows, 15, 1.0).startswith("NO RECOMMENDATION")
    rows[0]["on_target"] = True
    rows.append(dict(run="y", license="AGPL-3.0", risk="high", on_target=True, fps=40, dir_acc=0.905,
                     ww_f1=0.8, AP50_people=0.8))
    assert "**x**" in recommend(rows, 15, 1.0)      # permissive preferred within margin
    rows[1]["dir_acc"] = 0.95
    r = recommend(rows, 15, 1.0)
    assert "**y**" in r and "Enterprise" in r


# ------------------------------------------------------------------ end to end
def _args():
    return argparse.Namespace(op_thr=0.4, tracker={"high_thr": 0.5, "low_thr": 0.1, "new_thr": 0.6, "buffer_frames": 60})


def test_end_to_end_oracle(tmp_path):
    man = make_clip(tmp_path)
    clips = load_manifest(man, "test")
    det = OracleDetector(clips[0].label_dir / "gt.csv", jitter_px=1.0)
    res = evaluate_detector(det, clips, _args(), {})
    dp = res["direction_people"]["all"]
    assert dp["ww_recall"] == 1.0 and dp["ww_fp"] == 0
    assert dp["direction_accuracy"] == 1.0
    assert res["direction_bus"]["all"]["ww_fp"] == 0
    assert res["detection"]["all"]["person"]["AP50"] > 0.95
    assert res["tracking_people"]["all"]["IDSW"] == 0
    lat = res["decision_latency_s"]["density=medium"]
    assert lat["p50"] < 3.0


def test_end_to_end_noisy_detector_no_false_red(tmp_path):
    man = make_clip(tmp_path, seed=1)
    clips = load_manifest(man, "test")
    det = OracleDetector(clips[0].label_dir / "gt.csv", jitter_px=3.0, miss=0.15, fp_per_frame=2.0, seed=3)
    res = evaluate_detector(det, clips, _args(), {})
    dp = res["direction_people"]["all"]
    assert dp["ww_fp"] == 0            # random FP boxes must not become wrong-way alarms
    assert dp["direction_accuracy"] == 1.0


# ------------------------------------------------------------------ flow fallback
def _texture(seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    t = (rng.random((540, 960)) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(t, (0, 0), 2), cv2.COLOR_GRAY2BGR)


def test_flow_against_fraction_known_motion():
    from benchmark.flow import Farneback, cell_against_fraction
    cam = _cam()
    a = _texture()
    b = np.roll(a, -4, axis=1)        # scene moves -X (allowed is +X) -> against
    flow, sc = Farneback(0.5)(a, b)
    cells = cell_against_fraction(flow, sc, cam, a.shape, dt=0.04, cell_px=128, min_speed_mps=0.5)
    inner = [v for (cx, cy), v in cells.items() if 1 <= cx <= 5 and 1 <= cy <= 2]
    assert inner and np.mean([v[0] for v in inner]) > 0.9
    c = np.roll(a, 4, axis=1)          # with the flow -> ~0 % against
    flow, sc = Farneback(0.5)(a, c)
    cells = cell_against_fraction(flow, sc, cam, a.shape, dt=0.04, cell_px=128, min_speed_mps=0.5)
    assert np.mean([v[0] for v in cells.values()]) < 0.1
    # a masked bus removes its cells
    cells_bus = cell_against_fraction(flow, sc, cam, a.shape, dt=0.04, cell_px=128, min_speed_mps=0.5,
                                      bus_boxes=[(0, 0, 960, 540)])
    assert not cells_bus


# ------------------------------------------------------------------ OC-SORT / HOTA / points
from benchmark.metrics_det import PointAccumulator
from benchmark.ocsort import OCSort
from benchmark.points import pseudo_boxes


def _run_tracker(tr, n=80, gap=None, cross=False):
    te = TrackEval()
    for f in range(n):
        if cross:   # two heads walking through each other in opposite directions (counterflow)
            gt = np.array([[100 + 4 * f, 200, 120 + 4 * f, 222], [420 - 4 * f, 204, 440 - 4 * f, 226]], np.float32)
        else:
            gt = np.array([[10 + 3 * f, 10, 40 + 3 * f, 60], [500 - 3 * f, 100, 530 - 3 * f, 150]], np.float32)
        ids = [1, 2]
        if gap and gap[0] <= f < gap[1]:   # occlusion: no detections
            out = tr.update(f, np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0, int))
        else:
            out = tr.update(f, gt, np.array([0.9, 0.9]), np.array([0, 0]))
        te.add(ids, gt, [o[0] for o in out], np.array([o[2] for o in out]).reshape(-1, 4))
    return te.summary()


@pytest.mark.parametrize("make", [ByteTrack, OCSort])
def test_trackers_basic(make):
    s = _run_tracker(make())
    assert s["IDSW"] == 0 and s["IDF1"] > 0.9 and s["HOTA"] > 0.85


def test_ocsort_survives_occlusion():
    s = _run_tracker(OCSort(), gap=(30, 40))
    assert s["IDSW"] == 0


def test_ocsort_counterflow_crossing():
    s = _run_tracker(OCSort(), cross=True)
    assert s["IDSW"] == 0, s


def test_hota_drops_with_id_switch():
    te = TrackEval()
    b = np.array([[0, 0, 10, 10]], np.float32)
    for f in range(10):
        te.add([1], b, [1], b)
    perfect = te.hota()["HOTA"]
    te2 = TrackEval()
    for f in range(10):
        te2.add([1], b, [1 if f < 5 else 2], b)
    assert perfect == pytest.approx(1.0) and te2.hota()["HOTA"] < 0.8


def test_point_accumulator():
    pa = PointAccumulator()
    gt = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], np.float32)
    pa.add(np.array([[5, 5], [25, 5], [50, 50]]), np.array([0.9, 0.8, 0.7]), gt)
    s = pa.summary(op_thr=0.5)
    assert s["loc_recall@op"] == 1.0 and s["loc_precision@op"] == pytest.approx(2 / 3)
    pa2 = PointAccumulator()   # two points in one box: only one TP
    pa2.add(np.array([[4, 4], [6, 6]]), np.array([0.9, 0.8]), gt[:1])
    assert pa2.summary(0.5)["loc_precision@op"] == pytest.approx(0.5)


def test_pseudo_boxes_follow_spacing():
    pts = np.array([[0, 0], [10, 0], [200, 200]], np.float32)
    b = pseudo_boxes(pts)
    w = b[:, 2] - b[:, 0]
    assert w[0] == pytest.approx(10 * 0.5 * 2) and w[2] == pytest.approx(60)


class _PointOracle(OracleDetector):
    """Point-model stand-in: emits head-centre points + pseudo boxes (exercises the point path)."""
    point_model = True

    def __call__(self, img):
        d = super().__call__(img)
        pts = (d.boxes[:, :2] + d.boxes[:, 2:]) / 2
        out = Detections(pseudo_boxes(pts), d.scores, d.classes)
        out.points = pts
        return out


@pytest.mark.parametrize("trk", ["bytetrack", "ocsort"])
def test_end_to_end_trackers_and_point_model(tmp_path, trk):
    man = make_clip(tmp_path, seed=4)
    clips = load_manifest(man, "test")
    a = _args()
    a.tracker_name = trk
    for det in (OracleDetector(clips[0].label_dir / "gt.csv", jitter_px=1.5),
                _PointOracle(clips[0].label_dir / "gt.csv", jitter_px=1.5)):
        res = evaluate_detector(det, clips, a, {})
        dp = res["direction_people"]["all"]
        assert dp["ww_fp"] == 0 and dp["ww_recall"] == 1.0, (trk, type(det).__name__, dp)
        assert res["detection"]["all"]["person"]["loc_recall@op"] > 0.95
        assert res["tracking_people"]["all"]["HOTA_mean_over_clips"] > 0.5


# ------------------------------------------------------------------ preview demo
def test_demo_preview_flags_only_against_walkers(tmp_path):
    from benchmark.demo import SelfScaleCamera, run
    from benchmark.ocsort import OCSort
    import csv as _csv
    make_clip(tmp_path, speed=60, n_frames=150)       # 60 px/s on a 60 px tall person ~ 1.7 m/s
    lab = tmp_path / "labels" / "synth01"
    det = OracleDetector(lab / "gt.csv", jitter_px=1.0)
    arrow = [100, 270, 800, 270]                         # allowed: left -> right
    summ = run(tmp_path / "clips" / "synth01.avi", det, SelfScaleCamera(arrow, (540, 960)), OCSort(),
               tmp_path / "out.mp4", arrow=arrow, log=lambda *_: None)
    against = {int(r["track_id"]) for r in _csv.DictReader(open(lab / "directions.csv")) if r["direction"] == "against"}
    assert summ["wrong_way_tracks"] == len(against) == 3
    assert (tmp_path / "out.mp4").stat().st_size > 10_000
    assert all(e["first_red_s"] <= 3.0 for e in summ["events"])   # people decision latency ~2-3 s


# ------------------------------------------------------------------ camera motion / live engine / app
def _scene(seed=1, size=(540, 960)):
    import cv2
    rng = np.random.default_rng(seed)
    t = (rng.random(size) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.GaussianBlur(t, (0, 0), 3) * 3, cv2.COLOR_GRAY2BGR)


def _shift(img, dx, dy):
    import cv2
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)


def test_camera_motion_recovers_pan_and_detects_cut():
    from benchmark.stabilize import CameraMotion, transform_points
    cm = CameraMotion()
    a = _scene()
    assert cm.update(a).cut            # first frame starts a shot
    for k in range(1, 6):
        s = cm.update(_shift(a, 6 * k, -3 * k))
        assert s.ok and not s.cut
    # a point at x in the 5th frame was at x-30, y+15 in the reference frame
    p = transform_points(s.T, [[400, 200]])[0]
    assert abs(p[0] - 370) < 2.5 and abs(p[1] - 215) < 2.5
    other = _scene(seed=2)[..., ::-1] // 2
    assert cm.update(other).cut


def test_camera_motion_ignores_points_inside_boxes():
    """A moving block marked as a person must not be mistaken for camera motion."""
    from benchmark.stabilize import CameraMotion, transform_points
    cm = CameraMotion()
    a = _scene()
    box = np.array([[100, 100, 860, 440]], float)   # most of the frame is "crowd"
    cm.update(a, box)
    for k in range(1, 6):
        f = a.copy()
        f[100:440, 100:860] = _shift(a, 10 * k, 0)[100:440, 100:860]
        s = cm.update(f, box)
    assert abs(transform_points(s.T, [[50, 50]])[0][0] - 50) < 3


class _Script(Detector):
    """Person walking at a given image speed, optional camera pan added to every box."""
    classes = {"person"}
    name = "script"

    def __init__(self, vx, pan=0.0):
        self.vx, self.pan, self.f = vx, pan, 0

    def __call__(self, img):
        x = 300 + (self.vx + self.pan) * self.f
        self.f += 1
        return Detections(np.array([[x, 200, x + 24, 260]], np.float32), np.array([0.9], np.float32),
                          np.array([CLS_ID["person"]], np.int32))


def _play(det, arrow, frames=120, fps=25, pan=0.0, **kw):
    from benchmark.live import LivePipeline
    kw.setdefault("min_box_h", 0)      # _Script's person is 60 px tall
    a = _scene()
    p = LivePipeline(det, arrow, a.shape, **kw)
    out = []
    for i in range(frames):
        out.append(p.process(_shift(a, pan * i, 0), i / fps))
    return p, out


def _states(results):
    return [it[3] for r in results for it in r.items]


def test_live_marks_against_and_with_walkers():
    arrow = [100, 270, 800, 270]
    _, res = _play(_Script(2.4), arrow)                  # with the arrow
    assert "green" in _states(res) and "red" not in _states(res)
    p, res = _play(_Script(-2.4), arrow)                 # against the arrow
    assert "red" in _states(res) and "green" not in _states(res)
    assert len(p.red_ids) == 1


def test_live_panning_camera_does_not_make_a_standing_person_red():
    """A standing person looks like a walker while the camera pans; correction must cancel it."""
    arrow = [100, 270, 800, 270]
    _, res = _play(_Script(0.0, pan=-2.4), arrow, pan=-2.4, camera_moves=True)
    assert "red" not in _states(res)
    # Without correction the same scene IS a false red, which is why the correction exists.
    _, res = _play(_Script(0.0, pan=-2.4), arrow, pan=-2.4, camera_moves=False)
    assert "red" in _states(res)


def test_live_cut_resets_motion_and_no_arrow_is_all_gray():
    from benchmark.live import LivePipeline
    a, b = _scene(), _scene(seed=5)[..., ::-1] // 2
    p = LivePipeline(_Script(-2.4), [100, 270, 800, 270], a.shape, min_box_h=0)
    for i in range(80):
        r = p.process(a, i / 25)
    assert "red" in _states([r])
    assert p.process(b, 80 / 25).cut
    assert p.dirs.tracks == {} or all(s.state == "gray" for s in p.dirs.tracks.values())
    _, res = _play(_Script(-2.4), None)
    assert set(_states(res)) <= {"gray"}


def _serve(tmp_path, detector):
    import threading
    from app.server import serve
    httpd, app = serve(0, lambda model: detector, tmp_path / "up")
    port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, app, f"http://127.0.0.1:{port}"


def _req(url, method="GET", data=None, headers=None):
    import json as _json
    import urllib.error
    import urllib.request
    body = _json.dumps(data).encode() if isinstance(data, dict) else data
    rq = urllib.request.Request(url, data=body, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(rq, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_app_full_flow_upload_draw_watch(tmp_path):
    import json as _json
    import time as _t
    make_clip(tmp_path, speed=60, n_frames=100)
    video = (tmp_path / "clips" / "synth01.avi").read_bytes()
    det = OracleDetector(tmp_path / "labels" / "synth01" / "gt.csv", jitter_px=1.0)
    httpd, app, base = _serve(tmp_path, det)
    try:
        code, html = _req(base + "/")
        assert code == 200 and b"Wrong-way watch" in html
        code, body = _req(base + "/api/upload?name=../../evil.avi", "PUT", video)
        assert code == 200
        info = _json.loads(body)
        assert (tmp_path / "up" / f"{info['video']}.avi").exists() and not (tmp_path / "evil.avi").exists()
        code, jpg = _req(f"{base}/api/frame?video={info['video']}&t=1")
        assert code == 200 and jpg[:2] == b"\xff\xd8"
        # no arrow -> refused with a plain reason
        code, body = _req(base + "/api/start", "POST", {"video": info["video"], "arrow": None})
        assert code == 400 and b"draw" in body
        code, _ = _req(base + "/api/start", "POST", {"video": info["video"], "arrow": [100, 270, 800, 270],
                                                      "every_frame": True, "camera_moves": False, "min_box_h": 0})
        assert code == 200
        for _ in range(200):
            s = _json.loads(_req(base + "/api/status")[1])
            if s["state"] in ("finished", "error"):
                break
            _t.sleep(0.05)
        assert s["state"] == "finished", s
        assert s["frames_done"] == 100 and len(s["events"]) == 3
        code, thumb = _req(f"{base}/api/thumb?id={s['events'][0]['id']}")
        assert code == 200 and thumb[:2] == b"\xff\xd8"
        assert _req(base + "/api/events.csv")[1].count(b"\n") == 4
    finally:
        httpd.shutdown()


def test_app_refuses_other_sites_and_non_videos(tmp_path):
    httpd, app, base = _serve(tmp_path, None)
    try:
        assert _req(base + "/api/status", headers={"Origin": "https://evil.example"})[0] == 403
        assert _req(base + "/api/status", headers={"Host": "evil.example"})[0] == 403
        code, body = _req(base + "/api/upload?name=x.exe", "PUT", b"MZ")
        assert code == 400 and b"not a video" in body
        code, body = _req(base + "/api/upload?name=x.mp4", "PUT", b"not really a video")
        assert code == 400 and b"could not be opened" in body
        assert not list((tmp_path / "up").iterdir())
    finally:
        httpd.shutdown()


def test_app_rejects_bad_arrows():
    from app.server import clean_arrow
    assert clean_arrow(None, 100, 100) is None
    assert clean_arrow([1, 1, 50, 50], 100, 100) == [1.0, 1.0, 50.0, 50.0]
    for bad in ([1, 1, 2, 2], [1, 1, 500, 50], "x", [1, 2, 3]):
        with pytest.raises((ValueError, TypeError)):
            clean_arrow(bad, 100, 100)


def test_app_uses_next_port_when_taken(tmp_path):
    import socket
    from app.server import serve
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    httpd, _ = serve(port, lambda m: None, tmp_path / "up")
    try:
        assert httpd.server_address[0] == "127.0.0.1" and httpd.server_address[1] != port
    finally:
        httpd.server_close()
        blocker.close()


def test_app_stream_does_not_spin_while_model_loads(tmp_path):
    """Regression: the stream loop used 100% CPU before the first frame, starving model loading."""
    import json as _json
    import socket
    import threading
    import time as _t
    make_clip(tmp_path, n_frames=10)
    release = threading.Event()

    def slow_factory(model):
        release.wait(5)
        return OracleDetector(tmp_path / "labels" / "synth01" / "gt.csv")

    httpd, app = __import__("app.server", fromlist=["serve"]).serve(0, slow_factory, tmp_path / "up")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    try:
        info = _json.loads(_req(base + "/api/upload?name=a.avi", "PUT", (tmp_path / "clips" / "synth01.avi").read_bytes())[1])
        assert _req(base + "/api/start", "POST", {"video": info["video"], "arrow": [10, 10, 500, 10], "every_frame": True})[0] == 200
        s = socket.create_connection(("127.0.0.1", port))
        s.sendall(f"GET /stream HTTP/1.0\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode())
        _t.sleep(0.2)
        t0 = _t.process_time()
        _t.sleep(1.0)
        assert _t.process_time() - t0 < 0.3     # a spinning loop would burn about a full core here
        release.set()
        s.close()
    finally:
        release.set()
        httpd.shutdown()


def test_live_mark_is_held_briefly_then_released():
    from benchmark.live import LivePipeline
    a = _scene()
    p = LivePipeline(_Script(0), [100, 270, 800, 270], a.shape, hold_mark_s=2.0)
    assert p._shown(1, "red", 10.0) == "red"
    assert p._shown(1, "gray", 11.5) == "red"       # still visible
    assert p._shown(1, "gray", 12.5) == "gray"      # released
    assert p._shown(2, "gray", 10.0) == "gray"      # never decided: stays gray
    p._shown(3, "red", 10.0)
    assert p._shown(3, "green", 10.5) == "green"    # a new decision wins at once


# ------------------------------------------------------------------ accuracy fixes
def _crossing(tracker, feats, n=70):
    """Two people cross; while they overlap the detector sees only one of them (the usual failure)."""
    ends = {}
    for f in range(n):
        boxes = np.array([[100 + 6 * f, 200, 150 + 6 * f, 320], [520 - 6 * f, 202, 570 - 6 * f, 322]], np.float32)
        use = [0] if abs(boxes[0, 0] - boxes[1, 0]) < 40 else [0, 1]
        kw = {"features": feats[use]} if feats is not None else {}
        for tid, _c, box, _s, _q in tracker.update(f, boxes[use], np.full(len(use), 0.9), np.ones(len(use), int), **kw):
            ends.setdefault(tid, []).append((f, float(box[0])))
    return ends


def test_look_gate_keeps_ids_when_differently_dressed_people_cross():
    a, b = np.zeros(64, np.float32), np.zeros(64, np.float32)
    a[3], b[40] = 1, 1
    ends = _crossing(OCSort(look_gate=0.7, det_thr=.35, new_thr=.4, max_age=20), np.stack([a, b]))
    for path in ends.values():                   # nobody may change walking direction
        xs = [x for _, x in path]
        steps = np.diff(xs)
        assert (steps >= -8).all() or (steps <= 8).all(), path


def test_reid_tells_clothes_apart_but_not_the_same_person():
    from benchmark.reid import box_features
    img = np.zeros((200, 300, 3), np.uint8)
    img[:, :100] = (255, 255, 255)
    img[:, 100:200] = (30, 30, 200)
    img[:, 200:] = (250, 250, 250)
    f = box_features(img, np.array([[0, 0, 100, 200], [100, 0, 200, 200], [200, 0, 300, 200]], float))
    assert f[0] @ f[2] > 0.95 and f[0] @ f[1] < 0.5
    assert np.allclose(box_features(img, np.array([[50, 50, 50, 50]], float)), 0)   # empty box: no opinion


def test_straightness_keeps_a_wavering_person_gray():
    cam = _cam()
    kw = dict(window_s=1.5, min_disp_m=0.5, min_age_s=1.0)

    def play(params):
        dc = DirectionClassifier(cam, people=params)
        x, out = 100.0, []
        for i in range(200):
            x += 0.12 if (i // 12) % 2 else -0.2      # sways every half second, drifting against the arrow
            out.append(dc.update(i / 25, 1, "person", (10, 10), np.array([x, 100.0])))
        return out
    assert RED in play(ClassParams(**kw))             # without the check, the drift counts as wrong way
    assert RED not in play(ClassParams(**kw, min_straightness=0.6))


def test_small_people_and_poor_tracks_stay_gray():
    arrow = [100, 270, 800, 270]
    _, res = _play(_Script(-2.4), arrow, min_box_h=100)   # person is 60 px tall
    assert "red" not in _states(res)
    _, res = _play(_Script(-2.4), arrow, min_box_h=40)
    assert "red" in _states(res)
