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
