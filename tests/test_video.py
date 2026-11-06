import numpy as np
import pytest
from video_fixtures import write_video

from vitaltrace.video import pose
from vitaltrace.video.analyzer import VideoAnalyzer, VideoError, build_video_alerts
from vitaltrace.video.estimators import MotionBBoxEstimator, YoloPoseEstimator
from vitaltrace.video.events import analyze_observations
from vitaltrace.video.models import PersonObservation
from vitaltrace.models import Severity


def standing_keypoints(cx=100.0, top=20.0, height=160.0, knee_skew=0.0):
    """17 pontos COCO de uma pessoa em pé (x, y, conf)."""
    kp = [(cx, top, 0.9)] * 17
    kp = list(kp)
    kp[5], kp[6] = (cx - 15, top + 0.2 * height, 0.9), (cx + 15, top + 0.2 * height, 0.9)
    kp[11], kp[12] = (cx - 10, top + 0.55 * height, 0.9), (cx + 10, top + 0.55 * height, 0.9)
    kp[13], kp[14] = (cx - 10 + knee_skew, top + 0.75 * height, 0.9), (cx + 10, top + 0.75 * height, 0.9)
    kp[15], kp[16] = (cx - 10, top + height, 0.9), (cx + 10, top + height, 0.9)
    return kp


def lying_keypoints(cy=150.0, left=20.0, length=160.0):
    kp = [(left, cy, 0.9)] * 17
    kp = list(kp)
    kp[5], kp[6] = (left + 0.2 * length, cy - 8, 0.9), (left + 0.2 * length, cy + 8, 0.9)
    kp[11], kp[12] = (left + 0.55 * length, cy - 6, 0.9), (left + 0.55 * length, cy + 6, 0.9)
    return kp


def obs(t, kp, bbox):
    return PersonObservation(t=t, bbox=bbox, keypoints=kp, frame_size=(320, 240))


def test_posture_from_keypoints():
    assert pose.classify_posture(obs(0, standing_keypoints(), (80, 20, 120, 180))) == "standing"
    assert pose.classify_posture(obs(0, lying_keypoints(), (20, 140, 180, 160))) == "lying"


def test_posture_sitting_when_knees_close_to_hips():
    kp = standing_keypoints()
    kp[13] = (90, kp[11][1] + 10, 0.9)
    kp[14] = (110, kp[12][1] + 10, 0.9)
    assert pose.classify_posture(obs(0, kp, (80, 20, 120, 120))) == "sitting"


def test_posture_from_bbox_only():
    assert pose.classify_posture(obs(0, None, (0, 0, 40, 120))) == "upright"
    assert pose.classify_posture(obs(0, None, (0, 0, 120, 40))) == "lying"


def test_knee_angles_symmetric_vs_asymmetric():
    l, r = pose.knee_angles(obs(0, standing_keypoints(), (0, 0, 1, 1)))
    assert abs(l - r) < 1
    l, r = pose.knee_angles(obs(0, standing_keypoints(knee_skew=25), (0, 0, 1, 1)))
    assert abs(l - r) > 8


def timeline(n_walk, n_lying):
    items = []
    for i in range(n_walk):
        items.append(obs(i * 0.5, standing_keypoints(cx=100 + 12 * i), (80 + 12 * i, 20, 120 + 12 * i, 180)))
    t0 = n_walk * 0.5
    for j in range(n_lying):
        items.append(obs(t0 + j * 0.5, lying_keypoints(), (20, 140, 180, 160)))
    return items


def test_fall_from_keypoints_is_detected():
    items = timeline(10, 14)
    metrics, events = analyze_observations(items, [o.t for o in items], 12)
    falls = [e for e in events if e.type == "fall_suspected"]
    assert len(falls) == 1 and falls[0].severity == Severity.CRITICAL
    assert metrics.posture_seconds["lying"] > 5


def test_quick_crouch_is_not_a_fall():
    items = timeline(10, 2) + [obs(6 + k * 0.5, standing_keypoints(), (80, 20, 120, 180)) for k in range(8)]
    _, events = analyze_observations(items, [o.t for o in items], 10)
    assert not [e for e in events if e.type == "fall_suspected"]


def test_knee_asymmetry_event():
    items = [obs(i * 0.5, standing_keypoints(cx=100 + 10 * i, knee_skew=40), (80 + 10 * i, 20, 120 + 10 * i, 180)) for i in range(12)]
    metrics, events = analyze_observations(items, [o.t for o in items], 6)
    assert metrics.median_knee_diff_deg and metrics.median_knee_diff_deg > 15
    assert any(e.type == "knee_asymmetry" for e in events)


def test_no_person_is_inconclusive():
    _, events = analyze_observations([None] * 6, [i * 0.5 for i in range(6)], 3)
    assert events[0].type == "no_person"


class FakeTensor:
    def __init__(self, data):
        self._d = np.asarray(data, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._d


class FakeResult:
    def __init__(self):
        self.boxes = type("B", (), {"xyxy": FakeTensor([[10, 10, 50, 150], [0, 0, 20, 20]]), "conf": FakeTensor([0.9, 0.5]), "__len__": lambda s: 2})()
        self.keypoints = type("K", (), {"data": FakeTensor([[[1.0, 2.0, 0.9]] * 17, [[0.0, 0.0, 0.1]] * 17])})()


class FakeYolo:
    def predict(self, frame, **kwargs):
        assert kwargs["device"] == "cpu"
        return [FakeResult()]


def test_yolo_adapter_parses_ultralytics_results():
    estimator = YoloPoseEstimator(model=FakeYolo())
    o = estimator.estimate(np.zeros((240, 320, 3), np.uint8), 1.5)
    assert o.bbox == (10, 10, 50, 150) and len(o.keypoints) == 17 and o.t == 1.5
    assert o.keypoints[0] == (1.0, 2.0, 0.9)


@pytest.fixture(scope="module")
def analyzer():
    return VideoAnalyzer(MotionBBoxEstimator, sample_fps=2)


def test_synthetic_video_fall(analyzer, tmp_path):
    video = write_video(tmp_path / "queda.avi", [("empty", 3), ("walk", 8), ("fall", 10)])
    result = analyzer.analyze(video)
    falls = [e for e in result.events if e.type == "fall_suspected"]
    assert falls and falls[0].severity == Severity.CRITICAL
    assert result.keyframes and result.keyframes[0].image.startswith("data:image/jpeg;base64,")
    assert result.keyframes[0].event_index is not None and result.keyframes[0].t >= 0
    assert result.payload()["keyframes"][0]["label"]
    assert "Queda suspeita" in result.report
    alerts = build_video_alerts(7, result)
    assert alerts[0].severity == Severity.CRITICAL and alerts[0].patient_id == "7"


def test_synthetic_video_walking_has_no_events(analyzer, tmp_path):
    result = analyzer.analyze(write_video(tmp_path / "anda.avi", [("empty", 3), ("walk", 14)]))
    assert result.events == [] and result.metrics.mobility_index > 0.5


def test_synthetic_video_immobility(analyzer, tmp_path):
    result = analyzer.analyze(write_video(tmp_path / "parado.avi", [("empty", 3), ("walk", 3), ("still", 30)]))
    assert any(e.type == "prolonged_immobility" for e in result.events)


def test_invalid_video_raises(analyzer, tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"isto nao e um video")
    with pytest.raises(VideoError):
        analyzer.analyze(bad)
