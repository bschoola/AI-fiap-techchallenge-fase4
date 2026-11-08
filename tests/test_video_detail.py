import math

import numpy as np
import pytest
from video_fixtures import write_video

from vitaltrace.models import Severity
from vitaltrace.video.analyzer import VideoAnalyzer
from vitaltrace.video.detail import analyze_detail, smooth_keypoints
from vitaltrace.video.estimators import MotionBBoxEstimator
from vitaltrace.video.events import analyze_observations
from vitaltrace.video.models import PersonObservation


def person(t, *, knee_off=0.0, lean=0.0, sh_tilt=0.0, sway=0.0, cx=100.0):
    """Pessoa de frente (17 pontos COCO). knee_off dobra os joelhos; lean desloca os ombros; sh_tilt desnivela."""
    cx += sway
    kp = [(cx, 20.0, 0.9)] * 17
    kp = list(kp)
    kp[5], kp[6] = (cx - 15 + lean, 60 + sh_tilt, 0.9), (cx + 15 + lean, 60, 0.9)
    kp[11], kp[12] = (cx - 10, 100, 0.9), (cx + 10, 100, 0.9)
    kp[13], kp[14] = (cx - 10 + knee_off, 140, 0.9), (cx + 10 - knee_off, 140, 0.9)
    kp[15], kp[16] = (cx - 10, 180, 0.9), (cx + 10, 180, 0.9)
    return PersonObservation(t=t, bbox=(cx - 30, 20, cx + 30, 180), keypoints=kp, frame_size=(320, 240))


def run(frames, fps=5.0, segments=None):
    obs = [person(i / fps, **kw) for i, kw in enumerate(frames)]
    times = [o.t for o in obs]
    return analyze_detail(obs, times, segments or [0] * len(obs))


def reps(periods, amps, fps=5.0, still_s=0.0):
    """Joelhos dobrando e esticando: um ciclo por (período, amplitude)."""
    frames = []
    for period, amp in zip(periods, amps):
        n = int(period * fps)
        frames += [{"knee_off": amp * (1 - math.cos(2 * math.pi * k / n)) / 2} for k in range(n)]
    return frames


def types(result):
    return {e.type for e in result.events}


def test_regular_repetitions_have_no_rhythm_or_fatigue_events():
    result = run(reps([3.0] * 8, [30] * 8))
    assert result.curve and result.curve["n_reps"] >= 7
    assert not types(result) & {"rep_irregular", "rep_fatigue"}
    assert result.joints["joelho esquerdo"] > 40


def test_amplitude_drop_is_flagged_as_fatigue():
    result = run(reps([3.0] * 9, [34, 33, 31, 27, 22, 17, 13, 11, 10]))
    fatigue = [e for e in result.events if e.type == "rep_fatigue"]
    assert fatigue and fatigue[0].severity == Severity.MEDIUM


def test_irregular_rhythm_is_flagged():
    result = run(reps([2.0, 2.0, 2.0, 8.0, 2.0, 2.0, 2.0, 8.0, 2.0], [30] * 9))
    assert "rep_irregular" in types(result)


def test_sustained_trunk_lean_is_flagged_but_short_lean_is_not():
    long_lean = run([{"lean": 28}] * 40)  # 8 s a ~35°
    assert "trunk_lean" in types(long_lean)
    short = run([{}] * 20 + [{"lean": 28}] * 10 + [{}] * 20)  # 2 s: curvou e voltou
    assert "trunk_lean" not in types(short)


def test_shoulder_tilt_is_flagged_and_level_shoulders_are_not():
    assert "shoulder_tilt" in types(run([{"sh_tilt": 12}] * 30))
    assert not types(run([{}] * 30)) & {"shoulder_tilt", "hip_tilt", "trunk_lean", "postural_sway"}


def test_postural_sway_while_standing_still():
    frames = [{"sway": 8 * math.sin(2 * math.pi * (i / 5) / 6)} for i in range(60)]  # 12 s
    assert "postural_sway" in types(run(frames))


def test_rom_asymmetry_between_knees():
    obs = []
    for i in range(40):
        o = person(i / 5, knee_off=30 * (1 - math.cos(2 * math.pi * i / 15)) / 2)
        kp = list(o.keypoints)
        kp[14] = (kp[14][0] + (30 - 8) * 0 + (o.keypoints[13][0] - (100 - 10)) * 0.8, kp[14][1], kp[14][2])  # direito dobra pouco
        obs.append(o.model_copy(update={"keypoints": kp}))
    result = analyze_detail(obs, [o.t for o in obs], [0] * len(obs))
    assert "rom_asymmetry" in types(result)


def test_smoothing_removes_a_single_frame_jump():
    obs = [person(i / 5) for i in range(7)]
    kp = list(obs[3].keypoints)
    kp[13] = (400.0, 400.0, 0.9)
    obs[3] = obs[3].model_copy(update={"keypoints": kp})
    out = smooth_keypoints(obs, [0] * 7)
    assert abs(out[3].keypoints[13][0] - 90) < 1e-6


def test_scene_cut_is_not_a_fall():
    standing = [person(i / 5) for i in range(10)]
    lying_kp = [(20 + k, 150.0, 0.9) for k in range(17)]
    lying_kp[5], lying_kp[6] = (50, 142, 0.9), (50, 158, 0.9)
    lying_kp[11], lying_kp[12] = (110, 144, 0.9), (110, 156, 0.9)
    lying = [PersonObservation(t=2 + i / 5, bbox=(20, 140, 180, 160), keypoints=lying_kp, frame_size=(320, 240)) for i in range(25)]
    obs = standing + lying
    times = [o.t for o in obs]
    _, same_scene = analyze_observations(obs, times, 7)
    assert any(e.type == "fall_suspected" for e in same_scene)
    _, with_cut = analyze_observations(obs, times, 7, [0] * 10 + [1] * 25)
    assert not any(e.type == "fall_suspected" for e in with_cut)


@pytest.fixture(scope="module")
def analyzer():
    return VideoAnalyzer(MotionBBoxEstimator, sample_fps=2, detail_fps=5)


def test_video_cut_is_detected_and_not_reported_as_fall(analyzer, tmp_path):
    video = write_video(tmp_path / "corte.avi", [("empty", 2), ("walk", 6), ("dark_empty", 3), ("dark_lying", 8)])
    result = analyzer.analyze(video)
    assert result.quality.cuts >= 1
    assert any("corte" in w for w in result.quality.warnings)
    assert not any(e.type == "fall_suspected" for e in result.events)


def test_depth_changes_sampling_and_quality(analyzer, tmp_path):
    video = write_video(tmp_path / "anda.avi", [("empty", 2), ("walk", 12)])
    fast, detailed = analyzer.analyze(video, "fast"), analyzer.analyze(video, "detailed")
    assert fast.quality.depth == "fast" and detailed.quality.depth == "detailed"
    assert detailed.metrics.frames_analyzed > 2 * fast.metrics.frames_analyzed
    assert any("pose" in w.lower() for w in detailed.quality.warnings)  # MOG2 não tem esqueleto
    assert "quality" in detailed.payload() and "Análise detalhada" in detailed.report


def test_unknown_depth_falls_back_to_detailed(analyzer, tmp_path):
    video = write_video(tmp_path / "v.avi", [("empty", 2), ("walk", 8)])
    assert analyzer.analyze(video, "xyz").quality.depth == "detailed"
