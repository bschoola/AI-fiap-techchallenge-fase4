"""Rastreamento do paciente, limpeza do esqueleto, descrição da cena e o fluxo de duas passadas."""

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from video_fixtures import skeleton, write_video

from vitaltrace.video import clean, pose, tracking
from vitaltrace.video.analyzer import VideoAnalyzer
from vitaltrace.video.describe import describe_scene
from vitaltrace.video.detail import analyze_detail, smooth_keypoints
from vitaltrace.video.estimators import YoloPoseEstimator
from vitaltrace.video.narrative import AzureOpenAISceneNarrator, NarrativeError

FPS = 8.0


def squat(t, period=3.0, amp=28.0):
    return amp * (1 - math.cos(2 * math.pi * t / period)) / 2


def two_people(n=80, gap=()):
    """Terapeuta parado (maior, à esquerda) e paciente agachando (à direita). `gap`: quadros sem o paciente."""
    frames, segments = [], []
    tracker = tracking.PoseTracker()
    for i in range(n):
        t = i / FPS
        people = [skeleton(t, cx=70, top=20, height=200)]
        if i not in gap:
            people.append(skeleton(t, cx=230, top=60, height=150, knee_off=squat(t)))
        frames.append(tracker.update(people, t))
        segments.append(0)
    return frames, segments


# ------------------------------------------------------------- rastreador
def test_tracker_keeps_one_number_per_person():
    frames, _ = two_people()
    left = {p.track_id for people in frames for p in people if p.bbox[0] < 100}
    right = {p.track_id for people in frames for p in people if p.bbox[0] > 150}
    assert len(left) == 1 and len(right) == 1 and left != right
    assert all(p.people == 2 for p in frames[0])


def test_tracker_recovers_a_person_after_a_short_occlusion():
    frames, _ = two_people(gap=range(30, 36))  # 0,75 s encoberto
    right = {p.track_id for people in frames for p in people if p.bbox[0] > 150}
    assert len(right) == 1


def test_tracker_gives_a_new_number_to_a_newcomer_far_away():
    tracker = tracking.PoseTracker()
    a = tracker.update([skeleton(0.0, cx=60)], 0.0)[0]
    both = tracker.update([skeleton(0.125, cx=61), skeleton(0.125, cx=260)], 0.125)
    assert {p.track_id for p in both} == {a.track_id, a.track_id + 1}


def test_fall_does_not_become_a_new_person():
    tracker = tracking.PoseTracker()
    standing = tracker.update([skeleton(0.0, cx=100, top=20, height=160)], 0.0)[0]
    lying = skeleton(0.125, cx=120, top=150, height=40).model_copy(update={"bbox": (40, 150, 200, 190), "keypoints": None})
    assert tracker.update([lying], 0.125)[0].track_id == standing.track_id


# ------------------------------------------------------ escolha do paciente
def test_auto_choice_prefers_who_is_exercising_and_manual_choice_wins():
    frames, segments = two_people()
    stats = tracking.summarize(frames, segments)
    therapist = next(s for s in stats if s.height > 0.7)
    patient = next(s for s in stats if s.height < 0.7)
    assert patient.activity > therapist.activity
    assert tracking.choose(stats)[0][0] == patient.id
    assert tracking.choose(stats, prefer=therapist.id)[0][0] == therapist.id
    picked = tracking.pick(frames, segments, tracking.choose(stats))
    assert all(o is not None and o.track_id == patient.id for o in picked)


def test_candidates_ignore_small_people_in_the_background():
    frames, segments = two_people()
    extra = tracking.PoseTracker()
    for i, people in enumerate(frames):
        people.append(skeleton(i / FPS, cx=160, top=10, height=40).model_copy(update={"track_id": 99}))
    ids = {c.id for c in tracking.candidates(tracking.summarize(frames, segments))}
    assert 99 not in ids and len(ids) == 2
    assert extra is not None


def test_same_person_reacquired_with_a_new_number_is_merged_back():
    frames, segments = two_people(gap=range(30, 50))  # 2,5 s fora: volta com outro número
    assert len(tracking.summarize(frames, segments)) == 3
    frames = tracking.merge_fragments(frames, segments)
    stats = tracking.summarize(frames, segments)
    assert len(stats) == 2
    patient = next(s for s in stats if s.height < 0.7)
    picked = tracking.pick(frames, segments, tracking.choose(stats))
    assert sum(o is not None for o in picked) == 60 and {o.track_id for o in picked if o} == {patient.id}


def test_two_different_people_at_the_same_time_are_never_merged():
    frames, segments = two_people()
    assert tracking.merge_fragments(frames, segments) is frames


# ------------------------------------------------------------------ limpeza
def test_left_right_swap_is_undone():
    obs = [skeleton(i / FPS, knee_off=squat(i / FPS)) for i in range(20)]
    for i in (8, 9):
        obs[i] = obs[i].model_copy(update={"keypoints": clean._swap(obs[i].keypoints, clean._BODY + clean._FACE)})
    fixed, n = clean.fix_swaps(obs, [0] * 20)
    assert n == 2
    for i in (8, 9):
        assert fixed[i].keypoints[13][0] < fixed[i].keypoints[14][0]  # joelho esquerdo de volta à esquerda


def test_side_view_is_not_touched_by_the_swap_fix():
    obs = [skeleton(i / FPS, side=True) for i in range(12)]
    _, n = clean.fix_swaps(obs, [0] * 12)
    assert n == 0


def test_a_limb_that_suddenly_stretches_is_marked_unreliable():
    obs = [skeleton(i / FPS) for i in range(16)]
    kp = list(obs[7].keypoints)
    kp[15] = (kp[15][0] + 120, kp[15][1] + 30, 0.9)  # tornozelo esquerdo "voou"
    obs[7] = obs[7].model_copy(update={"keypoints": kp})
    out, n = clean.drop_implausible(obs, [0] * 16)
    assert n >= 1 and out[7].keypoints[15][2] == 0.0 and out[6].keypoints[15][2] == 0.9
    assert pose.joint_angle(out[7], pose.L_HIP, pose.L_KNEE, pose.L_ANKLE) is None


def test_smoothing_fills_a_single_missing_frame_but_not_a_long_gap():
    obs = [skeleton(i / FPS) for i in range(9)]

    def hide(i):
        kp = list(obs[i].keypoints)
        kp[13] = (0.0, 0.0, 0.05)
        obs[i] = obs[i].model_copy(update={"keypoints": kp})

    for i in (2, 5, 6):
        hide(i)
    out = smooth_keypoints(obs, [0] * 9)
    assert out[2].keypoints[13][2] >= pose.MIN_CONF and abs(out[2].keypoints[13][0] - obs[1].keypoints[13][0]) < 1
    assert out[5].keypoints[13][2] < pose.MIN_CONF and out[6].keypoints[13][2] < pose.MIN_CONF


def test_low_confidence_points_do_not_produce_angles():
    o = skeleton(0.0, conf=0.4)  # dá para desenhar (≥ 0,3), não para medir (< 0,5)
    assert pose.classify_posture(o) == "standing"
    assert pose.joint_angle(o, pose.L_HIP, pose.L_KNEE, pose.L_ANKLE) is None and pose.knee_angles(o) is None


# ----------------------------------------------------------- vista e cena
def test_view_classification():
    assert pose.view(skeleton(0.0)) == "frontal"
    assert pose.view(skeleton(0.0, side=True)) == "lateral"


def test_side_view_never_reports_asymmetry_between_sides():
    obs = []
    for i in range(60):
        o = skeleton(i / FPS, side=True)
        kp = list(o.keypoints)
        kp[13] = (kp[13][0] + squat(i / FPS, amp=40), kp[13][1], 0.9)  # só o joelho "esquerdo" se move
        obs.append(o.model_copy(update={"keypoints": kp}))
    result = analyze_detail(obs, [o.t for o in obs], [0] * 60)
    assert result.ranges["joelho esquerdo"]["rom"] > 30
    assert "rom_asymmetry" not in {e.type for e in result.events}


def run_scene(frames):
    times = [o.t for o in frames]
    found = analyze_detail(frames, times, [0] * len(frames))
    return found, describe_scene(frames, times, [0] * len(frames), found.series, found.postures, found.views, [2] * len(frames))


def test_scene_describes_squats_with_range_and_repetitions():
    found, scene = run_scene([skeleton(i / FPS, knee_off=squat(i / FPS)) for i in range(96)])  # 12 s, 4 ciclos
    assert len(scene) == 1
    s = scene[0]
    assert "agachamento" in s["exercise"] and s["joint"].startswith(("joelho", "quadril"))
    assert s["reps"] >= 3 and s["min"] < s["max"] and s["people"] == 2
    assert "2 pessoas em cena" in s["text"] and "vista frontal" in s["text"] and "em pé" in s["text"]
    assert {"min", "max", "rom"} <= set(found.ranges["joelho esquerdo"])


def test_scene_describes_arm_raises():
    _, scene = run_scene([skeleton(i / FPS, arm_up=0.5 * (1 - math.cos(2 * math.pi * i / FPS / 3))) for i in range(96)])
    assert "ombro" in scene[0]["exercise"] and scene[0]["joint"].startswith("ombro")


def test_scene_of_a_still_person_says_so():
    _, scene = run_scene([skeleton(i / FPS) for i in range(40)])
    assert "sem movimento repetitivo" in scene[0]["exercise"] and scene[0]["joint"] is None


# ------------------------------------------------- adaptador do YOLO (falso)
class FakeTensor:
    def __init__(self, data):
        self._d = np.asarray(data, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._d


class FakeYolo:
    """Duas pessoas: uma no canto (maior confiança) e uma no centro do quadro."""

    def __init__(self):
        self.calls = []

    def predict(self, frame, **kwargs):
        self.calls.append(kwargs)
        h, w = frame.shape[:2]
        boxes = [[0, 0, 0.2 * w, 0.3 * h], [0.3 * w, 0.1 * h, 0.7 * w, 0.9 * h]]
        kps = [[[5.0, 5.0, 0.9]] * 17, [[0.5 * w, 0.5 * h, 0.8]] * 17]
        res = type("R", (), {})()
        res.boxes = type("B", (), {"xyxy": FakeTensor(boxes), "conf": FakeTensor([0.95, 0.7]), "__len__": lambda s: 2})()
        res.keypoints = type("K", (), {"data": FakeTensor(kps)})()
        return [res]


def test_yolo_adapter_returns_everyone_and_refine_takes_the_centered_person():
    model = FakeYolo()
    estimator = YoloPoseEstimator(model=model, imgsz=640, refine_imgsz=480)
    people = estimator.estimate_all(np.zeros((240, 320, 3), np.uint8), 2.0)
    assert len(people) == 2 and people[0].conf == 0.95 and people[1].people == 2 and people[0].track_id is None
    assert model.calls[-1]["imgsz"] == 640
    keypoints, conf = estimator.refine(np.zeros((200, 100, 3), np.uint8))
    assert conf == 0.7 and keypoints[0][:2] == (50.0, 100.0)
    assert model.calls[-1]["imgsz"] == 480


# ------------------------------------------ analisador com duas pessoas
class TwoPeopleEstimator:
    """Estimador falso: terapeuta parado e paciente agachando, nas coordenadas do quadro analisado."""

    name = "falso (2 pessoas)"
    imgsz, refine_imgsz = 160, 480  # quadro inteiro em baixa resolução: o recorte compensa

    def __init__(self):
        self.refined = 0

    def reset(self):
        pass

    def estimate_all(self, frame, t):
        return [skeleton(t, cx=70, top=20, height=200), skeleton(t, cx=230, top=60, height=150, knee_off=squat(t))]

    def estimate(self, frame, t):
        return self.estimate_all(frame, t)[0]

    def refine(self, crop):
        self.refined += 1
        return None  # o recorte não achou nada melhor: fica a pose do quadro inteiro


@pytest.fixture
def clip(tmp_path):
    return write_video(tmp_path / "sessao.avi", [("walk", 14)])


def test_two_people_need_a_choice_and_only_the_patient_is_measured(clip, tmp_path):
    fake = TwoPeopleEstimator()
    analyzer = VideoAnalyzer(lambda: fake, sample_fps=2, detail_fps=5)
    scan = analyzer.scan(clip, "detailed")
    assert scan.needs_choice
    cards = scan.candidate_cards()
    assert len(cards) == 2 and cards[0]["suggested"] and cards[0]["image"].startswith("data:image/jpeg")
    patient_id, therapist_id = cards[0]["id"], cards[1]["id"]

    auto = analyzer.finish(clip, scan, None, annotated_to=tmp_path / "anotado")
    assert auto.people == {"max": 2, "analyzed": patient_id, "choice": "auto", "candidates": 2}
    assert auto.joint_ranges["joelho esquerdo"]["rom"] > 30 and auto.curve["n_reps"] >= 3
    assert "agachamento" in auto.scene[0]["exercise"] and "Pessoas em cena: até 2" in auto.report
    assert "O que foi visto" in auto.report and f"pessoa nº {patient_id}" in auto.report
    assert any("Mais de uma pessoa" in w for w in auto.quality.warnings)
    assert not {e.type for e in auto.events} & {"prolonged_immobility", "low_mobility"}  # agachar não é ficar parado
    assert fake.refined > 0 and auto.quality.refined_ratio == 0.0
    assert auto.annotated_path and Path(auto.annotated_path).suffix in (".mp4", ".webm")  # mp4 com ffmpeg; senão webm
    check = cv2.VideoCapture(auto.annotated_path)
    assert check.read()[0] and check.get(cv2.CAP_PROP_FRAME_COUNT) >= 60  # abre e tem os quadros analisados
    check.release()

    manual = analyzer.finish(clip, scan, therapist_id)
    assert manual.people["analyzed"] == therapist_id and manual.people["choice"] == "manual"
    assert not manual.curve or manual.curve["n_reps"] == 0
    assert manual.joint_ranges.get("joelho esquerdo", {"rom": 0})["rom"] < 5


def test_refined_pose_is_mapped_back_and_bad_refinements_are_rejected():
    o = skeleton(0.0, cx=100, top=40, height=100)  # coordenadas do quadro analisado (metade do original)
    frame = np.zeros((480, 640, 3), np.uint8)

    class Refiner:
        def __init__(self, conf, shift=0.0):
            self.conf, self.shift = conf, shift

        def refine(self, crop):
            self.crop = crop.shape
            # devolve os mesmos pontos, nas coordenadas do recorte (original = 2x)
            return [((x + 1 + self.shift) * 2 - self.x0, y * 2 - self.y0, self.conf) for x, y, _ in o.keypoints], 0.9

    good = Refiner(0.95)
    good.x0, good.y0 = int((80 - 6) * 2), int((40 - 10) * 2)
    better = VideoAnalyzer._refine_one(good, frame, o, 2.0)
    assert better is not None and abs(better.keypoints[13][0] - (o.keypoints[13][0] + 1)) < 0.6 and better.keypoints[13][2] == 0.95
    assert good.crop[0] == 240 and good.crop[1] == 104  # recorte com margem, na resolução original

    worse = Refiner(0.5)
    worse.x0, worse.y0 = good.x0, good.y0
    assert VideoAnalyzer._refine_one(worse, frame, o, 2.0) is None  # menos confiável que o original
    other = Refiner(0.95, shift=80)
    other.x0, other.y0 = good.x0, good.y0
    assert VideoAnalyzer._refine_one(other, frame, o, 2.0) is None  # esqueleto fora da caixa: outra pessoa


def test_refinement_only_when_the_crop_adds_pixels():
    refiner = type("R", (), {"imgsz": 640, "refine_imgsz": 480})()
    small = skeleton(0.0, top=100, height=150, frame=(960, 540))
    big = skeleton(0.0, top=10, height=520, frame=(960, 540))
    assert VideoAnalyzer._worth_refining(small, 2.0, (960, 540), refiner)
    assert not VideoAnalyzer._worth_refining(big, 2.0, (960, 540), refiner)


def test_unreliable_skeleton_drops_pose_findings_but_keeps_the_warning(clip):
    class Blurry(TwoPeopleEstimator):
        def estimate_all(self, frame, t):
            o = skeleton(t, cx=160, top=40, height=160, knee_off=squat(t), conf=0.32)
            kp = list(o.keypoints)
            kp[5] = (kp[5][0], kp[5][1] + 25, 0.32)  # ombros bem desnivelados
            return [o.model_copy(update={"keypoints": kp})]

    result = VideoAnalyzer(lambda: Blurry(), sample_fps=2, detail_fps=5).analyze(clip)
    assert not {e.type for e in result.events} & {"shoulder_tilt", "rom_asymmetry", "rep_irregular"}
    assert any("Esqueleto pouco confiável" in w for w in result.quality.warnings)


# ------------------------------------------------------------- web
def test_web_asks_who_is_the_patient_then_finishes(logged, clip):
    svc = logged.app.state.services
    svc.video = VideoAnalyzer(lambda: TwoPeopleEstimator(), sample_fps=2, detail_fps=5)
    with clip.open("rb") as f:
        r = logged.post("/patients/3/video", files={"file": ("sessao.avi", f, "video/x-msvideo")})
    assert r.status_code == 200 and "Quem é o paciente?" in r.text and "Analisar esta pessoa" in r.text
    token = r.text.split('name="token" value="')[1].split('"')[0]
    person = r.text.split('name="person" value="')[1].split('"')[0]

    assert logged.post("/patients/4/video", data={"token": token, "person": person}).status_code == 410  # outro paciente
    done = logged.post("/patients/3/video", data={"token": token, "person": person})
    assert done.status_code == 200 and "O que foi visto no vídeo" in done.text and "Ângulos por articulação" in done.text
    assert f"pessoa nº {person}" in done.text and "indicada pelo médico" in done.text
    url = done.text.split('<video class="annotated" controls playsinline muted preload="metadata" src="')[1].split('"')[0]
    video = logged.get(url)
    assert video.status_code == 200 and video.headers["content-type"] in ("video/mp4", "video/webm") and len(video.content) > 1000
    part = logged.get(url, headers={"Range": "bytes=0-99"})  # o player do navegador pede o arquivo em pedaços
    assert part.status_code == 206 and len(part.content) == 100
    assert logged.post("/patients/3/video", data={"token": token, "person": person}).status_code == 410  # já usado


# ------------------------------------------------- descrição por IA (opcional)
class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def test_scene_narrator_sends_frames_and_returns_one_paragraph(monkeypatch):
    sent = {}

    def post(url, **kwargs):
        sent["url"], sent["body"] = url, kwargs["json"]
        return FakeResponse(200, {"choices": [{"message": {"content": "Sala com maca.\n\nPaciente agacha."}}]})

    monkeypatch.setattr("vitaltrace.video.narrative.requests.post", post)
    narrator = AzureOpenAISceneNarrator("https://x.openai.azure.com/", "k", "visao", "2025-04-01-preview")
    text = narrator.describe([b"jpeg1", b"jpeg2"], "8 repetições")
    assert text == "Sala com maca. Paciente agacha." and "/deployments/visao/chat/completions" in sent["url"]
    content = sent["body"]["messages"][1]["content"]
    assert sum(c["type"] == "image_url" for c in content) == 2 and "8 repetições" in content[0]["text"]


def test_scene_narrator_failure_is_a_warning_not_an_error(monkeypatch, clip):
    monkeypatch.setattr("vitaltrace.video.narrative.requests.post", lambda url, **k: FakeResponse(404, text="DeploymentNotFound"))
    narrator = AzureOpenAISceneNarrator("https://x.openai.azure.com", "k", "visao", "v")
    with pytest.raises(NarrativeError):
        narrator.describe([b"x"], "")
    result = VideoAnalyzer(lambda: TwoPeopleEstimator(), sample_fps=2, detail_fps=5, narrator=narrator).analyze(clip)
    assert result.narrative == "" and any("Descrição da cena por IA indisponível" in w for w in result.quality.warnings)


def test_annotated_video_falls_back_to_webm_without_ffmpeg_and_rejects_empty_files(clip, tmp_path, monkeypatch):
    from vitaltrace.video import analyzer as module

    monkeypatch.setattr(module.shutil, "which", lambda name: None)  # máquina sem ffmpeg
    result = VideoAnalyzer(lambda: TwoPeopleEstimator(), sample_fps=2, detail_fps=5).analyze(clip, annotated_to=tmp_path / "a")
    assert result.annotated_path.endswith(".webm") and module._playable(Path(result.annotated_path))

    broken = tmp_path / "vazio.mp4"
    broken.write_bytes(b"\x00" * 4096)
    assert not module._playable(broken)
    monkeypatch.setattr(module, "_encode_h264", lambda *a: None)
    monkeypatch.setattr(module, "_encode_vp8", lambda *a: None)
    failed = VideoAnalyzer(lambda: TwoPeopleEstimator(), sample_fps=2, detail_fps=5).analyze(clip, annotated_to=tmp_path / "b")
    assert failed.annotated_path is None and any("vídeo com o esqueleto" in w for w in failed.quality.warnings)
