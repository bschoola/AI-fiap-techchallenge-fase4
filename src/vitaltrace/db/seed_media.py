"""Análises de áudio e vídeo de exemplo para a linha do tempo (dados 100% sintéticos).

Os resultados de áudio passam pelos detectores reais (termos críticos, ritmo de fala); só a
transcrição é escrita à mão. Os "quadros-chave" de vídeo são bonecos desenhados com OpenCV,
marcados como ilustrativos: não há paciente nem filmagem reais no projeto.
"""

import base64

import cv2
import numpy as np

from vitaltrace.audio.critical_terms import find_critical_terms
from vitaltrace.audio.metrics import compute_speech_metrics
from vitaltrace.audio.mock import MockTextAnalyzer
from vitaltrace.audio.models import AudioAnalysis, SpeechSegment, Transcription

_W, _H = 480, 270
_INK, _TEAL, _AMBER = (43, 42, 14), (110, 118, 15), (61, 184, 245)  # BGR


def audio_payload(segments: list[tuple[float, float, str]], filename: str, size_kb: int) -> dict:
    transcription = Transcription(
        text=" ".join(text for _, _, text in segments),
        language="pt-BR",
        segments=[SpeechSegment(start=a, end=b, text=t) for a, b, t in segments],
    )
    analysis = AudioAnalysis(
        patient_id="seed",
        transcription=transcription,
        text_analysis=MockTextAnalyzer().analyze(transcription.text),
        critical_terms=find_critical_terms(transcription.text),
        metrics=compute_speech_metrics(transcription),
    )
    payload = analysis.model_dump(mode="json", exclude={"alerts", "patient_id"})
    payload["patient_id"] = "seed"
    payload["input"] = {"type": "file", "filename": filename, "size_kb": size_kb, "mock": False}
    payload["seeded"] = True
    return payload


def _figure(img: np.ndarray, x: int, floor: int, lying: bool, knee_gap: int) -> None:
    """Boneco de palitos: em pé (ou deitado). `knee_gap` abre um joelho para ilustrar assimetria."""
    if lying:
        cv2.circle(img, (x - 70, floor - 18), 14, _INK, 3)
        cv2.line(img, (x - 56, floor - 18), (x + 40, floor - 18), _INK, 4)
        cv2.line(img, (x + 40, floor - 18), (x + 85, floor - 8), _INK, 4)
        cv2.line(img, (x - 40, floor - 18), (x - 20, floor - 40), _INK, 3)
        return
    cv2.circle(img, (x, floor - 150), 14, _INK, 3)
    cv2.line(img, (x, floor - 136), (x, floor - 70), _INK, 4)
    cv2.line(img, (x, floor - 128), (x - 28, floor - 96), _INK, 3)
    cv2.line(img, (x, floor - 128), (x + 28, floor - 96), _INK, 3)
    cv2.line(img, (x, floor - 70), (x - 14, floor - 36), _INK, 4)
    cv2.line(img, (x - 14, floor - 36), (x - 14, floor), _INK, 4)
    cv2.line(img, (x, floor - 70), (x + 14 + knee_gap, floor - 36), _INK, 4)
    cv2.line(img, (x + 14 + knee_gap, floor - 36), (x + 14, floor), _INK, 4)


def synthetic_frame(label: str, t: float, *, lying: bool = False, knee_gap: int = 0) -> str:
    img = np.full((_H, _W, 3), (247, 247, 244), np.uint8)
    floor = _H - 40
    cv2.line(img, (0, floor), (_W, floor), (216, 226, 226), 2)
    _figure(img, _W // 2, floor, lying, knee_gap)
    x1, y1, x2, y2 = (_W // 2 - 95, floor - 175, _W // 2 + 95, floor + 4) if not lying else (_W // 2 - 100, floor - 60, _W // 2 + 110, floor + 4)
    cv2.rectangle(img, (x1, y1), (x2, y2), _AMBER, 2)
    cv2.rectangle(img, (0, 0), (_W, 26), _INK, -1)
    cv2.putText(img, f"{label} - {t:.0f} s", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Quadro ilustrativo (dados sinteticos)", (8, _H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEAL, 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode() if ok else ""


def video_payload(
    mobility: float,
    *,
    filename: str,
    duration_s: float = 60.0,
    events: list[dict] | None = None,
    keyframes: list[dict] | None = None,
    knee_diff: float | None = None,
    immobility_s: float = 0.0,
) -> dict:
    events = events or []
    frames = int(duration_s * 2)
    metrics = {
        "duration_s": duration_s,
        "frames_analyzed": frames,
        "person_detected_ratio": 0.96,
        "mobility_index": mobility,
        "immobility_s": immobility_s,
        "posture_seconds": {"standing": round(duration_s * 0.8), "sitting": round(duration_s * 0.2)},
        "median_knee_diff_deg": knee_diff,
    }
    lines = [
        f"Vídeo de {duration_s:.0f} s (exemplo sintético): {frames} quadros amostrados, pessoa visível em 96% deles.",
        f"Índice de mobilidade: {mobility:.2f} (0 = parado, 1 = muito ativo).",
    ]
    if not events:
        lines.append("Nenhum desvio detectado nesta gravação.")
    else:
        lines.append(f"{len(events)} evento(s) detectado(s).")
        lines += [f"• {e['description']}" for e in events]
    return {
        "backend": "exemplo do seed (sintético)",
        "metrics": metrics,
        "events": events,
        "report": "\n".join(lines),
        "keyframes": keyframes or [],
        "input": {"filename": filename, "size_mb": 6.4},
        "seeded": True,
    }
