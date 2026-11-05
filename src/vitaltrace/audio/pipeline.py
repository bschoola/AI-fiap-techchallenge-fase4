"""Orquestra: transcrição → análise de texto → termos críticos → métricas → alertas."""

from pathlib import Path

from vitaltrace.alerts.engine import build_audio_alerts
from vitaltrace.audio.critical_terms import find_critical_terms
from vitaltrace.audio.language_service import TextAnalyzer
from vitaltrace.audio.metrics import compute_speech_metrics
from vitaltrace.audio.models import AudioAnalysis, Transcription
from vitaltrace.audio.speech_service import Transcriber
from vitaltrace.config import Settings


class AudioPipeline:
    def __init__(
        self, transcriber: Transcriber, text_analyzer: TextAnalyzer, settings: Settings
    ) -> None:
        self._transcriber = transcriber
        self._text_analyzer = text_analyzer
        self._settings = settings

    def analyze_audio(self, patient_id: str, wav_path: Path) -> AudioAnalysis:
        transcription = self._transcriber.transcribe(wav_path)
        return self._analyze(patient_id, transcription)

    def analyze_text(self, patient_id: str, text: str) -> AudioAnalysis:
        transcription = Transcription(
            text=text, language=self._settings.speech_language, segments=[]
        )
        return self._analyze(patient_id, transcription)

    def _analyze(self, patient_id: str, transcription: Transcription) -> AudioAnalysis:
        text_analysis = self._text_analyzer.analyze(transcription.text)
        terms = find_critical_terms(transcription.text)
        metrics = compute_speech_metrics(transcription)
        alerts = build_audio_alerts(
            patient_id, terms, text_analysis, metrics, self._settings
        )
        return AudioAnalysis(
            patient_id=patient_id,
            transcription=transcription,
            text_analysis=text_analysis,
            critical_terms=terms,
            metrics=metrics,
            alerts=alerts,
        )
