"""Orquestra: transcrição → texto → termos críticos → métricas → prosódia → tom de voz → alertas."""

import logging
import wave
from pathlib import Path

from vitaltrace.alerts.engine import build_audio_alerts
from vitaltrace.audio.critical_terms import find_critical_terms
from vitaltrace.audio.language_service import TextAnalyzer
from vitaltrace.audio.metrics import compute_speech_metrics
from vitaltrace.audio.models import AudioAnalysis, ProsodyMetrics, Transcription, VoiceTone
from vitaltrace.audio.prosody import analyze_prosody
from vitaltrace.audio.speech_service import Transcriber
from vitaltrace.audio.voice_tone import VoiceToneAnalyzer, VoiceToneError
from vitaltrace.config import Settings

log = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(
        self,
        transcriber: Transcriber,
        text_analyzer: TextAnalyzer,
        settings: Settings,
        tone_analyzer: VoiceToneAnalyzer | None = None,
        tone_unavailable: str | None = None,
    ) -> None:
        self._transcriber = transcriber
        self._text_analyzer = text_analyzer
        self._settings = settings
        self._tone_analyzer = tone_analyzer
        self._tone_unavailable = tone_unavailable  # motivo, quando foi pedido mas não pôde ser ligado

    def analyze_audio(self, patient_id: str, wav_path: Path) -> AudioAnalysis:
        transcription = self._transcriber.transcribe(wav_path)
        warnings: list[str] = []
        prosody = self._prosody(wav_path, warnings)
        voice_tone = self._voice_tone(wav_path, transcription.text, prosody, warnings)
        return self._analyze(patient_id, transcription, prosody, voice_tone, warnings)

    def analyze_text(self, patient_id: str, text: str) -> AudioAnalysis:
        transcription = Transcription(
            text=text, language=self._settings.speech_language, segments=[]
        )
        return self._analyze(patient_id, transcription)

    def _prosody(self, wav_path: Path, warnings: list[str]) -> ProsodyMetrics | None:
        """A prosódia nunca derruba a análise: sem WAV legível, segue só com a transcrição."""
        if not wav_path.is_file():
            return None
        try:
            return analyze_prosody(wav_path)
        except (OSError, ValueError, EOFError, wave.Error) as exc:
            log.warning("Prosódia indisponível para %s: %s", wav_path.name, exc)
            warnings.append("Não foi possível medir pausas e tom neste arquivo (formato de áudio não suportado).")
            return None

    def _voice_tone(
        self,
        wav_path: Path,
        transcript: str,
        prosody: ProsodyMetrics | None,
        warnings: list[str],
    ) -> VoiceTone | None:
        if self._tone_analyzer is None:
            if self._tone_unavailable:
                warnings.append(self._tone_unavailable)
            return None
        if not wav_path.is_file():
            return None
        try:
            return self._tone_analyzer.analyze(wav_path, transcript, prosody)
        except (VoiceToneError, OSError) as exc:
            log.warning("Tom de voz indisponível: %s", exc)
            warnings.append(f"Avaliação do tom de voz indisponível nesta análise ({exc}).")
            return None

    def _analyze(
        self,
        patient_id: str,
        transcription: Transcription,
        prosody: ProsodyMetrics | None = None,
        voice_tone: VoiceTone | None = None,
        warnings: list[str] | None = None,
    ) -> AudioAnalysis:
        text_analysis = self._text_analyzer.analyze(transcription.text)
        terms = find_critical_terms(transcription.text)
        metrics = compute_speech_metrics(transcription)
        alerts = build_audio_alerts(
            patient_id, terms, text_analysis, metrics, self._settings, prosody, voice_tone
        )
        return AudioAnalysis(
            patient_id=patient_id,
            transcription=transcription,
            text_analysis=text_analysis,
            critical_terms=terms,
            metrics=metrics,
            prosody=prosody,
            voice_tone=voice_tone,
            warnings=warnings or [],
            alerts=alerts,
        )
