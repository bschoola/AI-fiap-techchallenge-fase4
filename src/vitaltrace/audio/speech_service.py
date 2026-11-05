"""Transcrição com Azure AI Speech (Speech to Text)."""

import threading
from pathlib import Path
from typing import Protocol

from vitaltrace.audio.models import SpeechSegment, Transcription

_TICKS_PER_SECOND = 10_000_000  # o SDK mede offset/duração em unidades de 100 ns


class TranscriptionError(RuntimeError):
    pass


class Transcriber(Protocol):
    def transcribe(self, wav_path: Path) -> Transcription: ...


class AzureSpeechTranscriber:
    """Reconhecimento contínuo (suporta áudios longos, ao contrário do `recognize_once`)."""

    def __init__(
        self, key: str, region: str, language: str = "pt-BR", timeout_s: float = 300.0
    ) -> None:
        self._key = key
        self._region = region
        self._language = language
        self._timeout_s = timeout_s

    def transcribe(self, wav_path: Path) -> Transcription:
        import azure.cognitiveservices.speech as speechsdk  # import tardio: SDK é opcional no modo mock

        speech_config = speechsdk.SpeechConfig(
            subscription=self._key, region=self._region
        )
        speech_config.speech_recognition_language = self._language
        audio_config = speechsdk.audio.AudioConfig(filename=str(wav_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )

        segments: list[SpeechSegment] = []
        errors: list[str] = []
        done = threading.Event()

        def on_recognized(evt: "speechsdk.SpeechRecognitionEventArgs") -> None:
            result = evt.result
            if (
                result.reason == speechsdk.ResultReason.RecognizedSpeech
                and result.text
            ):
                start = result.offset / _TICKS_PER_SECOND
                segments.append(
                    SpeechSegment(
                        start=start,
                        end=start + result.duration / _TICKS_PER_SECOND,
                        text=result.text,
                    )
                )

        def on_canceled(evt: "speechsdk.SpeechRecognitionCanceledEventArgs") -> None:
            details = evt.cancellation_details
            if details.reason == speechsdk.CancellationReason.Error:
                errors.append(f"{details.error_code}: {details.error_details}")
            done.set()  # EndOfStream também chega aqui, sem erro

        recognizer.recognized.connect(on_recognized)
        recognizer.session_stopped.connect(lambda _evt: done.set())
        recognizer.canceled.connect(on_canceled)

        recognizer.start_continuous_recognition()
        finished = done.wait(self._timeout_s)
        recognizer.stop_continuous_recognition()

        if errors:
            raise TranscriptionError(f"Azure Speech falhou: {errors[0]}")
        if not finished:
            raise TranscriptionError("Tempo esgotado aguardando o Azure Speech.")

        return Transcription(
            text=" ".join(s.text for s in segments),
            language=self._language,
            segments=segments,
        )
