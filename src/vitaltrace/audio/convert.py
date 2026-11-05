"""Normalização de áudio para WAV PCM 16 kHz mono (formato ideal para o Azure Speech)."""

import shutil
import subprocess
from pathlib import Path


class AudioConversionError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def prepare_wav(src: Path, workdir: Path) -> Path:
    """Converte com ffmpeg quando disponível; sem ffmpeg, só aceita .wav."""
    if ffmpeg_available():
        dst = workdir / "normalized.wav"
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", str(dst),
        ]  # fmt: skip
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        except subprocess.CalledProcessError as exc:
            raise AudioConversionError(
                "Não foi possível converter o áudio (arquivo corrompido ou formato "
                "não suportado)."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AudioConversionError("Tempo esgotado ao converter o áudio.") from exc
        return dst

    if src.suffix.lower() == ".wav":
        return src
    raise AudioConversionError(
        "ffmpeg não encontrado: envie um arquivo .wav (PCM 16 kHz mono) ou instale o ffmpeg."
    )
