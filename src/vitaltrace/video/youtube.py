"""Baixa um trecho de vídeo do YouTube para a análise (recurso opcional, pensado para demonstração).

Cuidados:
- só aceita endereços do YouTube; a URL entregue ao yt-dlp é montada aqui a partir do ID de 11
  caracteres, então nada que o usuário digitou além do ID chega ao downloader;
- baixa só os primeiros `max_seconds`, em até 720p (configurável) e preferindo H.264 (o OpenCV decodifica melhor);
- o arquivo vive numa pasta temporária e é apagado pelo chamador: o vídeo não é guardado.

O yt-dlp muda com frequência. Se o download falhar do nada: `pip install -U "yt-dlp[default]"`.
Desde a versão 2025.11.12 o YouTube também exige um runtime JavaScript (Deno) instalado.
"""

import importlib.util
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlparse

_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be", "www.youtu.be"}
_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_PATH_ID = re.compile(r"^/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})(?:[/?]|$)")
_TITLE_MAX = 120

_UPDATE_HINT = 'Atualize com pip install -U "yt-dlp[default]" (e instale o Deno) ou envie o arquivo.'


class YouTubeError(RuntimeError):
    """Erro apresentável ao usuário. `status` é o código HTTP sugerido para a resposta."""

    def __init__(self, message: str, status: int = 422) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class YouTubeClip:
    path: Path
    url: str
    title: str
    seconds: float
    size_mb: float


def available() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def video_id(raw: str) -> str:
    """Extrai o ID de um link do YouTube ou levanta YouTubeError."""
    parsed = urlparse(raw.strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in ("http", "https") or host not in _HOSTS:
        raise YouTubeError("Link inválido: use um endereço do YouTube (youtube.com ou youtu.be).")
    candidate = ""
    if host.endswith("youtu.be"):
        candidate = parsed.path.lstrip("/").split("/")[0]
    elif parsed.path == "/watch":
        candidate = (parse_qs(parsed.query).get("v") or [""])[0]
    else:
        match = _PATH_ID.match(parsed.path)
        candidate = match.group(1) if match else ""
    if not _ID.match(candidate):
        raise YouTubeError("Não encontrei o vídeo no link. Cole o endereço de um vídeo, não de um canal ou lista.")
    return candidate


def canonical_url(raw: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id(raw)}"


def _make_ydl(options: dict):  # separado para os testes trocarem por um download simulado
    import yt_dlp

    return yt_dlp.YoutubeDL(options)


def _friendly(exc: Exception) -> YouTubeError:
    text = str(exc).lower()
    if "private video" in text:
        return YouTubeError("O vídeo é privado. Use um vídeo público.")
    if "confirm your age" in text or "age-restricted" in text or "inappropriate for some users" in text:
        return YouTubeError("O vídeo tem restrição de idade e exige login. Escolha outro vídeo.")
    if "not a bot" in text or "sign in to confirm" in text:
        return YouTubeError("O YouTube pediu verificação e bloqueou o download automático. Tente outro vídeo ou envie o arquivo.")
    if "unavailable" in text or "removed" in text or "terminated" in text or "does not exist" in text:
        return YouTubeError("Vídeo indisponível, removido ou bloqueado para este país.")
    if "requested format is not available" in text or "no video formats" in text:
        return YouTubeError(f"O YouTube não ofereceu um formato utilizável. {_UPDATE_HINT}")
    if any(k in text for k in ("urlopen", "timed out", "name or service", "connection", "network", "resolve")):
        return YouTubeError("Sem conexão com o YouTube a partir do servidor. Verifique a internet.", 502)
    return YouTubeError(f"Não foi possível baixar o vídeo do YouTube. {_UPDATE_HINT}")


def download_clip(raw_url: str, dest: Path, *, max_seconds: int, max_mb: int, max_height: int = 720) -> YouTubeClip:
    url = canonical_url(raw_url)
    if not available():
        raise YouTubeError("O download do YouTube não está instalado neste servidor (falta o yt-dlp).", 503)

    from yt_dlp.utils import DownloadError, download_range_func

    options = {
        "format": "bv*[vcodec!^=av01]/b[vcodec!^=av01]/bv*/b",  # só vídeo: o áudio não é usado
        "format_sort": [f"res:{max(144, int(max_height))}", "vcodec:h264"],  # a melhor resolução até o limite
        "noplaylist": True,
        "allowed_extractors": ["youtube"],
        "download_ranges": download_range_func(None, [(0, max_seconds)]),
        "outtmpl": str(dest / "youtube.%(ext)s"),
        "max_filesize": max_mb * 1024 * 1024,
        "socket_timeout": 20,
        "retries": 2,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with _make_ydl(options) as ydl:
            info = ydl.extract_info(url, download=False) or {}
            if info.get("is_live") or info.get("live_status") in ("is_live", "is_upcoming"):
                raise YouTubeError("Transmissões ao vivo não são suportadas. Use um vídeo gravado.")
            ydl.download([url])
    except YouTubeError:
        raise
    except DownloadError as exc:
        raise _friendly(exc) from exc
    except Exception as exc:  # noqa: BLE001 — qualquer falha do yt-dlp vira uma mensagem clara
        raise _friendly(exc) from exc

    files = sorted(p for p in dest.glob("youtube.*") if p.suffix not in (".part", ".ytdl") and p.is_file())
    if not files:
        raise YouTubeError(f"O download terminou sem gerar o vídeo. {_UPDATE_HINT}")
    path = files[0]
    size = path.stat().st_size
    if size == 0:
        raise YouTubeError("O download veio vazio. Tente novamente ou envie o arquivo.")
    if size > max_mb * 1024 * 1024:
        raise YouTubeError(f"O trecho baixado passa de {max_mb} MB. Escolha um vídeo mais curto.", 413)
    duration = float(info.get("duration") or max_seconds)
    return YouTubeClip(
        path=path,
        url=url,
        title=str(info.get("title") or "Vídeo do YouTube")[:_TITLE_MAX],
        seconds=min(duration, float(max_seconds)),
        size_mb=round(size / 1024 / 1024, 1),
    )
