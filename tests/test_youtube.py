"""Análise de vídeo por link do YouTube (o download é simulado: os testes não acessam a internet)."""

import shutil
from pathlib import Path

import pytest
from video_fixtures import write_video
from yt_dlp.utils import DownloadError

from vitaltrace.video import youtube

ID = "dQw4w9WgXcQ"
CANON = f"https://www.youtube.com/watch?v={ID}"


@pytest.mark.parametrize(
    "raw",
    [
        f"https://www.youtube.com/watch?v={ID}",
        f"http://youtube.com/watch?v={ID}&t=42s&list=PLxyz",
        f"https://m.youtube.com/watch?feature=share&v={ID}",
        f"https://youtu.be/{ID}?si=abc",
        f"https://www.youtube.com/shorts/{ID}",
        f"https://www.youtube.com/embed/{ID}",
        f"https://www.youtube.com/live/{ID}?feature=share",
        f"  https://youtu.be/{ID}  ",
    ],
)
def test_accepts_youtube_links_and_builds_canonical_url(raw):
    assert youtube.canonical_url(raw) == CANON


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "não é um link",
        f"https://vimeo.com/watch?v={ID}",
        f"https://youtube.com.evil.example/watch?v={ID}",
        f"https://evil.example/youtu.be/{ID}",
        f"https://evil.example/?u=https://youtu.be/{ID}",
        f"file:///etc/passwd#{ID}",
        f"javascript:alert('{ID}')",
        f"ftp://youtube.com/watch?v={ID}",
        "https://www.youtube.com/watch?v=curto",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/playlist?list=PL1234567890A",
        "https://www.youtube.com/@canal",
        "https://youtu.be/",
    ],
)
def test_rejects_everything_else(raw):
    with pytest.raises(youtube.YouTubeError):
        youtube.canonical_url(raw)


class FakeYDL:
    """Troca o yt-dlp: devolve metadados e 'baixa' um vídeo sintético para a pasta de destino."""

    source: Path | None = None
    info: dict = {"title": "Queda no corredor", "duration": 300, "is_live": False}
    error: Exception | None = None
    last: "FakeYDL | None" = None

    def __init__(self, options):
        self.options = options
        FakeYDL.last = self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=False):
        self.url = url
        if FakeYDL.error:
            raise FakeYDL.error
        return dict(FakeYDL.info)

    def download(self, urls):
        assert urls == [self.url]
        target = Path(self.options["outtmpl"].replace("%(ext)s", FakeYDL.source.suffix.lstrip(".")))
        shutil.copy(FakeYDL.source, target)


@pytest.fixture
def fake_ydl(monkeypatch, tmp_path):
    FakeYDL.source = write_video(tmp_path / "origem.avi", [("empty", 3), ("walk", 8), ("fall", 10)])
    FakeYDL.info = {"title": "Queda no corredor", "duration": 300, "is_live": False}
    FakeYDL.error = None
    FakeYDL.last = None
    monkeypatch.setattr(youtube, "_make_ydl", FakeYDL)
    return FakeYDL


def test_download_clip_uses_safe_options_and_returns_the_clip(fake_ydl, tmp_path):
    dest = tmp_path / "dl"
    dest.mkdir()
    clip = youtube.download_clip(f"https://youtu.be/{ID}?si=zzz&foo=bar", dest, max_seconds=120, max_mb=200)
    assert clip.path.exists() and clip.path.parent == dest and clip.url == CANON
    assert clip.title == "Queda no corredor" and clip.seconds == 120 and clip.size_mb >= 0
    used = fake_ydl.last
    assert used.url == CANON  # só o ID chegou ao yt-dlp; o resto do que foi digitado foi descartado
    assert used.options["noplaylist"] is True and used.options["allowed_extractors"] == ["youtube"]
    assert "download_ranges" in used.options and used.options["max_filesize"] == 200 * 1024 * 1024
    assert "res:720" in used.options["format_sort"]


def test_live_streams_are_rejected(fake_ydl, tmp_path):
    fake_ydl.info = {"title": "Ao vivo", "is_live": True}
    with pytest.raises(youtube.YouTubeError, match="ao vivo"):
        youtube.download_clip(CANON, tmp_path, max_seconds=120, max_mb=200)


@pytest.mark.parametrize(
    "message, expected, status",
    [
        ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", "privado", 422),
        ("ERROR: Sign in to confirm your age. This video may be inappropriate", "restrição de idade", 422),
        ("ERROR: Sign in to confirm you’re not a bot", "bloqueou o download", 422),
        ("ERROR: Video unavailable", "indisponível", 422),
        ("ERROR: Requested format is not available", "formato utilizável", 422),
        ("ERROR: Unable to download webpage: <urlopen error [Errno -3] Temporary failure in name resolution>", "Sem conexão", 502),
        ("ERROR: algo inesperado", "yt-dlp", 422),
    ],
)
def test_download_errors_become_friendly_messages(fake_ydl, tmp_path, message, expected, status):
    fake_ydl.error = DownloadError(message)
    with pytest.raises(youtube.YouTubeError, match=expected) as info:
        youtube.download_clip(CANON, tmp_path, max_seconds=120, max_mb=200)
    assert info.value.status == status


def test_oversized_clip_is_rejected(fake_ydl, tmp_path):
    with pytest.raises(youtube.YouTubeError) as info:
        youtube.download_clip(CANON, tmp_path, max_seconds=120, max_mb=0)
    assert info.value.status == 413


# ----------------------------------------------------------------- na aplicação
def test_video_page_offers_the_youtube_field(logged):
    html = logged.get("/patients/3/video").text
    assert "Link do YouTube" in html and 'name="url"' in html


def test_youtube_link_is_analyzed_and_shown_in_the_timeline(logged, fake_ydl):
    r = logged.post("/patients/3/video", data={"url": f"https://youtu.be/{ID}"})
    assert r.status_code == 200
    assert "Queda suspeita" in r.text and "vídeo do YouTube" in r.text and "Queda no corredor" in r.text
    html = logged.get("/patients/3/evolution").text
    assert f'href="{CANON}"' in html and 'rel="noopener noreferrer"' in html
    assert "Vídeo do YouTube" in html and "primeiros 120 s" in html
    assert any(a["patient_id"] == 3 and a["source"] == "video" and a["severity"] == "critical"
               for a in logged.get("/api/alerts").json())


def test_invalid_or_missing_youtube_input_is_rejected(logged, fake_ydl):
    r = logged.post("/patients/3/video", data={"url": "https://evil.example/watch?v=" + ID})
    assert r.status_code == 422 and "Link inválido" in r.text
    r = logged.post("/patients/3/video", data={"url": "  "})
    assert r.status_code == 400 and "link do YouTube" in r.text
    fake_ydl.error = DownloadError("ERROR: Private video")
    r = logged.post("/patients/3/video", data={"url": CANON})
    assert r.status_code == 422 and "privado" in r.text


def test_uploaded_file_wins_over_the_link(logged, fake_ydl, tmp_path):
    path = write_video(tmp_path / "anda.avi", [("empty", 3), ("walk", 14)])
    with path.open("rb") as f:
        r = logged.post("/patients/5/video", data={"url": CANON}, files={"file": ("anda.avi", f, "video/x-msvideo")})
    assert r.status_code == 200 and "arquivo <b>anda.avi</b>" in r.text
    assert fake_ydl.last is None  # o yt-dlp nem foi chamado


def test_youtube_can_be_disabled(logged, monkeypatch):
    monkeypatch.setattr(youtube, "available", lambda: False)
    assert "Link do YouTube" not in logged.get("/patients/3/video").text
    r = logged.post("/patients/3/video", data={"url": CANON})
    assert r.status_code == 503 and "não está habilitado" in r.text
