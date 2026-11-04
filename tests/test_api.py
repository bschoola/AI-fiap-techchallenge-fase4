import io
import wave

import pytest
from fastapi.testclient import TestClient

from medmonitor.api.main import create_app
from medmonitor.config import Settings


@pytest.fixture
def client(settings):
    return TestClient(create_app(settings))


def silent_wav(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * int(16000 * seconds))
    return buffer.getvalue()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["mock_azure"] is True


def test_text_analysis_creates_open_alerts_and_ack_flow(client):
    response = client.post(
        "/text/analyze", json={"patient_id": "p1", "text": "Estou com falta de ar"}
    )
    assert response.status_code == 200
    assert response.json()["alerts"][0]["severity"] == "critical"

    open_alerts = client.get("/alerts", params={"status": "open"}).json()
    assert len(open_alerts) >= 1

    alert_id = open_alerts[0]["id"]
    acked = client.post(f"/alerts/{alert_id}/ack").json()
    assert acked["status"] == "acknowledged"
    remaining = {a["id"] for a in client.get("/alerts", params={"status": "open"}).json()}
    assert alert_id not in remaining


def test_audio_upload_in_mock_mode(client):
    response = client.post(
        "/audio/analyze",
        data={"patient_id": "p9"},
        files={"file": ("consulta.wav", silent_wav(), "audio/wav")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["patient_id"] == "p9"
    assert any(a["severity"] == "critical" for a in body["alerts"])


def test_empty_upload_is_rejected(client):
    response = client.post(
        "/audio/analyze",
        data={"patient_id": "p1"},
        files={"file": ("vazio.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400


def test_unknown_alert_returns_404(client):
    assert client.post("/alerts/nao-existe/ack").status_code == 404


def test_missing_azure_config_fails_fast(tmp_path):
    settings = Settings(_env_file=None, mock_azure=False, database_path=str(tmp_path / "a.db"))
    with pytest.raises(RuntimeError, match="AZURE_SPEECH_KEY"):
        create_app(settings)
