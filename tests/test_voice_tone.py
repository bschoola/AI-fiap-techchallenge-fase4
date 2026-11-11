import pytest

from vitaltrace.audio import voice_tone
from vitaltrace.audio.models import ProsodyMetrics
from vitaltrace.audio.voice_tone import (
    AzureOpenAIVoiceToneAnalyzer,
    MockVoiceToneAnalyzer,
    VoiceToneError,
    parse_voice_tone,
)
from vitaltrace.config import Settings
from vitaltrace.services import build_voice_tone


def test_parse_accepts_code_fence_and_normalizes_levels():
    content = '```json\n{"respiratory_effort": "Intenso", "fatigue": "altíssimo", "emotional_tone": "abatida", "summary": "Puxa o ar entre as palavras."}\n```'
    tone = parse_voice_tone(content, "gpt-audio")

    assert tone.respiratory_effort == "intenso"
    assert tone.fatigue == "nenhum"  # valor fora da escala não é aceito
    assert tone.emotional_tone == "abatida"
    assert tone.source == "gpt-audio"


@pytest.mark.parametrize("content", ["", "não consigo ouvir o áudio", "{quebrado", "[1, 2]"])
def test_parse_rejects_non_json(content):
    with pytest.raises(VoiceToneError):
        parse_voice_tone(content, "gpt-audio")


class FakeResponse:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


def test_azure_analyzer_sends_audio_and_reads_json(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    sent = {}

    def fake_post(url, params, headers, json, timeout):
        sent.update(url=url, params=params, headers=headers, body=json)
        content = '{"respiratory_effort": "moderado", "fatigue": "leve", "emotional_tone": "ansiosa", "summary": "ok"}'
        return FakeResponse(200, {"choices": [{"message": {"content": content}}]})

    monkeypatch.setattr(voice_tone.requests, "post", fake_post)
    analyzer = AzureOpenAIVoiceToneAnalyzer("https://x.openai.azure.com/", "k", "gpt-audio", "2025-04-01-preview")
    tone = analyzer.analyze(wav, "texto", None)

    assert tone.respiratory_effort == "moderado"
    assert sent["url"] == "https://x.openai.azure.com/openai/deployments/gpt-audio/chat/completions"
    assert sent["headers"] == {"api-key": "k"}
    assert sent["body"]["modalities"] == ["text"]
    audio_part = sent["body"]["messages"][1]["content"][1]
    assert audio_part["type"] == "input_audio" and audio_part["input_audio"]["format"] == "wav"


def test_azure_analyzer_raises_on_http_error(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFFfake")
    monkeypatch.setattr(voice_tone.requests, "post", lambda *a, **k: FakeResponse(400, text="bad request"))
    analyzer = AzureOpenAIVoiceToneAnalyzer("https://x", "k", "gpt-audio", "v")
    with pytest.raises(VoiceToneError, match="HTTP 400"):
        analyzer.analyze(wav, "", None)


def test_mock_follows_measured_pauses(tmp_path):
    mock = MockVoiceToneAnalyzer()
    calm = mock.analyze(tmp_path / "x.wav", "", ProsodyMetrics(pause_ratio=0.2, reliable=True))
    tired = mock.analyze(tmp_path / "x.wav", "", ProsodyMetrics(pause_ratio=0.55, reliable=True))
    assert (calm.respiratory_effort, tired.respiratory_effort) == ("nenhum", "intenso")


def test_voice_tone_is_off_by_default():
    assert build_voice_tone(Settings(_env_file=None)) == (None, None)


def test_voice_tone_enabled_without_openai_config_only_warns():
    settings = Settings(_env_file=None, voice_tone_enabled=True, azure_openai_endpoint="", azure_openai_key="")
    analyzer, reason = build_voice_tone(settings)
    assert analyzer is None
    assert "AZURE_OPENAI_ENDPOINT" in reason and "AZURE_OPENAI_KEY" in reason


def test_voice_tone_enabled_in_mock_mode_uses_mock():
    analyzer, reason = build_voice_tone(Settings(_env_file=None, voice_tone_enabled=True, mock_azure=True))
    assert isinstance(analyzer, MockVoiceToneAnalyzer) and reason is None
