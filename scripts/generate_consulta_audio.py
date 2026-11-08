"""Gera 3 áudios SINTÉTICOS de uma paciente falando numa consulta, só a voz dela (sem o médico).

1. consulta_normal.wav       fala natural, ritmo normal, relato tranquilo
2. consulta_falta_de_ar.wav  voz lenta e cansada, respiração audível entre frases; nunca diz "falta de ar"
3. consulta_dor_no_peito.wav fala rápida e ansiosa, dor no peito e cansaço

A voz vem do Azure Neural TTS (pt-BR-FranciscaNeural); as respirações são ruído filtrado gerado aqui
(o TTS não produz sons de respiração) e entram entre as frases. Não são pacientes reais.

Uso: python scripts/generate_consulta_audio.py [--voice pt-BR-ThalitaNeural] [--out data/samples]
"""

import argparse
import io
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RATE = 16000  # Hz, mono, 16 bits: o formato que o Azure Speech reconhece melhor
B = "breath"  # marcador de respiração no roteiro


def pause(seconds: float) -> tuple:
    return ("pause", seconds)


def breath(seconds: float, strength: float) -> tuple:
    return (B, seconds, strength)


def say(text: str, rate: str = "0%", pitch: str = "0%") -> tuple:
    """`text` aceita <break time="…"/> para pausas no meio da frase."""
    return ("say", text, rate, pitch)


SCRIPTS: dict[str, list[tuple]] = {
    "consulta_normal.wav": [
        say("Bom dia, doutora. Então, essa semana eu tenho me sentido mais ou menos bem.", "-3%"),
        pause(0.5),
        say('Tive uma dorzinha de cabeça no começo da semana, <break time="300ms"/> mas passou sozinha.'),
        pause(0.4),
        say("Dormi bem na maioria das noites, só ontem que eu acordei umas duas vezes."),
        pause(0.5),
        say("O apetite está normal, e de manhã eu tenho conseguido caminhar um pouquinho no parque."),
        pause(0.4),
        say("Tomei os remédios direitinho, todos os dias, no mesmo horário."),
        pause(0.6),
        say("Era isso, doutora. No geral, eu estou bem, só queria confirmar se está tudo certo."),
    ],
    "consulta_falta_de_ar.wav": [
        breath(0.9, 0.9),
        say("Bom dia, doutora.", "-28%", "-4%"),
        breath(1.0, 1.0),
        say('Eu vim <break time="900ms"/> subindo a rampa <break time="700ms"/> bem devagar.', "-30%", "-4%"),
        breath(1.2, 1.0),
        say('Faz uns dias que eu <break time="800ms"/> não consigo <break time="700ms"/> fazer as coisas de casa.', "-30%", "-5%"),
        breath(1.3, 1.0),
        say('Parei pra descansar <break time="600ms"/> duas vezes <break time="800ms"/> só pra chegar aqui.', "-32%", "-5%"),
        breath(1.4, 1.0),
        say('À noite eu preciso <break time="700ms"/> dormir com dois travesseiros.', "-32%", "-5%"),
        breath(1.2, 0.9),
        say('Senão eu acordo <break time="900ms"/> assustada, <break time="700ms"/> sentada na cama.', "-33%", "-6%"),
        breath(1.5, 1.0),
        say('Minha voz <break time="800ms"/> está fraca, né?', "-35%", "-6%"),
        breath(1.0, 0.8),
    ],
    "consulta_dor_no_peito.wav": [
        say("Doutora, eu preciso falar logo, estou com uma dor no peito desde cedo.", "+22%", "+4%"),
        breath(0.35, 0.6),
        say("É um aperto aqui no meio, que vai pro braço esquerdo, e não passa.", "+25%", "+5%"),
        breath(0.35, 0.6),
        say("E eu estou muito cansada, cansada demais, qualquer coisinha que eu faço já me deixa exausta.", "+25%", "+5%"),
        breath(0.4, 0.7),
        say("Eu subi um lance de escada e achei que ia desmaiar, fiquei suando frio.", "+25%", "+6%"),
        breath(0.4, 0.7),
        say("Eu estou com muito medo, será que é o coração?", "+22%", "+6%"),
    ],
}


def breath_noise(seconds: float, strength: float, rng: np.random.Generator) -> np.ndarray:
    """Inspiração/expiração audível: ruído passa-banda (~300–3500 Hz) com envelope em arco."""
    n = int(seconds * RATE)
    noise = rng.standard_normal(n)
    spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    spec *= ((freqs > 300) & (freqs < 3500)) * (1 / (1 + freqs / 1200))
    shaped = np.fft.irfft(spec, n)
    x = np.linspace(0, 1, n)
    envelope = np.sin(np.pi * x) ** 1.5 * (0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * x + 1))  # leve ondulação
    shaped = shaped / (np.max(np.abs(shaped)) + 1e-9)
    return (shaped * envelope * 0.35 * strength * 32767).astype(np.int16)


def ssml(voice: str, text: str, rate: str, pitch: str) -> str:
    return (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">'
        f'<voice name="{voice}"><prosody rate="{rate}" pitch="{pitch}">{text}</prosody></voice></speak>'
    )


def assemble(script: list[tuple], synth, seed: int = 7) -> np.ndarray:
    """`synth(text, rate, pitch) -> int16 mono 16 kHz`. Junta fala, respirações e pausas."""
    rng = np.random.default_rng(seed)
    parts: list[np.ndarray] = []
    for item in script:
        if item[0] == "say":
            parts.append(synth(item[1], item[2], item[3]))
        elif item[0] == B:
            parts.append(breath_noise(item[1], item[2], rng))
        else:
            parts.append(np.zeros(int(item[1] * RATE), np.int16))
    return np.concatenate(parts)


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.astype(np.int16).tobytes())


def azure_synth(voice: str, key: str, region: str):
    import azure.cognitiveservices.speech as speechsdk

    config = speechsdk.SpeechConfig(subscription=key, region=region)
    config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm)

    def synth(text: str, rate: str, pitch: str) -> np.ndarray:
        synthesizer = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)  # em memória
        result = synthesizer.speak_ssml_async(ssml(voice, text, rate, pitch)).get()
        if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
            details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
            raise RuntimeError(details.error_details)
        with wave.open(io.BytesIO(result.audio_data)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16)

    return synth


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", default="pt-BR-FranciscaNeural")
    parser.add_argument("--out", default=str(ROOT / "data" / "samples"))
    args = parser.parse_args()

    from vitaltrace.config import Settings

    settings = Settings()
    missing = [m for m in settings.missing_azure_settings() if m.startswith("AZURE_SPEECH")]
    if missing:
        print(f"Variáveis ausentes no .env: {', '.join(missing)}")
        return 2
    synth = azure_synth(args.voice, settings.azure_speech_key, settings.azure_speech_region)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, script in SCRIPTS.items():
        try:
            samples = assemble(script, synth)
        except RuntimeError as exc:
            print(f"FALHOU {filename}: {exc}")
            return 1
        write_wav(out_dir / filename, samples)
        print(f"OK {out_dir / filename} ({len(samples) / RATE:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
