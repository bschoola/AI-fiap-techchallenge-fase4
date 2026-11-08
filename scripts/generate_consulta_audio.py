"""Gera 3 áudios SINTÉTICOS de uma paciente falando numa consulta (só a voz dela, sem o médico).

1. consulta_normal.wav        conversa leve e espontânea, sem queixas
2. consulta_falta_de_ar.wav   fala lenta, entrecortada, puxando o ar; NUNCA diz "falta de ar"
                              (o sistema precisa perceber pelo ritmo de fala e pelas pausas)
3. consulta_dor_no_peito.wav  fala rápida e assustada: dor no peito, braço esquerdo, suor frio

Cada áudio é gerado numa TOMADA ÚNICA (o texto inteiro numa chamada só). Gerar frase por frase e
colar ruído no meio quebra a entonação e deixa a respiração falsa; aqui a respiração, as hesitações
e a emoção saem do próprio modelo de voz.

Motores (--engine):
  audio   (recomendado) Azure OpenAI gpt-audio / gpt-audio-mini pela API de chat completions.
          Modelo de áudio completo: interpreta o personagem, respira, hesita e varia o ritmo.
          Deployment em AZURE_OPENAI_AUDIO_DEPLOYMENT. Usado por padrão quando há chaves do Azure OpenAI.
  tts     Azure OpenAI /audio/speech (gpt-4o-mini-tts, tts-hd ou tts). Deployment em
          AZURE_OPENAI_TTS_DEPLOYMENT. Só o gpt-4o-mini-tts aceita a direção de interpretação.
  speech  Azure AI Speech, voz HD pt-BR (Thalita DragonHD por padrão), também numa tomada única.

O motor `audio` pode parafrasear o texto. O script confere a transcrição que o modelo devolve e
tenta de novo (até --tries vezes) quando ela se afasta do roteiro.

Uso:
  python scripts/generate_consulta_audio.py                       # motor padrão, os 3 áudios
  python scripts/generate_consulta_audio.py --engine speech
  python scripts/generate_consulta_audio.py --only normal --voice cedar
"""

import argparse
import base64
import difflib
import io
import re
import sys
import time
import unicodedata
import wave
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Roteiros. A pontuação guia a interpretação: reticências = pausa/fôlego, travessão = mudança de ideia.
# Escritos como fala, não como texto: "tô", "pra", "né", frases que se corrigem no meio.
# Atenção aos termos críticos (src/vitaltrace/audio/critical_terms.py): o áudio normal não usa nenhum
# e o de falta de ar não diz "falta de ar", "não consigo respirar" nem "dificuldade para respirar".
SCRIPTS: dict[str, str] = {
    "consulta_normal.wav": (
        "Oi, doutora, bom dia! Tudo bem com a senhora? "
        "Ah, então... essa semana foi tranquila, viu? Graças a Deus. "
        "Eu tô dormindo bem — acordo uma vez de madrugada pra ir no banheiro, mas volto a dormir rapidinho. "
        "Tô comendo direitinho também. A minha filha tá passando uns dias lá em casa, então tem comida de verdade, né? "
        "Não é só aquele lanchinho que eu faço quando tô sozinha. "
        "E... ah, eu voltei a caminhar! De manhã, lá na pracinha perto de casa. Uns vinte, trinta minutinhos. "
        "No começo eu achei que não ia dar conta, mas agora eu já vou numa boa, até converso com as vizinhas no caminho. "
        "Os remédios eu tô tomando certinho, todo dia de manhã. Coloquei até um alarme no celular, que senão eu esqueço. "
        "Ah, e a pressão eu medi na farmácia na terça-feira, deu doze por oito. "
        "Então é isso, doutora, eu tô me sentindo bem. Só queria saber se os exames vieram bons, que eu fiquei curiosa."
    ),
    "consulta_falta_de_ar.wav": (
        "Bom dia, doutora... "
        "Desculpa, eu vou... sentar aqui um pouquinho primeiro. "
        "É que eu vim da recepção até aqui... e parei duas vezes... no corredor. "
        "Faz uns... uns quatro dias, eu acho... que tá assim. "
        "Eu começo a varrer a casa... e tenho que parar no meio. "
        "Fico ali... apoiada na vassoura... esperando passar. "
        "De noite, então... eu tô dormindo com três travesseiros... quase sentada. "
        "Se eu deito reto... parece que eu tô me afogando. "
        "Ontem eu acordei de madrugada... assustada... e fui até a janela... puxar o ar. "
        "E o pé... o pé tá inchado, olha... o sapato nem entra mais. "
        "Desculpa falar assim... devagar... é que não dá... pra falar tudo de uma vez."
    ),
    "consulta_dor_no_peito.wav": (
        "Doutora, desculpa entrar assim, mas eu preciso falar logo, tá? "
        "Desde cedo eu tô com uma dor no peito — é um aperto, bem aqui no meio, parece que tem alguém sentado em cima de mim — "
        "e vai descendo pro braço esquerdo, e não passa, não passa de jeito nenhum. "
        "Eu tomei um remédio e nada, nada! "
        "E eu tô cansada, cansada demais. Eu subi um lance de escada e achei que ia desmaiar, fiquei suando frio, a blusa ficou molhada. "
        "Meu marido falou pra eu vir direto pra cá, e eu vim. "
        "A minha mãe teve infarto com a minha idade, doutora... eu tô com muito medo. É o coração? "
        "Fala pra mim que não é o coração."
    ),
}

# Direção de interpretação (motores audio e tts).
CHARACTER = (
    "Você é uma atriz de voz brasileira gravando a fala de uma paciente de uns 60 anos, de São Paulo, "
    "numa consulta médica. Português do Brasil, sotaque paulistano, registro informal. "
    "Tem que soar como uma pessoa real falando, gravada por um celular em cima da mesa do consultório: "
    "nada de voz de locutora, nada de leitura. Respirações, hesitações e variações de ritmo naturais. "
)
DIRECTION: dict[str, str] = {
    "consulta_normal.wav": (
        "Ela está bem e de bom humor. Fala num ritmo normal de conversa, simpática, com um sorriso na voz, "
        "uma risadinha leve quando fala do lanchinho e das vizinhas. Pequenas hesitações naturais "
        "('ah', 'então'), sem pressa e sem dramatizar. Respiração normal, quase imperceptível."
    ),
    "consulta_falta_de_ar.wav": (
        "Ela está com muita dificuldade para respirar e exausta. Voz fraca, baixa e um pouco rouca. "
        "Fala MUITO devagar, em pedaços curtos de duas a cinco palavras, e entre os pedaços puxa o ar de forma "
        "AUDÍVEL: inspirações curtas e rápidas pela boca, às vezes um suspiro, como quem acabou de subir uma escada. "
        "Nas reticências, ela para e respira. A voz vai ficando mais fraca no fim de cada frase. "
        "Nunca aumenta o volume, nunca acelera."
    ),
    "consulta_dor_no_peito.wav": (
        "Ela está assustada, com dor e quase em pânico. Fala RÁPIDO, atropelando as palavras, voz tensa e um pouco "
        "trêmula, com a respiração curta e ofegante entre as frases. Na parte da mãe e do coração, a voz embarga, "
        "quase chorando. Urgência real, sem gritar."
    ),
}

AUDIO_SYSTEM = (
    CHARACTER
    + "Sua única tarefa é DIZER EM VOZ ALTA, palavra por palavra, o texto que vier na mensagem do usuário, "
    "do começo ao fim. Não responda ao texto, não comente, não cumprimente, não acrescente nem tire palavras, "
    "não leia a pontuação. Interpretação desta gravação: "
)

HD_VOICE = "pt-BR-Thalita:DragonHDLatestNeural"


def norm_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", text)


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm_words(a), norm_words(b)).ratio()


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Reamostragem por FFT (sem o 'abafado'/aliasing da interpolação linear)."""
    if src == dst:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst / src))
    spec = np.fft.rfft(x.astype(np.float64))
    keep = n_out // 2 + 1
    spec = spec[:keep] if len(spec) >= keep else np.pad(spec, (0, keep - len(spec)))
    return (np.fft.irfft(spec, n_out) * (n_out / len(x))).astype(np.float32)


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if width != 2:
        raise RuntimeError(f"WAV com {8 * width} bits não suportado")
    x = np.frombuffer(frames, np.int16).astype(np.float32)
    if channels > 1:
        x = x.reshape(-1, channels).mean(axis=1)
    return x, rate


def finish(x: np.ndarray, src_rate: int, out_rate: int) -> np.ndarray:
    """Reamostra, tira o silêncio excessivo das pontas, normaliza o pico em -1 dBFS e põe 0,3 s de folga."""
    x = resample(x, src_rate, out_rate)
    voiced = np.flatnonzero(np.abs(x) > 0.01 * (np.max(np.abs(x)) + 1e-9))
    if len(voiced):
        x = x[max(0, voiced[0] - int(0.05 * out_rate)): voiced[-1] + int(0.15 * out_rate)]
    x = x / (np.max(np.abs(x)) + 1e-9) * (10 ** (-1 / 20) * 32767)
    pad = np.zeros(int(0.3 * out_rate), np.float32)
    return np.concatenate([pad, x, pad]).astype(np.int16)


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())


def post(url: str, settings, body: dict) -> requests.Response:
    for attempt in range(4):
        resp = requests.post(
            url, params={"api-version": settings.azure_openai_api_version},
            headers={"api-key": settings.azure_openai_key}, json=body, timeout=180,
        )  # fmt: skip
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(3 * (attempt + 1))
            continue
        return resp
    return resp


def deployment_url(settings, deployment: str, path: str) -> str:
    return f"{settings.azure_openai_endpoint.rstrip('/')}/openai/deployments/{deployment}/{path}"


def synth_audio(settings, voice: str, filename: str, tries: int) -> tuple[np.ndarray, int]:
    """gpt-audio / gpt-audio-mini: chat completions com saída de áudio, conferindo a transcrição."""
    text = SCRIPTS[filename]
    url = deployment_url(settings, settings.azure_openai_audio_deployment, "chat/completions")
    body = {
        "model": settings.azure_openai_audio_deployment,
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "wav"},
        "messages": [
            {"role": "system", "content": AUDIO_SYSTEM + DIRECTION[filename]},
            {"role": "user", "content": text},
        ],
    }
    best: tuple[float, bytes] | None = None
    for attempt in range(1, tries + 1):
        resp = post(url, settings, body)
        if resp.status_code != 200:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
        audio = resp.json()["choices"][0]["message"].get("audio") or {}
        if not audio.get("data"):
            raise RuntimeError("o modelo não devolveu áudio (o deployment é gpt-audio / gpt-audio-mini?)")
        score = similarity(text, audio.get("transcript", ""))
        print(f"  tentativa {attempt}: fidelidade ao roteiro {score:.0%}")
        if best is None or score > best[0]:
            best = (score, base64.b64decode(audio["data"]))
        if score >= 0.9:
            break
    if best[0] < 0.75:
        print("  aviso: o modelo se afastou do roteiro; ouça o áudio antes de usar")
    return read_wav(best[1])


def synth_tts(settings, voice: str, filename: str) -> tuple[np.ndarray, int]:
    """/audio/speech (gpt-4o-mini-tts, tts-hd, tts). Manda a direção; se o modelo recusar, tenta sem."""
    url = deployment_url(settings, settings.azure_openai_tts_deployment, "audio/speech")
    body = {
        "model": settings.azure_openai_tts_deployment,
        "input": SCRIPTS[filename],
        "voice": voice,
        "instructions": CHARACTER + DIRECTION[filename],
        "response_format": "wav",
    }
    resp = post(url, settings, body)
    if resp.status_code == 400 and "instructions" in resp.text:
        print("  aviso: este modelo não aceita direção de interpretação; gerando sem ela")
        body.pop("instructions")
        resp = post(url, settings, body)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return read_wav(resp.content)


def synth_speech(settings, voice: str, filename: str) -> tuple[np.ndarray, int]:
    """Azure AI Speech: o texto inteiro num único SSML (entonação contínua)."""
    import azure.cognitiveservices.speech as speechsdk
    from xml.sax.saxutils import escape

    config = speechsdk.SpeechConfig(subscription=settings.azure_speech_key, region=settings.azure_speech_region)
    config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm)
    markup = (
        '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="pt-BR">'
        f'<voice name="{voice}">{escape(SCRIPTS[filename])}</voice></speak>'
    )
    result = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None).speak_ssml_async(markup).get()
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        raise RuntimeError(speechsdk.SpeechSynthesisCancellationDetails.from_result(result).error_details)
    return read_wav(result.audio_data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["audio", "tts", "speech"], default=None)
    parser.add_argument("--voice", default=None, help="audio/tts: marin, cedar, coral, sage, shimmer… | speech: nome da voz Azure")
    parser.add_argument("--only", choices=["normal", "falta_de_ar", "dor_no_peito"], default=None)
    parser.add_argument("--tries", type=int, default=3, help="motor audio: tentativas até bater com o roteiro")
    parser.add_argument("--rate", type=int, default=16000, help="taxa do WAV final (16000 é o ideal para o Azure Speech)")
    parser.add_argument("--out", default=str(ROOT / "data" / "samples"))
    args = parser.parse_args()

    from vitaltrace.config import Settings

    settings = Settings()
    have_openai = bool(settings.azure_openai_endpoint and settings.azure_openai_key)
    engine = args.engine or ("audio" if have_openai else "speech")
    if engine in ("audio", "tts") and not have_openai:
        print("Faltam AZURE_OPENAI_ENDPOINT e AZURE_OPENAI_KEY no .env (veja docs/azure-setup.md, seção Azure OpenAI).")
        return 2
    if engine == "speech" and not (settings.azure_speech_key and settings.azure_speech_region):
        print("Faltam AZURE_SPEECH_KEY e AZURE_SPEECH_REGION no .env.")
        return 2
    voice = args.voice or {"audio": "marin", "tts": "coral", "speech": HD_VOICE}[engine]
    deployment = {
        "audio": settings.azure_openai_audio_deployment,
        "tts": settings.azure_openai_tts_deployment,
        "speech": "-",
    }[engine]
    print(f"Motor: {engine} | deployment: {deployment} | voz: {voice}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in SCRIPTS:
        if args.only and filename != f"consulta_{args.only}.wav":
            continue
        print(f"{filename}…")
        try:
            if engine == "audio":
                x, rate = synth_audio(settings, voice, filename, args.tries)
            elif engine == "tts":
                x, rate = synth_tts(settings, voice, filename)
            else:
                x, rate = synth_speech(settings, voice, filename)
        except (RuntimeError, requests.RequestException) as exc:
            print(f"FALHOU {filename}: {exc}")
            return 1
        samples = finish(x, rate, args.rate)
        write_wav(out_dir / filename, samples, args.rate)
        print(f"OK {out_dir / filename} ({len(samples) / args.rate:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
