"""Regras que transformam achados do pipeline de áudio em alertas."""

from collections import defaultdict

from vitaltrace.audio.models import (
    CriticalTermMatch,
    ProsodyMetrics,
    SpeechMetrics,
    TextAnalysis,
    VoiceTone,
)
from vitaltrace.config import Settings
from vitaltrace.models import Alert, Severity


def build_audio_alerts(
    patient_id: str,
    terms: list[CriticalTermMatch],
    text_analysis: TextAnalysis,
    metrics: SpeechMetrics,
    settings: Settings,
    prosody: ProsodyMetrics | None = None,
    voice_tone: VoiceTone | None = None,
) -> list[Alert]:
    alerts: list[Alert] = []

    # 1) Termos críticos (um alerta por categoria clínica; termos negados são ignorados)
    by_category: dict[str, list[CriticalTermMatch]] = defaultdict(list)
    for term in terms:
        if not term.negated:
            by_category[term.category].append(term)

    for category, items in by_category.items():
        severity = max((i.severity for i in items), key=lambda s: s.rank)
        unique = sorted({i.term.lower() for i in items})
        alerts.append(
            Alert(
                patient_id=patient_id,
                source="audio",
                severity=severity,
                title=f"Termo crítico ({category}): {', '.join(unique)}",
                description=(
                    f"O paciente mencionou {', '.join(unique)} durante a consulta. "
                    "Revisar o trecho e avaliar o paciente."
                ),
                evidence={
                    "category": category,
                    "terms": unique,
                    "contexts": [i.context for i in items][:3],
                },
            )
        )

    # 2) Sentimento predominantemente negativo
    negative = text_analysis.sentiment.negative
    if negative >= settings.negative_sentiment_threshold:
        alerts.append(
            Alert(
                patient_id=patient_id,
                source="audio",
                severity=Severity.LOW,
                title="Sentimento predominantemente negativo",
                description=f"{negative:.0%} do discurso foi classificado como negativo.",
                evidence={"sentiment": text_analysis.sentiment.model_dump()},
            )
        )

    # 3) Como a pessoa fala: prosódia medida no áudio + tom de voz (quando disponíveis)
    breathless = _breathless_alert(patient_id, terms, metrics, prosody, voice_tone, settings)
    if breathless is not None:
        # O alerta combinado já carrega ritmo e pausas na evidência: não repete os alertas isolados
        alerts.append(breathless)
    else:
        alerts.extend(_speech_pattern_alerts(patient_id, metrics, prosody, settings))
        alerts.extend(_voice_tone_alerts(patient_id, voice_tone))

    alerts.sort(key=lambda a: a.severity.rank, reverse=True)
    return alerts


_EFFORT_RANK = {"nenhum": 0, "leve": 1, "moderado": 2, "intenso": 3}
_FATIGUE_TERMS = ("cansa", "fadiga", "exaust", "fraqueza")


def is_breathless_speech(prosody: ProsodyMetrics | None, settings: Settings) -> bool:
    """Fala entrecortada: muito tempo em silêncio, pausas longas e frequentes, trechos curtos."""
    if prosody is None or not prosody.reliable:
        return False
    values = (
        prosody.pause_ratio,
        prosody.median_pause_s,
        prosody.long_pauses_per_min,
        prosody.mean_phrase_s,
    )
    if any(v is None for v in values):
        return False
    return (
        prosody.pause_ratio >= settings.breathless_pause_ratio
        and prosody.median_pause_s >= settings.breathless_median_pause_s
        and prosody.long_pauses_per_min >= settings.breathless_long_pauses_per_min
        and prosody.mean_phrase_s <= settings.breathless_max_phrase_s
    )


def _breathless_alert(
    patient_id: str,
    terms: list[CriticalTermMatch],
    metrics: SpeechMetrics,
    prosody: ProsodyMetrics | None,
    voice_tone: VoiceTone | None,
    settings: Settings,
) -> Alert | None:
    """Fusão: o padrão acústico sozinho já é ALTO; confirmado por outra fonte, vira CRÍTICO."""
    if not is_breathless_speech(prosody, settings):
        return None

    corroboration: list[str] = []
    said = [t for t in terms if not t.negated]
    respiratory = sorted({t.term.lower() for t in said if t.category == "respiratório"})
    fatigue = sorted({t.term.lower() for t in said if t.term.lower().startswith(_FATIGUE_TERMS)})
    if respiratory:
        corroboration.append(f"queixa respiratória relatada ({', '.join(respiratory)})")
    if fatigue:
        corroboration.append(f"cansaço relatado ({', '.join(fatigue)})")
    if voice_tone is not None:
        effort = _EFFORT_RANK.get(voice_tone.respiratory_effort, 0)
        if effort >= 2:
            corroboration.append(
                f"tom de voz avaliado por IA: esforço respiratório {voice_tone.respiratory_effort}"
            )

    median_pause = f"{prosody.median_pause_s:.1f}".replace(".", ",")
    mean_phrase = f"{prosody.mean_phrase_s:.1f}".replace(".", ",")
    description = (
        f"{prosody.pause_ratio:.0%} do tempo em silêncio, {prosody.long_pause_count} pausa(s) de 1 s ou mais "
        f"(mediana de {median_pause} s) e trechos de fala de {mean_phrase} s em média: "
        "a pessoa interrompe a fala a cada poucas palavras."
    )
    if (
        prosody.pitch_variation_st is not None
        and prosody.pitch_variation_st < settings.monotone_pitch_st
    ):
        description += " Tom de voz monótono, com pouca variação."
    if corroboration:
        description += " Confirmado por: " + "; ".join(corroboration) + "."
    else:
        description += " O paciente não relatou a queixa em palavras: avaliar presencialmente."

    return Alert(
        patient_id=patient_id,
        source="audio",
        severity=Severity.CRITICAL if corroboration else Severity.HIGH,
        title="Fala entrecortada com pausas longas (possível esforço respiratório)",
        description=description,
        evidence={
            "prosody": prosody.model_dump(),
            "metrics": metrics.model_dump(),
            "voice_tone": voice_tone.model_dump() if voice_tone else None,
            "corroboration": corroboration,
        },
    )


def _speech_pattern_alerts(
    patient_id: str,
    metrics: SpeechMetrics,
    prosody: ProsodyMetrics | None,
    settings: Settings,
) -> list[Alert]:
    """Alertas isolados de ritmo e pausas (quando o padrão combinado não se confirma)."""
    alerts: list[Alert] = []
    acoustic = prosody is not None and prosody.reliable and prosody.pause_ratio is not None

    # Ritmo: só quando o trecho transcrito é longo o bastante para ser confiável
    wpm = metrics.words_per_minute
    if metrics.reliable and wpm is not None and wpm < settings.slow_speech_wpm:
        alerts.append(
            Alert(
                patient_id=patient_id,
                source="audio",
                severity=Severity.MEDIUM,
                title="Ritmo de fala reduzido (possível fadiga)",
                description=(
                    f"{wpm:.0f} palavras/min, abaixo do limiar de "
                    f"{settings.slow_speech_wpm:.0f}."
                ),
                evidence={"metrics": metrics.model_dump()},
            )
        )

    # Pausas: a medida acústica vale mais que a estimada pelos segmentos da transcrição
    pause_ratio = prosody.pause_ratio if acoustic else (metrics.pause_ratio if metrics.reliable else None)
    if pause_ratio is not None and pause_ratio > settings.high_pause_ratio:
        alerts.append(
            Alert(
                patient_id=patient_id,
                source="audio",
                severity=Severity.LOW,
                title="Pausas frequentes na fala",
                description=f"{pause_ratio:.0%} do tempo foi de pausa.",
                evidence={
                    "metrics": metrics.model_dump(),
                    "prosody": prosody.model_dump() if acoustic else None,
                },
            )
        )
    return alerts


def _voice_tone_alerts(patient_id: str, voice_tone: VoiceTone | None) -> list[Alert]:
    """O modelo de áudio sozinho gera, no máximo, um alerta MÉDIO (a resposta varia entre execuções)."""
    if voice_tone is None:
        return []
    effort = _EFFORT_RANK.get(voice_tone.respiratory_effort, 0)
    fatigue = _EFFORT_RANK.get(voice_tone.fatigue, 0)
    if effort < 2 and fatigue < 3:
        return []
    if effort >= 2:
        title = f"Tom de voz sugere esforço respiratório {voice_tone.respiratory_effort} (avaliação por IA)"
        severity = Severity.MEDIUM if effort >= 3 else Severity.LOW
    else:
        title = "Tom de voz sugere cansaço intenso (avaliação por IA)"
        severity = Severity.LOW
    return [
        Alert(
            patient_id=patient_id,
            source="audio",
            severity=severity,
            title=title,
            description=(
                (voice_tone.summary or "O modelo de áudio percebeu sinais de esforço na fala.")
                + " As pausas medidas no áudio não confirmaram o padrão: ouvir a gravação."
            ),
            evidence={"voice_tone": voice_tone.model_dump()},
        )
    ]
