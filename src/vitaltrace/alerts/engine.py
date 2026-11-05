"""Regras que transformam achados do pipeline de áudio em alertas."""

from collections import defaultdict

from vitaltrace.audio.models import CriticalTermMatch, SpeechMetrics, TextAnalysis
from vitaltrace.config import Settings
from vitaltrace.models import Alert, Severity


def build_audio_alerts(
    patient_id: str,
    terms: list[CriticalTermMatch],
    text_analysis: TextAnalysis,
    metrics: SpeechMetrics,
    settings: Settings,
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

    # 3) Padrões de fala (só quando o trecho é longo o bastante para ser confiável)
    if metrics.reliable:
        wpm = metrics.words_per_minute
        if wpm is not None and wpm < settings.slow_speech_wpm:
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
        if metrics.pause_ratio is not None and metrics.pause_ratio > settings.high_pause_ratio:
            alerts.append(
                Alert(
                    patient_id=patient_id,
                    source="audio",
                    severity=Severity.LOW,
                    title="Pausas frequentes na fala",
                    description=f"{metrics.pause_ratio:.0%} do tempo foi de pausa.",
                    evidence={"metrics": metrics.model_dump()},
                )
            )

    alerts.sort(key=lambda a: a.severity.rank, reverse=True)
    return alerts
