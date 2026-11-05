"""Resumo em linguagem natural dos achados, gerado por regras (determinístico, sem LLM)."""

from vitaltrace.anomaly.models import SOURCE_LABELS, Finding
from vitaltrace.fusion.risk import RiskAssessment
from vitaltrace.models import Severity

DISCLAIMER = (
    "Resumo gerado automaticamente por regras e modelos estatísticos, sem valor diagnóstico. "
    "A decisão clínica é sempre do médico."
)


def build_summary(patient_name: str, risk: RiskAssessment, findings: list[Finding]) -> str:
    if not findings:
        return (
            f"{patient_name}: nenhum desvio relevante nos sinais vitais, prescrições, movimento "
            f"ou anamnese. Risco {risk.label.lower()} ({risk.score:.0f}/100)."
        )
    n_serious = sum(1 for f in findings if f.severity.rank >= Severity.HIGH.rank)
    parts = [
        f"{patient_name}: risco {risk.label.lower()} ({risk.score:.0f}/100), com {len(findings)} "
        f"achado(s), dos quais {n_serious} de severidade alta ou crítica."
    ]
    for source in ("vitals", "prescription", "movement", "anamnesis", "audio", "video", "fusion"):
        items = [f for f in findings if f.source == source]
        if not items:
            continue
        top = sorted(items, key=lambda f: f.severity.rank, reverse=True)[:2]
        parts.append(f"{SOURCE_LABELS[source]}: " + "; ".join(f.title for f in top) + ".")
    return " ".join(parts)
