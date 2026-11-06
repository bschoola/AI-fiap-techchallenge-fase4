"""Linha do tempo do paciente: consultas, gravações de áudio, vídeos e análises de IA em uma lista só."""

from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from vitaltrace.anomaly.anamnesis import detect_anamnesis_anomalies
from vitaltrace.db import repos
from vitaltrace.db.orm import Analysis, Consultation
from vitaltrace.models import Severity

KINDS = ("all", "consult", "audio", "video", "anomaly")
KIND_LABELS = {
    "all": "Tudo",
    "consult": "Consultas",
    "audio": "Áudios",
    "video": "Vídeos",
    "anomaly": "Análises de IA",
}
KIND_TITLES = {
    "consult": "Consulta",
    "audio": "Gravação de áudio",
    "video": "Análise de vídeo",
    "anomaly": "Análise de anomalias com IA",
}
PAGE_SIZE = 10

_SEVERITY_LABELS = {"critical": "CRÍTICO", "high": "ALTO", "medium": "MÉDIO", "low": "BAIXO", "info": "INFO"}
_RISK_LABELS = {"critical": "CRÍTICO", "attention": "ATENÇÃO", "stable": "ESTÁVEL"}
_STABLE = ("stable", "SEM ACHADOS")


@dataclass
class Entry:
    kind: str
    obj_id: int
    at: datetime
    title: str
    summary: str
    level: str = "stable"  # classe do selo: critical | high | medium | low | attention | stable
    level_label: str = "SEM ACHADOS"
    payload: dict = field(default_factory=dict)
    consult: Consultation | None = None
    doctor: str = ""
    findings: list = field(default_factory=list)  # achados da anamnese (consultas)
    prev_mobility: float | None = None  # vídeo anterior, para comparar
    alerts: list[dict] = field(default_factory=list)  # alertas gerados + situação atual
    analysis_id: int | None = None
    full_summary: str = ""

    @property
    def key(self) -> str:
        return f"tl-{self.kind}-{self.obj_id}"

    @property
    def risk(self) -> dict | None:
        """Variação de risco registrada junto da análise (só existe para análises feitas no portal)."""
        r = self.payload.get("risk")
        return r if isinstance(r, dict) and "before" in r and "after" in r else None


@dataclass
class TimelinePage:
    entries: list[Entry]
    total: int
    next_offset: int | None
    kind: str


def _level_from_severities(values) -> tuple[str, str]:
    ranked = [Severity(v) for v in values if v]
    if not ranked:
        return _STABLE
    top = max(ranked, key=lambda s: s.rank)
    if top.rank <= Severity.INFO.rank:
        return _STABLE
    return top.value, _SEVERITY_LABELS[top.value]


def _first_chars(text: str, limit: int = 150) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _audio_entry(a: Analysis) -> Entry:
    p = a.payload or {}
    terms = [t for t in p.get("critical_terms", []) if not t.get("negated")]
    level, label = _level_from_severities([t["severity"] for t in terms] + [r["severity"] for r in p.get("alerts", [])])
    inp = p.get("input") or {}
    origin = inp.get("filename") or ("transcrição colada" if inp.get("type") == "text" else "")
    summary = a.summary or ""
    return Entry("audio", a.id, a.created_at, KIND_TITLES["audio"], summary + (f" Origem: {origin}." if origin else ""),
                 level, label, p, analysis_id=a.id)  # fmt: skip


def _video_entry(a: Analysis) -> Entry:
    p = a.payload or {}
    level, label = _level_from_severities([e["severity"] for e in p.get("events", [])])
    return Entry("video", a.id, a.created_at, KIND_TITLES["video"], a.summary or "", level, label, p, analysis_id=a.id)


def _anomaly_entry(a: Analysis) -> Entry:
    p = a.payload or {}
    risk = p.get("risk") or {}
    level = risk.get("level", "stable")
    return Entry("anomaly", a.id, a.created_at, KIND_TITLES["anomaly"], _first_chars(a.summary or "", 220),
                 level, _RISK_LABELS.get(level, level.upper()), p, analysis_id=a.id, full_summary=a.summary or "")  # fmt: skip


def _consult_entry(c: Consultation) -> Entry:
    findings = detect_anamnesis_anomalies([c.anamnesis])
    level, label = _level_from_severities([f.severity.value for f in findings])
    return Entry("consult", c.id, c.occurred_at, KIND_TITLES["consult"], _first_chars(c.anamnesis), level, label,
                 consult=c, doctor=c.doctor.name if c.doctor else "", findings=findings)  # fmt: skip


def all_entries(db: Session, patient_id: int) -> list[Entry]:
    """Todas as entradas, da mais recente para a mais antiga."""
    entries = [_consult_entry(c) for c in repos.consultations(db, patient_id)]
    rows = list(
        db.scalars(
            select(Analysis).where(Analysis.patient_id == patient_id, Analysis.kind.in_(("audio", "video", "anomaly")))
        )
    )
    builders = {"audio": _audio_entry, "video": _video_entry, "anomaly": _anomaly_entry}
    entries += [builders[a.kind](a) for a in rows]
    entries.sort(key=lambda e: (e.at, e.obj_id), reverse=True)

    # vídeo anterior de cada vídeo (a lista está do mais novo para o mais antigo)
    previous: float | None = None
    for e in reversed([e for e in entries if e.kind == "video"]):
        e.prev_mobility = previous
        mobility = (e.payload.get("metrics") or {}).get("mobility_index")
        previous = float(mobility) if mobility is not None else previous
    return entries


def build_timeline(
    db: Session, patient_id: int, *, kind: str = "all", offset: int = 0, limit: int = PAGE_SIZE,
    since: datetime | None = None,
) -> TimelinePage:  # fmt: skip
    """`since`: a primeira página inclui tudo a partir dessa data (para os marcadores dos gráficos)."""
    kind = kind if kind in KINDS else "all"
    entries = all_entries(db, patient_id)
    if kind != "all":
        entries = [e for e in entries if e.kind == kind]
    size = limit
    if since is not None and offset == 0:
        size = max(limit, sum(1 for e in entries if e.at >= since))
    page = entries[offset : offset + size]
    _attach_alert_status(db, page)
    nxt = offset + size
    return TimelinePage(page, len(entries), nxt if nxt < len(entries) else None, kind)


def _attach_alert_status(db: Session, entries: list[Entry]) -> None:
    refs = [r for e in entries for r in e.payload.get("alerts", []) if isinstance(r, dict)]
    rows = repos.alerts_by_id(db, [r["id"] for r in refs if r.get("id")])
    for e in entries:
        merged = []
        for r in e.payload.get("alerts", []):
            row = rows.get(r.get("id"))
            merged.append({
                **r,
                "status": row.status if row else "unknown",
                "acknowledged_by": row.acknowledged_by if row else None,
                "acknowledged_at": row.acknowledged_at if row else None,
            })  # fmt: skip
        e.alerts = merged


def chart_events(entries: list[Entry]) -> list[dict]:
    """Pontos de áudio/vídeo para marcar nos gráficos (cada um aponta para a entrada da linha do tempo)."""
    return [
        {"at": e.at, "kind": e.kind, "level": e.level, "label": f"{e.title} · {e.at:%d/%m %H:%M}", "href": f"#{e.key}"}
        for e in entries
        if e.kind in ("audio", "video")
    ]
