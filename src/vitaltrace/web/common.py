"""Utilitários compartilhados pelas rotas: sessão, templates e contextos."""

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session
from starlette.templating import Jinja2Templates

from vitaltrace.anomaly.models import SOURCE_LABELS
from vitaltrace.db import repos
from vitaltrace.db.orm import Doctor, Patient

BASE = Path(__file__).parent
SEVERITY_LABELS = {"critical": "CRÍTICO", "high": "ALTO", "medium": "MÉDIO", "low": "BAIXO", "info": "INFO"}
RISK_LABELS = {"critical": "CRÍTICO", "attention": "ATENÇÃO", "stable": "ESTÁVEL"}


class LoginRequired(Exception):
    pass


def _age(born: date) -> int:
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def _initials(name: str) -> str:
    words = name.replace("Dr. ", "").replace("Dra. ", "").split()
    return (words[0][0] + words[-1][0]).upper() if words else "?"


def _when(dt: datetime) -> str:
    return dt.strftime("%d/%m · %H:%M")


def _ago(dt: datetime) -> str:
    minutes = int((datetime.now() - dt).total_seconds() // 60)
    if minutes < 1:
        return "agora"
    if minutes < 60:
        return f"há {minutes} min"
    if minutes < 24 * 60:
        return f"há {minutes // 60} h"
    return f"há {minutes // (24 * 60)} d"


from vitaltrace.web.charts import angle_curve  # noqa: E402


def make_templates() -> Jinja2Templates:
    templates = Jinja2Templates(directory=str(BASE / "templates"))
    env = templates.env
    env.filters.update(
        age=_age, initials=_initials, when=_when, ago=_ago,
        sev=lambda s: SEVERITY_LABELS.get(s, str(s).upper()),
        risk=lambda s: RISK_LABELS.get(s, str(s).upper()),
        source=lambda s: SOURCE_LABELS.get(s, s),
        sex=lambda s: "Feminino" if s == "F" else "Masculino",
        angle_curve=angle_curve,
        num=lambda v, nd=0: "—" if v is None else f"{v:.{nd}f}".replace(".", ","),
    )  # fmt: skip
    return templates


# ---------------------------------------------------------------- helpers
def get_db(request: Request) -> Iterator[Session]:
    with request.app.state.services.db.session() as session:
        yield session


def require_doctor(request: Request, db: Session = Depends(get_db)) -> Doctor:
    doctor_id = request.session.get("doctor_id")
    doctor = repos.get_doctor(db, doctor_id) if doctor_id else None
    if doctor is None:
        raise LoginRequired
    return doctor


def render(request: Request, name: str, status_code: int = 200, **context):
    templates: Jinja2Templates = request.app.state.templates
    context.setdefault("hospital", request.app.state.services.settings.hospital_name)
    return templates.TemplateResponse(request, name, context, status_code=status_code)


def patient_list_context(db: Session, q: str, show_all: bool) -> dict:
    total = repos.count_patients(db)
    query = q.strip()
    if query:
        patients, title = repos.list_patients(db, query=query), f"Resultados para “{query}”"
        count = f"{len(patients)} paciente{'s' if len(patients) != 1 else ''} encontrado{'s' if len(patients) != 1 else ''}"
    elif show_all:
        patients, title = repos.list_patients(db), "Todos os pacientes"
        count = f"{total} pacientes, do atendimento mais recente ao mais antigo"
    else:
        patients, title = repos.list_patients(db, limit=5), "Atendimentos recentes"
        count = f"Mostrando {len(patients)} de {total} pacientes"
    return {
        "patients": patients, "list_title": title, "count_label": count, "query": query,
        "show_all": show_all, "total": total, "alert_counts": repos.open_alert_counts(db),
    }  # fmt: skip


def alerts_context(db: Session, patient_id: int | None = None) -> dict:
    alerts = repos.list_alerts(db, patient_id=patient_id)
    return {
        "alerts": alerts[:8],
        "alerts_total": len(alerts),
        "critical_alerts": [a for a in repos.list_alerts(db) if a.severity == "critical"],
        "alert_scope": patient_id,
    }


def patient_or_404(db: Session, patient_id: int) -> Patient:
    patient = repos.get_patient(db, patient_id)
    if patient is None:
        raise HTTPException(404, "Paciente não encontrado.")
    return patient
