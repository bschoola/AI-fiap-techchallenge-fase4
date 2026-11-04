"""Modelos de domínio compartilhados entre os módulos (alertas e severidade)."""

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return _SEVERITY_RANK[self]


_SEVERITY_RANK = {
    Severity.INFO: 0,
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class AlertStatus(str, Enum):
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Alert(BaseModel):
    """Alerta exibido para a equipe médica dentro da aplicação."""

    id: str = Field(default_factory=lambda: uuid4().hex)
    patient_id: str
    source: str  # "audio", "video", "vitals", "prescription", ...
    severity: Severity
    title: str
    description: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: AlertStatus = AlertStatus.OPEN
    created_at: datetime = Field(default_factory=utcnow)
    acknowledged_at: datetime | None = None
