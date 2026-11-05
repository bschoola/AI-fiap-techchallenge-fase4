"""Achado (anomalia) produzido pelos detectores."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from vitaltrace.models import Alert, Severity

SOURCE_LABELS = {
    "vitals": "Sinais vitais",
    "prescription": "Prescrição",
    "movement": "Movimento",
    "anamnesis": "Anamnese",
    "audio": "Áudio",
    "video": "Vídeo",
    "fusion": "Correlação",
}


class Finding(BaseModel):
    source: str  # vitals | prescription | movement | anamnesis | fusion
    detector: str  # rule | zscore | trend | isolation_forest | dose_change | ...
    severity: Severity
    title: str
    description: str
    ts: datetime | None = None
    evidence: dict[str, Any] = Field(default_factory=dict)

    def to_alert(self, patient_id: int | str) -> Alert:
        return Alert(
            patient_id=str(patient_id),
            source=self.source,
            severity=self.severity,
            title=self.title,
            description=self.description,
            evidence={"detector": self.detector, **self.evidence},
        )
