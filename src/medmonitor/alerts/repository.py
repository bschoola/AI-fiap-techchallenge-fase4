"""Persistência de alertas em SQLite (suficiente para a demo; sem dependências extras)."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from medmonitor.models import Alert, AlertStatus, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    source TEXT NOT NULL,
    severity TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    evidence TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    acknowledged_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_status_created ON alerts (status, created_at);
"""


class AlertRepository:
    def __init__(self, db_path: str | Path) -> None:
        self._path = str(db_path)
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn, conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _to_alert(row: sqlite3.Row) -> Alert:
        data = dict(row)
        data["evidence"] = json.loads(data["evidence"])
        return Alert(**data)

    def add_many(self, alerts: list[Alert]) -> None:
        if not alerts:
            return
        rows = [
            (
                a.id, a.patient_id, a.source, a.severity.value, a.title, a.description,
                json.dumps(a.evidence, ensure_ascii=False), a.status.value,
                a.created_at.isoformat(),
                a.acknowledged_at.isoformat() if a.acknowledged_at else None,
            )
            for a in alerts
        ]  # fmt: skip
        with closing(self._connect()) as conn, conn:
            conn.executemany(
                "INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)", rows
            )

    def list(
        self,
        status: AlertStatus | None = None,
        patient_id: str | None = None,
        limit: int = 100,
    ) -> list[Alert]:
        query = "SELECT * FROM alerts"
        clauses: list[str] = []
        params: list[object] = []
        if status:
            clauses.append("status = ?")
            params.append(status.value)
        if patient_id:
            clauses.append("patient_id = ?")
            params.append(patient_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with closing(self._connect()) as conn:
            return [self._to_alert(r) for r in conn.execute(query, params)]

    def get(self, alert_id: str) -> Alert | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM alerts WHERE id = ?", (alert_id,)
            ).fetchone()
        return self._to_alert(row) if row else None

    def acknowledge(self, alert_id: str) -> Alert | None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "UPDATE alerts SET status = ?, acknowledged_at = ? WHERE id = ?",
                (AlertStatus.ACKNOWLEDGED.value, utcnow().isoformat(), alert_id),
            )
        return self.get(alert_id)
