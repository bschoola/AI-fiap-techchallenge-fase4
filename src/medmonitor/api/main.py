"""API FastAPI do MedMonitor.

Execução (a partir da raiz do repositório):
    uvicorn medmonitor.api.main:create_app --factory --app-dir src --reload
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from medmonitor import __version__
from medmonitor.alerts.repository import AlertRepository
from medmonitor.audio.convert import AudioConversionError, prepare_wav
from medmonitor.audio.language_service import LanguageServiceError
from medmonitor.audio.models import AudioAnalysis
from medmonitor.audio.pipeline import AudioPipeline
from medmonitor.audio.speech_service import TranscriptionError
from medmonitor.config import Settings, get_settings
from medmonitor.models import Alert, AlertStatus
from medmonitor.services import build_audio_pipeline


class TextAnalyzeRequest(BaseModel):
    patient_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=20_000)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="MedMonitor API",
        version=__version__,
        description="Monitoramento multimodal de pacientes — FIAP Tech Challenge Fase 4",
    )
    app.state.settings = settings
    app.state.pipeline = build_audio_pipeline(settings)
    app.state.alerts = AlertRepository(settings.database_path)

    def pipeline_of(request: Request) -> AudioPipeline:
        return request.app.state.pipeline

    def repo_of(request: Request) -> AlertRepository:
        return request.app.state.alerts

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "mock_azure": settings.mock_azure, "version": __version__}

    @app.post("/audio/analyze", response_model=AudioAnalysis)
    def analyze_audio(
        request: Request,
        patient_id: str = Form(..., min_length=1),
        file: UploadFile = File(...),
    ) -> AudioAnalysis:
        content = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
        if not content:
            raise HTTPException(400, "Arquivo de áudio vazio.")
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Arquivo maior que {settings.max_upload_mb} MB.")

        suffix = Path(file.filename or "").suffix.lower()[:10] or ".bin"
        with tempfile.TemporaryDirectory(prefix="medmonitor_") as tmp:
            src = Path(tmp) / f"upload{suffix}"
            src.write_bytes(content)
            try:
                wav = prepare_wav(src, Path(tmp))
            except AudioConversionError as exc:
                raise HTTPException(415, str(exc)) from exc
            try:
                analysis = pipeline_of(request).analyze_audio(patient_id, wav)
            except (TranscriptionError, LanguageServiceError) as exc:
                raise HTTPException(502, str(exc)) from exc

        repo_of(request).add_many(analysis.alerts)
        return analysis

    @app.post("/text/analyze", response_model=AudioAnalysis)
    def analyze_text(body: TextAnalyzeRequest, request: Request) -> AudioAnalysis:
        """Analisa um texto já transcrito (útil para testar regras sem enviar áudio)."""
        try:
            analysis = pipeline_of(request).analyze_text(body.patient_id, body.text)
        except LanguageServiceError as exc:
            raise HTTPException(502, str(exc)) from exc
        repo_of(request).add_many(analysis.alerts)
        return analysis

    @app.get("/alerts", response_model=list[Alert])
    def list_alerts(
        request: Request,
        status: AlertStatus | None = None,
        patient_id: str | None = None,
        limit: int = Query(100, ge=1, le=500),
    ) -> list[Alert]:
        return repo_of(request).list(status=status, patient_id=patient_id, limit=limit)

    @app.post("/alerts/{alert_id}/ack", response_model=Alert)
    def acknowledge_alert(alert_id: str, request: Request) -> Alert:
        alert = repo_of(request).acknowledge(alert_id)
        if alert is None:
            raise HTTPException(404, "Alerta não encontrado.")
        return alert

    return app
