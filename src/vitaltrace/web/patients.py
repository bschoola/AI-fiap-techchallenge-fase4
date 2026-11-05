"""Rotas do paciente: visão geral, análise de áudio/vídeo e evolução clínica."""

import tempfile
from datetime import datetime, time
from html import escape
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import RedirectResponse
from markupsafe import Markup
from sqlalchemy.orm import Session

from vitaltrace.anomaly.service import analyze_patient
from vitaltrace.audio.convert import AudioConversionError, prepare_wav
from vitaltrace.audio.critical_terms import normalize
from vitaltrace.audio.language_service import LanguageServiceError
from vitaltrace.audio.models import AudioAnalysis
from vitaltrace.audio.speech_service import TranscriptionError
from vitaltrace.db import repos
from vitaltrace.db.orm import Doctor
from vitaltrace.video.analyzer import VideoError, build_video_alerts
from vitaltrace.video.report import type_label
from vitaltrace.web import charts
from vitaltrace.web.common import alerts_context, get_db, patient_or_404, render, require_doctor

router = APIRouter(prefix="/patients")

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".m4v"}


def _error(request: Request, message: str, status_code: int):
    return render(request, "_error.html", status_code=status_code, message=message)


def _num(value: str | None) -> float | None:
    try:
        return float(str(value).replace(",", ".")) if value not in (None, "") else None
    except ValueError:
        return None


def highlight_terms(text: str, matches) -> Markup:
    """Marca os termos críticos no texto. Só é seguro quando a normalização preserva posições."""
    if len(text) != len(normalize(text)):
        return Markup(escape(text))
    out, cursor = [], 0
    for m in sorted(matches, key=lambda m: m.position):
        start, end = m.position, m.position + len(m.term)
        if start < cursor or normalize(text[start:end]) != normalize(m.term):
            continue
        sev = "low" if m.negated else m.severity.value
        note = f"{m.category} — negado" if m.negated else m.category
        out.append(escape(text[cursor:start]))
        out.append(f'<mark class="t {sev}" title="{escape(note)}">{escape(text[start:end])}</mark>')
        cursor = end
    out.append(escape(text[cursor:]))
    return Markup("".join(out))


# ------------------------------------------------------------ visão geral
@router.get("/{patient_id}")
def patient_page(
    patient_id: int, request: Request,
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    history = repos.consultations(db, patient_id)
    return render(
        request, "patient.html", doctor=doctor, patient=patient,
        last=history[0] if history else None, **alerts_context(db, patient_id),
    )  # fmt: skip


# ------------------------------------------------------------------ áudio
@router.get("/{patient_id}/audio")
def audio_page(
    patient_id: int, request: Request,
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    svc = request.app.state.services
    return render(
        request, "audio.html", doctor=doctor, patient=patient, audio_error=svc.audio_error,
        mock=svc.settings.mock_azure, max_mb=svc.settings.max_upload_mb,
    )  # fmt: skip


@router.post("/{patient_id}/audio")
def audio_analyze(
    patient_id: int, request: Request,
    file: UploadFile | None = File(None), text: str = Form(""),
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    svc = request.app.state.services
    if svc.audio is None:
        return _error(request, svc.audio_error or "Análise de áudio indisponível.", 503)

    limit = svc.settings.max_upload_mb * 1024 * 1024
    try:
        if file is not None and file.filename:
            content = file.file.read(limit + 1)
            if not content:
                return _error(request, "O arquivo de áudio está vazio.", 400)
            if len(content) > limit:
                return _error(request, f"Arquivo maior que {svc.settings.max_upload_mb} MB.", 413)
            suffix = Path(file.filename).suffix.lower()[:10] or ".bin"
            with tempfile.TemporaryDirectory(prefix="vitaltrace_") as tmp:
                src = Path(tmp) / f"upload{suffix}"
                src.write_bytes(content)
                wav = prepare_wav(src, Path(tmp))
                analysis: AudioAnalysis = svc.audio.analyze_audio(str(patient.id), wav)
        elif text.strip():
            analysis = svc.audio.analyze_text(str(patient.id), text.strip()[:20_000])
        else:
            return _error(request, "Envie um arquivo de áudio ou cole uma transcrição.", 400)
    except AudioConversionError as exc:
        return _error(request, str(exc), 415)
    except (TranscriptionError, LanguageServiceError) as exc:
        return _error(request, f"Falha no serviço do Azure: {exc}", 502)

    created = repos.add_alerts(db, analysis.alerts)
    summary = (
        f"{len(analysis.critical_terms)} termo(s) crítico(s), sentimento "
        f"{analysis.text_analysis.sentiment.label}, {len(created)} alerta(s) novo(s)."
    )
    repos.save_analysis(db, patient.id, "audio", summary, analysis.model_dump(mode="json", exclude={"alerts"}))
    db.flush()
    report = analyze_patient(db, patient)
    db.commit()  # confirma antes de responder: o painel (polling) já precisa ver os alertas
    response = render(
        request, "_audio_result.html", analysis=analysis, patient=patient, new_alerts=created,
        transcript=highlight_terms(analysis.transcription.text, analysis.critical_terms),
        risk=report.risk, from_file=bool(file and file.filename), mock=svc.settings.mock_azure,
    )  # fmt: skip
    response.headers["HX-Trigger"] = "alerts-changed"
    return response


# ------------------------------------------------------------------ vídeo
@router.get("/{patient_id}/video")
def video_page(
    patient_id: int, request: Request,
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    svc = request.app.state.services
    return render(
        request, "video.html", doctor=doctor, patient=patient, video_error=svc.video_error,
        backend=svc.video.backend if svc.video else None, max_mb=svc.settings.max_video_mb,
        max_seconds=svc.settings.video_max_seconds,
    )  # fmt: skip


@router.post("/{patient_id}/video")
def video_analyze(
    patient_id: int, request: Request, file: UploadFile = File(...),
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    svc = request.app.state.services
    if svc.video is None:
        return _error(request, svc.video_error or "Análise de vídeo indisponível.", 503)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _VIDEO_SUFFIXES:
        return _error(request, "Formato não suportado. Envie MP4, AVI, MOV, MKV ou WebM.", 415)

    limit = svc.settings.max_video_mb * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="vitaltrace_") as tmp:
        src = Path(tmp) / f"upload{suffix}"
        size = 0
        with src.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    return _error(request, f"Vídeo maior que {svc.settings.max_video_mb} MB.", 413)
                out.write(chunk)
        if size == 0:
            return _error(request, "O arquivo de vídeo está vazio.", 400)
        try:
            analysis = svc.video.analyze(src)
        except VideoError as exc:
            return _error(request, str(exc), 422)

    analysis.patient_id = str(patient.id)
    analysis.alerts = build_video_alerts(patient.id, analysis)
    created = repos.add_alerts(db, analysis.alerts)
    summary = f"{len(analysis.events)} evento(s), mobilidade {analysis.metrics.mobility_index}."
    repos.save_analysis(db, patient.id, "video", summary, analysis.payload())
    db.flush()
    report = analyze_patient(db, patient)
    db.commit()
    previous = [a.payload.get("metrics", {}).get("mobility_index") for a in repos.latest_analyses(db, patient.id, "video")[1:4]]
    response = render(
        request, "_video_result.html", analysis=analysis, patient=patient, new_alerts=created,
        risk=report.risk, type_label=type_label, previous=[p for p in previous if p is not None],
    )  # fmt: skip
    response.headers["HX-Trigger"] = "alerts-changed"
    return response


# -------------------------------------------------------------- evolução
def _evolution_context(db: Session, patient) -> dict:
    df = repos.vitals_frame(db, patient.id)
    rx = repos.prescriptions(db, patient.id)
    markers = []
    by_drug: dict[str, list] = {}
    for item in rx:
        by_drug.setdefault(item.drug, []).append(item)
    drugs = []
    for drug, rows in by_drug.items():
        rows.sort(key=lambda r: r.start_date)
        for prev, cur in zip(rows, rows[1:]):
            if cur.dose_mg != prev.dose_mg:
                markers.append((
                    datetime.combine(cur.start_date, time(0, 1)),
                    f"{drug} {prev.dose_mg:g}→{cur.dose_mg:g} mg",
                ))  # fmt: skip
        current = rows[-1]
        drugs.append({
            "drug": drug,
            "current": f"{current.dose_mg:g} mg".replace(".", ","),
            "frequency": current.frequency,
            "active": current.end_date is None,
            "since": current.start_date,
            "history": " → ".join(f"{r.dose_mg:g} mg".replace(".", ",") for r in rows),
            "changed": len(rows) > 1,
        })  # fmt: skip
    return {
        "history": repos.consultations(db, patient.id),
        "spo2_chart": charts.line_chart(df, "spo2", title="Saturação de oxigênio", unit="%", y_min=84, y_max=100,
                                        thresholds=[(92, "limiar 92%")], markers=markers),
        "hr_chart": charts.line_chart(df, "heart_rate", title="Frequência cardíaca", unit="bpm", y_min=40, y_max=130,
                                      thresholds=[(110, "limiar 110")], markers=markers),
        "drugs": drugs,
    }  # fmt: skip


@router.get("/{patient_id}/evolution")
def evolution_page(
    patient_id: int, request: Request,
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    latest = repos.latest_analyses(db, patient_id, "anomaly", limit=1)
    return render(
        request, "evolution.html", doctor=doctor, patient=patient, error=None, draft="",
        last_analysis=latest[0] if latest else None, **_evolution_context(db, patient),
    )  # fmt: skip


@router.post("/{patient_id}/evolution")
def add_consultation(
    patient_id: int, request: Request,
    anamnesis: str = Form(""), spo2: str = Form(""), heart_rate: str = Form(""),
    systolic: str = Form(""), diastolic: str = Form(""),
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    if len(anamnesis.strip()) < 10:
        return render(
            request, "evolution.html", status_code=422, doctor=doctor, patient=patient, draft=anamnesis,
            error="Descreva a anamnese com pelo menos 10 caracteres.", last_analysis=None,
            **_evolution_context(db, patient),
        )  # fmt: skip
    repos.add_consultation(
        db, patient=patient, doctor_id=doctor.id, anamnesis=anamnesis[:5000],
        spo2=_num(spo2), heart_rate=_num(heart_rate), systolic=_num(systolic), diastolic=_num(diastolic),
    )  # fmt: skip
    db.commit()
    return RedirectResponse(f"/patients/{patient_id}/evolution#historico", status_code=303)


@router.post("/{patient_id}/evolution/analyze")
def run_anomaly_analysis(
    patient_id: int, request: Request,
    doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
):  # fmt: skip
    patient = patient_or_404(db, patient_id)
    report = analyze_patient(db, patient)
    db.commit()
    by_source: dict[str, list] = {}
    for f in report.findings:
        by_source.setdefault(f.source, []).append(f)
    response = render(request, "_anomaly_result.html", report=report, patient=patient, by_source=by_source)
    response.headers["HX-Trigger"] = "alerts-changed"
    return response
