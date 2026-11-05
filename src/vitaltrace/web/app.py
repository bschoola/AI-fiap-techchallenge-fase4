"""Aplicação web (FastAPI + Jinja2 + HTMX).

Execução (a partir da raiz do repositório):
    uvicorn vitaltrace.web.app:create_app --factory --app-dir src --reload
"""

from fastapi import Depends, FastAPI, Form, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.middleware.sessions import SessionMiddleware

from vitaltrace import __version__
from vitaltrace.config import Settings, get_settings
from vitaltrace.db import repos
from vitaltrace.db.orm import Doctor
from vitaltrace.db.seed import DEMO_EMAIL, DEMO_PASSWORD, seed_database
from vitaltrace.security import verify_password
from vitaltrace.services import Services, build_services
from vitaltrace.web import patients
from vitaltrace.web.common import (
    BASE,
    LoginRequired,
    alerts_context,
    get_db,
    make_templates,
    patient_list_context,
    render,
    require_doctor,
)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or get_settings()
    services = services or build_services(settings)
    if settings.seed_on_startup:
        seed_database(services.db)

    app = FastAPI(
        title="VitalTrace",
        version=__version__,
        description="Portal médico com análise multimodal — FIAP Tech Challenge Fase 4",
    )
    app.state.services = services
    app.state.templates = make_templates()
    app.add_middleware(
        SessionMiddleware,
        secret_key=settings.secret_key,
        session_cookie="vitaltrace_session",
        same_site="lax",
        max_age=60 * 60 * 8,
    )
    app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")

    @app.exception_handler(LoginRequired)
    async def _login_required(request: Request, exc: LoginRequired):
        if request.headers.get("HX-Request"):
            return Response(status_code=204, headers={"HX-Redirect": "/login"})
        return RedirectResponse("/login", status_code=303)

    app.include_router(patients.router)

    # ------------------------------------------------------------- páginas
    @app.get("/health", include_in_schema=False)
    def health() -> dict:
        return {
            "status": "ok",
            "version": __version__,
            "mock_azure": settings.mock_azure,
            "audio": services.audio is not None,
            "video": services.video.backend if services.video else None,
        }

    @app.get("/login")
    def login_page(request: Request):
        return render(request, "login.html", demo_email=DEMO_EMAIL, demo_password=DEMO_PASSWORD, error=None)

    @app.post("/login")
    def login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
        doctor = repos.get_doctor_by_email(db, email)
        if doctor is None or not verify_password(password, doctor.password_hash):
            return render(
                request, "login.html", demo_email=DEMO_EMAIL, demo_password=DEMO_PASSWORD,
                error="E-mail ou senha incorretos.", status_code=401,
            )  # fmt: skip
        request.session.clear()
        request.session["doctor_id"] = doctor.id
        return RedirectResponse("/", status_code=303)

    @app.get("/logout")
    def logout(request: Request):
        request.session.clear()
        return RedirectResponse("/login", status_code=303)

    @app.get("/")
    def home(
        request: Request,
        q: str = "",
        all: bool = False,
        doctor: Doctor = Depends(require_doctor),
        db: Session = Depends(get_db),
    ):
        ctx = patient_list_context(db, q, all)
        return render(request, "home.html", doctor=doctor, **ctx, **alerts_context(db))

    @app.get("/patients-list")
    def patients_partial(
        request: Request, q: str = "", all: bool = False,
        doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
    ):  # fmt: skip
        return render(request, "_patient_list.html", **patient_list_context(db, q, all))

    # ------------------------------------------------------------- alertas
    @app.get("/alerts/panel")
    def alerts_panel(
        request: Request, patient_id: int | None = Query(None),
        doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
    ):  # fmt: skip
        return render(request, "_alerts_panel.html", **alerts_context(db, patient_id))

    @app.get("/alerts/banner")
    def alerts_banner(request: Request, doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db)):
        ctx = alerts_context(db)
        return render(request, "_banner.html", critical=ctx["critical_alerts"])

    @app.post("/alerts/{alert_id}/ack")
    def acknowledge(
        alert_id: str, request: Request, patient_id: int | None = Query(None),
        doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db),
    ):  # fmt: skip
        if repos.acknowledge_alert(db, alert_id, doctor.name) is None:
            return JSONResponse({"detail": "Alerta não encontrado."}, status_code=404)
        db.commit()  # o dependency só confirma depois de enviar a resposta
        response = render(request, "_alerts_panel.html", **alerts_context(db, patient_id))
        response.headers["HX-Trigger"] = "alerts-changed"
        return response

    @app.get("/api/alerts")
    def api_alerts(doctor: Doctor = Depends(require_doctor), db: Session = Depends(get_db)):
        return [
            {
                "id": a.id, "patient_id": a.patient_id, "patient": a.patient.name, "source": a.source,
                "severity": a.severity, "title": a.title, "status": a.status,
                "created_at": a.created_at.isoformat(),
            }
            for a in repos.list_alerts(db)
        ]  # fmt: skip

    return app


