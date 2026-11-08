import pytest
from video_fixtures import write_video

from vitaltrace.db.seed import DEMO_EMAIL


def test_requires_login(client):
    r = client.get("/", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/login"
    assert client.get("/patients/1", follow_redirects=False).status_code == 303
    assert client.get("/api/alerts", follow_redirects=False).status_code == 303


def test_htmx_request_without_session_gets_redirect_header(client):
    r = client.get("/alerts/panel", headers={"HX-Request": "true"})
    assert r.status_code == 204 and r.headers["HX-Redirect"] == "/login"


def test_login_page_has_no_care_location_selector(client):
    html = client.get("/login").text
    assert "Entrar no portal" in html and DEMO_EMAIL in html
    assert "Hospital" not in html.split("<form")[1] and "Clínica" not in html


def test_wrong_password_is_rejected(client):
    r = client.post("/login", data={"email": DEMO_EMAIL, "password": "errada"})
    assert r.status_code == 401 and "incorretos" in r.text
    assert client.get("/", follow_redirects=False).status_code == 303


def test_home_lists_five_recent_with_show_more(logged):
    html = logged.get("/").text
    assert "Atendimentos recentes" in html and "Mostrando 5 de 30 pacientes" in html
    assert html.count('class="item"') == 5
    assert "Ver todos os 30 pacientes" in html
    assert "Helena Martins Duarte" in html and "Sérgio Luiz Machado" not in html


def test_home_show_all(logged):
    html = logged.get("/?all=true").text
    assert html.count('class="item"') == 30 and "Mostrar só os recentes" in html


def test_search_by_name_is_accent_insensitive(logged):
    html = logged.get("/patients-list", params={"q": "antonio"}).text
    assert "José Antônio Ribeiro" in html and "Antônio Carlos Medeiros" in html
    assert html.count('class="item"') == 2 and "2 pacientes encontrados" in html
    assert "Nenhum paciente encontrado" in logged.get("/patients-list", params={"q": "xyz"}).text


def test_critical_banner_and_acknowledge_flow(logged):
    assert "alertas críticos em aberto" in logged.get("/alerts/banner").text
    html = logged.get("/alerts/panel").text
    assert "Estou ciente" in html and "Reconhecer" not in html
    alerts = logged.get("/api/alerts").json()
    first = alerts[0]
    r = logged.post(f"/alerts/{first['id']}/ack")
    assert r.status_code == 200 and r.headers["HX-Trigger"] == "alerts-changed"
    remaining = {a["id"] for a in logged.get("/api/alerts").json()}
    assert first["id"] not in remaining and len(remaining) == len(alerts) - 1
    assert logged.post("/alerts/naoexiste/ack").status_code == 404


def test_patient_page_and_404(logged):
    html = logged.get("/patients/1").text
    assert "Helena Martins Duarte" in html and "Analisar gravação" in html
    assert "Análise de vídeo" in html and "Evolução clínica" in html and "00482" in html
    assert logged.get("/patients/9999").status_code == 404


def test_audio_text_analysis_creates_alerts_and_updates_risk(logged):
    before = len(logged.get("/api/alerts").json())
    r = logged.post("/patients/6/audio", data={"text": "Doutor, estou com falta de ar e muita dor no peito."})
    assert r.status_code == 200
    assert '<mark class="t critical"' in r.text and "falta de ar" in r.text
    assert len(logged.get("/api/alerts").json()) > before
    assert "CRÍTICO" in logged.get("/patients/6").text  # risco do paciente subiu


def test_audio_negated_terms_do_not_alert(logged):
    r = logged.post("/patients/4/audio", data={"text": "Estou bem, sem falta de ar e sem dor no peito."})
    assert r.status_code == 200 and "Nenhum alerta" in r.text


def test_audio_upload_with_mock_azure(logged):
    r = logged.post("/patients/5/audio", files={"file": ("consulta.wav", b"RIFF....", "audio/wav")})
    assert r.status_code in (200, 415)  # 415 se o ffmpeg do ambiente rejeitar o WAV falso


def test_audio_requires_input(logged):
    r = logged.post("/patients/5/audio", data={"text": "  "})
    assert r.status_code == 400 and "Envie um arquivo" in r.text


def test_video_upload_detects_fall_and_raises_alert(logged, tmp_path):
    path = write_video(tmp_path / "queda.avi", [("empty", 3), ("walk", 8), ("fall", 10)])
    with path.open("rb") as f:
        r = logged.post("/patients/3/video", files={"file": ("queda.avi", f, "video/x-msvideo")})
    assert r.status_code == 200
    assert "Queda suspeita" in r.text and "data:image/jpeg;base64" in r.text
    alerts = logged.get("/api/alerts").json()
    assert any(a["patient_id"] == 3 and a["source"] == "video" and a["severity"] == "critical" for a in alerts)


def test_video_rejects_bad_files(logged):
    r = logged.post("/patients/3/video", files={"file": ("x.txt", b"abc", "text/plain")})
    assert r.status_code == 415
    r = logged.post("/patients/3/video", files={"file": ("x.mp4", b"nao e video", "video/mp4")})
    assert r.status_code == 422


def test_evolution_page_has_charts_and_prescriptions(logged):
    html = logged.get("/patients/1/evolution").text
    assert html.count("<svg class=\"chart\"") == 2 and "Furosemida" in html
    assert "Furosemida 40→80 mg" in html and "alterada" in html
    assert "Analisar anomalias com IA" in html


def test_add_consultation_and_validation(logged):
    r = logged.post("/patients/4/evolution", data={"anamnesis": "curta"})
    assert r.status_code == 422 and "pelo menos 10" in r.text
    r = logged.post(
        "/patients/4/evolution",
        data={"anamnesis": "Refere melhora completa da cefaleia, sem novas queixas.", "spo2": "98", "heart_rate": "73"},
        follow_redirects=False,
    )
    assert r.status_code == 303
    page = logged.get("/patients/4/evolution").text
    assert "melhora completa da cefaleia" in page and "<b>73</b>" in page
    # a última consulta passa a ser a nova
    assert "melhora completa" in logged.get("/patients/4").text


def test_anomaly_analysis_for_helena(logged):
    r = logged.post("/patients/1/evolution/analyze")
    assert r.status_code == 200
    for expected in ("CRÍTICO", "Sinais vitais", "Prescrição", "Correlação", "Furosemida", "sem valor diagnóstico"):
        assert expected in r.text, expected


def test_anomaly_analysis_is_quiet_for_stable_patient(logged):
    r = logged.post("/patients/4/evolution/analyze")
    assert "ESTÁVEL" in r.text and "Nenhum desvio" in r.text


def test_anomaly_analysis_does_not_duplicate_open_alerts(logged):
    logged.post("/patients/1/evolution/analyze")
    count = len(logged.get("/api/alerts").json())
    logged.post("/patients/1/evolution/analyze")
    assert len(logged.get("/api/alerts").json()) == count


def test_new_consultation_text_feeds_the_analysis(logged):
    logged.post("/patients/4/evolution", data={"anamnesis": "Paciente refere desmaio ontem e falta de ar ao deitar."})
    r = logged.post("/patients/4/evolution/analyze")
    assert "CRÍTICO" in r.text and "Anamnese" in r.text


def test_logout(logged):
    assert logged.get("/logout", follow_redirects=False).status_code == 303
    assert logged.get("/", follow_redirects=False).status_code == 303


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["mock_azure"] is True and body["video"]


def test_audio_and_video_pages_use_the_dropzone(logged):
    for path, kind in (("/patients/1/audio", "audio"), ("/patients/1/video", "video")):
        html = logged.get(path).text
        assert 'data-drop' in html and f'data-kind="{kind}"' in html and "Arraste" in html
        assert 'type="file"' in html and "/static/js/dropzone.js" in html
    assert logged.get("/static/js/dropzone.js").status_code == 200


def test_video_page_has_depth_selector_and_result_shows_details(logged, tmp_path):
    assert 'name="depth"' in logged.get("/patients/3/video").text
    path = write_video(tmp_path / "anda.avi", [("empty", 2), ("walk", 12)])
    with path.open("rb") as f:
        r = logged.post("/patients/3/video", files={"file": ("anda.avi", f, "video/x-msvideo")}, data={"depth": "detailed"})
    assert r.status_code == 200 and "Detalhes da análise" in r.text and "Análise detalhada" in r.text
