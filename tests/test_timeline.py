"""Linha do tempo do paciente: consultas, áudios, vídeos e análises de IA com os detalhes de cada um."""

import re


from video_fixtures import write_video

from vitaltrace.db import repos
from vitaltrace.db.orm import Analysis
from vitaltrace.web import timeline

HELENA, SOFIA, CARLOS = 1, 3, 4


def _entries(html: str, kind: str) -> list[str]:
    return re.findall(rf'id="(tl-{kind}-\d+)"', html)


def test_evolution_page_shows_seeded_audio_and_video_with_details(logged):
    html = logged.get("/patients/1/evolution").text
    assert "Linha do tempo do paciente" in html
    assert _entries(html, "audio") and _entries(html, "video") and _entries(html, "consult")
    # o que foi analisado e o resultado, no próprio registro
    assert "consulta_helena_manha.wav" in html and "O que foi analisado" in html
    assert '<mark class="t critical"' in html and "falta de ar" in html
    assert "Em aberto" in html  # situação do alerta gerado pelo áudio
    assert "Quadro ilustrativo" not in html  # a imagem vem de um endpoint, não embutida na página
    assert "/patients/1/analyses/" in html and "/frames/0" in html


def test_charts_get_clickable_markers_for_audio_and_video(logged):
    html = logged.get("/patients/1/evolution").text
    links = re.findall(r'<a href="#(tl-(?:audio|video)-\d+)"', html)
    assert links, "o gráfico deveria marcar o áudio/vídeo"
    for anchor in set(links):
        assert f'id="{anchor}"' in html  # cada marcador aponta para um registro presente na página
    assert "Áudio" in html and "Vídeo" in html  # legenda


def test_filter_and_show_more(logged):
    audio = logged.get("/patients/1/timeline?kind=audio").text
    assert _entries(audio, "audio") and not _entries(audio, "consult") and 'aria-selected="true"' in audio
    assert not _entries(logged.get("/patients/1/timeline?kind=video").text, "audio")
    everything = logged.get("/patients/1/timeline?kind=all").text
    assert "Mostrar mais" not in everything or "offset=10" in everything
    rest = logged.get("/patients/1/timeline?kind=all&offset=1000").text
    assert "<li" in rest and not _entries(rest, "audio")  # além do fim: só a mensagem de vazio
    assert logged.get("/patients/999/timeline").status_code == 404
    assert logged.get("/patients/1/timeline?kind=qualquer").status_code == 200


def test_live_audio_analysis_appears_with_risk_change_and_alert_status(logged):
    r = logged.post("/patients/6/audio", data={"text": "Doutor, estou com falta de ar e muita dor no peito."})
    assert r.status_code == 200
    html = logged.get("/patients/6/evolution").text
    entry = _entries(html, "audio")[0]  # o mais novo vem primeiro
    assert "transcrição colada" in html and "Risco do paciente após esta análise" in html
    assert "Em aberto" in html
    alert_id = next(a["id"] for a in logged.get("/api/alerts").json() if a["patient_id"] == 6)
    assert logged.post(f"/alerts/{alert_id}/ack").status_code == 200
    assert "Ciente" in logged.get("/patients/6/evolution").text
    assert entry.startswith("tl-audio-")


def test_audio_and_video_do_not_add_extra_ai_entries(logged, tmp_path):
    before = _entries(logged.get("/patients/5/timeline?kind=anomaly").text, "anomaly")
    logged.post("/patients/5/audio", data={"text": "Tudo bem por aqui, sem queixas."})
    path = write_video(tmp_path / "anda.avi", [("empty", 3), ("walk", 14)])
    with path.open("rb") as f:
        assert logged.post("/patients/5/video", files={"file": ("anda.avi", f, "video/x-msvideo")}).status_code == 200
    assert _entries(logged.get("/patients/5/timeline?kind=anomaly").text, "anomaly") == before
    logged.post("/patients/5/evolution/analyze")
    assert len(_entries(logged.get("/patients/5/timeline?kind=anomaly").text, "anomaly")) == len(before) + 1


def test_video_upload_stores_keyframes_served_by_endpoint(logged, tmp_path):
    path = write_video(tmp_path / "queda.avi", [("empty", 3), ("walk", 8), ("fall", 10)])
    with path.open("rb") as f:
        assert logged.post("/patients/3/video", files={"file": ("queda.avi", f, "video/x-msvideo")}).status_code == 200
    html = logged.get("/patients/3/evolution").text
    assert "queda.avi" in html and "Queda suspeita" in html and "O vídeo original não é guardado" in html
    match = re.search(r'src="(/patients/3/analyses/\d+/frames/0)"', html)
    assert match
    image = logged.get(match.group(1))
    assert image.status_code == 200 and image.headers["content-type"] == "image/jpeg" and image.content[:2] == b"\xff\xd8"
    assert logged.get(match.group(1).replace("/patients/3/", "/patients/4/")).status_code == 404  # outro paciente
    assert logged.get(re.sub(r"frames/0", "frames/99", match.group(1))).status_code == 404


def test_consultation_entry_shows_text_analysis_result(logged):
    logged.post("/patients/4/evolution", data={"anamnesis": "Paciente refere falta de ar ao subir escadas desde ontem."})
    html = logged.get("/patients/4/evolution").text
    assert "Resultado da análise do texto" in html and "Anamnese menciona" in html


def test_legacy_analyses_without_new_fields_still_render(logged):
    with logged.app.state.services.db.session() as s:
        repos.save_analysis(s, 8, "video", "sessão antiga", {"metrics": {"mobility_index": 0.5}})
        repos.save_analysis(s, 8, "audio", "áudio antigo", {})
        s.commit()
    r = logged.get("/patients/8/evolution")
    assert r.status_code == 200 and "Sem quadros-chave" in r.text and "áudio antigo" in r.text


def test_timeline_service_orders_newest_first_and_compares_videos(db):
    with db.session() as s:
        entries = timeline.all_entries(s, 3)  # Sofia
        stamps = [e.at for e in entries]
        assert stamps == sorted(stamps, reverse=True)
        videos = [e for e in entries if e.kind == "video"]
        assert len(videos) >= 2 and videos[0].prev_mobility is not None and videos[-1].prev_mobility is None
        page = timeline.build_timeline(s, 3, limit=2)
        assert len(page.entries) == 2 and page.next_offset == 2
