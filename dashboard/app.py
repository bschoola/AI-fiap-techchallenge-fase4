"""Painel Streamlit do MedMonitor.

Execução: streamlit run dashboard/app.py
Os avisos para a equipe médica aparecem dentro da própria aplicação (banner + toast),
atualizados automaticamente a cada poucos segundos.
"""

import os

import requests
import streamlit as st

SEVERITY_ORDER = ["critical", "high", "medium", "low", "info"]
SEVERITY_LABEL = {
    "critical": ("🔴", "CRÍTICO"),
    "high": ("🟠", "ALTO"),
    "medium": ("🟡", "MÉDIO"),
    "low": ("🔵", "BAIXO"),
    "info": ("⚪", "INFO"),
}
AUDIO_TYPES = ["wav", "mp3", "m4a", "ogg", "flac", "webm"]

st.set_page_config(page_title="MedMonitor", page_icon="🩺", layout="wide")
st.title("🩺 MedMonitor — monitoramento multimodal")

api_url = st.sidebar.text_input("URL da API", os.getenv("API_URL", "http://localhost:8000"))
patient_id = st.sidebar.text_input("ID do paciente", "paciente-001")


def call_api(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{api_url}{path}", timeout=120, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Erro da API: {detail}")
    except requests.RequestException as exc:
        st.error(f"API indisponível em {api_url}: {exc}")
    return None


def sort_alerts(alerts: list[dict]) -> list[dict]:
    # mais recentes primeiro; depois (ordenação estável) críticos primeiro
    newest_first = sorted(alerts, key=lambda a: a["created_at"], reverse=True)
    return sorted(newest_first, key=lambda a: SEVERITY_ORDER.index(a["severity"]))


@st.fragment(run_every="3s")
def alerts_panel() -> None:
    try:
        response = requests.get(
            f"{api_url}/alerts", params={"status": "open", "limit": 100}, timeout=5
        )
        response.raise_for_status()
        alerts = response.json()
    except requests.RequestException:
        st.warning(f"API indisponível em {api_url}")
        return

    # Notificação dentro da aplicação: toast para alertas novos desde a última atualização.
    seen = st.session_state.get("seen_alert_ids")
    current_ids = {a["id"] for a in alerts}
    if seen is not None:
        for alert in alerts:
            if alert["id"] not in seen:
                icon, label = SEVERITY_LABEL[alert["severity"]]
                st.toast(f"{label}: {alert['title']}", icon=icon)
    st.session_state["seen_alert_ids"] = current_ids

    critical = sum(a["severity"] == "critical" for a in alerts)
    if critical:
        st.error(f"🚨 {critical} alerta(s) CRÍTICO(S) em aberto — atenção imediata.")
    elif alerts:
        st.warning(f"{len(alerts)} alerta(s) em aberto.")
    else:
        st.success("Nenhum alerta em aberto.")

    for alert in sort_alerts(alerts):
        icon, label = SEVERITY_LABEL[alert["severity"]]
        with st.container(border=True):
            left, right = st.columns([6, 1])
            left.markdown(
                f"**{icon} {label}** · `{alert['source']}` · paciente "
                f"**{alert['patient_id']}**  \n{alert['title']}"
            )
            left.caption(f"{alert['description']}  ·  {alert['created_at'][:19]} UTC")
            if right.button("Reconhecer", key=f"ack-{alert['id']}"):
                requests.post(f"{api_url}/alerts/{alert['id']}/ack", timeout=5)
                st.rerun(scope="fragment")


def render_analysis(result: dict) -> None:
    st.subheader("Transcrição")
    st.write(result["transcription"]["text"] or "_(vazia)_")

    sentiment = result["text_analysis"]["sentiment"]
    metrics = result["metrics"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sentimento", sentiment["label"])
    col2.metric("Negativo", f"{sentiment['negative']:.0%}")
    wpm = metrics["words_per_minute"]
    col3.metric("Palavras/min", "—" if wpm is None else f"{wpm:.0f}")
    pause = metrics["pause_ratio"]
    col4.metric("Pausas", "—" if pause is None else f"{pause:.0%}")

    phrases = result["text_analysis"]["key_phrases"]
    if phrases:
        st.markdown("**Frases-chave:** " + ", ".join(phrases))

    terms = result["critical_terms"]
    if terms:
        st.markdown("**Termos clínicos detectados**")
        st.dataframe(
            [
                {
                    "termo": t["term"],
                    "categoria": t["category"],
                    "severidade": t["severity"],
                    "negado": t["negated"],
                    "contexto": t["context"],
                }
                for t in terms
            ]
        )
    st.caption(f"{len(result['alerts'])} alerta(s) gerado(s) — veja a aba Alertas.")


tab_alerts, tab_audio, tab_text = st.tabs(["🚨 Alertas", "🎙️ Analisar áudio", "📝 Analisar texto"])

with tab_alerts:
    alerts_panel()

with tab_audio:
    st.write("Envie a gravação de uma consulta (WAV recomendado).")
    upload = st.file_uploader("Arquivo de áudio", type=AUDIO_TYPES)
    if upload and st.button("Analisar áudio", type="primary"):
        with st.spinner("Transcrevendo e analisando..."):
            result = call_api(
                "POST",
                "/audio/analyze",
                data={"patient_id": patient_id},
                files={"file": (upload.name, upload.getvalue(), upload.type)},
            )
        if result:
            render_analysis(result)

with tab_text:
    text = st.text_area(
        "Texto transcrito",
        "Doutor, estou com muita falta de ar e uma dor no peito desde ontem.",
        height=140,
    )
    if st.button("Analisar texto"):
        with st.spinner("Analisando..."):
            result = call_api(
                "POST", "/text/analyze", json={"patient_id": patient_id, "text": text}
            )
        if result:
            render_analysis(result)
