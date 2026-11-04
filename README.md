# MedMonitor — FIAP Tech Challenge · Fase 4

Monitoramento multimodal de pacientes (áudio, vídeo e texto) com detecção de sinais precoces de risco e alertas para a equipe médica, usando serviços gerenciados do **Azure AI**. Projeto da pós-graduação *IA para Devs* (FIAP).

## Estado atual

| Módulo | Estado |
|---|---|
| Áudio: Azure Speech + Azure Language + termos críticos + métricas de fala | ✅ primeiro slice |
| Alertas dentro da aplicação (API + painel Streamlit) | ✅ |
| Vídeo (YOLOv8 + pose, em CPU) | ⏳ |
| Anomalias (sinais vitais, prescrições, movimentação) | ⏳ |
| Fusão multimodal | ⏳ |

Cronograma até 16/11: [docs/roadmap.md](docs/roadmap.md) · Arquitetura: [docs/architecture.md](docs/architecture.md) · Azure: [docs/azure-setup.md](docs/azure-setup.md)

## Como rodar (Windows / PowerShell)

Requer Python 3.11+ (recomendado 3.11 ou 3.12) e, opcionalmente, o [ffmpeg](https://ffmpeg.org) (`winget install Gyan.FFmpeg`) para aceitar mp3/m4a etc.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
copy .env.example .env
```

**Sem Azure (para testar o fluxo):** no `.env`, defina `MOCK_AZURE=true`. O transcritor simulado ignora o áudio e devolve um texto fixo.

**Com Azure:** siga [docs/azure-setup.md](docs/azure-setup.md), preencha o `.env` e valide:

```powershell
python scripts/azure_smoke_test.py
python scripts/generate_sample_audio.py    # gera áudios sintéticos em data/samples/
```

Em dois terminais:

```powershell
# 1) API
uvicorn medmonitor.api.main:create_app --factory --app-dir src --reload

# 2) Painel
streamlit run dashboard/app.py
```

Abra <http://localhost:8501>, envie um áudio (ou use a aba de texto) e acompanhe o banner/toast de alertas na aba **Alertas**. Documentação interativa da API: <http://localhost:8000/docs>.

### Docker

```powershell
copy .env.example .env   # preencha ou use MOCK_AZURE=true
docker compose up --build
```

API em `:8000`, painel em `:8501`.

## Testes

```powershell
pytest
```

Os testes não dependem do Azure (usam serviços simulados e falsos).

## Estrutura

```
src/medmonitor/
├── audio/      # Speech, Language, termos críticos, métricas, pipeline
├── alerts/     # regras de alerta e persistência (SQLite)
├── api/        # FastAPI
├── video/      # (a implementar) YOLOv8 / pose
├── anomaly/    # (a implementar) séries temporais, prescrições
├── fusion/     # (a implementar) score de risco multimodal
├── config.py · models.py · services.py
dashboard/app.py      # painel Streamlit
scripts/              # smoke test do Azure e geração de áudios de exemplo
tests/                # pytest
docs/                 # arquitetura, Azure, cronograma, relatório técnico
data/samples/         # áudios/vídeos de exemplo (sintéticos ou públicos)
```

## Avisos

Projeto acadêmico. As regras e limiares são heurísticas ilustrativas, sem validação clínica, e o sistema apoia, não substitui, a decisão da equipe médica. Use apenas dados públicos ou sintéticos; nunca dados reais de pacientes. Nunca faça commit do `.env`.
