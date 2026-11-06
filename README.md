# VitalTrace — FIAP Tech Challenge · Fase 4

Portal médico com análise multimodal de pacientes (áudio, vídeo, sinais vitais, prescrições e anamnese), detecção de anomalias e alertas para a equipe dentro da própria aplicação. Usa **Azure AI Speech** e **Azure AI Language** para voz/texto e **YOLOv8-pose em CPU** para vídeo. Projeto da pós-graduação *IA para Devs* (FIAP).

> Dados 100% fictícios e sintéticos. As regras e limiares são heurísticas didáticas, sem validação clínica: o sistema apoia, não substitui, a decisão médica.

## O que o portal faz

| Tela | Função |
|---|---|
| Login | Acesso do médico (sem escolha de local de atendimento) |
| Pacientes | 5 atendimentos mais recentes, "Ver todos os 30 pacientes" e busca por nome (sem acento/maiúscula) |
| Alertas | Painel que se atualiza a cada 3 s, faixa vermelha para críticos e botão **Estou ciente** |
| Paciente → Analisar gravação | Upload de áudio (ou transcrição colada): Azure Speech → Azure Language → termos críticos, sentimento, ritmo de fala → alertas |
| Paciente → Análise de vídeo | Upload de vídeo: pose/movimento em CPU → queda, imobilidade, assimetria de joelhos, mobilidade → relatório automático |
| Paciente → Evolução clínica | **Linha do tempo** com consultas, áudios, vídeos e análises de IA (cada registro abre mostrando o que foi analisado e o resultado), gráficos de SpO₂/FC com mudanças de dose e marcadores de áudio/vídeo, prescrições e **Análise de anomalias com IA** |

Anomalias detectadas: séries de sinais vitais (regras clínicas + desvio robusto da linha de base + tendência + Isolation Forest), evolução de prescrições (saltos e aumentos repetidos de dose), movimento (queda de mobilidade e quedas no vídeo) e texto da anamnese. A **fusão** cruza as modalidades e calcula o risco (0–100) de cada paciente.

Arquitetura: [docs/architecture.md](docs/architecture.md) · Azure: [docs/azure-setup.md](docs/azure-setup.md) · Relatório técnico: [docs/relatorio-tecnico.md](docs/relatorio-tecnico.md) · Cronograma e roteiro do vídeo: [docs/roadmap.md](docs/roadmap.md)

## Como rodar

### Opção A — Docker (Postgres + portal)

```powershell
copy .env.example .env      # preencha o Azure, ou use MOCK_AZURE=true
docker compose up --build
```

Portal em <http://localhost:8000>. O banco sobe vazio e é semeado automaticamente com 1 médico e 30 pacientes. A imagem inclui o YOLOv8-pose (PyTorch só-CPU); para uma imagem leve, use `docker compose build --build-arg WITH_YOLO=false` (o vídeo cai no plano B, OpenCV).

### Opção B — Python local (SQLite)

Requer Python 3.11+ e, opcionalmente, [ffmpeg](https://ffmpeg.org) (`winget install Gyan.FFmpeg`) para mp3/m4a.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
# opcional, para pose com YOLOv8 (CPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-video.txt

copy .env.example .env      # MOCK_AZURE=true para testar sem Azure
uvicorn vitaltrace.web.app:create_app --factory --app-dir src --reload
```

Abra <http://localhost:8000>.

**Acesso de demonstração:** `rafael.menezes@hospitalmodelo.demo` · senha `vitaltrace123`.

### Azure

Siga [docs/azure-setup.md](docs/azure-setup.md), preencha o `.env`, defina `MOCK_AZURE=false` e valide:

```powershell
python scripts/azure_smoke_test.py
python scripts/generate_sample_audio.py    # gera 3 áudios sintéticos em data/samples/
```

Em modo simulado (`MOCK_AZURE=true`) a transcrição é fixa e ignora o áudio; para textos próprios use o campo de transcrição colada.

## Vídeo por link do YouTube (opcional)

Na tela **Análise de vídeo** dá para colar um link do YouTube em vez de enviar o arquivo. O servidor baixa só os primeiros 120 s, em até 480p, analisa e apaga o arquivo (o vídeo não é guardado). Só aceita endereços do `youtube.com` e `youtu.be`, e apenas o ID do vídeo chega ao `yt-dlp`.

- **Docker:** já vem pronto (o `yt-dlp` e o Deno entram na imagem). O container precisa de internet.
- **Local (sem Docker):** o `yt-dlp` vem no `requirements.txt`, e o YouTube exige também um runtime JavaScript: no Windows, `winget install DenoLand.Deno`.
- **Se o download falhar do nada:** o YouTube muda com frequência. Rode `pip install -U "yt-dlp[default]"` (no Docker, `docker compose build --no-cache web`). Vídeos privados, com restrição de idade ou que peçam login não funcionam.
- **Desligar:** `YOUTUBE_ENABLED=false` no `.env` esconde o campo.
- **Uso:** só para demonstração, com vídeos públicos. Os Termos de Serviço do YouTube restringem baixar vídeos fora dos recursos que ele oferece. Para a análise funcionar bem, prefira clipes de câmera parada e uma pessoa em cena.

## Testes

```powershell
pytest
```

Rodam sem Azure e sem YOLO: usam serviços simulados, um modelo falso no adaptador do YOLO e vídeos sintéticos gerados na hora (pessoa = retângulo que anda, cai ou fica parada).

## Estrutura

```
src/vitaltrace/
├── web/        # portal: FastAPI + Jinja2 + HTMX (templates, CSS, gráficos SVG)
├── audio/      # Azure Speech, Azure Language, termos críticos, métricas de fala
├── video/      # pose (YOLOv8) ou movimento (OpenCV), eventos, relatório
├── anomaly/    # sinais vitais, prescrições, movimento, anamnese + orquestração
├── fusion/     # correlação entre modalidades, score de risco, resumo
├── alerts/     # regras que transformam achados de áudio em alertas
├── db/         # SQLAlchemy (Postgres/SQLite), repositórios, seed de 30 pacientes
└── config.py · models.py · security.py · services.py
scripts/        # smoke test do Azure e geração de áudios de exemplo
tests/          # 88 testes
docs/           # arquitetura, Azure, cronograma, relatório técnico
```

## Avisos

Projeto acadêmico. Use apenas dados públicos ou sintéticos; nunca dados reais de pacientes. Nunca faça commit do `.env`. Troque `SECRET_KEY` fora da sua máquina.
