# VitalTrace — FIAP Tech Challenge · Fase 4

Portal médico com análise multimodal de pacientes (áudio, vídeo, sinais vitais, prescrições e anamnese), detecção de anomalias e alertas para a equipe dentro da própria aplicação. Usa **Azure AI Speech** e **Azure AI Language** para voz/texto e **YOLO-pose em CPU** para vídeo. Projeto da pós-graduação *IA para Devs* (FIAP).

> Dados 100% fictícios e sintéticos. As regras e limiares são heurísticas didáticas, sem validação clínica: o sistema apoia, não substitui, a decisão médica.

## O que o portal faz

| Tela | Função |
|---|---|
| Login | Acesso do médico (sem escolha de local de atendimento) |
| Pacientes | 5 atendimentos mais recentes, "Ver todos os 30 pacientes" e busca por nome (sem acento/maiúscula) |
| Alertas | Painel que se atualiza a cada 3 s, faixa vermelha para críticos e botão **Estou ciente** |
| Paciente → Analisar gravação | Upload de áudio (ou transcrição colada): Azure Speech → Azure Language → termos críticos, sentimento, ritmo de fala → alertas |
| Paciente → Análise de vídeo | Upload de vídeo ou link do YouTube: rastreia as pessoas, fixa o paciente (o médico escolhe quando há mais de uma), pose em CPU → descrição do que foi visto, ângulos por articulação, repetições, queda, imobilidade, assimetrias → relatório automático e vídeo com o esqueleto |
| Paciente → Evolução clínica | **Linha do tempo** com consultas, áudios, vídeos e análises de IA (cada registro abre mostrando o que foi analisado e o resultado), gráficos de SpO₂/FC com mudanças de dose e marcadores de áudio/vídeo, prescrições e **Análise de anomalias com IA** |

Anomalias detectadas: séries de sinais vitais (regras clínicas + desvio robusto da linha de base + tendência + Isolation Forest), evolução de prescrições (saltos e aumentos repetidos de dose), movimento (queda de mobilidade e quedas no vídeo) e texto da anamnese. A **fusão** cruza as modalidades e calcula o risco (0–100) de cada paciente.

Arquitetura: [docs/architecture.md](docs/architecture.md) · Azure: [docs/azure-setup.md](docs/azure-setup.md) · Relatório técnico: [docs/relatorio-tecnico.md](docs/relatorio-tecnico.md) · Cronograma e roteiro do vídeo: [docs/roadmap.md](docs/roadmap.md)

## Modelos e técnicas por análise

Só quatro partes usam aprendizado de máquina; o restante é regra ou estatística, o que facilita explicar e auditar cada alerta. Nenhum modelo foi treinado com dados reais de pacientes: o YOLO usa os pesos pré-treinados da Ultralytics, o Isolation Forest aprende na hora com a série de cada paciente, e todos os limiares são pontos de partida.

### Áudio
| Etapa | Modelo / técnica |
|---|---|
| Transcrição | **Azure AI Speech**, reconhecimento contínuo em pt-BR (serve para áudios longos). Com `MOCK_AZURE=true`, um transcritor simulado de texto fixo. |
| Sentimento e frases-chave | **Azure AI Language** (`analyze_sentiment` e `extract_key_phrases`). Com `MOCK_AZURE=true`, um analisador simulado por palavras-chave. |
| Termos críticos | **Regras**: dicionário com severidade e categoria, ignorando termos negados ("sem falta de ar"). |
| Ritmo de fala | **Métricas calculadas** a partir dos tempos da transcrição (palavras por minuto e proporção de pausa); só geram alerta em trechos longos o bastante. Com a marcação **por palavra** do Azure Speech (`request_word_level_timestamps`), as pausas dentro de cada frase entram na conta e sai também a velocidade "sem as pausas". |
| Prosódia (como a pessoa fala) | **Processamento de sinal** com numpy direto no WAV (`audio/prosody.py`): tempo em silêncio, número e duração das pausas, duração média de cada trecho falado e variação do tom (F0 por autocorrelação). Não depende do Azure nem da transcrição. |
| Tom de voz (opcional) | **Azure OpenAI `gpt-audio`** ouvindo a gravação (`VOICE_TONE_ENABLED=true`): esforço respiratório, cansaço e tom emocional em JSON. É uma segunda opinião: sozinho gera, no máximo, alerta médio. |
| Fusão | **Regra**: muito silêncio + pausas longas e frequentes + trechos curtos = "Fala entrecortada" com severidade **alta**, mesmo que o paciente não diga nenhum termo crítico; vira **crítica** quando confirmada por queixa relatada ou pelo tom de voz. Limiares em `config.py` (`breathless_*`). |

### Vídeo
| Etapa | Modelo / técnica |
|---|---|
| Pessoas e pose | **YOLO26s-pose** (`yolo26s-pose.pt`, `YOLO_POSE_WEIGHTS`) via `ultralytics`, em CPU: quadro reduzido a 960 px no maior lado, entrada de 640 px, caixa e 17 pontos de **todas** as pessoas. Se os pesos não carregarem, cai para `yolo11s-pose.pt` e `yolov8n-pose.pt`. |
| Quem é o paciente | **Rastreador próprio** (`video/tracking.py`): associa as pessoas entre quadros por sobreposição da caixa e distância do esqueleto (algoritmo húngaro) e dá um número estável a cada uma. Uma pessoa por trecho é analisada: a indicada pelo médico ou, sem indicação, a que fica mais tempo em cena, maior e se movimentando mais. Depois de um corte, é reencontrada pela cor da roupa. |
| Pose em dois estágios | No modo detalhado, a pose do paciente é refeita só no **recorte dele, tirado do vídeo em resolução original** (entrada de 480 px), quando isso dá mais pixels do que o quadro inteiro deu. |
| Limpeza do esqueleto | `video/clean.py`: desfaz trocas esquerda/direita, descarta pontos que fazem um osso "esticar" de repente, e os ângulos só usam pontos com confiança ≥ 0,5. |
| Plano B | **OpenCV**: subtração de fundo MOG2 e a maior região em movimento, sem rede neural. Fica ativo se o YOLO não estiver instalado. |
| Postura, queda, imobilidade, joelhos | **Regras geométricas** sobre a pose: ângulo do tronco, queda de em pé para deitado em até 2 s, imobilidade abaixo de 0,03 altura-do-corpo/s, ângulo e assimetria dos joelhos. |
| Índice de mobilidade | Mediana do movimento, normalizada por uma referência fixa. |

### Anomalias clínicas
| Etapa | Modelo / técnica |
|---|---|
| Sinais vitais | Regras clínicas (por exemplo SpO₂ < 90) + desvio robusto em relação à linha de base do próprio paciente (mediana e MAD) + tendência linear de 3 dias + **Isolation Forest** do scikit-learn (100 árvores, `StandardScaler`) sobre as últimas 24 h. É o único modelo de ML treinado nesta parte. |
| Prescrições | **Regras**: salto de dose, aumentos repetidos, medicamento novo e polifarmácia. |
| Movimento | **Regras** sobre o histórico de vídeo: queda de 25% ou mais no índice de mobilidade e quedas no vídeo mais recente. |
| Anamnese | **Regras**: termos críticos (o mesmo dicionário do áudio) e linguagem de piora em consultas seguidas. |

### Fusão e relatório
| Etapa | Modelo / técnica |
|---|---|
| Risco de 0 a 100 | **Pontuação por regra**: pontos por severidade, teto por modalidade e correlações entre elas. |
| Resumo textual | **Texto montado por regra**, não gerado por LLM. O botão "Análise de anomalias com IA" usa essa combinação, não um modelo de linguagem. |


### Análise detalhada de um único vídeo

> **Tempo em CPU:** com `yolo26s-pose`, 8 quadros/s e refinamento, um vídeo de 2 minutos pode levar alguns minutos. Para acelerar: `YOLO_POSE_WEIGHTS=yolo26n-pose.pt`, `VIDEO_DETAIL_FPS=5` ou `VIDEO_REFINE_POSE=false`.

Não precisa de histórico do paciente: o vídeo é examinado por dentro. Na tela de vídeo, **Detalhada** (padrão, 8 quadros/s, `VIDEO_DETAIL_FPS`) ou **Rápida** (2 quadros/s, só os eventos básicos, sem refinamento).

**Mais de uma pessoa em cena:** a primeira passada rastreia todo mundo. Se duas ou mais candidatas aparecem ao mesmo tempo, a tela mostra a miniatura de cada uma e o médico clica em quem é o paciente; só então as medidas são feitas e salvas. As outras pessoas aparecem em cinza no vídeo anotado e não entram em nenhuma medida.

**O que foi visto:** cada análise detalhada descreve o vídeo trecho a trecho (quantas pessoas, postura, vista da câmera, exercício provável, articulação que mais trabalhou com menor e maior ângulo, repetições), traz a tabela de ângulos de joelho, quadril, ombro e cotovelo dos dois lados e guarda um **vídeo reduzido com o esqueleto desenhado** (`data/video_annotated/`, desligue com `VIDEO_ANNOTATED=false`). O exercício é estimado por regras, não por um classificador treinado. Opcionalmente (`VIDEO_NARRATIVE_ENABLED=true` + `AZURE_OPENAI_VISION_DEPLOYMENT`), 6 quadros vão a um modelo multimodal do Azure OpenAI, que descreve ambiente e cena; esse texto não entra nas medidas nem nos alertas.

| Etapa / achado | Como funciona | Gravidade |
|---|---|---|
| Cortes de cena | histograma de cinza + miniatura; velocidade e queda nunca atravessam um corte | aviso de qualidade |
| Paciente | rastreamento com número estável por pessoa; escolha do médico ou automática (tempo em cena × tamanho × movimento) | aviso de qualidade |
| Vista da câmera | largura dos ombros ÷ tronco: frontal, oblíqua ou lateral; simetria entre os lados só é avaliada com os dois lados à vista | — |
| Esqueleto pouco confiável | confiança média dos pontos < 0,40: achados de postura e amplitude não são emitidos | aviso de qualidade |
| Suavização | mediana móvel de 3 quadros por ponto (só entre quadros em que o ponto é confiável); falha de 1 quadro é preenchida | — |
| Tronco inclinado | > 20° da vertical por ≥ 5 s seguidos | baixa (≥ 30°: média) |
| Ombros / quadril desalinhados | mediana > 8° (só visão frontal) | baixa (≥ 14°: média) |
| Oscilação postural | desvio lateral do quadril > 0,10 do tronco, parado em pé ≥ 6 s | baixa (≥ 0,20: média) |
| Assimetria de amplitude | amplitude (p95–p5) de um lado < 65% do outro (joelhos, quadris, ombros, cotovelos), fora da vista lateral | baixa (< 50%: média) |
| Imobilidade × exercício | "imobilidade" e "mobilidade reduzida" são descartadas se alguma articulação trabalhou ≥ 25° no intervalo | — |
| Ritmo irregular | variação (CV) da duração das repetições > 35%, ≥ 4 repetições | baixa (> 60%: média) |
| Fadiga | amplitude do último terço ≥ 25% menor que a do primeiro, ≥ 6 repetições | baixa (≥ 30%: média) |

Achados de gravidade **média ou alta geram alerta** no painel da equipe; os de baixa aparecem no resultado e na linha do tempo. Cada análise guarda um **relatório de qualidade** (cortes, pose utilizável, mais de uma pessoa, vídeo curto, sem repetições) e a **curva de ângulo** da articulação mais ativa. Limitações: ângulos 2D de uma câmera só, sem calibração, não validados clinicamente; inclinação de câmera e visão lateral afetam ombros/quadril; em exercícios, inclinar o tronco pode ser o movimento esperado. É apoio à revisão humana, não diagnóstico.

## Como rodar

### Opção A — Docker (Postgres + portal)

```powershell
copy .env.example .env      # preencha o Azure, ou use MOCK_AZURE=true
docker compose up --build
```

Portal em <http://localhost:8000>. O banco sobe vazio e é semeado automaticamente com 1 médico e 30 pacientes. A imagem inclui o YOLO-pose (PyTorch só-CPU); para uma imagem leve, use `docker compose build --build-arg WITH_YOLO=false` (o vídeo cai no plano B, OpenCV).

### Opção B — Python local (SQLite)

Requer Python 3.11+ e, opcionalmente, [ffmpeg](https://ffmpeg.org) (`winget install Gyan.FFmpeg`) para mp3/m4a.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
# opcional, para pose com YOLO (CPU):
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

Na tela **Análise de vídeo** dá para colar um link do YouTube em vez de enviar o arquivo. O servidor baixa só os primeiros 120 s, em até 720p (`YOUTUBE_MAX_HEIGHT`), analisa e apaga o arquivo original (fica só a versão reduzida com o esqueleto, se `VIDEO_ANNOTATED=true`). Só aceita endereços do `youtube.com` e `youtu.be`, e apenas o ID do vídeo chega ao `yt-dlp`.

- **Docker:** já vem pronto (o `yt-dlp` e o Deno entram na imagem). O container precisa de internet.
- **Local (sem Docker):** o `yt-dlp` vem no `requirements.txt`, e o YouTube exige também um runtime JavaScript: no Windows, `winget install DenoLand.Deno`.
- **Se o download falhar do nada:** o YouTube muda com frequência. Rode `pip install -U "yt-dlp[default]"` (no Docker, `docker compose build --no-cache web`). Vídeos privados, com restrição de idade ou que peçam login não funcionam.
- **Desligar:** `YOUTUBE_ENABLED=false` no `.env` esconde o campo.
- **Uso:** só para demonstração, com vídeos públicos. Os Termos de Serviço do YouTube restringem baixar vídeos fora dos recursos que ele oferece. Para a análise funcionar bem, prefira clipes de câmera parada e corpo inteiro em cena; com mais de uma pessoa, a tela pede para indicar o paciente.

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
├── video/      # pose (YOLO) ou movimento (OpenCV), rastreamento do paciente, limpeza, eventos, descrição da cena, relatório
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
