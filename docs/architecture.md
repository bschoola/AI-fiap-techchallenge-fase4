# Arquitetura

```mermaid
flowchart LR
    M[Médico] --> W[Portal web<br/>FastAPI + Jinja2 + HTMX]
    W --> A[Pipeline de áudio]
    W --> V[Pipeline de vídeo]
    W --> E[Evolução clínica<br/>anamnese · sinais vitais · prescrições]
    A -->|fala| SP[Azure AI Speech]
    A -->|texto| LG[Azure AI Language]
    A --> T[Termos críticos · ritmo de fala]
    V --> Y[YOLO-pose em CPU + rastreamento<br/>ou OpenCV]
    V --> EV[Queda · imobilidade · assimetria · mobilidade]
    E --> AN[Detectores de anomalia<br/>regras · z-score · tendência · Isolation Forest · doses]
    T --> F[Fusão: correlação e score de risco]
    EV --> F
    AN --> F
    F --> AL[(Alertas)]
    F --> R[(Risco do paciente)]
    AL --> W
    DB[(Postgres)] <--> W
```

## Decisões

- **Alertas na própria aplicação**, não por e-mail/Teams: ficam no banco, o painel faz *polling* (HTMX) a cada 3 s, uma faixa vermelha avisa quando há crítico em aberto e o médico confirma com **Estou ciente** (registra quem e quando).
- **Front sem build**: FastAPI + Jinja2 + HTMX, tudo em Python. Gráficos são SVG gerados no servidor.
- **CPU**: YOLO26s-pose em quadros reduzidos a 960 px (2 quadros/s no modo rápido, 8 no detalhado), rastreador próprio para fixar o paciente e segunda passada de pose no recorte dele. Sem `ultralytics`, o plano B usa subtração de fundo do OpenCV (bounding box, sem articulações).
- **Anomalias leves e explicáveis**: regras clínicas, desvio robusto (mediana/MAD) contra a linha de base do próprio paciente, tendência por regressão linear e Isolation Forest treinado na linha de base. Cada achado traz o motivo e a evidência.
- **Risco**: pontos por severidade, teto de 50 por modalidade (o score exige convergência) e nível "crítico" só com algum achado crítico. Correlações entre modalidades geram achados próprios (ex.: *piora clínica apesar do ajuste de dose*).
- **Resumo por regras**, determinístico e sem LLM, sempre com aviso de que não é diagnóstico.
- **Serviços atrás de interfaces** (`Transcriber`, `TextAnalyzer`, `PersonEstimator`): trocar Azure por simulado, ou YOLO por OpenCV, não toca no resto.
- **Dados**: SQLAlchemy 2. Postgres no Docker; SQLite local e nos testes. Seed idempotente na subida.

## Fluxo de um achado

1. Upload (áudio/vídeo) ou clique em *Analisar anomalias com IA*.
2. O módulo gera achados (`Finding`: origem, severidade, título, descrição, evidência).
3. A fusão correlaciona e calcula o risco; achados de severidade alta/crítica viram alertas (sem duplicar os que já estão em aberto).
4. O painel mostra o alerta; **Estou ciente** o encerra.
