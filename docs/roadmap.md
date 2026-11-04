# Cronograma — entrega em 16/11/2026 (segunda-feira)

> Confirme no portal da FIAP o horário-limite. O vídeo e o repositório devem estar prontos antes.

| Data | Meta | Entrega |
|---|---|---|
| Qua 04/11 – Qui 05/11 | Configurar Azure, rodar o primeiro slice com áudios de exemplo | `docs/azure-setup.md` concluído; alerta crítico no painel |
| Sex 06/11 – Dom 08/11 | **Vídeo**: pose (ângulos, simetria, queda) + YOLOv8n, relatório automático | `src/medmonitor/video`, 2–3 clipes de teste |
| Seg 09/11 – Ter 10/11 | **Anomalias**: sinais vitais (PhysioNet), prescrições sintéticas, movimentação | `src/medmonitor/anomaly`, métricas com anomalias injetadas |
| Qua 11/11 – Qui 12/11 | **Fusão + painel**: alertas de todas as modalidades, score de risco por paciente | `src/medmonitor/fusion`, painel completo |
| Sex 13/11 | **Relatório técnico** (fluxo, modelos, resultados, exemplos de anomalias) | `docs/relatorio-tecnico.md` |
| Sáb 14/11 | **Roteiro e gravação** do vídeo (até 15 min) e upload (YouTube/Vimeo, não listado) | link do vídeo |
| Dom 15/11 | Buffer: revisar README, rodar do zero em máquina limpa, corrigir pendências | repositório final |
| **Seg 16/11** | **Entrega** | repositório + link do vídeo |

## Roteiro sugerido do vídeo (≤ 15 min)

1. Contexto e arquitetura (1–2 min)
2. Áudio: upload de `paciente_critico.wav` → transcrição (Azure Speech) → termos/sentimento (Azure Language) → alerta (3 min)
3. Vídeo: clipe com desvio postural → relatório automático (3 min)
4. Anomalias: série de sinais vitais com anomalia detectada (3 min)
5. Painel: alertas chegando, banner/toast, **Reconhecer** (2 min)
6. Integração dos serviços Azure no portal e resultados (2 min)
7. Limitações e próximos passos (1 min)

## Divisão sugerida (grupo)

Um módulo por pessoa (áudio ✅ · vídeo · anomalias) e alguém na integração/painel/relatório. Combine os contratos cedo: todo módulo devolve objetos `Alert` (`medmonitor/models.py`) e usa `AlertRepository`.
