# Relatório Técnico — Tech Challenge Fase 4

> Esqueleto a preencher durante o desenvolvimento. Estrutura exigida pelo enunciado: fluxo multimodal, modelos por tipo de dado, resultados e exemplos de anomalias.

## 1. Introdução e objetivo
<!-- Problema: monitoramento contínuo de pacientes com dados multimodais (áudio, vídeo, texto). -->

## 2. Fluxo multimodal
<!-- Diagrama (ver docs/architecture.md), descrição da entrada, processamento, fusão e alerta. -->

## 3. Modelos e serviços aplicados por tipo de dado

| Dado | Serviço / modelo | Por quê | Saída |
|---|---|---|---|
| Áudio (voz) | Azure AI Speech (Speech to Text, pt-BR) | Transcrição gerenciada | Texto + segmentos com tempo |
| Texto | Azure AI Language (sentimento, frases-chave) + dicionário de termos críticos | Sentimento e termos clínicos | Sentimento, termos, alertas |
| Fala (vocal) | Métricas de ritmo/pausa (palavras/min, % de pausa) | Indício de fadiga/disartria | Alertas heurísticos |
| Vídeo | YOLOv8n-pose / YOLOv8n (CPU) | _preencher_ | _preencher_ |
| Sinais vitais | _regras + z-score + Isolation Forest_ | _preencher_ | _preencher_ |
| Prescrições | _preencher_ | _preencher_ | _preencher_ |

## 4. Dados utilizados
<!-- PhysioNet, AudioSet, dados sintéticos (áudios gerados com Azure TTS; prescrições sintéticas). Deixar claro o que é sintético. -->

## 5. Resultados
<!-- Métricas, tabelas, gráficos. -->

## 6. Exemplos de anomalias detectadas
<!-- Capturas do painel, trechos de transcrição, gráficos de séries temporais com anomalia marcada. -->

## 7. Integração com serviços em nuvem (Azure)
<!-- Recursos criados, região, tier, como foram usados. -->

## 8. Alertas à equipe médica
<!-- Regras de severidade, fluxo de reconhecimento no painel. -->

## 9. Limitações, ética e LGPD
<!-- Heurísticas sem validação clínica; apoio à decisão; dados públicos/sintéticos. -->

## 10. Conclusão e próximos passos
