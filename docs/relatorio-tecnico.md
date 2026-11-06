# Relatório Técnico — VitalTrace (Tech Challenge Fase 4)

> Itens marcados com **[preencher]** dependem de execuções que só você pode fazer (Azure com suas chaves, gravação do vídeo da demonstração). O restante descreve o que está implementado e testado no repositório.

## 1. Introdução e objetivo

Equipes médicas acompanham muitos pacientes entre consultas e perdem sinais precoces de piora que aparecem em fontes diferentes: a voz na gravação da consulta, o movimento em vídeo, as séries de sinais vitais, a evolução das prescrições e o texto da anamnese. O **VitalTrace** é um portal médico que ingere essas fontes, detecta anomalias em cada uma, cruza as modalidades e avisa a equipe **dentro da aplicação**, com o motivo de cada alerta.

## 2. Fluxo multimodal

```
Áudio ──► Azure Speech ──► Azure Language ──► termos críticos + ritmo de fala ─┐
Vídeo ──► YOLOv8-pose (CPU) ──► postura, movimento, quedas ────────────────────┤
Sinais vitais ──► regras · z-score robusto · tendência · Isolation Forest ─────┼─► Fusão ─► risco 0–100 ─► alertas ─► "Estou ciente"
Prescrições ──► saltos e repetição de aumentos de dose ────────────────────────┤
Anamnese (texto) ──► termos críticos e linguagem de piora ─────────────────────┘
```

Diagrama completo e decisões em [architecture.md](architecture.md). Cada módulo devolve `Finding`s (origem, severidade, título, descrição, evidência). A fusão correlaciona achados entre modalidades, calcula o risco e transforma os de severidade alta/crítica em alertas, sem duplicar os que já estão em aberto.

## 3. Modelos e serviços por tipo de dado

| Dado | Serviço / modelo | Por quê | Saída |
|---|---|---|---|
| Voz | Azure AI Speech (pt-BR, reconhecimento contínuo) | Transcrição gerenciada, com tempo por trecho | Texto e segmentos |
| Texto da fala | Azure AI Language (sentimento, frases-chave) | Serviço gerenciado em pt | Sentimento e frases |
| Texto clínico | Dicionário pt-BR de termos críticos (sem acento, com negação e severidade) | Detecta "falta de ar", "dor no peito" e ignora "sem falta de ar" | Termos, categoria, severidade |
| Ritmo de fala | Palavras/min e % de pausa a partir dos tempos da transcrição | Indício de fadiga/disartria | Alertas médios/baixos |
| Vídeo | YOLOv8n-pose (ultralytics, CPU, 480 px) e, sem ele, subtração de fundo MOG2 (OpenCV) | Leve o bastante para CPU; pose permite ângulos e postura | Postura, mobilidade, queda, imobilidade, assimetria de joelhos |
| Sinais vitais | Regras clínicas, mediana/MAD contra a linha de base do paciente, regressão linear (3 dias) e Isolation Forest (100 árvores, treinado na linha de base) | Camadas explicáveis; o Isolation Forest cobre combinações incomuns | Achados com evidência |
| Prescrições | Regras sobre a linha do tempo de doses | Mudanças rápidas de dose sugerem resposta insuficiente | Achados por fármaco |
| Fusão | Pontos por severidade (teto de 50 por modalidade) + correlações | Exige convergência entre fontes | Risco 0–100, nível e resumo |

Parâmetros relevantes: vídeo amostrado a 2 quadros/s; queda = de pé para deitado em ≤ 2 s e permanência ≥ 3 s; imobilidade = movimento < 0,03 alturas/s por ≥ 10 s; assimetria = Δ mediano de joelhos > 15°; Isolation Forest só avalia as últimas 24 h e só quando algum sinal já está fora do habitual (pré-filtro).

## 4. Dados utilizados

- **Linha do tempo do paciente** (`web/timeline.py`): consultas, gravações de áudio, vídeos e análises de IA em ordem cronológica, com filtro por tipo. Cada registro abre mostrando **o que foi analisado** (transcrição com os termos críticos destacados; quadros-chave do vídeo com a pessoa detectada) e **o resultado** (sentimento, ritmo de fala, eventos, alertas gerados com a situação atual, "Estou ciente" e a variação do risco antes/depois). Os gráficos recebem marcadores de áudio (círculo) e vídeo (losango) que levam ao registro. Por privacidade, o áudio e o vídeo originais **não são guardados**: só a transcrição e até 3 quadros reduzidos por vídeo, no JSON da análise (sem mudança de schema).
- **Vídeo por link do YouTube** (`video/youtube.py`, opcional): baixa os primeiros 120 s em até 480p com o `yt-dlp` e passa pelo mesmo analisador do upload; o arquivo é apagado em seguida. Aceita só `youtube.com`/`youtu.be` e entrega ao downloader uma URL montada a partir do ID do vídeo. É um recurso de demonstração: os Termos de Serviço do YouTube restringem baixar vídeos, o `yt-dlp` precisa de atualização frequente (e de um runtime JavaScript, o Deno), e vídeos editados ou com câmera em movimento degradam a detecção, que pressupõe câmera parada e uma pessoa em cena. A demonstração principal usa arquivo local.
- **Todos os dados do portal são sintéticos**: 30 pacientes fictícios, 14 dias de sinais vitais (6 aferições/dia), 3–5 consultas por paciente, prescrições e histórico de vídeo. O gerador é determinístico (`db/seed.py`).
- **PhysioNet / AudioSet**: o enunciado os cita como sugestão. Nesta versão **não foram usados** (o ambiente de desenvolvimento não acessa esses repositórios). A estrutura aceita séries reais: basta gravar linhas em `vital_signs`. Para a banca, declarar isso é mais honesto do que apresentar os dados sintéticos como reais.
- **Áudio**: três amostras geradas com Azure TTS (`scripts/generate_sample_audio.py`): crítico, estável e fadiga.
- **Vídeo**: nos testes, vídeos sintéticos (retângulo que anda, cai ou fica parado). Para a demonstração use gravações suas, com autorização, ou clipes públicos.

## 5. Resultados

**Testes automatizados:** 88 testes (`pytest`), sem Azure nem YOLO.

**Detecção em séries de sinais vitais** (`python scripts/evaluate_anomalies.py 150`; 150 séries sintéticas por perfil, nível calculado só com sinais vitais):

| Perfil gerado | Estável | Atenção | Crítico |
|---|---|---|---|
| Estável | 100,0 % | 0,0 % | 0,0 % |
| Deriva leve (SpO₂ −3, FC +9 em 5 dias) | 44,0 % | 55,3 % | 0,7 % |
| Deriva grave (SpO₂ −7,5, FC +28 em 4 dias) | 0,0 % | 49,3 % | 50,7 % |

Leitura: nenhum falso alarme nas séries estáveis (limiares conservadores), mas só 56 % das derivas leves e 51 % das graves chegam ao nível esperado. O nível "crítico" exige cruzar um limiar clínico duro (ex.: SpO₂ < 90 %); derivas graves que ficam acima dele são classificadas como atenção. Os rótulos vêm do gerador, então isto checa os limiares, não prova validade clínica. Limiares mais sensíveis aumentariam a detecção ao custo de falsos alarmes.

**Vídeo** (vídeos sintéticos, motor OpenCV): queda de 10 s detectada como crítica; caminhada de 14 s sem eventos; 30 s parado gera imobilidade prolongada; agachar por 1,5 s não gera queda. O adaptador do YOLOv8-pose é testado com um modelo falso que imita a saída do ultralytics; **a execução com pesos reais não foi medida** **[preencher: tempo por vídeo e exemplos em uma gravação real]**.

**Áudio com Azure**: **[preencher após `python scripts/azure_smoke_test.py` e o upload dos três áudios de exemplo: transcrição, sentimento, termos e alertas obtidos]**.

## 6. Exemplos de anomalias detectadas

Paciente **Helena Martins Duarte** (67 anos, insuficiência cardíaca), análise do seed:

- Sinais vitais: SpO₂ em 89 % (limiar 90 %), queda progressiva de ≈ 6 pontos em 3 dias (R² = 0,94), FC +25 bpm, FR +6 irpm, combinação incomum de sinais (7 de 7 aferições do último dia).
- Prescrição: furosemida de 40 para 80 mg (2×), segundo aumento em 60 dias.
- Anamnese: "falta de ar", "tontura", relatos de piora em consultas seguidas.
- Áudio (exemplo): termos críticos respiratório e cardiovascular.
- Movimento: queda de 38 % no índice de mobilidade.
- **Correlação**: *piora clínica apesar do ajuste de tratamento* e *sintomas relatados coincidem com alterações medidas*.
- Resultado: risco **crítico, 100/100**, com 22 achados e 4 alertas críticos em aberto.

Contraste: **Carlos Eduardo Pereira** (cefaleia recorrente) tem 0 achados e risco estável. **Roberto Almeida Lima** fica em *atenção* (SpO₂ ≈ 90 %, dose de furosemida dobrada) sem chegar a crítico. **[preencher: capturas de tela do portal para o vídeo e o relatório]**

## 7. Integração com serviços em nuvem (Azure)

Dois recursos: **Azure AI Speech** (Speech to Text pt-BR, reconhecimento contínuo para áudios longos) e **Azure AI Language** (sentimento e frases-chave, com divisão do texto em blocos de até 4000 caracteres e lotes de até 10 documentos). Chaves e endpoints vêm do `.env`; passo a passo em [azure-setup.md](azure-setup.md). **[preencher: região, tier e custo observado]**

## 8. Alertas à equipe médica

- Severidades: crítico, alto, médio, baixo. Só alto e crítico viram alertas; os demais ficam no relatório de anomalias.
- Fora do portal não há e-mail nem webhook: o alerta fica no banco, o painel consulta a cada 3 s, uma faixa vermelha avisa sobre críticos e o botão **Estou ciente** registra quem confirmou e quando.
- Alertas duplicados (mesmo paciente, origem e título, ainda em aberto) não são recriados.

## 9. Limitações, ética e LGPD

- Heurísticas e limiares didáticos, sem validação clínica; o resumo é gerado por regras e avisa que não é diagnóstico.
- Detecção de pessoas assume câmera parada e uma pessoa em cena; o plano B (OpenCV) não vê articulações e uma pessoa totalmente imóvel pode ser absorvida pelo fundo.
- O dicionário de termos é pequeno e ilustrativo.
- Dados fictícios. Com dados reais seriam necessários consentimento, minimização, criptografia, trilha de auditoria e controle de acesso por papel (hoje há apenas login de médico com senha em scrypt).
- Sem testes automáticos contra o Postgres do Docker nem contra o Azure; os testes usam SQLite e serviços simulados.

## 10. Conclusão e próximos passos

O portal cobre as quatro modalidades pedidas, com alertas internos e explicações por achado. Próximos passos: importar séries reais do PhysioNet, ajustar limiares com dados rotulados, ampliar o dicionário clínico com especialistas, detectar pose com mais de uma pessoa e adicionar auditoria e perfis de acesso.
