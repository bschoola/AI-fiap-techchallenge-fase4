# Configuração do Azure — passo a passo

Para o primeiro slice você precisa de **dois recursos** do Azure AI Services, ambos com camada gratuita (F0):

| Recurso | Serve para | Variáveis no `.env` |
|---|---|---|
| **Speech** | Speech to Text (transcrição) e Text to Speech (gerar áudios de exemplo) | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` |
| **Language** | Sentimento e frases-chave (o antigo *Text Analytics*) | `AZURE_LANGUAGE_KEY`, `AZURE_LANGUAGE_ENDPOINT` |

> Os limites e preços mudam; confirme a camada F0 atual na página de preços de cada serviço antes de usar volume.

## 1. Conta e assinatura

1. Acesse <https://portal.azure.com> e entre com sua conta Microsoft.
2. Se ainda não tiver assinatura: crie uma conta gratuita (<https://azure.microsoft.com/free>) ou, se a sua instituição for elegível, o **Azure for Students** (<https://azure.microsoft.com/free/students>), que dispensa cartão.
3. Confira em **Subscriptions** que há uma assinatura ativa.

## 2. Grupo de recursos (facilita apagar tudo no final)

1. Portal → **Resource groups** → **Create**.
2. Subscription: a sua. Resource group: `rg-techchallenge-fase4`. Region: **East US**.
3. **Review + create** → **Create**.

> Use a **mesma região** nos dois recursos. East US tem os dois serviços e a camada F0. Como os dados do projeto são públicos/sintéticos, a região não é um problema.

## 3. Recurso Speech

1. Portal → **Create a resource** → busque **Speech** → *Speech* (Microsoft) → **Create**.
2. Preencha:
   - **Subscription**: a sua · **Resource group**: `rg-techchallenge-fase4`
   - **Region**: `East US`
   - **Name**: `speech-tc4-<algo-unico>` (ex.: `speech-tc4-bruno`)
   - **Pricing tier**: **Free F0**
3. **Review + create** → **Create** → **Go to resource**.
4. No menu lateral: **Resource Management → Keys and Endpoint**.
5. Copie **KEY 1** e o campo **Location/Region** (ex.: `eastus`).

## 4. Recurso Language

1. Portal → **Create a resource** → busque **Language service** → **Create**.
2. Na tela de *Select additional features*, clique em **Continue to create your resource** (não precisa de recursos extras).
3. Preencha:
   - **Resource group**: `rg-techchallenge-fase4` · **Region**: `East US`
   - **Name**: `lang-tc4-<algo-unico>`
   - **Pricing tier**: **Free F0**
   - Marque o aviso de *Responsible AI*.
4. **Review + create** → **Create** → **Go to resource**.
5. **Resource Management → Keys and Endpoint**: copie **KEY 1** e o **Endpoint** (`https://<nome>.cognitiveservices.azure.com/`).

## 5. Preencher o `.env`

```powershell
copy .env.example .env
```

Edite o `.env`:

```
AZURE_SPEECH_KEY=<KEY 1 do Speech>
AZURE_SPEECH_REGION=eastus
AZURE_LANGUAGE_KEY=<KEY 1 do Language>
AZURE_LANGUAGE_ENDPOINT=https://lang-tc4-bruno.cognitiveservices.azure.com/
MOCK_AZURE=false
```

## 6. Validar

```powershell
python scripts/azure_smoke_test.py
```

Saída esperada: `OK` para **Azure AI Language** e **Azure AI Speech**. Depois gere os áudios de exemplo (sintéticos):

```powershell
python scripts/generate_sample_audio.py
```

Isso cria `paciente_critico.wav`, `paciente_estavel.wav` e `paciente_fadiga.wav` em `data/samples/`.

## Alternativa: tudo via Azure CLI

```bash
az login
az group create -n rg-techchallenge-fase4 -l eastus

az cognitiveservices account create -n speech-tc4-bruno -g rg-techchallenge-fase4 \
  --kind SpeechServices --sku F0 -l eastus --yes

az cognitiveservices account create -n lang-tc4-bruno -g rg-techchallenge-fase4 \
  --kind TextAnalytics --sku F0 -l eastus --custom-domain lang-tc4-bruno --yes

# chaves e endpoint
az cognitiveservices account keys list -n speech-tc4-bruno -g rg-techchallenge-fase4
az cognitiveservices account keys list -n lang-tc4-bruno   -g rg-techchallenge-fase4
az cognitiveservices account show      -n lang-tc4-bruno   -g rg-techchallenge-fase4 --query properties.endpoint -o tsv
```

## Segurança e custos

- **Nunca** faça commit do `.env` (já está no `.gitignore`). Se uma chave vazar: recurso → *Keys and Endpoint* → **Regenerate Key**.
- Crie um alerta de orçamento em **Cost Management + Billing → Budgets** (mesmo no F0, por segurança).
- Terminada a entrega, apague tudo de uma vez: `az group delete -n rg-techchallenge-fase4` (ou pelo portal).
- Compartilhe a chave com o grupo por um canal privado; ou cada integrante cria o seu recurso F0.

## Azure OpenAI (opcional): áudios de demonstração mais naturais

Usado **só** por `scripts/generate_consulta_audio.py`, para gerar as três consultas de exemplo (normal, falta de ar, dor no peito). A aplicação em si não depende dele. Cada áudio sai numa **tomada única**: a respiração, as hesitações e a emoção vêm do próprio modelo, sem colar trechos.

Modelo recomendado: **`gpt-audio`** (ou **`gpt-audio-mini`**, mais barato). Não é um TTS simples: é um modelo de áudio completo, que recebe a direção da cena ("voz fraca, puxando o ar entre as frases") e interpreta o texto. Ele é usado pela API de *chat completions* com saída de áudio.

1. No portal Azure, **Create a resource** → **Azure OpenAI** → *Create*. Mesmo grupo de recursos; região **East US 2** (ou outra que liste o modelo na tabela de disponibilidade); preço **Standard S0**.
2. Abra o recurso → **Go to Microsoft Foundry** → *Deployments* → **Deploy model** → **gpt-audio** (ou **gpt-audio-mini**) → tipo **Global Standard**. Em *Deployment name* use o mesmo valor do `.env`.
3. Copie **KEY 1** e o **Endpoint** em *Keys and Endpoint*.
4. No `.env`:
   ```
   AZURE_OPENAI_ENDPOINT=https://SEU-RECURSO.openai.azure.com/
   AZURE_OPENAI_KEY=<KEY 1>
   AZURE_OPENAI_AUDIO_DEPLOYMENT=gpt-audio
   AZURE_OPENAI_API_VERSION=2025-04-01-preview
   ```
5. Rode `python scripts/generate_consulta_audio.py`. Voz padrão `marin`; teste `--voice cedar`, `coral` ou `sage`. Para refazer só um áudio: `--only normal` (ou `falta_de_ar`, `dor_no_peito`).

O `gpt-audio` às vezes muda uma palavra do roteiro. O script compara a transcrição devolvida com o texto e repete a geração (até `--tries`, padrão 3) quando a fidelidade fica abaixo de 90%.

Outros motores: `--engine tts` usa o endpoint `/audio/speech` (`AZURE_OPENAI_TTS_DEPLOYMENT`: `gpt-4o-mini-tts`, `tts-hd` ou `tts`; só o primeiro aceita direção de interpretação). `--engine speech` usa o Azure AI Speech com a voz HD `pt-BR-Thalita:DragonHDLatestNeural`, sem Azure OpenAI.

Erros comuns: `404` = nome do deployment diferente do `.env`; `400` mencionando `modalities` ou `audio` = o deployment não é um modelo de áudio, ou a `AZURE_OPENAI_API_VERSION` é antiga demais; `401` = chave/endpoint de outro recurso; `429` = cota, o script espera e tenta de novo. Custo: alguns centavos pelos três áudios; apague o recurso ao terminar.

## Problemas comuns

| Sintoma | Causa provável |
|---|---|
| `401` / *Access denied* | Chave errada, ou região do `.env` diferente da do recurso Speech |
| `Connection failed` no Speech | Região incorreta (`AZURE_SPEECH_REGION` deve ser o código, ex. `eastus`) |
| `404` no Language | Endpoint sem `https://` ou de outro recurso |
| `429` / *quota exceeded* | Cota gratuita do mês esgotada: aguarde ou use `MOCK_AZURE=true` para desenvolver |
| Transcrição vazia | Áudio sem fala/ruído; use WAV 16 kHz mono (o ffmpeg normaliza automaticamente) |
| `ffmpeg não encontrado` | Instale (`winget install Gyan.FFmpeg`) ou envie apenas `.wav` PCM 16 kHz mono |
