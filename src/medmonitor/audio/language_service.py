"""Análise de texto com Azure AI Language (sentimento e frases-chave)."""

from typing import Protocol

from medmonitor.audio.models import SentimentResult, TextAnalysis

MAX_CHARS = 4000  # limite do serviço por documento: 5.120 caracteres
MAX_DOCS_PER_REQUEST = 10


class LanguageServiceError(RuntimeError):
    pass


class TextAnalyzer(Protocol):
    def analyze(self, text: str) -> TextAnalysis: ...


def chunk_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Divide o texto em pedaços de até `max_chars`, quebrando em espaços."""
    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for word in text.split():
        extra = len(word) + (1 if current else 0)
        if current and size + extra > max_chars:
            chunks.append(" ".join(current))
            current, size, extra = [], 0, len(word)
        # palavra gigante isolada: corta para respeitar o limite
        while len(word) > max_chars:
            chunks.append(word[:max_chars])
            word = word[max_chars:]
            extra = len(word)
        current.append(word)
        size += extra
    if current:
        chunks.append(" ".join(current))
    return chunks


def _batches(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


class AzureLanguageAnalyzer:
    def __init__(self, endpoint: str, key: str, language: str = "pt") -> None:
        from azure.ai.textanalytics import TextAnalyticsClient
        from azure.core.credentials import AzureKeyCredential

        self._client = TextAnalyticsClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )
        self._language = language

    def analyze(self, text: str) -> TextAnalysis:
        chunks = chunk_text(text)
        if not chunks:
            return TextAnalysis()

        weighted = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
        total_chars = 0
        phrases: list[str] = []

        for batch in _batches(chunks, MAX_DOCS_PER_REQUEST):
            try:
                sentiments = self._client.analyze_sentiment(
                    batch, language=self._language
                )
                keys = self._client.extract_key_phrases(batch, language=self._language)
            except Exception as exc:  # erros de rede/autenticação do SDK
                raise LanguageServiceError(f"Azure Language falhou: {exc}") from exc

            for chunk, doc in zip(batch, sentiments):
                if doc.is_error:
                    raise LanguageServiceError(doc.error.message)
                scores = doc.confidence_scores
                weighted["positive"] += scores.positive * len(chunk)
                weighted["neutral"] += scores.neutral * len(chunk)
                weighted["negative"] += scores.negative * len(chunk)
                total_chars += len(chunk)
            for doc in keys:
                if doc.is_error:
                    raise LanguageServiceError(doc.error.message)
                phrases.extend(doc.key_phrases)

        averages = {k: v / total_chars for k, v in weighted.items()}
        label = max(averages, key=averages.get)  # type: ignore[arg-type]
        return TextAnalysis(
            sentiment=SentimentResult(label=label, **averages),
            key_phrases=_dedupe(phrases),
        )


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
