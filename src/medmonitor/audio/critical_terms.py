"""Detecção de termos clínicos críticos em transcrições (pt-BR).

O dicionário abaixo é ilustrativo e deve ser validado por profissionais de saúde antes de
qualquer uso real. A busca ignora acentos/maiúsculas e descarta termos negados
("não tenho falta de ar").
"""

import re
import unicodedata
from dataclasses import dataclass

from medmonitor.audio.models import CriticalTermMatch
from medmonitor.models import Severity

NEGATIONS = frozenset(
    {"sem", "nao", "nunca", "nega", "negou", "nenhum", "nenhuma", "ausencia"}
)
NEGATION_WINDOW = 3  # palavras anteriores consideradas dentro da mesma oração
_CLAUSE_SPLIT = re.compile(r"[.,;:!?\n]")
_CONTEXT_CHARS = 40


@dataclass(frozen=True)
class TermRule:
    pattern: str  # regex sobre texto normalizado (minúsculo e sem acentos)
    category: str
    severity: Severity


C, H, M, L = Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW

DEFAULT_RULES: tuple[TermRule, ...] = (
    # Crítico
    TermRule(r"\bdor\s+(?:no|do)\s+peito\b", "cardiovascular", C),
    TermRule(r"\baperto\s+no\s+peito\b", "cardiovascular", C),
    TermRule(r"\bfalta\s+de\s+ar\b", "respiratório", C),
    TermRule(r"\bnao\s+consigo\s+respirar\b", "respiratório", C),
    TermRule(r"\bdificuldade\s+(?:para|de|em)\s+respirar\b", "respiratório", C),
    TermRule(r"\bperdi\s+a\s+consciencia\b", "neurológico", C),
    TermRule(r"\bdesmai\w*", "neurológico", C),
    TermRule(r"\bfala\s+enrolada\b", "neurológico", C),
    TermRule(r"\bboca\s+torta\b", "neurológico", C),
    # Alto
    TermRule(r"\bformigamento\s+(?:no|na)\s+(?:braco|perna|rosto)\b", "neurológico", H),
    TermRule(r"\bconfus(?:ao\s+mental|[oa])\b", "neurológico", H),
    TermRule(r"\bsangramento\b|\bsangrando\b|\bsangue\b", "sangramento", H),
    TermRule(r"\bpalpitac\w*|\bcoracao\s+disparado\b", "cardiovascular", H),
    TermRule(r"\bfebre\s+alta\b", "geral", H),
    # Médio
    TermRule(r"\btontur\w*|\btont[oa]\b", "neurológico", M),
    TermRule(r"\bfebre\b", "geral", M),
    TermRule(r"\bchiado\b", "respiratório", M),
    TermRule(r"\btosse\b|\btossindo\b", "respiratório", M),
    TermRule(r"\bfraqueza\b", "geral", M),
    TermRule(r"\bnausea\w*|\benjo\w*|\bvomit\w*", "gastrointestinal", M),
    TermRule(r"\bdor\s+de\s+cabeca\b", "dor", M),
    # Baixo
    TermRule(r"\bcansaco\b|\bcansad[oa]\b|\bfadiga\b|\bexaust\w*", "geral", L),
    TermRule(r"\bsonolenc\w*|\bsonolent\w*", "geral", L),
    TermRule(r"\bdor\b", "dor", L),
)


def normalize(text: str) -> str:
    """Minúsculas e sem acentos (preserva o comprimento para textos pré-compostos)."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _is_negated(norm: str, start: int) -> bool:
    clause = _CLAUSE_SPLIT.split(norm[:start])[-1]
    tokens = re.findall(r"[a-z0-9]+", clause)[-NEGATION_WINDOW:]
    return any(token in NEGATIONS for token in tokens)


def find_critical_terms(
    text: str, rules: tuple[TermRule, ...] = DEFAULT_RULES
) -> list[CriticalTermMatch]:
    norm = normalize(text)
    source = text if len(text) == len(norm) else norm

    taken: list[tuple[int, int]] = []
    found: list[CriticalTermMatch] = []
    # Regras mais graves primeiro: "dor no peito" ganha de "dor" genérica.
    for rule in sorted(rules, key=lambda r: r.severity.rank, reverse=True):
        for match in re.finditer(rule.pattern, norm):
            start, end = match.span()
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            taken.append((start, end))
            ctx_start = max(0, start - _CONTEXT_CHARS)
            found.append(
                CriticalTermMatch(
                    term=source[start:end],
                    category=rule.category,
                    severity=rule.severity,
                    negated=_is_negated(norm, start),
                    context=source[ctx_start : end + _CONTEXT_CHARS].strip(),
                    position=start,
                )
            )
    found.sort(key=lambda m: m.position)
    return found
