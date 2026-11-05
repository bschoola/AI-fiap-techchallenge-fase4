from vitaltrace.audio.critical_terms import find_critical_terms
from vitaltrace.models import Severity


def terms(text):
    return {m.term.lower(): m for m in find_critical_terms(text)}


def test_detects_critical_term_ignoring_case_and_accents():
    found = terms("Estou com FALTA DE AR e Dor no Peito")
    assert found["falta de ar"].severity == Severity.CRITICAL
    assert found["dor no peito"].category == "cardiovascular"


def test_negated_term_is_flagged():
    found = terms("Eu não tenho falta de ar.")
    assert found["falta de ar"].negated is True


def test_negation_does_not_leak_across_clauses():
    found = terms("Não dormi bem, cansaço forte.")
    assert found["cansaço"].negated is False


def test_specific_term_wins_over_generic_overlap():
    found = find_critical_terms("Sinto dor no peito")
    assert [m.term.lower() for m in found] == ["dor no peito"]


def test_generic_pain_is_low_severity():
    found = terms("Tenho dor nas costas")
    assert found["dor"].severity == Severity.LOW


def test_results_are_ordered_by_position():
    found = find_critical_terms("Febre alta e depois tosse")
    assert [m.term.lower() for m in found] == ["febre alta", "tosse"]


def test_no_terms_in_healthy_text():
    assert find_critical_terms("Hoje acordei bem disposto e tomei café.") == []
