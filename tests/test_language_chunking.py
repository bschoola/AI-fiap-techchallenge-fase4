from vitaltrace.audio.language_service import chunk_text


def test_short_text_is_single_chunk():
    assert chunk_text("olá mundo") == ["olá mundo"]


def test_empty_text_has_no_chunks():
    assert chunk_text("   ") == []


def test_long_text_is_split_within_limit_without_losing_words():
    text = " ".join(["palavra"] * 1000)
    chunks = chunk_text(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_giant_word_is_cut():
    chunks = chunk_text("x" * 250, max_chars=100)
    assert [len(c) for c in chunks] == [100, 100, 50]
