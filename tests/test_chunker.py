import pytest

from core.chunker import chunk_text, estimate_tokens


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_scales_with_length():
    short = estimate_tokens("hello world")
    long = estimate_tokens("hello world " * 100)
    assert long > short


def test_chunk_text_short_document_returns_single_chunk():
    text = "This is a short clause about payment terms."
    chunks = chunk_text(text, max_chars=1000)
    assert chunks == [text]


def test_chunk_text_empty_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_splits_long_document_on_paragraphs():
    paragraphs = [f"Clause {i}. " + ("word " * 50) for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=400, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 400 + 50 + 20  # tolerance for overlap seams


def test_chunk_text_preserves_all_content_across_chunks():
    paragraphs = [f"UNIQUEMARKER{i}" for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = chunk_text(text, max_chars=60, overlap=10)
    joined = " ".join(chunks)
    for i in range(20):
        assert f"UNIQUEMARKER{i}" in joined


def test_chunk_text_rejects_invalid_max_chars():
    with pytest.raises(ValueError):
        chunk_text("some text", max_chars=0)


def test_chunk_text_rejects_overlap_ge_max_chars():
    with pytest.raises(ValueError):
        chunk_text("some text", max_chars=100, overlap=100)


def test_chunk_text_handles_single_oversized_paragraph():
    text = "word " * 5000
    chunks = chunk_text(text, max_chars=500, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 500 + 5
