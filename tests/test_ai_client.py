"""Unit tests for LegalAIClient using a mocked Gemini SDK.

No real network calls are made here - the google.genai.Client is
replaced with a stand-in so we can test retry logic, error message
selection, and chunked map-reduce processing deterministically and
offline. Network-dependent, non-deterministic API behavior is
intentionally out of scope for unit tests; these tests instead verify
that OUR code reacts correctly to every response/error shape the SDK
can hand back.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from core.ai_client import AIClientError, LegalAIClient
from core.chunker import chunk_text
from core.ai_client import SINGLE_CALL_CHAR_LIMIT


def _fake_response(text: str | None):
    """Build an object shaped like the SDK's GenerateContentResponse."""
    return SimpleNamespace(text=text)


@pytest.fixture
def client():
    with patch("core.ai_client.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        instance = LegalAIClient(api_key="test-key-not-real")
        yield instance


def test_call_returns_stripped_text_on_success(client):
    client._client.models.generate_content.return_value = _fake_response(
        "  Here is the simplified summary.  "
    )
    result = client._call("some prompt")
    assert result == "Here is the simplified summary."


def test_call_raises_ai_client_error_on_empty_response(client):
    client._client.models.generate_content.return_value = _fake_response(None)
    with pytest.raises(AIClientError, match="no content"):
        client._call("some prompt")


def test_call_retries_on_429_then_succeeds(client):
    from google.genai import errors as genai_errors

    rate_limited = genai_errors.ClientError(
        code=429, response_json={"error": {"message": "rate limited"}}
    )
    client._client.models.generate_content.side_effect = [
        rate_limited,
        _fake_response("Success after retry"),
    ]
    with patch("core.ai_client.time.sleep"):  # skip real backoff delay
        result = client._call("some prompt")
    assert result == "Success after retry"
    assert client._client.models.generate_content.call_count == 2


def test_call_does_not_retry_on_403_and_raises_auth_message(client):
    from google.genai import errors as genai_errors

    forbidden = genai_errors.ClientError(
        code=403, response_json={"error": {"message": "forbidden"}}
    )
    client._client.models.generate_content.side_effect = forbidden

    with pytest.raises(AIClientError, match="unauthorized"):
        client._call("some prompt")
    # Should fail fast on a 403, not retry MAX_RETRIES times.
    assert client._client.models.generate_content.call_count == 1


def test_call_raises_model_not_found_message_on_404(client):
    from google.genai import errors as genai_errors

    not_found = genai_errors.ClientError(
        code=404, response_json={"error": {"message": "model not found"}}
    )
    client._client.models.generate_content.side_effect = not_found

    with pytest.raises(AIClientError, match="model was not found"):
        client._call("some prompt")


def test_call_gives_up_after_max_retries_on_persistent_429(client):
    from google.genai import errors as genai_errors

    rate_limited = genai_errors.ClientError(
        code=429, response_json={"error": {"message": "rate limited"}}
    )
    client._client.models.generate_content.side_effect = rate_limited

    with patch("core.ai_client.time.sleep"):
        with pytest.raises(AIClientError, match="rate limit"):
            client._call("some prompt")
    assert client._client.models.generate_content.call_count == 3  # MAX_RETRIES


def test_simplify_document_short_text_makes_single_call(client):
    client._client.models.generate_content.return_value = _fake_response("Summary.")
    result = client.simplify_document("A short document.")
    assert result == "Summary."
    assert client._client.models.generate_content.call_count == 1


def test_simplify_document_long_text_chunks_and_merges(client):
    # Build text long enough to force multiple chunks.
    long_text = ("Clause. " * 50 + "\n\n") * 400
    num_chunks = len(chunk_text(long_text, max_chars=SINGLE_CALL_CHAR_LIMIT))
    assert num_chunks > 1

    # One call per chunk, plus one final merge call.
    per_chunk_responses = [
        _fake_response(f"Part {i} summary") for i in range(1, num_chunks + 1)
    ]
    client._client.models.generate_content.side_effect = per_chunk_responses + [
        _fake_response("Merged final summary")
    ]
    result = client.simplify_document(long_text)
    assert result == "Merged final summary"
    assert client._client.models.generate_content.call_count == num_chunks + 1


def test_compare_documents_truncates_each_side_independently(client):
    client._client.models.generate_content.return_value = _fake_response(
        "Comparison result"
    )
    huge_a = "A" * 50000
    huge_b = "B" * 50000
    result = client.compare_documents(huge_a, huge_b, "Doc A", "Doc B")
    assert result == "Comparison result"

    call_args = client._client.models.generate_content.call_args
    sent_prompt = call_args.kwargs["contents"]
    # Each side should be truncated, not sent in full.
    assert len(sent_prompt) < len(huge_a) + len(huge_b)


def test_missing_api_key_raises_helpful_error():
    with patch("core.ai_client.os.environ.get", return_value=None):
        with pytest.raises(AIClientError, match="No Google AI Studio API key"):
            LegalAIClient(api_key=None)
