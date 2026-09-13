"""Thin, testable wrapper around the Anthropic Claude API.

Design goals:
- All GenAI calls live in exactly one place, so rate limiting, retries,
  logging, and error handling are consistent everywhere they're used.
- Never leak raw exceptions (which can include request internals) to
  the UI layer; convert everything to AIClientError with a safe message.
- Support long documents transparently by chunking + map-reduce style
  summarization, instead of failing once a doc exceeds context limits.
"""

from __future__ import annotations

import os
import time

import anthropic

from core import prompts
from core.chunker import chunk_text

DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
MAX_OUTPUT_TOKENS = 2000
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
# Above this size we switch to chunked map-reduce processing.
SINGLE_CALL_CHAR_LIMIT = 12000


class AIClientError(Exception):
    """Safe, user-facing error raised for any AI-call failure."""


class LegalAIClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AIClientError(
                "No Anthropic API key configured. Set ANTHROPIC_API_KEY "
                "in your environment or Streamlit secrets."
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    # ------------------------------------------------------------------
    # Low-level call with retry/backoff
    # ------------------------------------------------------------------
    def _call(self, user_prompt: str, system: str = prompts.SYSTEM_PROMPT) -> str:
        last_error: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=system,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text_parts = [
                    block.text for block in response.content
                    if getattr(block, "type", None) == "text"
                ]
                return "\n".join(text_parts).strip()
            except anthropic.RateLimitError as exc:
                last_error = exc
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            except anthropic.APIStatusError as exc:
                last_error = exc
                if exc.status_code and exc.status_code < 500:
                    break  # client error - retrying won't help
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break

        raise AIClientError(
            "The AI service could not complete this request right now. "
            "Please try again in a moment."
        ) from last_error

    # ------------------------------------------------------------------
    # Long-document handling (map-reduce over chunks)
    # ------------------------------------------------------------------
    def _process_long_document(self, document_text: str, prompt_builder) -> str:
        chunks = chunk_text(document_text, max_chars=SINGLE_CALL_CHAR_LIMIT)
        if len(chunks) == 1:
            return self._call(prompt_builder(chunks[0]))

        partial_results = []
        for i, chunk in enumerate(chunks, start=1):
            partial = self._call(prompt_builder(chunk))
            partial_results.append(f"[Section {i} of {len(chunks)}]\n{partial}")

        combined = "\n\n".join(partial_results)
        reduce_prompt = (
            "The following are analyses of consecutive sections of one "
            "long legal document. Merge them into a single, coherent, "
            "de-duplicated result using the same structure/headings as "
            "the section analyses below:\n\n" + combined
        )
        return self._call(reduce_prompt)

    # ------------------------------------------------------------------
    # Public capabilities
    # ------------------------------------------------------------------
    def simplify_document(self, document_text: str) -> str:
        return self._process_long_document(document_text, prompts.build_simplify_prompt)

    def extract_key_clauses(self, document_text: str) -> str:
        return self._process_long_document(document_text, prompts.build_risk_clause_prompt)

    def compare_documents(self, document_a: str, document_b: str,
                           label_a: str = "Document A",
                           label_b: str = "Document B") -> str:
        # Comparison needs both docs together, so we truncate defensively
        # rather than chunk independently (chunking would break alignment).
        max_each = SINGLE_CALL_CHAR_LIMIT // 2
        prompt = prompts.build_compare_prompt(
            document_a[:max_each], document_b[:max_each], label_a, label_b
        )
        return self._call(prompt)

    def answer_question(self, document_text: str, question: str,
                         chat_history: str = "") -> str:
        # For Q&A we keep the most relevant window rather than the whole
        # doc, to stay fast and cheap; truncate rather than silently fail.
        context = document_text[:SINGLE_CALL_CHAR_LIMIT]
        prompt = prompts.build_qa_prompt(context, question, chat_history)
        return self._call(prompt)

    def generate_checklist(self, document_text: str) -> str:
        return self._process_long_document(document_text, prompts.build_checklist_prompt)

    def prepare_lawyer_questions(self, document_text: str, user_context: str = "") -> str:
        prompt = prompts.build_lawyer_prep_prompt(
            document_text[:SINGLE_CALL_CHAR_LIMIT], user_context
        )
        return self._call(prompt)
