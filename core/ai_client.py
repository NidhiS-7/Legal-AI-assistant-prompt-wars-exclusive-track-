from __future__ import annotations

import os
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from core import prompts
from core.chunker import chunk_text

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
MAX_OUTPUT_TOKENS = 2000
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
# Above this size we switch to chunked map-reduce processing.
SINGLE_CALL_CHAR_LIMIT = 12000


class AIClientError(Exception):
    """Safe, user-facing error raised for any AI-call failure."""


class LegalAIClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise AIClientError(
                "No Google AI Studio API key configured. Get a free key at "
                "https://aistudio.google.com/app/apikey and set "
                "GOOGLE_API_KEY in your environment or Streamlit secrets."
            )
        self._client = genai.Client(api_key=api_key)
        self._model_name = model
        self._config = genai_types.GenerateContentConfig(
            system_instruction=prompts.SYSTEM_PROMPT,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    # Low-level call with retry/backoff
    def _call(self, user_prompt: str) -> str:
        last_error: Exception | None = None
        last_code: int | None = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=user_prompt,
                    config=self._config,
                )
                text = getattr(response, "text", None)
                if text:
                    return text.strip()
                raise AIClientError(
                    "The AI service returned no content for this request. "
                    "This can happen if the document triggered a safety "
                    "filter; try a different document or section."
                )
            except genai_errors.ClientError as exc:
                # 4xx: includes rate limiting (429), auth errors (401/403),
                # bad model name / bad request (400/404). Only 429 is worth
                # retrying - the rest won't succeed no matter how many times
                # we try again.
                last_error = exc
                last_code = getattr(exc, "code", None)
                if last_code == 429:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                    continue
                break
            except genai_errors.ServerError as exc:
                last_error = exc
                last_code = getattr(exc, "code", None)
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
            except AIClientError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break

        raise AIClientError(self._friendly_message_for(last_code, last_error)) from last_error

    @staticmethod
    def _friendly_message_for(code: int | None, error: Exception | None) -> str:
        """Turn a Gemini API error code into an actionable message, instead
        of a single generic string that hides what actually went wrong."""
        if code == 429:
            return (
                "You've hit the Gemini free-tier rate limit (too many "
                "requests per minute, or the daily quota is used up). "
                "Wait a minute and try again, or try again tomorrow if "
                "you've hit the daily cap."
            )
        if code in (401, 403):
            return (
                "The Gemini API rejected this request as unauthorized. "
                "Your GOOGLE_API_KEY is likely missing, invalid, or was "
                "typed with extra characters/whitespace. Double-check the "
                "key in Streamlit Secrets against "
                "https://aistudio.google.com/app/apikey."
            )
        if code == 404:
            return (
                "The requested Gemini model was not found for this API "
                "key/region. Try a different model via the GEMINI_MODEL "
                "environment variable (e.g. 'gemini-1.5-flash')."
            )
        if code == 400:
            return (
                "The Gemini API rejected this request as malformed. This "
                "can happen with unusual file content; try a different "
                "document or a smaller excerpt."
            )
        return (
            f"The AI service could not complete this request right now"
            f"{f' (error code {code})' if code else ''}. Please try again "
            f"in a moment. If this keeps happening, check the app logs "
            f"for the underlying error: {error}"
        )
    # Long-document handling (map-reduce over chunks)
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
    # Public capabilities
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
