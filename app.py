"""Legal Document Assistant - Streamlit application.

Vertical: Individual / Consumer legal help (renters, freelancers,
employees, small-business owners) who receive contracts and legal
documents they don't have easy professional access to.

This file is intentionally UI-only. All business logic lives in core/
and is unit-tested independently in tests/.
"""

from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from core.ai_client import AIClientError, LegalAIClient
from core.document_parser import (
    DocumentParseError,
    DocumentTooLargeError,
    UnsupportedFileTypeError,
    parse_document,
)
from core.prompts import DISCLAIMER

load_dotenv()

MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "8"))
# Session-level cap on AI calls: protects the shared free-tier quota from
# a single session exhausting it (accidental double-clicks, runaway
# loops), and caps the cost/latency blast radius of any one browser tab.
MAX_AI_CALLS_PER_SESSION = int(os.environ.get("MAX_AI_CALLS_PER_SESSION", "30"))

st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="\u2696\ufe0f",
    layout="wide",
)


# ----------------------------------------------------------------------
# API key resolution: Streamlit secrets first (for cloud deploys), then
# environment variable (for local/.env use). Never hardcoded.
# ----------------------------------------------------------------------
def get_api_key() -> str | None:
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY")


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> LegalAIClient:
    return LegalAIClient(api_key=api_key)


def render_disclaimer_banner() -> None:
    st.markdown(
        f"""<div role="note" aria-label="Legal disclaimer"
        style="background-color:#FEF3C7;border-left:4px solid #D97706;
        padding:0.75rem 1rem;border-radius:4px;margin-bottom:1rem;
        color:#78350F;font-size:0.95rem;">
        <span aria-hidden="true">\u26a0\ufe0f</span> {DISCLAIMER}</div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def _parse_document_cached(filename: str, file_bytes: bytes, max_mb: int):
    """Cache-wrapped document parsing.

    Streamlit reruns this whole script on nearly every widget
    interaction. Without caching, re-selecting a mode or touching an
    unrelated widget would silently re-run PDF/DOCX extraction on the
    same bytes every time. Keying the cache on the file's own content
    (via file_bytes, which st.cache_data hashes) makes repeat parses of
    the same upload free.
    """
    return parse_document(filename, file_bytes, max_mb=max_mb)


def read_upload(uploaded_file) -> str | None:
    """Parse an uploaded file (cached) and surface friendly errors in the UI."""
    if uploaded_file is None:
        return None
    try:
        parsed = _parse_document_cached(
            uploaded_file.name, uploaded_file.getvalue(), MAX_UPLOAD_MB
        )
        st.success(
            f"Loaded **{parsed.filename}** "
            f"({parsed.char_count:,} characters"
            + (f", {parsed.page_count} pages" if parsed.page_count else "")
            + ")."
        )
        return parsed.text
    except (DocumentTooLargeError, UnsupportedFileTypeError, DocumentParseError) as exc:
        st.error(str(exc))
        return None


def _ai_cache_key(*parts: str) -> str:
    """Build a stable cache key from document text + task parameters.

    Avoids re-spending a Gemini call (time, quota, and - on paid tiers -
    money) if the user re-runs the exact same task on the exact same
    document, e.g. after switching modes and switching back.
    """
    joined = "\x1f".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def call_ai(cache_key: str, fn) -> str:
    """Run an AI call through a session-scoped cache and a session-scoped
    rate limit, so repeat requests are free and a single session can't
    silently exhaust the shared free-tier quota."""
    if "ai_cache" not in st.session_state:
        st.session_state.ai_cache = {}
    if "ai_call_count" not in st.session_state:
        st.session_state.ai_call_count = 0

    if cache_key in st.session_state.ai_cache:
        return st.session_state.ai_cache[cache_key]

    if st.session_state.ai_call_count >= MAX_AI_CALLS_PER_SESSION:
        raise AIClientError(
            f"You've reached this session's limit of "
            f"{MAX_AI_CALLS_PER_SESSION} AI requests, which protects the "
            f"shared free-tier quota. Please refresh the page to start a "
            f"new session."
        )

    result = fn()
    st.session_state.ai_call_count += 1
    st.session_state.ai_cache[cache_key] = result
    return result


def main() -> None:
    st.markdown(
        '<h1><span aria-hidden="true">\u2696\ufe0f</span> Legal Document Assistant</h1>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Understand, compare, and navigate legal documents in plain "
        "language \u2014 built for renters, freelancers, employees, and "
        "small business owners."
    )
    render_disclaimer_banner()

    api_key = get_api_key()
    if not api_key:
        st.warning(
            "No API key found. Get a free key at "
            "[aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) "
            "and add `GOOGLE_API_KEY` to your environment (`.env` file) or "
            "to Streamlit secrets before using the assistant. See `.env.example`."
        )
        st.stop()

    try:
        client = get_client(api_key)
    except AIClientError as exc:
        st.error(str(exc))
        st.stop()

    mode = st.sidebar.radio(
        "Choose what you'd like to do",
        [
            "Simplify a document",
            "Highlight risks & obligations",
            "Compare two documents",
            "Ask questions about a document",
            "Generate an action checklist",
            "Prepare for a lawyer meeting",
        ],
        help="Pick the task that matches what you need right now.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Supported files:** PDF, DOCX, TXT\n\n"
        f"**Max size:** {MAX_UPLOAD_MB} MB\n\n"
        "Files are processed in-memory for this session only and are "
        "not stored on any server."
    )

    if mode == "Simplify a document":
        run_simplify(client)
    elif mode == "Highlight risks & obligations":
        run_risk_highlight(client)
    elif mode == "Compare two documents":
        run_compare(client)
    elif mode == "Ask questions about a document":
        run_qa(client)
    elif mode == "Generate an action checklist":
        run_checklist(client)
    elif mode == "Prepare for a lawyer meeting":
        run_lawyer_prep(client)


def run_simplify(client: LegalAIClient) -> None:
    st.header("Simplify a document")
    uploaded = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
    text = read_upload(uploaded)
    if text and st.button("Simplify", type="primary"):
        with st.spinner("Reading and simplifying your document..."):
            try:
                key = _ai_cache_key("simplify", text)
                result = call_ai(key, lambda: client.simplify_document(text))
                st.markdown(result)
                st.download_button(
                    "Download plain-language summary",
                    result,
                    file_name="simplified_summary.md",
                )
            except AIClientError as exc:
                st.error(str(exc))


def run_risk_highlight(client: LegalAIClient) -> None:
    st.header("Highlight risks & obligations")
    uploaded = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
    text = read_upload(uploaded)
    if text and st.button("Analyze clauses", type="primary"):
        with st.spinner("Scanning for important clauses..."):
            try:
                key = _ai_cache_key("risk", text)
                result = call_ai(key, lambda: client.extract_key_clauses(text))
                st.markdown(result)
                st.download_button(
                    "Download clause analysis", result, file_name="clause_analysis.md"
                )
            except AIClientError as exc:
                st.error(str(exc))


def run_compare(client: LegalAIClient) -> None:
    st.header("Compare two documents")
    col1, col2 = st.columns(2)
    with col1:
        file_a = st.file_uploader("Document A", type=["pdf", "docx", "txt"], key="doc_a")
        text_a = read_upload(file_a)
    with col2:
        file_b = st.file_uploader("Document B", type=["pdf", "docx", "txt"], key="doc_b")
        text_b = read_upload(file_b)

    if text_a and text_b and st.button("Compare", type="primary"):
        with st.spinner("Comparing documents..."):
            try:
                key = _ai_cache_key("compare", text_a, text_b, file_a.name, file_b.name)
                result = call_ai(
                    key,
                    lambda: client.compare_documents(
                        text_a, text_b, file_a.name, file_b.name
                    ),
                )
                st.markdown(result)
                st.download_button(
                    "Download comparison", result, file_name="comparison.md"
                )
            except AIClientError as exc:
                st.error(str(exc))


def run_qa(client: LegalAIClient) -> None:
    st.header("Ask questions about a document")
    uploaded = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
    text = read_upload(uploaded)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if text:
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        question = st.chat_input("Ask a question about this document...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)

            history_text = "\n".join(
                f"{role}: {msg}" for role, msg in st.session_state.chat_history[:-1]
            )
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Q&A is intentionally NOT cached - the same
                        # question can legitimately get a different
                        # answer depending on prior chat history, so
                        # caching by question text alone would be wrong.
                        answer = client.answer_question(text, question, history_text)
                        st.markdown(answer)
                        st.session_state.chat_history.append(("assistant", answer))
                    except AIClientError as exc:
                        st.error(str(exc))


def run_checklist(client: LegalAIClient) -> None:
    st.header("Generate an action checklist")
    uploaded = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
    text = read_upload(uploaded)
    if text and st.button("Generate checklist", type="primary"):
        with st.spinner("Building your checklist..."):
            try:
                key = _ai_cache_key("checklist", text)
                result = call_ai(key, lambda: client.generate_checklist(text))
                st.markdown(result)
                st.download_button(
                    "Download checklist", result, file_name="checklist.md"
                )
            except AIClientError as exc:
                st.error(str(exc))


def run_lawyer_prep(client: LegalAIClient) -> None:
    st.header("Prepare for a lawyer meeting")
    uploaded = st.file_uploader("Upload a legal document", type=["pdf", "docx", "txt"])
    text = read_upload(uploaded)
    context = st.text_area(
        "Optional: add context about your situation "
        "(e.g. 'I'm being asked to sign this as a new hire and I'm unsure "
        "about the non-compete clause').",
        height=100,
    )
    if text and st.button("Prepare questions", type="primary"):
        with st.spinner("Preparing your consultation notes..."):
            try:
                key = _ai_cache_key("lawyer_prep", text, context)
                result = call_ai(
                    key, lambda: client.prepare_lawyer_questions(text, context)
                )
                st.markdown(result)
                st.download_button(
                    "Download lawyer-prep notes", result, file_name="lawyer_prep.md"
                )
            except AIClientError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
