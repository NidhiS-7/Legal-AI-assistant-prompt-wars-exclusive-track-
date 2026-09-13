"""Utilities for splitting long documents into model-friendly chunks.

Legal documents (leases, MSAs, ToS) can easily exceed a single prompt's
comfortable context budget. Rather than truncating silently and giving
wrong answers, we chunk on paragraph boundaries with a small overlap so
clauses that straddle a chunk boundary aren't lost, and we give callers
an honest token estimate so the UI can warn the user for very large
files instead of failing opaquely mid-request.
"""

from __future__ import annotations

# Rough heuristic: ~4 characters per token for English legal prose.
CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate used for UI warnings only."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def chunk_text(
    text: str,
    max_chars: int = 6000,
    overlap: int = 300,
) -> list[str]:
    """Split text into overlapping chunks on paragraph boundaries.

    - Never splits mid-sentence when a paragraph fits within max_chars.
    - Falls back to hard character slicing only for a single paragraph
      that itself exceeds max_chars (e.g. densely formatted tables).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be >= 0 and smaller than max_chars")

    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}" if current else paragraph

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current.strip())
            # seed next chunk with a small overlap for continuity
            current = current[-overlap:] + "\n\n" + paragraph
        else:
            current = paragraph

        # Handle a single oversized paragraph via hard slicing.
        while len(current) > max_chars:
            chunks.append(current[:max_chars].strip())
            current = current[max_chars - overlap:]

    if current.strip():
        chunks.append(current.strip())

    return chunks
