"""Safe extraction of plain text from user-uploaded documents.

Security notes:
- We never execute, eval, or shell out on uploaded content.
- We enforce a hard size limit before doing any parsing work (defends
  against memory-exhaustion / zip-bomb-style PDF attacks).
- We only accept a small allow-list of file extensions.
- All parsing failures are caught and converted into a single, safe
  ValueError so internal library errors/tracebacks are never leaked
  to the UI.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

from pypdf import PdfReader
from docx import Document

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}
DEFAULT_MAX_MB = 8


class DocumentTooLargeError(ValueError):
    """Raised when an uploaded file exceeds the configured size limit."""


class UnsupportedFileTypeError(ValueError):
    """Raised when a file extension is not in the allow-list."""


class DocumentParseError(ValueError):
    """Raised when a file cannot be parsed into text."""


@dataclass
class ParsedDocument:
    filename: str
    text: str
    char_count: int
    page_count: int | None = None


def _get_extension(filename: str) -> str:
    if "." not in filename:
        raise UnsupportedFileTypeError("File has no extension.")
    return filename.rsplit(".", 1)[-1].lower()


def _extract_pdf_text(file_bytes: bytes) -> tuple[str, int]:
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - convert to safe error
            raise DocumentParseError(
                "This PDF is password-protected. Please upload an "
                "unlocked copy."
            ) from exc

    pages_text = []
    for page in reader.pages:
        pages_text.append(page.extract_text() or "")
    return "\n\n".join(pages_text).strip(), len(reader.pages)


def _extract_docx_text(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs]
    # Also pull text out of tables, which plain paragraph extraction misses
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                if cell.text.strip():
                    paragraphs.append(cell.text)
    return "\n".join(paragraphs).strip()


def _extract_txt_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(encoding).strip()
        except UnicodeDecodeError:
            continue
    raise DocumentParseError("Could not decode this text file's encoding.")


def parse_document(
    filename: str,
    file_bytes: bytes,
    max_mb: int = DEFAULT_MAX_MB,
) -> ParsedDocument:
    """Parse an uploaded document into plain text.

    Raises DocumentTooLargeError, UnsupportedFileTypeError or
    DocumentParseError on invalid input; never raises a raw library
    exception.
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        raise DocumentTooLargeError(
            f"File is {size_mb:.1f} MB, which exceeds the {max_mb} MB limit."
        )

    extension = _get_extension(filename)
    if extension not in ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'.{extension}' files are not supported. "
            f"Please upload one of: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        )

    try:
        if extension == "pdf":
            text, page_count = _extract_pdf_text(file_bytes)
        elif extension == "docx":
            text, page_count = _extract_docx_text(file_bytes), None
        else:  # txt
            text, page_count = _extract_txt_text(file_bytes), None
    except (DocumentParseError,):
        raise
    except Exception as exc:  # noqa: BLE001 - never leak internal errors
        raise DocumentParseError(
            "This file could not be read. It may be corrupted, empty, "
            "or an image-only (scanned) document with no extractable text."
        ) from exc

    if not text or not text.strip():
        raise DocumentParseError(
            "No readable text was found in this document. If it is a "
            "scanned image, please upload a text-based version instead."
        )

    return ParsedDocument(
        filename=filename,
        text=text,
        char_count=len(text),
        page_count=page_count,
    )
