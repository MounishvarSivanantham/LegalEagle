# ============================================================
# services/pdf_parser.py  –  PDF Text Extraction Module
# ============================================================
# This module handles everything related to reading PDF files:
#   • Extracting text from every page
#   • Preserving page numbers
#   • Graceful handling of edge cases (empty, scanned,
#     corrupted, and password-protected PDFs)
#
# Library used: PyMuPDF (imported as "fitz")
#   - PyMuPDF is a fast, lightweight PDF library.
#   - The import name "fitz" comes from the underlying
#     MuPDF rendering engine's original project name.
# ============================================================

import pymupdf  # PyMuPDF — the pip package is called "PyMuPDF"


class PDFParseResult:
    """
    A simple container that holds everything we learn from a PDF.

    Attributes
    ----------
    filename : str
        The original name of the uploaded file.
    page_count : int
        Total number of pages in the PDF.
    pages : list[dict]
        One entry per page.  Each dict has:
          - "page_number" (int): 1-based page number
          - "text" (str): the raw text extracted from that page
    total_chars : int
        Sum of characters across all pages (handy for quick stats).
    warnings : list[str]
        Human-readable messages about anything unusual we detected
        (e.g. pages with no text, possible scanned content).
    error : str | None
        If extraction failed entirely, this holds the reason.
    """

    def __init__(self):
        self.filename: str = ""
        self.page_count: int = 0
        self.pages: list[dict] = []
        self.total_chars: int = 0
        self.warnings: list[str] = []
        self.error: str | None = None

    # Convenience: lets us do `if result:` to check for success
    def __bool__(self):
        return self.error is None and self.page_count > 0


DEFAULT_MAX_PDF_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB max limit to prevent memory exhaustion


def extract_text_from_pdf(
    file_bytes: bytes,
    filename: str = "document.pdf",
    max_size_bytes: int = DEFAULT_MAX_PDF_SIZE_BYTES,
) -> PDFParseResult:
    """
    Extract text from a PDF given its raw bytes.

    Parameters
    ----------
    file_bytes : bytes
        The raw content of the PDF file (e.g. from Streamlit's
        uploaded_file.read()).
    filename : str
        The original filename — used for display purposes only.
    max_size_bytes : int
        Maximum permitted file size in bytes (default: 25 MB).

    Returns
    -------
    PDFParseResult
        A result object containing pages, stats, and any warnings.

    How it works
    ------------
    1. Validate file size to prevent memory exhaustion attacks.
    2. Open the PDF from memory (no temp file needed).
    3. Loop through every page and call page.get_text().
    4. Track which pages returned no text and flag them.
    5. Catch common errors: wrong password, corrupt data, etc.
    """

    result = PDFParseResult()
    result.filename = filename

    # ── Security Check: File Size Limit ─────────────────────
    if file_bytes is None:
        result.error = "No file data provided."
        return result

    if len(file_bytes) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        actual_mb = len(file_bytes) / (1024 * 1024)
        result.error = (
            f"File size ({actual_mb:.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB."
        )
        return result

    # ── Step 1: Try to open the PDF ─────────────────────────
    try:
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as e:
        # The file bytes aren't a valid PDF or something else went wrong
        result.error = (
            "The uploaded file appears to be corrupted or is not a valid PDF. "
            f"Details: {e}"
        )
        return result

    # ── Step 2: Check for password protection ───────────────
    # PyMuPDF can open some encrypted PDFs if no user password
    # is set, but we should check and warn.
    if doc.is_encrypted:
        # Try opening with an empty password (works for some
        # PDFs that only have an owner password, not a user one)
        if not doc.authenticate(""):
            result.error = (
                "This PDF is password-protected. "
                "Please remove the password and re-upload."
            )
            doc.close()
            return result
        else:
            result.warnings.append(
                "This PDF has owner-level encryption but no user password. "
                "Text was extracted successfully."
            )

    # ── Step 3: Extract text page by page ───────────────────
    result.page_count = len(doc)

    if result.page_count == 0:
        result.error = "The PDF has zero pages — it appears to be empty."
        doc.close()
        return result

    empty_page_numbers = []  # track pages that yield no text

    for page_index in range(result.page_count):
        page = doc[page_index]
        page_number = page_index + 1  # humans count from 1

        # get_text() extracts selectable text from the page.
        # It does NOT perform OCR on scanned/image-only pages.
        text = page.get_text().strip()

        result.pages.append({
            "page_number": page_number,
            "text": text,
        })

        result.total_chars += len(text)

        if not text:
            empty_page_numbers.append(page_number)

    doc.close()

    # ── Step 4: Generate warnings for edge cases ────────────
    if empty_page_numbers:
        if len(empty_page_numbers) == result.page_count:
            # EVERY page is blank → almost certainly a scanned PDF
            result.warnings.append(
                "No text could be extracted from any page. "
                "This PDF likely contains scanned images rather than "
                "selectable text. OCR is not implemented yet, so "
                "scanned documents cannot be processed at this time."
            )
        else:
            # Only some pages are blank
            pages_str = ", ".join(str(p) for p in empty_page_numbers)
            result.warnings.append(
                f"Pages with no extractable text: {pages_str}. "
                "These pages may contain images or scanned content."
            )

    return result


def get_full_text(result: PDFParseResult) -> str:
    """
    Combine all page texts into a single string.

    Each page is separated by a clear marker like:
        --- Page 3 ---

    This is useful for previewing the full document.
    """
    sections = []
    for page in result.pages:
        header = f"--- Page {page['page_number']} ---"
        sections.append(f"{header}\n{page['text']}")
    return "\n\n".join(sections)


def get_text_preview(result: PDFParseResult, max_chars: int = 500) -> str:
    """
    Return a short preview of the extracted text.

    Takes up to `max_chars` characters from the beginning of
    the full document text.  Adds "…" if it was truncated.
    """
    full = get_full_text(result)
    if len(full) <= max_chars:
        return full
    return full[:max_chars].rstrip() + " …"
