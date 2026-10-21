# ============================================================
# services/chunking.py  –  Text Chunking Module
# ============================================================
# In Retrieval-Augmented Generation (RAG), language models cannot
# easily read an entire 50-page legal contract at once.
#
# Instead, we break documents into smaller, bite-sized pieces
# called "chunks". Each chunk carries:
#   • The chunk text
#   • Accurate source metadata (document name, page number, chunk ID)
#
# This allows the AI to later cite the exact page and clause
# when answering a user's question.
#
# Defaults:
#   - Chunk size:    ~500 words
#   - Chunk overlap: ~80 words
#
# Overlap ensures that key legal context crossing the boundary
# between two adjacent chunks is not cut in half and lost.
# ============================================================

import re
from typing import Any, Dict, List, Optional


# ── Configuration constants ─────────────────────────────────
DEFAULT_CHUNK_SIZE: int = 500       # Target word count per chunk
DEFAULT_CHUNK_OVERLAP: int = 80     # Word overlap between consecutive chunks


def create_chunk_id(filename: str, page_number: int, chunk_index: int) -> str:
    """
    Generate a clean, unique, human-readable chunk ID.

    Example:
        create_chunk_id("sample_contract.pdf", 2, 1) -> "sample_contract_p2_c1"
    """
    # Clean the filename for use in IDs (remove path and non-alphanumeric chars)
    base_name = filename.rsplit(".", 1)[0]
    safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", base_name)
    return f"{safe_name}_p{page_number}_c{chunk_index}"


def chunk_text(
    text: str,
    filename: str = "document.pdf",
    page_number: int = 1,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_chunk_index: int = 1,
) -> List[Dict[str, Any]]:
    """
    Split a single string of text into word-based chunks with overlap.

    Parameters
    ----------
    text : str
        The raw text from a page or section.
    filename : str
        Source filename for metadata.
    page_number : int
        Source 1-based page number for metadata.
    chunk_size : int
        Number of words per chunk (default: 500).
    chunk_overlap : int
        Number of overlapping words between consecutive chunks (default: 80).
    start_chunk_index : int
        Starting index for chunk numbering on this page (default: 1).

    Returns
    -------
    List[Dict[str, Any]]
        List of chunk dictionaries. Returns empty list if text has no words.
    """
    # Defensive checks: handle empty, whitespace-only, or invalid input
    if not text or not isinstance(text, str):
        return []

    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    # Ensure valid overlap (overlap cannot be >= chunk_size)
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    step = max(1, chunk_size - chunk_overlap)
    chunks: List[Dict[str, Any]] = []
    chunk_idx = start_chunk_index

    # If the text is shorter than or equal to chunk_size, it forms a single chunk
    if total_words <= chunk_size:
        clean_content = " ".join(words).strip()
        if clean_content:
            chunks.append({
                "chunk_id": create_chunk_id(filename, page_number, chunk_idx),
                "text": clean_content,
                "page_number": page_number,
                "filename": filename,
                "word_count": total_words,
                "char_count": len(clean_content),
            })
        return chunks

    # Sliding window chunking
    for start_idx in range(0, total_words, step):
        end_idx = min(start_idx + chunk_size, total_words)
        chunk_words = words[start_idx:end_idx]

        if not chunk_words:
            break

        clean_content = " ".join(chunk_words).strip()
        if clean_content:
            chunks.append({
                "chunk_id": create_chunk_id(filename, page_number, chunk_idx),
                "text": clean_content,
                "page_number": page_number,
                "filename": filename,
                "word_count": len(chunk_words),
                "char_count": len(clean_content),
            })
            chunk_idx += 1

        # If we reached the end of the text, stop sliding
        if end_idx >= total_words:
            break

    return chunks


def chunk_pdf_pages(
    pages: List[Dict[str, Any]],
    filename: str = "document.pdf",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Split extracted PDF pages into chunks, keeping page references 100% accurate.

    Each page's text is chunked individually, guaranteeing that:
      1. Every chunk's `page_number` accurately points to the exact PDF page.
      2. No empty pages produce empty chunks.
      3. Long pages spanning >500 words are broken down with the configured overlap.

    Parameters
    ----------
    pages : List[Dict[str, Any]]
        List of dicts with keys "page_number" (int) and "text" (str),
        as returned by `services.pdf_parser.extract_text_from_pdf`.
    filename : str
        Source filename.
    chunk_size : int
        Target word count per chunk.
    chunk_overlap : int
        Word overlap between chunks.

    Returns
    -------
    List[Dict[str, Any]]
        All valid non-empty chunks across all pages.
    """
    all_chunks: List[Dict[str, Any]] = []

    for page_info in pages:
        page_num = page_info.get("page_number", 1)
        raw_text = page_info.get("text", "")

        # Skip empty or whitespace-only pages
        if not raw_text or not raw_text.strip():
            continue

        page_chunks = chunk_text(
            text=raw_text,
            filename=filename,
            page_number=page_num,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_chunk_index=1,
        )

        all_chunks.extend(page_chunks)

    return all_chunks


def chunk_parse_result(
    parse_result: Any,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Convenience wrapper that directly accepts a `PDFParseResult` instance
    from `services.pdf_parser`.

    Parameters
    ----------
    parse_result : PDFParseResult
        The result object from `extract_text_from_pdf`.

    Returns
    -------
    List[Dict[str, Any]]
        List of chunks, or empty list if parse_result is None or has errors.
    """
    if not parse_result or getattr(parse_result, "error", None) is not None:
        return []

    pages = getattr(parse_result, "pages", [])
    filename = getattr(parse_result, "filename", "document.pdf")

    return chunk_pdf_pages(
        pages=pages,
        filename=filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
