# ============================================================
# tests/test_pdf_parser.py  –  Test the PDF Extraction Module
# ============================================================
# This script tests the pdf_parser module against:
#   1. A normal multi-page PDF    (should extract text)
#   2. An empty PDF               (should report 0 pages)
#   3. A PDF with a blank page    (should warn about it)
#
# Run with:
#     python tests/test_pdf_parser.py
#
# Prerequisites:
#     python tests/create_sample_pdf.py   (creates the sample)
# ============================================================

import sys
import os

# Add the project root to the Python path so we can import
# from the "services" package even when running this script
# directly from the tests/ folder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pymupdf  # PyMuPDF — needed to create test PDFs on the fly
from services.pdf_parser import extract_text_from_pdf, get_text_preview, get_full_text


def divider(title: str):
    """Print a section divider for readability."""
    print(f"\n{'=' * 60}")
    print(f"  TEST: {title}")
    print(f"{'=' * 60}")


def test_normal_pdf():
    """Test extraction on the sample contract PDF."""
    divider("Normal multi-page PDF")

    sample_path = os.path.join("data", "sample_contract.pdf")
    if not os.path.exists(sample_path):
        print("[FAIL] Sample PDF not found. Run this first:")
        print("     python tests/create_sample_pdf.py")
        return False

    with open(sample_path, "rb") as f:
        pdf_bytes = f.read()

    result = extract_text_from_pdf(pdf_bytes, "sample_contract.pdf")

    print(f"  Filename:     {result.filename}")
    print(f"  Page count:   {result.page_count}")
    print(f"  Total chars:  {result.total_chars:,}")
    print(f"  Warnings:     {result.warnings}")
    print(f"  Error:        {result.error}")
    print(f"\n  Preview (first 300 chars):")
    print(f"  {get_text_preview(result, 300)}")

    # Basic assertions
    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.page_count == 3, f"Expected 3 pages, got {result.page_count}"
    assert result.total_chars > 100, f"Too few characters: {result.total_chars}"
    assert len(result.pages) == 3, f"Expected 3 page entries, got {len(result.pages)}"

    # Check page numbers are correct
    for i, page in enumerate(result.pages):
        assert page["page_number"] == i + 1, f"Page number mismatch at index {i}"

    print("\n  [OK] PASSED")
    return True


def test_empty_pdf():
    """Test extraction on a minimal PDF-like input.

    PyMuPDF won't save a zero-page document to bytes, so
    instead we pass minimal bytes that it may reject — testing
    that the parser handles the failure gracefully.
    """
    divider("Empty / minimal PDF")

    # A minimal valid PDF header but no actual content —
    # PyMuPDF should either open it with 0 pages or reject it.
    minimal_pdf = b"%PDF-1.0\n%%EOF\n"
    result = extract_text_from_pdf(minimal_pdf, "empty.pdf")

    print(f"  Page count: {result.page_count}")
    print(f"  Error:      {result.error}")

    # We expect either an error OR zero chars with zero pages
    if result.error:
        print(f"  -> Parser correctly rejected the file.")
    else:
        assert result.page_count == 0 or result.total_chars == 0

    print("\n  [OK] PASSED")
    return True


def test_blank_page_pdf():
    """Test extraction on a PDF where one page has no text."""
    divider("PDF with a blank page")

    doc = pymupdf.open()
    # Page 1: has text
    page1 = doc.new_page()
    page1.insert_text((72, 100), "This page has some text.")
    # Page 2: blank (no text inserted)
    doc.new_page()
    # Page 3: has text
    page3 = doc.new_page()
    page3.insert_text((72, 100), "This page also has text.")

    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf(pdf_bytes, "partial_blank.pdf")

    print(f"  Page count:  {result.page_count}")
    print(f"  Total chars: {result.total_chars}")
    print(f"  Warnings:    {result.warnings}")

    assert result.error is None, f"Unexpected error: {result.error}"
    assert result.page_count == 3
    assert len(result.warnings) > 0, "Expected a warning about the blank page"
    assert "2" in result.warnings[0], "Warning should mention page 2"

    print("\n  [OK] PASSED")
    return True


def test_corrupted_data():
    """Test extraction on random bytes (not a valid PDF)."""
    divider("Corrupted / invalid file")

    garbage = b"This is not a PDF file at all! Just random text."
    result = extract_text_from_pdf(garbage, "corrupted.pdf")

    print(f"  Error: {result.error}")

    assert result.error is not None, "Expected an error for corrupted data"

    print("\n  [OK] PASSED")
    return True


def test_all_blank_pages():
    """Test a PDF where ALL pages have no text (simulates scanned)."""
    divider("All-blank pages (simulates scanned PDF)")

    doc = pymupdf.open()
    doc.new_page()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf(pdf_bytes, "scanned.pdf")

    print(f"  Page count:  {result.page_count}")
    print(f"  Total chars: {result.total_chars}")
    print(f"  Warnings:    {result.warnings}")

    assert result.error is None  # not an error, just a warning
    assert result.total_chars == 0
    assert len(result.warnings) > 0
    assert "scanned" in result.warnings[0].lower() or "no text" in result.warnings[0].lower()

    print("\n  [OK] PASSED")
    return True


# ── Run all tests ───────────────────────────────────────────
if __name__ == "__main__":
    results = []
    results.append(test_normal_pdf())
    results.append(test_empty_pdf())
    results.append(test_blank_page_pdf())
    results.append(test_corrupted_data())
    results.append(test_all_blank_pages())

    print(f"\n{'=' * 60}")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 60}")

    if passed < total:
        sys.exit(1)
