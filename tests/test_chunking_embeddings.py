# ============================================================
# tests/test_chunking_embeddings.py
# ============================================================
# Test suite for Stage 3:
#   1. Chunking logic (500 words, ~80 words overlap)
#   2. Empty text and blank page handling
#   3. Metadata preservation (filename, page numbers, unique chunk IDs)
#   4. Embedding generation and vector dimensions (384 dimensions)
#   5. End-to-end test on the sample contract PDF
#
# Run with:
#     python tests/test_chunking_embeddings.py
# ============================================================

import sys
import os

# Add root directory to path so we can import services
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.chunking import (
    chunk_text,
    chunk_pdf_pages,
    chunk_parse_result,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from services.embeddings import (
    load_embedding_model,
    generate_single_embedding,
    generate_chunk_embeddings,
    EXPECTED_EMBEDDING_DIM,
)
from services.pdf_parser import extract_text_from_pdf


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST: {title}")
    print(f"{'=' * 60}")


def test_chunking_word_count_and_overlap():
    divider("Chunking word count and ~80-word overlap")

    # Generate a dummy contract clause of 1,000 words
    words = [f"word{i}" for i in range(1, 1001)]
    long_text = " ".join(words)

    chunks = chunk_text(
        text=long_text,
        filename="test_contract.pdf",
        page_number=1,
        chunk_size=500,
        chunk_overlap=80,
    )

    print(f"  Total words in source: 1,000")
    print(f"  Number of chunks created: {len(chunks)}")
    for i, c in enumerate(chunks, 1):
        print(f"    Chunk {i}: {c['chunk_id']} | words={c['word_count']} | chars={c['char_count']}")

    # Assertions
    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"
    assert chunks[0]["word_count"] == 500, f"Expected 500 words in chunk 1, got {chunks[0]['word_count']}"

    # Verify overlap: last 80 words of chunk 1 should match first 80 words of chunk 2
    words_c1 = chunks[0]["text"].split()
    words_c2 = chunks[1]["text"].split()
    overlap_c1 = words_c1[-80:]
    overlap_c2 = words_c2[:80]
    assert overlap_c1 == overlap_c2, "Overlap between chunk 1 and chunk 2 does not match!"

    print("  [OK] Word counts and overlap verified successfully.")
    return True


def test_empty_text_and_blank_pages():
    divider("Empty text and blank page handling")

    # Test empty string
    empty_chunks = chunk_text("", filename="empty.pdf", page_number=1)
    assert empty_chunks == [], "Empty string must return 0 chunks"

    # Test whitespace-only string
    whitespace_chunks = chunk_text("   \n\t   ", filename="blank.pdf", page_number=1)
    assert whitespace_chunks == [], "Whitespace string must return 0 chunks"

    # Test pages list with blank pages
    pages = [
        {"page_number": 1, "text": "This page has valid legal terms."},
        {"page_number": 2, "text": ""},  # blank page
        {"page_number": 3, "text": "   "},  # whitespace page
        {"page_number": 4, "text": "Another page with valid terms."},
    ]
    result_chunks = chunk_pdf_pages(pages, filename="mixed.pdf")

    print(f"  Input pages: 4 (2 non-empty, 2 blank)")
    print(f"  Result chunks: {len(result_chunks)}")
    for c in result_chunks:
        print(f"    {c['chunk_id']}: page={c['page_number']}, words={c['word_count']}")

    assert len(result_chunks) == 2, f"Expected 2 chunks, got {len(result_chunks)}"
    assert result_chunks[0]["page_number"] == 1
    assert result_chunks[1]["page_number"] == 4

    print("  [OK] Empty texts and blank pages handled gracefully without empty chunks.")
    return True


def test_metadata_preservation():
    divider("Metadata preservation and accurate page references")

    pages = [
        {"page_number": 1, "text": "Page one agreement intro."},
        {"page_number": 2, "text": "Page two indemnification clauses."},
        {"page_number": 3, "text": "Page three signature lines."},
    ]
    chunks = chunk_pdf_pages(pages, filename="master_agreement.pdf")

    assert len(chunks) == 3
    chunk_ids = set()

    for idx, c in enumerate(chunks, 1):
        # Verify required keys
        for key in ["chunk_id", "text", "page_number", "filename", "word_count", "char_count"]:
            assert key in c, f"Missing key {key} in chunk metadata"

        assert c["filename"] == "master_agreement.pdf"
        assert c["page_number"] == idx, f"Expected page_number {idx}, got {c['page_number']}"
        assert c["chunk_id"] not in chunk_ids, f"Duplicate chunk_id found: {c['chunk_id']}"
        chunk_ids.add(c["chunk_id"])

    print(f"  Unique chunk IDs: {chunk_ids}")
    print("  [OK] Metadata and page numbers 100% accurate and preserved.")
    return True


def test_embedding_dimensions_and_errors():
    divider("Embedding generation and dimensions (all-MiniLM-L6-v2)")

    # Test empty text handling
    empty_emb = generate_single_embedding("")
    assert empty_emb is None, "Empty text must return None"

    empty_chunk_emb, err = generate_chunk_embeddings([])
    assert empty_chunk_emb == [] and err is None

    # Test single embedding generation
    test_phrase = "This Agreement shall be governed by the laws of the State of Delaware."
    emb = generate_single_embedding(test_phrase)

    assert emb is not None, "Failed to generate embedding for test phrase"
    assert isinstance(emb, list), "Embedding should be a list of floats"
    assert len(emb) == EXPECTED_EMBEDDING_DIM, (
        f"Expected {EXPECTED_EMBEDDING_DIM} dimensions, got {len(emb)}"
    )

    print(f"  Single embedding dimension: {len(emb)} (Expected: {EXPECTED_EMBEDDING_DIM})")
    print(f"  First 5 vector values: {[round(x, 4) for x in emb[:5]]}")

    # Test batch chunk embedding
    sample_chunks = [
        {
            "chunk_id": "test_p1_c1",
            "text": "Acme Corporation agrees to pay Global Services LLC.",
            "page_number": 1,
            "filename": "contract.pdf",
            "word_count": 8,
            "char_count": 52,
        },
        {
            "chunk_id": "test_p2_c1",
            "text": "Either party may terminate this agreement with 30 days notice.",
            "page_number": 2,
            "filename": "contract.pdf",
            "word_count": 10,
            "char_count": 62,
        },
    ]

    embedded_chunks, error = generate_chunk_embeddings(sample_chunks)
    assert error is None, f"Embedding generation failed: {error}"
    assert len(embedded_chunks) == 2, f"Expected 2 embedded chunks, got {len(embedded_chunks)}"

    for c in embedded_chunks:
        assert "embedding" in c, "Chunk missing 'embedding' key"
        assert len(c["embedding"]) == EXPECTED_EMBEDDING_DIM
        assert c["embedding_dim"] == EXPECTED_EMBEDDING_DIM
        print(f"    {c['chunk_id']}: embedding_dim={c['embedding_dim']}, text='{c['text'][:35]}...'")

    print(f"  [OK] Embeddings generated successfully with exact {EXPECTED_EMBEDDING_DIM} dimensions.")
    return True


def test_end_to_end_sample_contract():
    divider("End-to-End: Sample contract PDF -> Extract -> Chunk -> Embed")

    pdf_path = os.path.join("data", "sample_contract.pdf")
    if not os.path.exists(pdf_path):
        print("  [SKIP] data/sample_contract.pdf not found. Run tests/create_sample_pdf.py first.")
        return True

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 1. Extract
    parse_result = extract_text_from_pdf(pdf_bytes, "sample_contract.pdf")
    assert parse_result.error is None
    print(f"  1. Extracted: {parse_result.page_count} pages, {parse_result.total_chars} chars")

    # 2. Chunk
    chunks = chunk_parse_result(parse_result, chunk_size=500, chunk_overlap=80)
    print(f"  2. Chunked into: {len(chunks)} chunks")
    for c in chunks:
        print(f"     • {c['chunk_id']} (Page {c['page_number']}): {c['word_count']} words, {c['char_count']} chars")

    # 3. Embed
    embedded_chunks, error = generate_chunk_embeddings(chunks)
    assert error is None, f"Failed to embed chunks: {error}"
    print(f"  3. Embedded: {len(embedded_chunks)} chunks successfully generated.")

    print("\n  [OK] End-to-end test passed completely!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  LEGALEAGLE STAGE 3 TEST SUITE: CHUNKING & EMBEDDINGS")
    print("=" * 60)

    t1 = test_chunking_word_count_and_overlap()
    t2 = test_empty_text_and_blank_pages()
    t3 = test_metadata_preservation()
    t4 = test_embedding_dimensions_and_errors()
    t5 = test_end_to_end_sample_contract()

    if all([t1, t2, t3, t4, t5]):
        print(f"\n{'=' * 60}")
        print("  ALL 5/5 STAGE 3 TESTS PASSED!")
        print(f"{'=' * 60}")
    else:
        print("\n[FAIL] Some tests failed.")
        sys.exit(1)
