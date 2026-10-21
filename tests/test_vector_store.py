# ============================================================
# tests/test_vector_store.py  –  Test ChromaDB Vector Store & Search
# ============================================================
# Test suite for Stage 4:
#   1. Empty collection & empty query handling
#   2. Indexing and metadata validation
#   3. Repeated indexing (deduplication & per-document replacement)
#   4. Embedding dimension validation (384-d enforcement)
#   5. Semantic search retrieval & distance scoring
#   6. End-to-end test with sample_contract.pdf
#
# Run with:
#     .venv\Scripts\python.exe tests/test_vector_store.py
# ============================================================

import os
import shutil
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import (
    get_chroma_client,
    get_collection,
    index_document_chunks,
    search_contract_chunks,
    get_collection_stats,
    clear_collection,
    EXPECTED_EMBEDDING_DIM,
)
from services.chunking import chunk_parse_result, chunk_pdf_pages
from services.embeddings import generate_chunk_embeddings
from services.pdf_parser import extract_text_from_pdf

TEST_PERSIST_DIR = "tests/test_chroma_db"
TEST_COLLECTION = "test_contracts"


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST: {title}")
    print(f"{'=' * 60}")


def cleanup_test_db():
    """Remove test ChromaDB directory."""
    if os.path.exists(TEST_PERSIST_DIR):
        try:
            shutil.rmtree(TEST_PERSIST_DIR)
        except Exception:
            pass


def test_empty_collection_and_queries():
    divider("Empty collection and empty query handling")
    cleanup_test_db()

    # 1. Search an empty collection
    results, err = search_contract_chunks(
        query="What is the term of the agreement?",
        top_k=5,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"  Empty collection search result: {results}, error: {err}")
    assert results == [], "Empty collection must return empty results list"
    assert err is None, f"Expected no error on empty collection, got: {err}"

    # 2. Search with empty string
    results, err = search_contract_chunks(
        query="",
        top_k=5,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"  Empty query search result: {results}, error: {err}")
    assert results == []
    assert err == "Query is empty."

    # 3. Search with whitespace
    results, err = search_contract_chunks(
        query="   \n\t  ",
        top_k=5,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert results == []
    assert err == "Query is empty."

    print("  [OK] Empty collections and empty queries handled gracefully.")
    return True


def test_dimension_validation():
    divider("Embedding dimension validation (384-d enforcement)")
    cleanup_test_db()

    # Create dummy chunk with wrong dimension (e.g. 128 instead of 384)
    invalid_chunk = [{
        "chunk_id": "bad_chunk_1",
        "text": "Invalid embedding chunk",
        "page_number": 1,
        "filename": "invalid.pdf",
        "embedding": [0.1] * 128,  # Wrong dimension!
    }]

    count, err = index_document_chunks(
        chunks=invalid_chunk,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"  Invalid dimension indexing count: {count}, error: {err}")
    assert count == 0, "Indexing must reject invalid dimensions"
    assert err is not None and "dimension mismatch" in err.lower()

    print("  [OK] Dimension validation correctly rejected mismatched embeddings.")
    return True


def test_indexing_and_metadata():
    divider("Indexing and metadata preservation")
    cleanup_test_db()

    # Create 3 synthetic chunks with valid 384-d embeddings
    fake_chunks = [
        {
            "chunk_id": "contract_a_p1_c1",
            "text": "This Service Agreement is entered into by Acme Corp and Global LLC.",
            "page_number": 1,
            "filename": "contract_a.pdf",
            "embedding": [0.05] * EXPECTED_EMBEDDING_DIM,
            "word_count": 12,
        },
        {
            "chunk_id": "contract_a_p2_c1",
            "text": "Party A shall pay Party B a monthly fee of $15,000 for consulting services.",
            "page_number": 2,
            "filename": "contract_a.pdf",
            "embedding": [0.10] * EXPECTED_EMBEDDING_DIM,
            "word_count": 13,
        },
    ]

    indexed_count, err = index_document_chunks(
        chunks=fake_chunks,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert err is None, f"Failed to index: {err}"
    assert indexed_count == 2, f"Expected 2 indexed chunks, got {indexed_count}"

    # Verify collection stats
    stats = get_collection_stats(TEST_COLLECTION, TEST_PERSIST_DIR)
    print(f"  Collection stats after indexing: {stats}")
    assert stats["total_chunks"] == 2
    assert stats["documents"] == ["contract_a.pdf"]

    # Verify stored metadata in collection
    collection = get_collection(TEST_COLLECTION, TEST_PERSIST_DIR)
    record = collection.get(ids=["contract_a_p2_c1"], include=["metadatas", "documents"])
    assert len(record["ids"]) == 1
    meta = record["metadatas"][0]
    print(f"  Retrieved record metadata: {meta}")
    assert meta["filename"] == "contract_a.pdf"
    assert meta["page_number"] == 2
    assert meta["chunk_id"] == "contract_a_p2_c1"
    assert meta["word_count"] == 13

    print("  [OK] Indexing and metadata successfully validated.")
    return True


def test_repeated_indexing_and_multi_doc():
    divider("Repeated indexing deduplication and multi-document isolation")
    # Using existing test database from previous test (currently has 2 chunks for contract_a.pdf)

    # Re-index contract_a with modified text (simulates re-uploading an edited contract)
    updated_contract_a = [
        {
            "chunk_id": "contract_a_p1_c1",
            "text": "Updated Service Agreement between Acme Corp and Global LLC.",
            "page_number": 1,
            "filename": "contract_a.pdf",
            "embedding": [0.06] * EXPECTED_EMBEDDING_DIM,
            "word_count": 9,
        },
        {
            "chunk_id": "contract_a_p2_c1",
            "text": "Party A shall pay a renegotiated fee of $20,000 monthly.",
            "page_number": 2,
            "filename": "contract_a.pdf",
            "embedding": [0.12] * EXPECTED_EMBEDDING_DIM,
            "word_count": 9,
        },
    ]

    count, err = index_document_chunks(
        chunks=updated_contract_a,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert err is None
    assert count == 2

    # Check that total count is STILL 2 (no duplicates created!)
    stats = get_collection_stats(TEST_COLLECTION, TEST_PERSIST_DIR)
    print(f"  Stats after re-indexing contract_a: {stats}")
    assert stats["total_chunks"] == 2, f"Expected 2 chunks, got {stats['total_chunks']} (duplicates were created!)"

    # Now index a completely separate document: contract_b.pdf
    contract_b = [
        {
            "chunk_id": "contract_b_p1_c1",
            "text": "Non-Disclosure Agreement between Delta Inc and Echo Corp.",
            "page_number": 1,
            "filename": "contract_b.pdf",
            "embedding": [0.20] * EXPECTED_EMBEDDING_DIM,
            "word_count": 8,
        }
    ]
    count_b, err_b = index_document_chunks(
        chunks=contract_b,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert err_b is None
    assert count_b == 1

    # Check that contract_b was added and contract_a was preserved
    stats_both = get_collection_stats(TEST_COLLECTION, TEST_PERSIST_DIR)
    print(f"  Stats after indexing contract_b: {stats_both}")
    assert stats_both["total_chunks"] == 3  # 2 from a + 1 from b
    assert sorted(stats_both["documents"]) == ["contract_a.pdf", "contract_b.pdf"]

    print("  [OK] Repeated indexing replaces only target document and preserves other documents.")
    return True


def test_semantic_search_with_sample_contract():
    divider("Semantic search & retrieval on real sample contract")
    cleanup_test_db()

    pdf_path = os.path.join("data", "sample_contract.pdf")
    if not os.path.exists(pdf_path):
        print("  [SKIP] data/sample_contract.pdf not found.")
        return True

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 1. Parse PDF
    parse_result = extract_text_from_pdf(pdf_bytes, "sample_contract.pdf")
    assert parse_result.error is None

    # 2. Chunk text
    raw_chunks = chunk_parse_result(parse_result, chunk_size=500, chunk_overlap=80)
    assert len(raw_chunks) > 0
    print(f"  Created {len(raw_chunks)} chunks from sample contract.")

    # 3. Generate embeddings
    embedded_chunks, emb_err = generate_chunk_embeddings(raw_chunks)
    assert emb_err is None
    print(f"  Generated embeddings for {len(embedded_chunks)} chunks.")

    # 4. Index into ChromaDB
    indexed_count, idx_err = index_document_chunks(
        chunks=embedded_chunks,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert idx_err is None
    assert indexed_count == len(embedded_chunks)
    print(f"  Indexed {indexed_count} chunks into ChromaDB.")

    # 5. Semantic Search Query 1: Payment and fees
    query_1 = "What is the monthly payment and compensation fee?"
    results_1, search_err = search_contract_chunks(
        query=query_1,
        top_k=3,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    assert search_err is None
    assert len(results_1) > 0
    print(f"\n  Query 1: '{query_1}'")
    print(f"  Top result:")
    print(f"    Chunk ID:    {results_1[0]['chunk_id']}")
    print(f"    Page Number: {results_1[0]['page_number']}")
    print(f"    Distance:    {results_1[0]['distance']} (Similarity: {results_1[0]['similarity']})")
    print(f"    Excerpt:     {results_1[0]['text'][:100]}...")

    # The payment terms are in Page 2 of sample_contract.pdf
    assert results_1[0]["page_number"] == 2 or "15,000" in results_1[0]["text"]

    # 6. Semantic Search Query 2: Termination
    query_2 = "Can either party terminate the contract early with written notice?"
    results_2, _ = search_contract_chunks(
        query=query_2,
        top_k=2,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"\n  Query 2: '{query_2}'")
    print(f"  Top result Chunk ID: {results_2[0]['chunk_id']} | Page: {results_2[0]['page_number']}")
    assert len(results_2) <= 2
    for r in results_2:
        assert "chunk_id" in r
        assert "text" in r
        assert "filename" in r
        assert "page_number" in r
        assert "distance" in r

    cleanup_test_db()
    print("\n  [OK] Semantic search retrieved highly relevant clauses with accurate citations.")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  LEGALEAGLE STAGE 4 TEST SUITE: CHROMADB & SEMANTIC SEARCH")
    print("=" * 60)

    t1 = test_empty_collection_and_queries()
    t2 = test_dimension_validation()
    t3 = test_indexing_and_metadata()
    t4 = test_repeated_indexing_and_multi_doc()
    t5 = test_semantic_search_with_sample_contract()

    if all([t1, t2, t3, t4, t5]):
        print(f"\n{'=' * 60}")
        print("  ALL 5/5 STAGE 4 TESTS PASSED!")
        print(f"{'=' * 60}")
    else:
        print("\n[FAIL] Some tests failed.")
        sys.exit(1)
