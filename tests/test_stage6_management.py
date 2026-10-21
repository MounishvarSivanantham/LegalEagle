# ============================================================
# tests/test_stage6_management.py  –  Step 6 Tests
# ============================================================
# This test suite validates:
#   1. Document filtering in ChromaDB semantic search.
#   2. Safe document deletion (deleting only target document; never clearing DB).
#   3. Detailed document metadata listing (page count, chunk count, words).
#   4. Chat history and follow-up context integration in QA.
#   5. Conversation export to Markdown and Text with citations and without secrets.
#
# Run with:
#     python tests/test_stage6_management.py
# ============================================================

import sys
import os
import shutil
import unittest
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import (
    index_document_chunks,
    search_contract_chunks,
    get_document_metadata_list,
    delete_document_vectors,
    get_collection_stats,
    clear_collection,
)
from services.qa import (
    answer_contract_question,
    export_conversation_to_markdown,
    export_conversation_to_text,
    QAResult,
)

TEST_DB_DIR = os.path.join("tests", "test_chroma_db_stage6")
TEST_COLLECTION = "test_stage6_contracts"


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST: {title}")
    print(f"{'=' * 60}")


def clean_test_db():
    if os.path.exists(TEST_DB_DIR):
        try:
            shutil.rmtree(TEST_DB_DIR)
        except Exception:
            pass


def test_document_filtering_and_metadata():
    divider("Document filtering & metadata aggregation")
    clean_test_db()

    # Create dummy embeddings of length 384
    dummy_vec_1 = [0.05] * 384
    dummy_vec_2 = [-0.05] * 384

    doc_a_chunks = [
        {
            "chunk_id": "emp_p1_c1",
            "text": "The employee annual base salary is $120,000 paid semi-monthly.",
            "page_number": 1,
            "filename": "employment_agreement.pdf",
            "word_count": 9,
            "embedding": dummy_vec_1,
        },
        {
            "chunk_id": "emp_p2_c1",
            "text": "Employment is at-will and may be terminated with 2 weeks notice.",
            "page_number": 2,
            "filename": "employment_agreement.pdf",
            "word_count": 10,
            "embedding": dummy_vec_1,
        },
    ]

    doc_b_chunks = [
        {
            "chunk_id": "lease_p1_c1",
            "text": "The tenant shall pay monthly rent of $3,500 due on the first of each month.",
            "page_number": 1,
            "filename": "commercial_lease.pdf",
            "word_count": 14,
            "embedding": dummy_vec_2,
        },
        {
            "chunk_id": "lease_p3_c1",
            "text": "Security deposit of $7,000 shall be held in an interest-bearing escrow account.",
            "page_number": 3,
            "filename": "commercial_lease.pdf",
            "word_count": 11,
            "embedding": dummy_vec_2,
        },
    ]

    # Index both documents
    count_a, err_a = index_document_chunks(
        doc_a_chunks, filename="employment_agreement.pdf",
        collection_name=TEST_COLLECTION, persist_directory=TEST_DB_DIR
    )
    count_b, err_b = index_document_chunks(
        doc_b_chunks, filename="commercial_lease.pdf",
        collection_name=TEST_COLLECTION, persist_directory=TEST_DB_DIR
    )

    assert err_a is None and count_a == 2
    assert err_b is None and count_b == 2

    # Verify metadata list
    meta_list = get_document_metadata_list(
        collection_name=TEST_COLLECTION, persist_directory=TEST_DB_DIR
    )
    assert len(meta_list) == 2, f"Expected 2 documents, got {len(meta_list)}"

    doc_map = {d["filename"]: d for d in meta_list}
    assert "employment_agreement.pdf" in doc_map
    assert "commercial_lease.pdf" in doc_map

    emp_meta = doc_map["employment_agreement.pdf"]
    assert emp_meta["chunk_count"] == 2
    assert emp_meta["page_count"] == 2
    assert emp_meta["pages"] == [1, 2]
    assert emp_meta["total_words"] == 19

    lease_meta = doc_map["commercial_lease.pdf"]
    assert lease_meta["chunk_count"] == 2
    assert lease_meta["page_count"] == 2
    assert lease_meta["pages"] == [1, 3]

    print(f"  Indexed 2 documents with 4 chunks total.")
    print(f"  Metadata verified: {meta_list}")

    # Test filtering by employment_agreement.pdf
    emp_results, err = search_contract_chunks(
        query="salary and compensation",
        top_k=5,
        filename_filter="employment_agreement.pdf",
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_DB_DIR,
    )
    assert err is None
    assert len(emp_results) > 0
    for r in emp_results:
        assert r["filename"] == "employment_agreement.pdf", (
            f"Expected only employment_agreement.pdf, got {r['filename']}"
        )
    print("  [OK] Document filter strictly isolates results to employment_agreement.pdf.")

    # Test filtering by commercial_lease.pdf
    lease_results, err = search_contract_chunks(
        query="monthly rent and deposit",
        top_k=5,
        filename_filter="commercial_lease.pdf",
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_DB_DIR,
    )
    assert err is None
    assert len(lease_results) > 0
    for r in lease_results:
        assert r["filename"] == "commercial_lease.pdf", (
            f"Expected only commercial_lease.pdf, got {r['filename']}"
        )
    print("  [OK] Document filter strictly isolates results to commercial_lease.pdf.")

    return True


def test_safe_document_deletion():
    divider("Safe document deletion (targeted delete, no accidental clear)")

    # Currently we have employment_agreement.pdf and commercial_lease.pdf in TEST_DB_DIR
    stats_before = get_collection_stats(
        collection_name=TEST_COLLECTION, persist_directory=TEST_DB_DIR
    )
    assert stats_before["total_chunks"] == 4
    assert len(stats_before["documents"]) == 2

    # Safely delete employment_agreement.pdf
    deleted_count, del_err = delete_document_vectors(
        filename="employment_agreement.pdf",
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_DB_DIR,
    )
    assert del_err is None
    assert deleted_count == 2, f"Expected 2 deleted chunks, got {deleted_count}"

    # Verify state after deletion:
    # 1. commercial_lease.pdf must STILL BE INTACT
    # 2. Total chunks must be 2
    # 3. employment_agreement.pdf must be completely gone
    stats_after = get_collection_stats(
        collection_name=TEST_COLLECTION, persist_directory=TEST_DB_DIR
    )
    assert stats_after["total_chunks"] == 2, f"Expected 2 remaining chunks, got {stats_after['total_chunks']}"
    assert stats_after["documents"] == ["commercial_lease.pdf"], (
        f"Expected only commercial_lease.pdf to remain, got {stats_after['documents']}"
    )

    # Search in commercial_lease.pdf should still work
    lease_results, err = search_contract_chunks(
        query="rent",
        top_k=2,
        filename_filter="commercial_lease.pdf",
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_DB_DIR,
    )
    assert err is None and len(lease_results) > 0
    assert lease_results[0]["filename"] == "commercial_lease.pdf"

    # Search in employment_agreement.pdf should return empty results
    emp_results, err = search_contract_chunks(
        query="salary",
        top_k=2,
        filename_filter="employment_agreement.pdf",
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_DB_DIR,
    )
    assert err is None and len(emp_results) == 0

    print("  [OK] Deleting employment_agreement.pdf removed only its records.")
    print("  [OK] commercial_lease.pdf was preserved completely without database wipe.")
    clean_test_db()
    return True


def test_chat_history_and_followup():
    divider("Chat history & follow-up context integration in QA")
    clean_test_db()

    # Index 1 chunk
    chunk = [{
        "chunk_id": "c1",
        "text": "Clause 4.2 states that notice must be provided via registered mail.",
        "page_number": 4,
        "filename": "agreement.pdf",
        "word_count": 11,
        "embedding": [0.01] * 384,
    }]
    index_document_chunks(
        chunk, filename="agreement.pdf",
        collection_name=TEST_COLLECTION, persist_directory=TEST_DB_DIR
    )

    # Mock OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Notice must be given via registered mail under [agreement.pdf, Page 4]."))
    ]
    mock_client.chat.completions.create.return_value = mock_response

    # Simulated past conversation history
    prior_history = [
        {"role": "user", "content": "How do we give notice?"},
        {"role": "assistant", "content": "Notice is covered under Section 4."},
    ]

    result = answer_contract_question(
        question="Can we send notice by email instead?",
        filename="agreement.pdf",
        openai_client=mock_client,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_DB_DIR,
        chat_history=prior_history,
    )

    assert result.error is None
    assert "registered mail" in result.answer

    # Verify that mock_client was called with conversation messages
    calls = mock_client.chat.completions.create.call_args
    passed_messages = calls.kwargs["messages"]

    roles = [m["role"] for m in passed_messages]
    # Should have: system, user, assistant, user
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert passed_messages[1]["content"] == "How do we give notice?"
    assert roles[2] == "assistant"
    assert passed_messages[2]["content"] == "Notice is covered under Section 4."
    assert roles[3] == "user"
    assert "Can we send notice by email instead?" in passed_messages[3]["content"]

    print("  Messages sent to OpenAI API:")
    for idx, m in enumerate(passed_messages):
        preview = m['content'][:45].replace('\n', ' ')
        print(f"    {idx}. [{m['role']}]: {preview}...")

    print("  [OK] Chat history and follow-up context correctly passed to LLM.")
    clean_test_db()
    return True


def test_conversation_export():
    divider("Conversation export (Markdown & Text) with citations and no secrets")

    sample_history = [
        {
            "role": "user",
            "content": "What is the termination period?",
            "timestamp": "2026-10-21 10:15:00",
        },
        {
            "role": "assistant",
            "content": "Either party may terminate with 30 days notice [sample_contract.pdf, Page 2].",
            "timestamp": "2026-10-21 10:15:05",
            "sources": [
                {
                    "filename": "sample_contract.pdf",
                    "page_number": 2,
                    "similarity": 0.8842,
                    "text": "Either party may terminate this agreement upon 30 days prior written notice.",
                }
            ],
        },
        {
            "role": "user",
            "content": "What about confidentiality?",
            "timestamp": "2026-10-21 10:16:00",
        },
        {
            "role": "assistant",
            "content": "Confidentiality obligations survive for 2 years [sample_contract.pdf, Page 3].",
            "timestamp": "2026-10-21 10:16:04",
            "sources": [
                {
                    "filename": "sample_contract.pdf",
                    "page_number": 3,
                    "similarity": 0.8210,
                    "text": "Confidentiality covenants shall survive termination for a period of 2 years.",
                }
            ],
        },
    ]

    # Test Markdown export
    md_output = export_conversation_to_markdown(
        sample_history, document_filter="sample_contract.pdf"
    )

    assert "# LegalDocuLex-AI" in md_output
    assert "What is the termination period?" in md_output
    assert "Either party may terminate with 30 days notice" in md_output
    assert "`sample_contract.pdf` (Page 2)" in md_output
    assert "What about confidentiality?" in md_output
    assert "Confidentiality obligations survive for 2 years" in md_output
    assert "`sample_contract.pdf` (Page 3)" in md_output
    assert "Legal Disclaimer" in md_output

    # Test Plain Text export
    txt_output = export_conversation_to_text(
        sample_history, document_filter="sample_contract.pdf"
    )

    assert "LEGALDOCULEX-AI AI CONTRACT ASSISTANT" in txt_output
    assert "QUESTION 1" in txt_output
    assert "What is the termination period?" in txt_output
    assert "LEGALDOCULEX-AI ANSWER" in txt_output
    assert "File: sample_contract.pdf | Page: 2" in txt_output
    assert "LEGAL DISCLAIMER" in txt_output

    # Ensure NO secrets or API key indicators exist in exports
    for sensitive in ["sk-", "OPENAI_API_KEY", "bearer", "secret", "password"]:
        assert sensitive not in md_output.lower(), f"Sensitive token '{sensitive}' found in Markdown export!"
        assert sensitive not in txt_output.lower(), f"Sensitive token '{sensitive}' found in Text export!"

    print("  Markdown export length:", len(md_output), "chars")
    print("  Text export length:    ", len(txt_output), "chars")
    print("  [OK] Export includes questions, answers, exact page citations, disclaimer, and 0 secrets.")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  LEGALDOCULEX-AI STAGE 6 TEST SUITE: INTERFACE & MANAGEMENT")
    print("=" * 60)

    t1 = test_document_filtering_and_metadata()
    t2 = test_safe_document_deletion()
    t3 = test_chat_history_and_followup()
    t4 = test_conversation_export()

    if all([t1, t2, t3, t4]):
        print(f"\n{'=' * 60}")
        print("  ALL 4/4 STAGE 6 TESTS PASSED!")
        print(f"{'=' * 60}")
    else:
        print("\n[FAIL] Some tests failed.")
        sys.exit(1)
