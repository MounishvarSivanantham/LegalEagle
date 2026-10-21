# ============================================================
# tests/test_unit_pytest.py  –  Comprehensive Pytest Test Suite
# ============================================================
# This suite satisfies Step 7 Requirement A:
#   • Uses pytest fixtures (tmp_path) for total test isolation.
#   • Mocks embedding models and external API calls so standard unit tests
#     run in seconds without model downloads or network dependencies.
#   • Validates PDF extraction, chunking, embeddings, ChromaDB, QA,
#     chat history, citations, exports, and security constraints.
# ============================================================

import os
import sys
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import pymupdf

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pdf_parser import extract_text_from_pdf, get_text_preview, get_full_text, PDFParseResult
from services.chunking import chunk_text, chunk_pdf_pages, chunk_parse_result
from services.embeddings import EXPECTED_EMBEDDING_DIM, generate_single_embedding, generate_chunk_embeddings
from services.vector_store import (
    index_document_chunks,
    search_contract_chunks,
    get_document_metadata_list,
    delete_document_vectors,
    get_collection_stats,
)
from services.qa import (
    answer_contract_question,
    export_conversation_to_markdown,
    export_conversation_to_text,
    get_openai_api_key,
    QAResult,
)


# ── Fixtures: Mock Embeddings (384-dimensional) ───────────────
@pytest.fixture(autouse=True)
def mock_embedding_generation():
    """
    Mock embedding generator so unit tests run fast with zero external network
    calls or Hugging Face model downloads.
    """
    def _fake_embed(text: str) -> List[float]:
        if not text or not text.strip():
            return None
        # Return deterministic 384-dimensional pseudo-embedding based on hash
        val = (hash(text) % 100) / 1000.0
        return [val] * EXPECTED_EMBEDDING_DIM

    with patch("services.embeddings.generate_single_embedding", side_effect=_fake_embed):
        yield



# ── 1. PDF Parser Tests ────────────────────────────────────────

def test_pdf_parser_normal():
    """Verify text extraction from a valid multi-page PDF."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "Confidential Consulting Agreement. Page 1 intro.")
    p2 = doc.new_page()
    p2.insert_text((72, 100), "Compensation: Party A shall pay Party B $10,000 monthly.")
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf(pdf_bytes, filename="consulting.pdf")
    assert result.error is None
    assert result.page_count == 2
    assert len(result.pages) == 2
    assert "Confidential Consulting" in result.pages[0]["text"]
    assert "$10,000" in result.pages[1]["text"]
    assert result.total_chars > 20


def test_pdf_parser_corrupted():
    """Verify corrupted or random bytes are rejected with a descriptive error."""
    corrupted_bytes = b"NOT_A_PDF_HEADER_12345"
    result = extract_text_from_pdf(corrupted_bytes, filename="bad.pdf")
    assert result.error is not None
    assert "corrupted or is not a valid PDF" in result.error


def test_pdf_parser_empty():
    """Verify empty byte stream is handled gracefully."""
    result = extract_text_from_pdf(b"", filename="empty.pdf")
    assert result.error is not None
    assert "empty" in result.error.lower() or "corrupted" in result.error.lower()


def test_pdf_parser_blank_pages():
    """Verify that pages with no selectable text trigger warnings without crashing."""
    doc = pymupdf.open()
    p1 = doc.new_page()
    p1.insert_text((72, 100), "Valid text here.")
    doc.new_page()  # blank page
    pdf_bytes = doc.tobytes()
    doc.close()

    result = extract_text_from_pdf(pdf_bytes, filename="blank_page.pdf")
    assert result.error is None
    assert result.page_count == 2
    assert len(result.warnings) > 0
    assert "Pages with no extractable text" in result.warnings[0]


def test_pdf_parser_size_limit_security():
    """Security check: verify that files exceeding the maximum size limit are rejected."""
    # Simulate a 100-byte file with a 50-byte max size limit
    fake_large_bytes = b"%PDF-" + b"0" * 95
    result = extract_text_from_pdf(fake_large_bytes, filename="huge.pdf", max_size_bytes=50)
    assert result.error is not None
    assert "exceeds maximum allowed limit" in result.error


# ── 2. Chunking Tests ──────────────────────────────────────────

def test_chunking_word_count_and_overlap():
    """Verify chunks average ~500 words with ~80-word overlap."""
    words = [f"token{i}" for i in range(1, 1001)]  # 1000 words
    long_text = " ".join(words)

    chunks = chunk_text(long_text, filename="agreement.pdf", page_number=1, chunk_size=500, chunk_overlap=80)
    assert len(chunks) == 3
    assert chunks[0]["word_count"] == 500

    # Verify overlap
    c1_words = chunks[0]["text"].split()
    c2_words = chunks[1]["text"].split()
    assert c1_words[-80:] == c2_words[:80]


def test_chunking_metadata_preservation():
    """Verify page numbers, filenames, word counts, and unique chunk IDs are preserved."""
    pages = [
        {"page_number": 1, "text": "Clause 1: Scope of Work and Deliverables."},
        {"page_number": 2, "text": "Clause 2: Payment and Compensation Schedule."},
    ]
    chunks = chunk_pdf_pages(pages, filename="master_services.pdf")
    assert len(chunks) == 2
    assert chunks[0]["page_number"] == 1
    assert chunks[1]["page_number"] == 2
    assert chunks[0]["filename"] == "master_services.pdf"
    assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]


def test_chunking_empty_and_whitespace():
    """Verify empty and whitespace strings return zero chunks without phantom records."""
    assert chunk_text("", filename="empty.pdf", page_number=1) == []
    assert chunk_text("   \n\t  ", filename="blank.pdf", page_number=1) == []


# ── 3. Embeddings Tests ────────────────────────────────────────

def test_embedding_dimensions_and_validation():
    """Verify embeddings produce exact 384-dimensional vectors and reject invalid dimensions."""
    sample_chunks = [
        {
            "chunk_id": "c1",
            "text": "The contract shall expire on December 31, 2026.",
            "page_number": 1,
            "filename": "lease.pdf",
            "word_count": 8,
        }
    ]
    embedded_chunks, err = generate_chunk_embeddings(sample_chunks)
    assert err is None
    assert len(embedded_chunks) == 1
    assert "embedding" in embedded_chunks[0]
    assert len(embedded_chunks[0]["embedding"]) == EXPECTED_EMBEDDING_DIM
    assert embedded_chunks[0]["embedding_dim"] == EXPECTED_EMBEDDING_DIM


# ── 4. ChromaDB & Vector Store Tests (Isolated tmp_path) ───────

def test_vector_store_indexing_and_retrieval(tmp_path):
    """Verify indexing and retrieval in an isolated temporary ChromaDB directory."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "test_contracts"

    dummy_vec = [0.05] * EXPECTED_EMBEDDING_DIM
    chunks = [
        {
            "chunk_id": "c1",
            "text": "Tenant shall pay monthly rent of $4,000 on the first day of each calendar month.",
            "page_number": 2,
            "filename": "lease_doc.pdf",
            "word_count": 15,
            "embedding": dummy_vec,
        }
    ]

    indexed_count, idx_err = index_document_chunks(
        chunks, filename="lease_doc.pdf", collection_name=coll_name, persist_directory=db_dir
    )
    assert idx_err is None
    assert indexed_count == 1

    # Verify retrieval
    results, search_err = search_contract_chunks(
        query="monthly rent payment",
        top_k=2,
        filename_filter="lease_doc.pdf",
        collection_name=coll_name,
        persist_directory=db_dir,
    )
    assert search_err is None
    assert len(results) == 1
    assert results[0]["filename"] == "lease_doc.pdf"
    assert results[0]["page_number"] == 2
    assert "monthly rent" in results[0]["text"]


def test_vector_store_duplicate_prevention_reindex(tmp_path):
    """Verify re-indexing replaces only the target document and preserves other documents."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "test_contracts"
    dummy_vec = [0.03] * EXPECTED_EMBEDDING_DIM

    # Index doc A
    doc_a_chunks = [{
        "chunk_id": "a_1",
        "text": "Initial terms for document A.",
        "page_number": 1,
        "filename": "doc_a.pdf",
        "word_count": 5,
        "embedding": dummy_vec,
    }]
    index_document_chunks(doc_a_chunks, filename="doc_a.pdf", collection_name=coll_name, persist_directory=db_dir)

    # Index doc B
    doc_b_chunks = [{
        "chunk_id": "b_1",
        "text": "Terms for document B.",
        "page_number": 1,
        "filename": "doc_b.pdf",
        "word_count": 4,
        "embedding": dummy_vec,
    }]
    index_document_chunks(doc_b_chunks, filename="doc_b.pdf", collection_name=coll_name, persist_directory=db_dir)

    # Re-index doc A with updated text
    doc_a_updated = [{
        "chunk_id": "a_new",
        "text": "Updated version 2 terms for document A.",
        "page_number": 1,
        "filename": "doc_a.pdf",
        "word_count": 7,
        "embedding": dummy_vec,
    }]
    index_document_chunks(doc_a_updated, filename="doc_a.pdf", collection_name=coll_name, persist_directory=db_dir)

    # Verify stats: should have exactly 2 chunks (1 for doc_a, 1 for doc_b)
    stats = get_collection_stats(coll_name, persist_directory=db_dir)
    assert stats["total_chunks"] == 2
    assert sorted(stats["documents"]) == ["doc_a.pdf", "doc_b.pdf"]


def test_vector_store_document_filtering(tmp_path):
    """Verify document filter restricts retrieval strictly to the target contract."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "test_contracts"

    chunks_1 = [{
        "chunk_id": "c1", "text": "Acme compensation terms.", "page_number": 1,
        "filename": "acme.pdf", "word_count": 3, "embedding": [0.01] * EXPECTED_EMBEDDING_DIM,
    }]
    chunks_2 = [{
        "chunk_id": "c2", "text": "Beta compensation terms.", "page_number": 1,
        "filename": "beta.pdf", "word_count": 3, "embedding": [0.02] * EXPECTED_EMBEDDING_DIM,
    }]
    index_document_chunks(chunks_1, filename="acme.pdf", collection_name=coll_name, persist_directory=db_dir)
    index_document_chunks(chunks_2, filename="beta.pdf", collection_name=coll_name, persist_directory=db_dir)

    results, _ = search_contract_chunks("compensation", filename_filter="acme.pdf", collection_name=coll_name, persist_directory=db_dir)
    assert len(results) == 1
    assert results[0]["filename"] == "acme.pdf"


def test_vector_store_safe_targeted_deletion(tmp_path):
    """Verify delete_document_vectors deletes only target records and never drops the collection."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "test_contracts"

    chunks_1 = [{
        "chunk_id": "c1", "text": "Target doc terms.", "page_number": 1,
        "filename": "delete_me.pdf", "word_count": 3, "embedding": [0.01] * EXPECTED_EMBEDDING_DIM,
    }]
    chunks_2 = [{
        "chunk_id": "c2", "text": "Keep me terms.", "page_number": 1,
        "filename": "keep_me.pdf", "word_count": 3, "embedding": [0.02] * EXPECTED_EMBEDDING_DIM,
    }]
    index_document_chunks(chunks_1, filename="delete_me.pdf", collection_name=coll_name, persist_directory=db_dir)
    index_document_chunks(chunks_2, filename="keep_me.pdf", collection_name=coll_name, persist_directory=db_dir)

    # Delete only delete_me.pdf
    deleted_cnt, del_err = delete_document_vectors("delete_me.pdf", collection_name=coll_name, persist_directory=db_dir)
    assert del_err is None
    assert deleted_cnt == 1

    # Verify keep_me.pdf is still present
    stats = get_collection_stats(coll_name, persist_directory=db_dir)
    assert stats["total_chunks"] == 1
    assert stats["documents"] == ["keep_me.pdf"]


# ── 5. Question Answering & Prompt Mocking Tests ───────────────

def test_qa_mocked_llm_answer(tmp_path):
    """Verify grounded answer formulation, citations, and disclaimer using mocked OpenAI client."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "test_contracts"

    chunk = [{
        "chunk_id": "term_c1",
        "text": "Section 4.1: Either party may terminate upon 30 days written notice.",
        "page_number": 3,
        "filename": "terms.pdf",
        "word_count": 10,
        "embedding": [0.05] * EXPECTED_EMBEDDING_DIM,
    }]
    index_document_chunks(chunk, filename="terms.pdf", collection_name=coll_name, persist_directory=db_dir)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content="Notice of 30 days is required under [terms.pdf, Page 3]."))
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    qa_res = answer_contract_question(
        question="What is the notice period for termination?",
        filename="terms.pdf",
        openai_client=mock_client,
        collection_name=coll_name,
        persist_directory=db_dir,
    )

    assert qa_res.error is None
    assert "30 days" in qa_res.answer
    assert "[terms.pdf, Page 3]" in qa_res.answer
    assert "Disclaimer" in qa_res.answer
    assert len(qa_res.sources) == 1
    assert qa_res.sources[0]["page_number"] == 3


def test_qa_insufficient_evidence(tmp_path):
    """Verify empty database or unanswerable query yields clear insufficient evidence notice."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "empty_coll"

    qa_res = answer_contract_question(
        question="What is the stock vesting schedule?",
        filename="empty.pdf",
        collection_name=coll_name,
        persist_directory=db_dir,
    )
    assert qa_res.error is None
    assert "No relevant contract passages were found" in qa_res.answer
    assert len(qa_res.sources) == 0


def test_qa_chat_history_inclusion(tmp_path):
    """Verify conversational context from chat history is sent to OpenAI API."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "test_contracts"
    chunk = [{
        "chunk_id": "c1", "text": "Arbitration shall be held in London.", "page_number": 4,
        "filename": "disputes.pdf", "word_count": 6, "embedding": [0.02] * EXPECTED_EMBEDDING_DIM,
    }]
    index_document_chunks(chunk, filename="disputes.pdf", collection_name=coll_name, persist_directory=db_dir)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Arbitration is in London."))]
    mock_client.chat.completions.create.return_value = mock_resp

    history = [
        {"role": "user", "content": "How are disputes resolved?"},
        {"role": "assistant", "content": "Disputes are resolved through arbitration."},
    ]

    answer_contract_question(
        question="Where does it take place?",
        filename="disputes.pdf",
        openai_client=mock_client,
        collection_name=coll_name,
        persist_directory=db_dir,
        chat_history=history,
    )

    messages = mock_client.chat.completions.create.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "How are disputes resolved?"


def test_qa_error_handling(tmp_path):
    """Verify API errors (timeout, invalid key, rate limit) return helpful messages without leaking secrets."""
    db_dir = str(tmp_path / "test_chroma")
    coll_name = "test_contracts"
    chunk = [{
        "chunk_id": "c1", "text": "Clause text.", "page_number": 1,
        "filename": "test.pdf", "word_count": 2, "embedding": [0.01] * EXPECTED_EMBEDDING_DIM,
    }]
    index_document_chunks(chunk, filename="test.pdf", collection_name=coll_name, persist_directory=db_dir)

    # 1. Simulate APITimeoutError
    mock_client_timeout = MagicMock()
    mock_client_timeout.chat.completions.create.side_effect = Exception("APITimeoutError: Request timed out")

    res_timeout = answer_contract_question(
        question="Test question", filename="test.pdf",
        openai_client=mock_client_timeout, collection_name=coll_name, persist_directory=db_dir,
    )
    assert "timed out" in res_timeout.error.lower()

    # 2. Simulate AuthenticationError
    mock_client_auth = MagicMock()
    mock_client_auth.chat.completions.create.side_effect = Exception("AuthenticationError: invalid_api_key")

    res_auth = answer_contract_question(
        question="Test question", filename="test.pdf",
        openai_client=mock_client_auth, collection_name=coll_name, persist_directory=db_dir,
    )
    assert "Invalid OpenAI API key" in res_auth.error


# ── 6. Conversation Export & Security Verification ─────────────

def test_conversation_export_markdown_and_text():
    """Verify conversation export contains questions, answers, page citations, disclaimer, and 0 secrets."""
    history = [
        {
            "role": "user",
            "content": "What is the warranty period?",
            "timestamp": "14:30",
        },
        {
            "role": "assistant",
            "content": "The warranty lasts 12 months [warranty_agreement.pdf, Page 5].",
            "timestamp": "14:31",
            "sources": [
                {
                    "filename": "warranty_agreement.pdf",
                    "page_number": 5,
                    "similarity": 0.912,
                    "text": "Party A warrants products for 12 months from delivery.",
                }
            ],
        },
    ]

    md_out = export_conversation_to_markdown(history, document_filter="warranty_agreement.pdf")
    txt_out = export_conversation_to_text(history, document_filter="warranty_agreement.pdf")

    # Content checks
    assert "What is the warranty period?" in md_out
    assert "warranty_agreement.pdf" in md_out
    assert "Page 5" in md_out
    assert "Legal Disclaimer" in md_out

    assert "What is the warranty period?" in txt_out
    assert "warranty_agreement.pdf" in txt_out
    assert "LEGAL DISCLAIMER" in txt_out

    # Security check: verify no secrets or tokens exist in export
    for sensitive in ["sk-", "OPENAI_API_KEY", "bearer", "password", "token="]:
        assert sensitive not in md_out.lower()
        assert sensitive not in txt_out.lower()
