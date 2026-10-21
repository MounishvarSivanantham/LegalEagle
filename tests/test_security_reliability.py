# ============================================================
# tests/test_security_reliability.py  –  Security & Reliability Tests
# ============================================================
# Step 7 Requirements C & D:
#   • Security: API key storage, .gitignore, PDF validation,
#     untrusted input handling, document isolation, zero secret leakage.
#   • Reliability: Missing keys, timeouts, rate limits, empty queries,
#     vector DB resilience, safe error messages.
# ============================================================

import os
import sys
import re
from unittest.mock import MagicMock, patch

import pytest
import pymupdf

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.pdf_parser import extract_text_from_pdf, DEFAULT_MAX_PDF_SIZE_BYTES
from services.vector_store import (
    index_document_chunks,
    search_contract_chunks,
    delete_document_vectors,
    get_collection_stats,
    EXPECTED_EMBEDDING_DIM,
)
from services.qa import (
    answer_contract_question,
    get_openai_api_key,
    export_conversation_to_markdown,
    export_conversation_to_text,
    SYSTEM_PROMPT,
)


@pytest.fixture(autouse=True)
def mock_embedding_generation():
    """Mock single embedding so tests run fast without model downloads."""
    def _fake_embed(text: str):
        if not text or not text.strip():
            return None
        val = (hash(text) % 100) / 1000.0
        return [val] * EXPECTED_EMBEDDING_DIM

    with patch("services.embeddings.generate_single_embedding", side_effect=_fake_embed):
        yield



# ── 1. Security Tests ──────────────────────────────────────────

def test_gitignore_protects_secrets():
    """Verify .gitignore includes .env and secret files."""
    gitignore_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".gitignore")
    assert os.path.exists(gitignore_path), ".gitignore file must exist in project root"

    with open(gitignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert ".env" in content
    assert "*.secret" in content or ".secret" in content
    assert "secrets.toml" in content


def test_no_hardcoded_api_keys_in_codebase():
    """Verify no live API keys are hardcoded in application source code files."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    services_dir = os.path.join(root_dir, "services")

    api_key_regex = re.compile(r'sk-[a-zA-Z0-9_\-]{32,}')

    for folder in [services_dir, os.path.join(root_dir, "app.py")]:
        if os.path.isfile(folder):
            files = [folder]
        else:
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".py")]

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                matches = api_key_regex.findall(content)
                # Ensure no live API key patterns exist in python source code
                assert not matches, f"Potential hardcoded API key found in {file_path}: {matches}"


def test_pdf_size_limit_security():
    """Verify that uploaded files exceeding maximum size limit are safely rejected."""
    # 30 MB simulated payload
    fake_large_payload = b"%PDF-1.4\n" + b"A" * (DEFAULT_MAX_PDF_SIZE_BYTES + 1024)
    result = extract_text_from_pdf(fake_large_payload, filename="overflow.pdf")

    assert result.error is not None
    assert "exceeds maximum allowed limit" in result.error
    assert result.page_count == 0


def test_untrusted_input_prompt_hardening():
    """Verify system prompt explicitly instructs model to treat contract text as untrusted data."""
    assert "untrusted" in SYSTEM_PROMPT.lower()
    assert "never invent" in SYSTEM_PROMPT.lower() or "do not assume" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower()


def test_zero_secret_leakage_in_exports():
    """Verify conversation exports never include API keys or secrets."""
    mock_history = [
        {"role": "user", "content": "What are the payment terms?", "timestamp": "10:00"},
        {
            "role": "assistant",
            "content": "Payment is Net 30 [contract.pdf, Page 2].",
            "timestamp": "10:01",
            "sources": [{"filename": "contract.pdf", "page_number": 2, "text": "Net 30 terms apply."}],
        },
    ]

    md = export_conversation_to_markdown(mock_history, document_filter="contract.pdf")
    txt = export_conversation_to_text(mock_history, document_filter="contract.pdf")

    for secret_marker in ["sk-", "OPENAI_API_KEY", "bearer", "secret", "password"]:
        assert secret_marker not in md.lower()
        assert secret_marker not in txt.lower()


# ── 2. Reliability Tests ───────────────────────────────────────

def test_empty_query_and_whitespace_reliability(tmp_path):
    """Verify empty or whitespace-only queries are handled gracefully without crashing."""
    db_dir = str(tmp_path / "chroma")
    res1, err1 = search_contract_chunks("", persist_directory=db_dir)
    assert res1 == []
    assert err1 == "Query is empty."

    res2, err2 = search_contract_chunks("    \n\t  ", persist_directory=db_dir)
    assert res2 == []
    assert err2 == "Query is empty."


def test_missing_api_key_reliability(tmp_path):
    """Verify question answering cleanly reports missing API key without raising unhandled exceptions."""
    db_dir = str(tmp_path / "chroma")
    chunk = [{
        "chunk_id": "c1", "text": "Some text", "page_number": 1,
        "filename": "f.pdf", "word_count": 2, "embedding": [0.01] * EXPECTED_EMBEDDING_DIM
    }]
    index_document_chunks(chunk, filename="f.pdf", persist_directory=db_dir)

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
        qa_res = answer_contract_question(
            question="What is the payment fee?",
            filename="f.pdf",
            openai_client=None,
            persist_directory=db_dir,
        )
        assert qa_res.error is not None
        assert "API key not found" in qa_res.error


def test_api_timeout_reliability(tmp_path):
    """Verify OpenAI timeouts return a polite user-facing error message."""
    db_dir = str(tmp_path / "chroma")
    chunk = [{
        "chunk_id": "c1", "text": "Some text", "page_number": 1,
        "filename": "f.pdf", "word_count": 2, "embedding": [0.01] * EXPECTED_EMBEDDING_DIM
    }]
    index_document_chunks(chunk, filename="f.pdf", persist_directory=db_dir)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("APITimeoutError: Request timed out after 30s")

    qa_res = answer_contract_question(
        question="What is the payment fee?",
        filename="f.pdf",
        openai_client=mock_client,
        persist_directory=db_dir,
    )
    assert qa_res.error is not None
    assert "timed out" in qa_res.error.lower()


def test_rate_limit_reliability(tmp_path):
    """Verify OpenAI quota / rate limit errors return a clean user-facing error message."""
    db_dir = str(tmp_path / "chroma")
    chunk = [{
        "chunk_id": "c1", "text": "Some text", "page_number": 1,
        "filename": "f.pdf", "word_count": 2, "embedding": [0.01] * EXPECTED_EMBEDDING_DIM
    }]
    index_document_chunks(chunk, filename="f.pdf", persist_directory=db_dir)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("RateLimitError: You exceeded your current quota")

    qa_res = answer_contract_question(
        question="What is the payment fee?",
        filename="f.pdf",
        openai_client=mock_client,
        persist_directory=db_dir,
    )
    assert qa_res.error is not None
    assert "rate limit or quota exceeded" in qa_res.error.lower()
