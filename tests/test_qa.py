# ============================================================
# tests/test_qa.py  –  Test LLM Question Answering (Mocked & Unit)
# ============================================================
# Test suite for Stage 5:
#   1. Successful answers with Mocked OpenAI client
#   2. Empty retrieval handling (no LLM call wasted)
#   3. Missing API key error handling
#   4. API errors (AuthenticationError, RateLimitError, ConnectionError)
#   5. Empty question validation
#   6. Prompt formatting and citation preservation
#
# Run with:
#     .venv\Scripts\python.exe tests/test_qa.py
# ============================================================

import os
import shutil
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.qa import (
    answer_contract_question,
    format_context_for_prompt,
    get_openai_api_key,
    QAResult,
    DEFAULT_OPENAI_MODEL,
)
from services.vector_store import index_document_chunks, clear_collection, EXPECTED_EMBEDDING_DIM

TEST_PERSIST_DIR = "tests/test_chroma_db_qa"
TEST_COLLECTION = "test_qa_contracts"


def divider(title: str):
    print(f"\n{'=' * 60}")
    print(f"  TEST: {title}")
    print(f"{'=' * 60}")


def cleanup_test_db():
    if os.path.exists(TEST_PERSIST_DIR):
        try:
            shutil.rmtree(TEST_PERSIST_DIR)
        except Exception:
            pass


def create_mock_openai_response(text: str):
    """Helper to build a mock OpenAI chat completion response."""
    choice = MagicMock()
    choice.message.content = text
    response = MagicMock()
    response.choices = [choice]
    return response


def setup_sample_vector_data():
    """Populate a small test vector collection with sample contract clauses."""
    cleanup_test_db()
    chunks = [
        {
            "chunk_id": "service_p1_c1",
            "text": "SERVICE AGREEMENT between Acme Corp and Global Services LLC dated Jan 1 2025.",
            "page_number": 1,
            "filename": "service_agreement.pdf",
            "embedding": [0.05] * EXPECTED_EMBEDDING_DIM,
            "word_count": 12,
        },
        {
            "chunk_id": "service_p2_c1",
            "text": "COMPENSATION: Party A shall pay Party B a monthly fee of $15,000 within 30 days of invoice.",
            "page_number": 2,
            "filename": "service_agreement.pdf",
            "embedding": [0.10] * EXPECTED_EMBEDDING_DIM,
            "word_count": 17,
        },
    ]
    index_document_chunks(
        chunks=chunks,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )


def test_empty_question_validation():
    divider("Empty question validation")

    res = answer_contract_question("", collection_name=TEST_COLLECTION, persist_directory=TEST_PERSIST_DIR)
    assert res.error is not None
    assert "enter a question" in res.error.lower()

    res_spaces = answer_contract_question("   \n\t  ", collection_name=TEST_COLLECTION, persist_directory=TEST_PERSIST_DIR)
    assert res_spaces.error is not None
    print("  [OK] Empty questions rejected properly.")
    return True


def test_empty_retrieval_handling():
    divider("Empty retrieval handling (empty database)")
    cleanup_test_db()  # Empty database

    # Test with empty vector store
    res = answer_contract_question(
        question="What is the monthly fee?",
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"  Empty retrieval answer: {res.answer}")
    assert res.error is None
    assert "No relevant contract passages" in res.answer
    assert res.sources == []
    print("  [OK] Empty database returned informative message without error.")
    return True


def test_missing_api_key_handling():
    divider("Missing API key handling")
    setup_sample_vector_data()

    # Clear only OPENAI_API_KEY (keeping USERPROFILE/system env intact)
    orig_key = os.environ.get("OPENAI_API_KEY")
    try:
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        res = answer_contract_question(
            question="What is the payment amount?",
            collection_name=TEST_COLLECTION,
            persist_directory=TEST_PERSIST_DIR,
            openai_client=None,
        )
        print(f"  Missing key error: {res.error}")
        assert res.error is not None
        assert "OpenAI API key not found" in res.error
        # Sources should still be populated from Step 4 retrieval!
        assert len(res.sources) > 0
        print("  [OK] Missing API key cleanly reported with retrieved sources preserved.")
    finally:
        if orig_key is not None:
            os.environ["OPENAI_API_KEY"] = orig_key
    return True


def test_mocked_successful_answer():
    divider("Successful question answering (Mocked OpenAI SDK)")
    setup_sample_vector_data()

    mock_answer = (
        "Under Section 2 of the agreement [service_agreement.pdf, Page 2], "
        "Party A must pay Party B a monthly fee of $15,000 for consulting services, "
        "payable within 30 days of receiving the invoice."
    )

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = create_mock_openai_response(mock_answer)

    res = answer_contract_question(
        question="What is the monthly consulting fee and payment schedule?",
        filename="service_agreement.pdf",
        top_k=2,
        openai_client=mock_client,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )

    print(f"  Question: {res.question}")
    print(f"  Answer:   {res.answer}")
    print(f"  Sources:  {len(res.sources)} passages retrieved")

    assert res.error is None
    assert "$15,000" in res.answer
    assert "[service_agreement.pdf, Page 2]" in res.answer
    assert len(res.sources) > 0
    assert res.sources[0]["page_number"] in [1, 2]
    assert "Disclaimer:" in res.answer

    # Verify mock was called with system prompt and retrieved context
    call_args = mock_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]
    assert len(messages) == 2
    assert "LegalEagle" in messages[0]["content"]
    assert "service_agreement.pdf" in messages[1]["content"]

    print("  [OK] Mocked OpenAI response grounded, cited, and returned accurately.")
    return True


def test_api_errors_handling():
    divider("API error handling (AuthenticationError, RateLimitError, APIConnectionError)")
    setup_sample_vector_data()

    # 1. Authentication Error
    mock_client_auth = MagicMock()
    from openai import AuthenticationError
    mock_client_auth.chat.completions.create.side_effect = AuthenticationError(
        message="Incorrect API key provided",
        response=MagicMock(status_code=401),
        body={"error": {"message": "Invalid API key"}},
    )

    res_auth = answer_contract_question(
        question="What are the terms?",
        openai_client=mock_client_auth,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"  Auth error handling: {res_auth.error}")
    assert "Invalid OpenAI API key" in res_auth.error

    # 2. Rate Limit Error
    mock_client_rate = MagicMock()
    from openai import RateLimitError
    mock_client_rate.chat.completions.create.side_effect = RateLimitError(
        message="You exceeded your current quota",
        response=MagicMock(status_code=429),
        body={"error": {"message": "Rate limit reached"}},
    )

    res_rate = answer_contract_question(
        question="What are the terms?",
        openai_client=mock_client_rate,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"  Rate limit handling: {res_rate.error}")
    assert "rate limit or quota" in res_rate.error.lower()

    # 3. Connection Error
    mock_client_conn = MagicMock()
    from openai import APIConnectionError
    mock_client_conn.chat.completions.create.side_effect = APIConnectionError(
        message="Connection refused",
        request=MagicMock(),
    )

    res_conn = answer_contract_question(
        question="What are the terms?",
        openai_client=mock_client_conn,
        collection_name=TEST_COLLECTION,
        persist_directory=TEST_PERSIST_DIR,
    )
    print(f"  Connection error handling: {res_conn.error}")
    assert "network connection" in res_conn.error.lower()

    print("  [OK] All OpenAI exceptions caught and translated to user-friendly messages.")
    return True


def test_prompt_formatting_and_citations():
    divider("Prompt context formatting & citation integrity")

    sample_passages = [
        {
            "filename": "lease_contract.pdf",
            "page_number": 3,
            "chunk_id": "lease_p3_c1",
            "text": "Tenant shall keep premises in good repair and condition.",
        },
        {
            "filename": "lease_contract.pdf",
            "page_number": 7,
            "chunk_id": "lease_p7_c1",
            "text": "Security deposit of $5,000 refundable within 14 days.",
        },
    ]

    formatted = format_context_for_prompt(sample_passages)
    print("  Formatted Prompt Block:")
    for line in formatted.splitlines()[:5]:
        print(f"    {line}")

    assert "Document: lease_contract.pdf" in formatted
    assert "Page: 3" in formatted
    assert "Page: 7" in formatted
    assert "Tenant shall keep premises" in formatted
    assert "Security deposit of $5,000" in formatted

    cleanup_test_db()
    print("  [OK] Prompt format includes exact filenames, page citations, and chunk IDs.")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("  LEGALEAGLE STAGE 5 TEST SUITE: LLM QUESTION ANSWERING")
    print("=" * 60)

    t1 = test_empty_question_validation()
    t2 = test_empty_retrieval_handling()
    t3 = test_missing_api_key_handling()
    t4 = test_mocked_successful_answer()
    t5 = test_api_errors_handling()
    t6 = test_prompt_formatting_and_citations()

    if all([t1, t2, t3, t4, t5, t6]):
        print(f"\n{'=' * 60}")
        print("  ALL 6/6 STAGE 5 TESTS PASSED!")
        print(f"{'=' * 60}")
    else:
        print("\n[FAIL] Some tests failed.")
        sys.exit(1)
