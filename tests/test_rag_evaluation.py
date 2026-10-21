# ============================================================
# tests/test_rag_evaluation.py  –  Pytest RAG Evaluation Test
# ============================================================
# Automated test that executes the quantitative RAG benchmark
# and asserts that Recall@K, Citation Accuracy, and Insufficient
# Evidence detection meet deterministic thresholds.
# ============================================================

import os
import sys
import pytest

# Add root directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.evaluation.rag_evaluator import run_rag_evaluation, print_evaluation_report


@pytest.mark.integration
def test_rag_evaluation_benchmarks():
    """
    Execute quantitative RAG evaluation and verify recall, citation correctness,
    and out-of-domain insufficient evidence detection.
    """
    metrics = run_rag_evaluation(top_k=3)
    print_evaluation_report(metrics)

    # 1. Retrieval Recall@3 on synthetic dataset must be >= 80% (typically 100%)
    assert metrics["recall_at_k"] >= 80.0, f"Recall@3 too low: {metrics['recall_at_k']}%"

    # 2. Citation Accuracy must be >= 80%
    assert metrics["citation_accuracy"] >= 80.0, f"Citation accuracy too low: {metrics['citation_accuracy']}%"

    # 3. Grounded Fact Coverage must be >= 80%
    assert metrics["fact_coverage_rate"] >= 80.0, f"Fact coverage rate too low: {metrics['fact_coverage_rate']}%"

    # 4. Out-of-Domain / Insufficient Evidence must be 100%
    assert metrics["insufficient_evidence_rate"] == 100.0, (
        f"Insufficient evidence handling failed: {metrics['insufficient_evidence_rate']}%"
    )
