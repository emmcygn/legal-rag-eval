"""Keep changed metric semantics visible across serialization and reports."""

import json

from legal_rag_eval.metrics.retrieval import compute_retrieval_metrics
from legal_rag_eval.models import (
    AnnotatedQuery,
    BenchmarkResult,
    EmbeddingModelName,
    Jurisdiction,
    RetrievalResult,
    StrategyName,
)
from legal_rag_eval.reporting.json_export import export_json_string, reconstruct_benchmark_result


def test_metric_version_survives_roundtrip_and_legacy_is_not_relabelled() -> None:
    query = AnnotatedQuery("control", "question", ["doc"], Jurisdiction.UK, (), "control")
    retrieval = RetrievalResult(query, StrategyName.LEXICHUNK, EmbeddingModelName.MINILM, (), 0, 0)
    metrics = compute_retrieval_metrics(retrieval)
    result = BenchmarkResult("test", retrieval_metrics=[metrics])
    data = json.loads(export_json_string(result))
    assert data["retrieval_metrics"][0]["ndcg_metric"] == "evidence_assignment_ndcg_v1"
    restored = reconstruct_benchmark_result(data)
    assert restored.retrieval_metrics[0].ndcg_metric == "evidence_assignment_ndcg_v1"
    del data["retrieval_metrics"][0]["ndcg_metric"]
    legacy = reconstruct_benchmark_result(data)
    assert legacy.retrieval_metrics[0].ndcg_metric == "legacy_unversioned"
