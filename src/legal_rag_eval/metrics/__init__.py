"""Quality metrics.

``gold`` holds the structural metrics that are reported: they score a chunk set against
hand-checked annotations in ``gold/`` by character-span overlap. ``structural`` holds the
superseded metrics that derived their ground truth from LexiChunk itself; they are exported
under ``legacy_`` names and are not part of the default report.
"""

from legal_rag_eval.metrics.gold import compute_gold_structural_metrics
from legal_rag_eval.metrics.retrieval import compute_retrieval_metrics
from legal_rag_eval.metrics.structural import compute_legacy_structural_metrics

__all__ = [
    "compute_gold_structural_metrics",
    "compute_legacy_structural_metrics",
    "compute_retrieval_metrics",
]
