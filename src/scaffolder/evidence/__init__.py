"""Anchored-evidence benchmark interfaces."""

from scaffolder.evidence.benchmark import (
    EvidenceScore,
    SelectedSpan,
    score_selected_spans,
)
from scaffolder.evidence.schema import DatasetError, EvidenceDataset, load_dataset

__all__ = [
    "DatasetError",
    "EvidenceDataset",
    "EvidenceScore",
    "SelectedSpan",
    "load_dataset",
    "score_selected_spans",
]
