"""External evaluations of LexiChunk against public legal datasets.

Two evaluations live here, both runnable from the command line:

* ``python -m scaffolder.evals ledgar`` — clause-type classification accuracy
  on LEDGAR (LexGLUE), against a majority-class and a supervised baseline.
* ``python -m scaffolder.evals cuad`` — gold answer-span containment on CUAD,
  against RecursiveCharacterTextSplitter and sentence-window baselines.

This package is intentionally independent of ``scaffolder.metrics``,
``scaffolder.queries`` and ``scaffolder.reporting``.
"""

from __future__ import annotations

from scaffolder.evals.common import DEFAULT_RESULTS_DIR, Interval, bootstrap_ci, markdown_table
from scaffolder.evals.label_map import OUT_OF_SCOPE, LabelMap, load_label_map

__all__ = [
    "DEFAULT_RESULTS_DIR",
    "OUT_OF_SCOPE",
    "Interval",
    "LabelMap",
    "bootstrap_ci",
    "load_label_map",
    "markdown_table",
]
