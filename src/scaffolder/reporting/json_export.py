"""JSON export for benchmark results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scaffolder.models import BenchmarkResult


def _serialize(obj: Any) -> Any:
    """Custom serializer for dataclass fields."""
    if isinstance(obj, Path):
        return str(obj)
    # Import here to avoid circular imports at module level
    from datetime import datetime as dt_cls

    if isinstance(obj, dt_cls):
        return obj.isoformat()
    if isinstance(obj, set):
        return sorted(obj)
    # scipy/numpy return numpy scalars (e.g. numpy.bool_ for `p_value < alpha`),
    # which json cannot encode. Unwrap them to their Python equivalents.
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def export_json(
    result: BenchmarkResult,
    output_path: str | Path,
    indent: int = 2,
) -> Path:
    """Export benchmark results to a JSON file.

    Args:
        result: The complete benchmark result to export.
        output_path: Path to write the JSON file. Parent dirs are created.
        indent: JSON indentation level.

    Returns:
        The Path where the file was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = asdict(result)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=indent, default=_serialize)

    return output_path


def export_json_string(result: BenchmarkResult, indent: int = 2) -> str:
    """Export benchmark results as a JSON string.

    Useful for Streamlit download buttons and API responses.
    """
    data = asdict(result)
    return json.dumps(data, indent=indent, default=_serialize)


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a previously exported benchmark result from JSON.

    Returns the raw dict — caller is responsible for reconstructing
    dataclass instances if needed.
    """
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]
