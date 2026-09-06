"""Tests for dependency boundaries between core reporting and the dashboard."""

import subprocess
import sys
from importlib.metadata import requires
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_plotly_is_a_core_dependency() -> None:
    """The standard HTML report must be installable without the dashboard extra."""
    dependencies = requires("legal_rag_eval") or []

    assert any(
        dependency.startswith("plotly") and "extra ==" not in dependency
        for dependency in dependencies
    )


def test_reporting_package_imports_without_dashboard_dependency() -> None:
    """Importing reporting must not require Streamlit or other dashboard packages."""
    result = subprocess.run(
        [sys.executable, "-c", "import legal_rag_eval.reporting"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
