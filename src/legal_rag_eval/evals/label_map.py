"""Loading and validation of the LEDGAR -> LexiChunk clause-type mapping."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

#: Sentinel used in the YAML for LEDGAR labels with no LexiChunk counterpart.
OUT_OF_SCOPE = "__out_of_scope__"

#: Repository-relative location of the shipped mapping.
DEFAULT_MAP_PATH = Path("evals") / "ledgar_label_map.yaml"


@dataclass(frozen=True, slots=True)
class MappingEntry:
    """One LEDGAR label and the LexiChunk clause type it maps onto."""

    ledgar_label: str
    lexichunk: str
    rationale: str

    @property
    def in_scope(self) -> bool:
        """True when this LEDGAR label has a LexiChunk counterpart."""
        return self.lexichunk != OUT_OF_SCOPE


@dataclass(frozen=True, slots=True)
class LabelMap:
    """A validated many-to-one LEDGAR -> LexiChunk clause-type mapping."""

    entries: dict[str, MappingEntry]
    version: int
    source_dataset: str

    def get(self, ledgar_label: str) -> MappingEntry | None:
        """Return the entry for ``ledgar_label``, or None if unmapped."""
        return self.entries.get(ledgar_label)

    def target(self, ledgar_label: str) -> str | None:
        """Return the LexiChunk clause type, or None when out of scope/absent."""
        entry = self.entries.get(ledgar_label)
        if entry is None or not entry.in_scope:
            return None
        return entry.lexichunk

    @property
    def in_scope_labels(self) -> set[str]:
        """LEDGAR labels that map onto a LexiChunk clause type."""
        return {k for k, v in self.entries.items() if v.in_scope}

    @property
    def targets(self) -> set[str]:
        """The distinct LexiChunk clause types the mapping can produce."""
        return {v.lexichunk for v in self.entries.values() if v.in_scope}


def valid_clause_types() -> set[str]:
    """Return the set of LexiChunk ``ClauseType`` values, as strings.

    Imported lazily so this module (and the mapping tests) stay importable in
    environments where LexiChunk is not installed.
    """
    from lexichunk.models import ClauseType

    return {member.value for member in ClauseType}


def load_label_map(
    path: str | Path = DEFAULT_MAP_PATH,
    *,
    known_clause_types: set[str] | None = None,
) -> LabelMap:
    """Load and validate the mapping YAML.

    Args:
        path: Path to the mapping file.
        known_clause_types: Override the set of accepted LexiChunk clause
            types. Defaults to the values of :class:`lexichunk.models.ClauseType`.

    Returns:
        A validated :class:`LabelMap`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the document is malformed, an entry lacks a rationale,
            or a target is not a known LexiChunk clause type.
    """
    map_path = Path(path)
    if not map_path.is_file():
        msg = f"label map not found: {map_path}"
        raise FileNotFoundError(msg)
    with open(map_path, encoding="utf-8") as handle:
        raw: Any = yaml.safe_load(handle)
    return parse_label_map(raw, known_clause_types=known_clause_types, origin=str(map_path))


def parse_label_map(
    raw: Any,
    *,
    known_clause_types: set[str] | None = None,
    origin: str = "<memory>",
) -> LabelMap:
    """Validate an already-parsed mapping document.

    Split out from :func:`load_label_map` so tests can exercise validation on
    in-memory dictionaries without touching the filesystem.
    """
    if not isinstance(raw, dict):
        msg = f"{origin}: top level must be a mapping, got {type(raw).__name__}"
        raise ValueError(msg)

    allowed = known_clause_types if known_clause_types is not None else valid_clause_types()
    mappings = raw.get("mappings")
    if not isinstance(mappings, dict) or not mappings:
        msg = f"{origin}: 'mappings' must be a non-empty mapping"
        raise ValueError(msg)

    entries: dict[str, MappingEntry] = {}
    for label, value in mappings.items():
        if not isinstance(value, dict):
            msg = f"{origin}: entry {label!r} must be a mapping with 'lexichunk' and 'rationale'"
            raise ValueError(msg)
        target = value.get("lexichunk")
        rationale = value.get("rationale")
        if not isinstance(target, str) or not target:
            msg = f"{origin}: entry {label!r} has no 'lexichunk' target"
            raise ValueError(msg)
        if not isinstance(rationale, str) or not rationale.strip():
            msg = f"{origin}: entry {label!r} has no 'rationale'"
            raise ValueError(msg)
        if target != OUT_OF_SCOPE and target not in allowed:
            msg = (
                f"{origin}: entry {label!r} maps to {target!r}, which is not a "
                f"LexiChunk ClauseType (known: {sorted(allowed)})"
            )
            raise ValueError(msg)
        if str(label) in entries:
            msg = f"{origin}: duplicate entry for {label!r}"
            raise ValueError(msg)
        entries[str(label)] = MappingEntry(
            ledgar_label=str(label), lexichunk=target, rationale=rationale.strip()
        )

    version = raw.get("version", 1)
    if not isinstance(version, int):
        msg = f"{origin}: 'version' must be an integer"
        raise ValueError(msg)
    dataset = raw.get("dataset", "")
    if not isinstance(dataset, str):
        msg = f"{origin}: 'dataset' must be a string"
        raise ValueError(msg)

    return LabelMap(entries=entries, version=version, source_dataset=dataset)
