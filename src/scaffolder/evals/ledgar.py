"""LEDGAR evaluation: how accurate is LexiChunk's keyword clause classifier?

LEDGAR (as distributed in the LexGLUE benchmark) is ~80k contract provisions
drawn from EDGAR filings, each labelled with one of 100 clause-type classes.
Those 100 classes are collapsed onto LexiChunk's ``ClauseType`` enum through an
explicit, hand-written mapping (``evals/ledgar_label_map.yaml``); LEDGAR labels
with no LexiChunk counterpart are reported as out-of-scope coverage loss rather
than being silently scored.

Two reference points put the keyword classifier's number in context:

* a **majority-class** baseline (predict the most frequent mapped class), and
* a **TF-IDF + logistic regression** classifier trained on LEDGAR's train split.

The second is important: it is a weak, cheap, *supervised* model, so it marks
roughly what "learning the task from the data" buys over hand-written keywords.
"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

from scaffolder.evals.classification import (
    Calibration,
    ClassScore,
    accuracy,
    calibration_curve,
    confusion_pairs,
    macro_f1,
    per_class_scores,
)
from scaffolder.evals.common import (
    DEFAULT_BOOTSTRAP,
    Interval,
    bootstrap_ci,
    cache_dir,
    deterministic_sample,
    markdown_table,
)
from scaffolder.evals.label_map import DEFAULT_MAP_PATH, LabelMap, load_label_map

#: Pinned revision of ``coastalcph/lex_glue`` used for every reported number.
LEDGAR_REVISION = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"

#: Dataset sources tried in order, as ``(dataset_id, config, revision)``. The
#: first that loads is recorded in the results JSON so the run is reproducible.
LEDGAR_SOURCES: tuple[tuple[str, str | None, str | None], ...] = (
    ("coastalcph/lex_glue", "ledgar", LEDGAR_REVISION),
    ("coastalcph/lex_glue", "ledgar", None),
    ("lex_glue", "ledgar", None),
)

#: Position handed to the classifier for isolated provisions. LEDGAR gives no
#: document position, and LexiChunk only applies its end-of-document bonus
#: above 0.75, so 0.5 is a neutral choice (identical in effect to 0.0).
NEUTRAL_POSITION = 0.5


@dataclass(frozen=True, slots=True)
class LedgarItem:
    """One LEDGAR provision after mapping onto a LexiChunk clause type."""

    text: str
    ledgar_label: str
    gold: str


@dataclass(slots=True)
class SystemResult:
    """Scores for one classifier (LexiChunk or a baseline) on the same items."""

    name: str
    accuracy: Interval
    macro_f1: Interval
    per_class: list[ClassScore] = field(default_factory=list)
    confusions: list[tuple[str, str, int]] = field(default_factory=list)
    calibration: Calibration | None = None
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "name": self.name,
            "accuracy": self.accuracy.as_dict(),
            "macro_f1": self.macro_f1.as_dict(),
            "per_class": [s.as_dict() for s in self.per_class],
            "confusions": [
                {"gold": g, "predicted": p, "count": n} for g, p, n in self.confusions
            ],
            "calibration": self.calibration.as_dict() if self.calibration else None,
            "seconds": self.seconds,
        }


@dataclass(slots=True)
class LedgarResult:
    """Everything the LEDGAR evaluation produces."""

    dataset_id: str
    dataset_config: str | None
    dataset_revision: str | None
    split: str
    seed: int
    requested_sample: int
    split_size: int
    evaluated: int
    out_of_scope_items: int
    coverage: float
    mapped_labels: int
    total_labels: int
    label_coverage: float
    class_distribution: dict[str, int]
    systems: list[SystemResult]
    relative_position: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly mapping."""
        return {
            "evaluation": "ledgar",
            "dataset": {
                "id": self.dataset_id,
                "config": self.dataset_config,
                "revision": self.dataset_revision,
                "split": self.split,
                "split_size": self.split_size,
            },
            "sampling": {
                "seed": self.seed,
                "requested": self.requested_sample,
                "evaluated": self.evaluated,
                "out_of_scope_items": self.out_of_scope_items,
                "item_coverage": self.coverage,
            },
            "label_map": {
                "mapped_labels": self.mapped_labels,
                "total_labels": self.total_labels,
                "label_coverage": self.label_coverage,
            },
            "class_distribution": self.class_distribution,
            "relative_position": self.relative_position,
            "systems": [s.as_dict() for s in self.systems],
        }


@dataclass(frozen=True, slots=True)
class LedgarSplit:
    """A loaded LEDGAR split plus the provenance of where it came from."""

    dataset_id: str
    config: str | None
    revision: str | None
    texts: list[str]
    labels: list[str]


def load_ledgar_split(
    split: str,
    *,
    sources: Sequence[tuple[str, str | None, str | None]] = LEDGAR_SOURCES,
) -> LedgarSplit:
    """Load a LEDGAR split, trying each source in turn.

    Returns:
        A :class:`LedgarSplit` whose ``labels`` are gold LEDGAR class *names*
        (not integer ids), aligned with ``texts``.

    Raises:
        RuntimeError: If no source could be loaded; the message lists every
            error encountered so the failure is diagnosable offline.
    """
    from datasets import load_dataset  # imported lazily: heavy, optional

    errors: list[str] = []
    for dataset_id, config, revision in sources:
        try:
            data = load_dataset(
                dataset_id,
                config,
                split=split,
                revision=revision,
                cache_dir=str(cache_dir()),
            )
        except Exception as exc:  # noqa: BLE001 - we genuinely want to try the next source
            errors.append(f"{dataset_id}/{config}@{revision}: {type(exc).__name__}: {exc}")
            continue
        names = data.features["label"].names
        return LedgarSplit(
            dataset_id=dataset_id,
            config=config,
            revision=revision,
            texts=list(data["text"]),
            labels=[names[i] for i in data["label"]],
        )
    joined = "\n  ".join(errors)
    msg = f"could not load LEDGAR from any source:\n  {joined}"
    raise RuntimeError(msg)


def map_items(
    texts: Sequence[str],
    labels: Sequence[str],
    label_map: LabelMap,
) -> tuple[list[LedgarItem], int]:
    """Project LEDGAR rows onto LexiChunk clause types.

    Returns:
        ``(in_scope_items, out_of_scope_count)``.
    """
    items: list[LedgarItem] = []
    dropped = 0
    for text, label in zip(texts, labels, strict=True):
        target = label_map.target(label)
        if target is None:
            dropped += 1
            continue
        items.append(LedgarItem(text=text, ledgar_label=label, gold=target))
    return items, dropped


def predict_lexichunk(
    items: Sequence[LedgarItem],
    *,
    relative_position: float = NEUTRAL_POSITION,
) -> tuple[list[str], list[float], float]:
    """Run LexiChunk's ``ClauseTypeClassifier`` over ``items``.

    Returns:
        ``(predictions, confidences, elapsed_seconds)``.
    """
    from lexichunk.enrichment.clause_type import ClauseTypeClassifier

    classifier = ClauseTypeClassifier()
    preds: list[str] = []
    confs: list[float] = []
    start = time.perf_counter()
    for item in items:
        result = classifier.classify_detailed(item.text, relative_position=relative_position)
        preds.append(result.clause_type.value)
        confs.append(float(result.confidence))
    return preds, confs, time.perf_counter() - start


def predict_majority(items: Sequence[LedgarItem], majority: str) -> list[str]:
    """Predict a single class for every item."""
    return [majority] * len(items)


def train_tfidf_baseline(
    train_items: Sequence[LedgarItem],
    test_items: Sequence[LedgarItem],
    *,
    seed: int = 0,
) -> tuple[list[str], float]:
    """Train TF-IDF + multinomial logistic regression on LEDGAR's train split.

    Returns:
        ``(predictions_for_test_items, elapsed_seconds)``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    start = time.perf_counter()
    pipeline = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000),
        LogisticRegression(max_iter=1000, random_state=seed, n_jobs=1),
    )
    pipeline.fit([i.text for i in train_items], [i.gold for i in train_items])
    preds = [str(p) for p in pipeline.predict([i.text for i in test_items])]
    return preds, time.perf_counter() - start


def score_system(
    name: str,
    items: Sequence[LedgarItem],
    preds: Sequence[str],
    *,
    labels: Sequence[str],
    seed: int,
    n_boot: int,
    confidences: Sequence[float] | None = None,
    seconds: float = 0.0,
) -> SystemResult:
    """Compute accuracy, macro-F1 (both with bootstrap CIs) and diagnostics."""
    gold = [i.gold for i in items]
    paired = list(zip(gold, preds, strict=True))

    acc = bootstrap_ci(
        paired,
        lambda rows: accuracy([t for t, _ in rows], [p for _, p in rows]),
        n_boot=n_boot,
        seed=seed,
    )
    f1 = bootstrap_ci(
        paired,
        lambda rows: macro_f1([t for t, _ in rows], [p for _, p in rows], labels),
        n_boot=n_boot,
        seed=seed + 1,
    )
    calibration = None
    if confidences is not None:
        correct = [t == p for t, p in paired]
        calibration = calibration_curve(confidences, correct)

    return SystemResult(
        name=name,
        accuracy=acc,
        macro_f1=f1,
        per_class=per_class_scores(gold, preds, labels),
        confusions=confusion_pairs(gold, preds),
        calibration=calibration,
        seconds=seconds,
    )


def run_ledgar(
    *,
    sample: int = 5000,
    seed: int = 0,
    split: str = "test",
    label_map_path: str | Path = DEFAULT_MAP_PATH,
    n_boot: int = DEFAULT_BOOTSTRAP,
    relative_position: float = NEUTRAL_POSITION,
    train_sample: int = 20000,
    skip_tfidf: bool = False,
    provider: Callable[[str], LedgarSplit] | None = None,
) -> LedgarResult:
    """Run the full LEDGAR evaluation.

    Args:
        sample: Number of test provisions to score (0 = the whole split).
            Sampling happens *before* the out-of-scope filter, so the scored
            count is smaller than ``sample`` by the out-of-scope rate.
        seed: Seed for sampling and for the bootstrap.
        split: LEDGAR split to score.
        label_map_path: Path to the LEDGAR -> LexiChunk mapping.
        n_boot: Bootstrap resamples per interval.
        relative_position: Position hint passed to the classifier.
        train_sample: Cap on training rows for the TF-IDF baseline.
        skip_tfidf: Skip the supervised baseline (much faster).
        provider: Override for split loading, keyed by split name. Used by
            ``--smoke`` and by the tests to run the whole pipeline on synthetic
            rows without touching the network.

    Returns:
        A :class:`LedgarResult`.
    """
    label_map = load_label_map(label_map_path)
    load = provider if provider is not None else load_ledgar_split
    loaded = load(split)
    split_size = len(loaded.texts)

    indices = deterministic_sample(list(range(split_size)), sample, seed)
    sampled_texts = [loaded.texts[i] for i in indices]
    sampled_labels = [loaded.labels[i] for i in indices]

    items, dropped = map_items(sampled_texts, sampled_labels, label_map)
    if not items:
        msg = "no in-scope LEDGAR items after mapping; check evals/ledgar_label_map.yaml"
        raise RuntimeError(msg)

    total_sampled = len(sampled_texts)
    distribution = Counter(i.gold for i in items)
    scored_labels = sorted(distribution)

    systems: list[SystemResult] = []

    preds, confs, elapsed = predict_lexichunk(items, relative_position=relative_position)
    systems.append(
        score_system(
            "lexichunk-keyword",
            items,
            preds,
            labels=scored_labels,
            seed=seed,
            n_boot=n_boot,
            confidences=confs,
            seconds=elapsed,
        )
    )

    majority = distribution.most_common(1)[0][0]
    systems.append(
        score_system(
            f"majority ({majority})",
            items,
            predict_majority(items, majority),
            labels=scored_labels,
            seed=seed,
            n_boot=n_boot,
        )
    )

    if not skip_tfidf:
        train = load("train")
        train_idx = deterministic_sample(list(range(len(train.texts))), train_sample, seed)
        train_items, _ = map_items(
            [train.texts[i] for i in train_idx],
            [train.labels[i] for i in train_idx],
            label_map,
        )
        tfidf_preds, tfidf_seconds = train_tfidf_baseline(train_items, items, seed=seed)
        systems.append(
            score_system(
                "tfidf+logreg (supervised)",
                items,
                tfidf_preds,
                labels=scored_labels,
                seed=seed,
                n_boot=n_boot,
                seconds=tfidf_seconds,
            )
        )

    return LedgarResult(
        dataset_id=loaded.dataset_id,
        dataset_config=loaded.config,
        dataset_revision=loaded.revision,
        split=split,
        seed=seed,
        requested_sample=sample,
        split_size=split_size,
        evaluated=len(items),
        out_of_scope_items=dropped,
        coverage=len(items) / total_sampled if total_sampled else 0.0,
        mapped_labels=len(label_map.in_scope_labels),
        total_labels=len(label_map.entries),
        label_coverage=(
            len(label_map.in_scope_labels) / len(label_map.entries) if label_map.entries else 0.0
        ),
        class_distribution=dict(sorted(distribution.items())),
        systems=systems,
        relative_position=relative_position,
    )


def render_markdown(result: LedgarResult, *, top_classes: int = 15) -> str:
    """Render the LEDGAR result as markdown tables."""
    lines: list[str] = ["## LEDGAR: clause-type classification accuracy", ""]
    lines.append(
        f"Dataset `{result.dataset_id}`"
        + (f" (config `{result.dataset_config}`)" if result.dataset_config else "")
        + (f" at revision `{result.dataset_revision}`" if result.dataset_revision else "")
        + f", split `{result.split}` ({result.split_size:,} rows). "
        f"Sampled {result.requested_sample or result.split_size:,} rows with seed "
        f"{result.seed}; {result.evaluated:,} were in scope "
        f"({result.coverage * 100:.1f}% item coverage, "
        f"{result.out_of_scope_items:,} dropped as out of scope). "
        f"{result.mapped_labels}/{result.total_labels} LEDGAR labels map onto a "
        f"LexiChunk clause type ({result.label_coverage * 100:.0f}% label coverage), "
        f"spanning {len(result.class_distribution)} distinct LexiChunk classes."
    )
    lines += ["", "### Headline", ""]
    lines.append(
        markdown_table(
            ["System", "Accuracy (95% CI)", "Macro-F1 (95% CI)", "Seconds"],
            [
                [s.name, s.accuracy.format(), s.macro_f1.format(), f"{s.seconds:.1f}"]
                for s in result.systems
            ],
        )
    )

    primary = result.systems[0]
    lines += ["", f"### Per-class scores — {primary.name}", ""]
    lines.append(
        markdown_table(
            ["LexiChunk class", "Support", "Predicted", "Precision", "Recall", "F1"],
            [
                [
                    s.label,
                    f"{s.support:,}",
                    f"{s.predicted:,}",
                    f"{s.precision * 100:.1f}%",
                    f"{s.recall * 100:.1f}%",
                    f"{s.f1 * 100:.1f}%",
                ]
                for s in primary.per_class
            ],
        )
    )

    total_errors = round(result.evaluated * (1.0 - primary.accuracy.point))
    lines += ["", f"### Dominant confusions — {primary.name}", ""]
    lines.append(
        markdown_table(
            ["Gold", "Predicted", "Count", "% of all errors"],
            [
                [g, p, f"{n:,}", f"{n / max(total_errors, 1) * 100:.1f}%"]
                for g, p, n in primary.confusions[:top_classes]
            ],
        )
    )
    lines += [
        "",
        f"({total_errors:,} misclassifications in total; the table lists the "
        f"{min(top_classes, len(primary.confusions))} most frequent pairs.)",
    ]

    if primary.calibration is not None:
        cal = primary.calibration
        lines += ["", f"### Confidence calibration — {primary.name}", ""]
        lines.append(
            markdown_table(
                ["Confidence bin", "n", "Mean confidence", "Accuracy"],
                [
                    [
                        f"[{b.low:.1f}, {b.high:.1f})",
                        f"{b.count:,}",
                        f"{b.mean_confidence:.3f}",
                        f"{b.accuracy * 100:.1f}%",
                    ]
                    for b in cal.bins
                ],
            )
        )
        lines += [
            "",
            f"Expected calibration error: {cal.expected_calibration_error:.3f}. "
            f"Accuracy is {'monotonically non-decreasing' if cal.monotonic else 'NOT monotonic'} "
            f"across bins. Spearman rank correlation between confidence and correctness: "
            f"{cal.spearman:.3f}.",
        ]
    return "\n".join(lines) + "\n"
