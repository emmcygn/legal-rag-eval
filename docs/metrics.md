# Metrics

The authoritative benchmark metrics are defined once in `scaffolder.evidence.benchmark` and consumed unchanged by the CLI JSON report and dashboard viewer.

## Source-span Evidence Recall

For answerable queries: unioned evidence characters covered by selected source spans divided by all unioned evidence characters.

## Source-span Evidence Precision

For answerable queries: unioned evidence characters covered divided by all selected context characters. Duplicate or overlapping selected spans remain fully charged.

## Abstention Accuracy

For unanswerable queries only: one when retrieval selects no context, otherwise zero. It records context emission, not correctness of a legal answer.

## Custom Legacy Assignment Metric

`evidence_assignment_ndcg_v1` is retained for regression compatibility. It performs one-to-one evidence assignment with ordered discounts and is not classical NDCG. It is not used by the primary anchored-evidence benchmark.

Legacy circular structural scores are diagnostic only and are not authoritative evidence.
