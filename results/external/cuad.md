## CUAD: gold-span containment under different chunkers

Dataset `theatticusproject/cuad-qa` at revision `refs/convert/parquet`, split `train+test`; 100 of 510 contracts sampled with seed 0, carrying 2,458 verified gold answer spans (mean 242 chars, median 182). Mean contract length 47,432 chars.

### Containment

| Strategy          | Containment (95% CI) | Mean chunk chars | Chunks/contract | Length-matched grid | Lift    | s/contract |
| ----------------- | -------------------- | ---------------- | --------------- | ------------------- | ------- | ---------- |
| lexichunk-512tok  | 98.1% [97.5, 98.6]   | 1,306            | 36.3            | 81.8%               | +16.3pp | 0.714      |
| lexichunk-1024tok | 98.9% [98.5, 99.3]   | 2,044            | 23.2            | 88.2%               | +10.7pp | 0.670      |
| lexichunk-128tok  | 83.7% [82.2, 85.2]   | 346              | 137.1           | 47.0%               | +36.7pp | 0.656      |
| rcts-512          | 72.1% [70.2, 74.0]   | 382              | 123.0           | 49.9%               | +22.2pp | 0.007      |
| rcts-1024         | 89.3% [88.1, 90.6]   | 744              | 63.4            | 69.5%               | +19.9pp | 0.005      |
| sentence-window-3 | 98.9% [98.5, 99.3]   | 647              | 109.3           | 65.6%               | +33.3pp | 0.001      |

`Length-matched grid` is the containment a fixed-stride splitter with the same mean chunk length would get on these spans; `Lift` is the strategy's containment minus that. Lift, not raw containment, is the number that reflects boundary quality.

### How many chunks a gold span is split across

| Strategy          | 1     | 2     | 3    | 4+   | Mean |
| ----------------- | ----- | ----- | ---- | ---- | ---- |
| lexichunk-512tok  | 98.1% | 1.6%  | 0.2% | 0.0% | 1.02 |
| lexichunk-1024tok | 98.9% | 0.9%  | 0.1% | 0.0% | 1.01 |
| lexichunk-128tok  | 83.7% | 12.0% | 2.4% | 2.0% | 1.25 |
| rcts-512          | 72.1% | 24.4% | 2.6% | 0.9% | 1.33 |
| rcts-1024         | 89.3% | 10.0% | 0.4% | 0.2% | 1.12 |
| sentence-window-3 | 54.7% | 44.4% | 0.8% | 0.1% | 1.47 |

### LexiChunk structure recall on CUAD

| Metric                                 | Value      |
| -------------------------------------- | ---------- |
| Contracts with >= 5 TOP-LEVEL clauses  | 14 (14.0%) |
| Contracts with >= 5 nodes at ANY level | 52 (52.0%) |
| Fell back to flat text                 | 48 (48.0%) |
| Mean / median top-level nodes          | 2.8 / 1    |
| Mean / median total nodes              | 22.8 / 6   |
| Contracts yielding a single chunk      | 5          |
| Exceptions during parsing              | 0          |

Same contracts, both jurisdiction profiles (a gap means the corpus's numbering style is recognised by one profile and not the other):

| Measure                 | Value |
| ----------------------- | ----- |
| us_parse_rate_top_level | 14.0% |
| uk_parse_rate_top_level | 31.0% |
| us_mean_top_level_nodes | 2.8   |
| uk_mean_top_level_nodes | 4.8   |

### Failures

No exceptions were raised on any sampled contract.
