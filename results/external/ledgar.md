## LEDGAR: clause-type classification accuracy

Dataset `coastalcph/lex_glue` (config `ledgar`) at revision `c23fdff1a6bf74e0e1a71cb86f1e781d37da888c`, split `test` (10,000 rows). Sampled 5,000 rows with seed 0; 3,955 were in scope (79.1% item coverage, 1,045 dropped as out of scope). 67/100 LEDGAR labels map onto a LexiChunk clause type (67% label coverage), spanning 19 distinct LexiChunk classes.

### Headline

| System                    | Accuracy (95% CI)  | Macro-F1 (95% CI)  | Seconds |
| ------------------------- | ------------------ | ------------------ | ------- |
| lexichunk-keyword         | 39.6% [38.0, 41.2] | 42.5% [40.7, 44.1] | 0.7     |
| majority (boilerplate)    | 21.6% [20.3, 22.8] | 1.9% [1.8, 2.0]    | 0.0     |
| tfidf+logreg (supervised) | 90.8% [89.9, 91.7] | 89.9% [88.5, 91.0] | 21.5    |

### Per-class scores — lexichunk-keyword

| LexiChunk class       | Support | Predicted | Precision | Recall | F1    |
| --------------------- | ------- | --------- | --------- | ------ | ----- |
| boilerplate           | 854     | 214       | 79.9%     | 20.0%  | 32.0% |
| representations       | 591     | 128       | 50.0%     | 10.8%  | 17.8% |
| payment               | 379     | 395       | 54.7%     | 57.0%  | 55.8% |
| governing_law         | 316     | 594       | 42.8%     | 80.4%  | 55.8% |
| covenants             | 251     | 320       | 8.1%      | 10.4%  | 9.1%  |
| entire_agreement      | 232     | 169       | 91.7%     | 66.8%  | 77.3% |
| notices               | 226     | 217       | 73.3%     | 70.4%  | 71.8% |
| severability          | 200     | 113       | 93.8%     | 53.0%  | 67.7% |
| assignment            | 152     | 205       | 57.6%     | 77.6%  | 66.1% |
| dispute_resolution    | 145     | 42        | 26.2%     | 7.6%   | 11.8% |
| amendment             | 133     | 354       | 24.0%     | 63.9%  | 34.9% |
| definitions           | 88      | 81        | 12.3%     | 11.4%  | 11.8% |
| indemnification       | 82      | 59        | 79.7%     | 57.3%  | 66.7% |
| confidentiality       | 74      | 76        | 52.6%     | 54.1%  | 53.3% |
| insurance             | 65      | 51        | 84.3%     | 66.2%  | 74.1% |
| warranties            | 53      | 34        | 2.9%      | 1.9%   | 2.3%  |
| termination           | 50      | 126       | 26.2%     | 66.0%  | 37.5% |
| conditions            | 38      | 10        | 20.0%     | 5.3%   | 8.3%  |
| intellectual_property | 26      | 68        | 36.8%     | 96.2%  | 53.2% |

### Dominant confusions — lexichunk-keyword

| Gold               | Predicted     | Count | % of all errors |
| ------------------ | ------------- | ----- | --------------- |
| representations    | unknown       | 192   | 8.0%            |
| boilerplate        | covenants     | 153   | 6.4%            |
| boilerplate        | amendment     | 146   | 6.1%            |
| boilerplate        | unknown       | 125   | 5.2%            |
| covenants          | unknown       | 83    | 3.5%            |
| dispute_resolution | governing_law | 80    | 3.3%            |
| representations    | governing_law | 67    | 2.8%            |
| boilerplate        | payment       | 60    | 2.5%            |
| payment            | unknown       | 52    | 2.2%            |
| covenants          | governing_law | 51    | 2.1%            |
| representations    | payment       | 50    | 2.1%            |
| representations    | amendment     | 47    | 2.0%            |
| definitions        | unknown       | 46    | 1.9%            |
| governing_law      | unknown       | 44    | 1.8%            |
| boilerplate        | definitions   | 41    | 1.7%            |

(2,389 misclassifications in total; the table lists the 15 most frequent pairs.)

### Confidence calibration — lexichunk-keyword

| Confidence bin | n     | Mean confidence | Accuracy |
| -------------- | ----- | --------------- | -------- |
| [0.0, 0.1)     | 675   | 0.006           | 0.7%     |
| [0.1, 0.2)     | 307   | 0.137           | 27.4%    |
| [0.2, 0.3)     | 1,202 | 0.246           | 49.2%    |
| [0.3, 0.4)     | 578   | 0.339           | 24.7%    |
| [0.4, 0.5)     | 150   | 0.445           | 49.3%    |
| [0.5, 0.6)     | 586   | 0.513           | 53.4%    |
| [0.6, 0.7)     | 65    | 0.649           | 67.7%    |
| [0.7, 0.8)     | 199   | 0.747           | 78.9%    |
| [0.8, 0.9)     | 68    | 0.822           | 72.1%    |
| [0.9, 1.0)     | 125   | 1.000           | 84.8%    |

Expected calibration error: 0.113. Accuracy is NOT monotonic across bins. Spearman rank correlation between confidence and correctness: 0.375.
