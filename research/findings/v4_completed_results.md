# Completed V4 run: verified results

Five training epochs, seed 0, NVIDIA L4. Status `complete`; checkpoint selected at epoch 5. The detached recovery attempt took 3,356.76 seconds, including verification/recovery and final evaluation; this excludes earlier interrupted training. No containers remained active at verification.

Comparison uses the existing CADETS test population of 268,153 nodes and 68 malicious nodes. Four truth nodes have no scored test events. ORTHRUS here uses matched calibrated node evidence, not the paper's released top-30 decision rule. The 20-epoch arm has a different training budget and is shown separately.

| Model | Precision (%) | Recall (%) | F1 (%) | AP (%) | TP | FP | Node accuracy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 (5 epochs) | 0.4194 | 35.29 | 0.8289 | 0.7797 | 24 | 5,699 | 97.86 |
| V4 (5 epochs) | 0.2151 | 45.59 | 0.4283 | 0.7661 | 31 | 14,378 | 94.62 |
| ORTHRUS (5 epochs) | 0.1752 | 25.00 | 0.3479 | 0.2339 | 17 | 9,687 | 96.37 |
| ORTHRUS (20 epochs) | 0.2856 | 32.35 | 0.5662 | 0.1515 | 22 | 7,681 | 97.12 |

V4 improves recall over V3 (31 vs 24 detected malicious nodes), but false positives grow from 5,699 to 14,378. F1 falls by approximately 48.3%; AP falls by approximately 1.74%. It therefore does not improve the overall V3 detection/efficiency tradeoff in this run. It exceeds the five-epoch ORTHRUS baseline on F1/AP, but its F1 is below the 20-epoch ORTHRUS result. These are single-seed observations, not statistical superiority or results under the published ORTHRUS alert protocol.

Accuracy is dominated by benign nodes: an all-benign predictor would achieve approximately 99.975% node accuracy here. Accuracy alone cannot support the detection claim.

| Model | Parameters | Median epoch (s) | Peak allocated CUDA memory (GiB) | Final scoring pipeline (s) |
|---|---:|---:|---:|---:|
| V3 (5 epochs) | 785,327 | 182.4 | 3.47 | 58.4 |
| V4 (5 epochs) | 560,897 | 758.1 | 12.54 | 196.0 |
| ORTHRUS (5 epochs) | 764,202 | 58.8 | 3.74 | 32.6 |
| ORTHRUS (20 epochs) | 764,202 | 57.3 | 3.74 | 32.4 |

V4 has 28.6% fewer parameters than V3, but approximately 4.16x its median epoch time and 3.62x its recorded peak CUDA allocation. V4 epoch 1 combines interrupted interpreted execution and compiled continuation; fully compiled epochs 2–5 took 745.8, 753.7, 758.1 and 806.8 seconds. Scoring includes validation/test evaluation and calibration; these timings are not isolated batch inference latency, and runs were not a simultaneous controlled timing experiment.

The prior 3.70x acceleration applies to compiled versus interpreted V4 on a paired short-prefix L4 training benchmark. It does not mean V4 trains faster than V3.

V4 final validation CE: 0.02803057; test CE: 0.04694959. The smaller checkpoint-selection CE (0.02594608) is from the validation selection prefix. Neither is attack F1.

This result corresponds to the original running source snapshot. The later local last-position attention optimization and optional CPU prefetch were not deployed into this run. Their new L4 gains remain unmeasured.

Next priority: analyze false positives and validation-to-test score drift using saved edge scores, and evaluate any calibration change on validation only before a new held-out comparison. Keep V3 as the stronger current reference; retain V4 as an experimental architecture. No additional training was launched for this check.

Sources: each path is recorded in `artifacts/sliding_v4/completed_comparison.json`; raw V4 result is `artifacts/sliding_v4/fast_resume.json`. Confusion-matrix-derived precision, recall and F1 and common population counts were checked arithmetically. AP/AUC are saved pipeline outputs; recomputation from raw score files was not performed in this check.
