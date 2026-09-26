# Temporal CNN study

Verified completed runs: 40/40. Five seeds and 200 epochs per arm.

Q = local history GRU; A = additional attention; C = causal 1D temporal CNN. All arms include rehearsal.

| Model | Seeds | F1 mean ± SD (%) | PR-AP mean ± SD (%) | Parameters |
|---|---:|---:|---:|---:|
| GNN_H | 5 | 86.61 ± 4.84 | 88.18 ± 5.30 | 35458 |
| TGNN | 5 | 84.52 ± 2.63 | 94.22 ± 0.49 | 35714 |
| TGNN_Q | 5 | 85.45 ± 1.38 | 93.64 ± 0.66 | 134786 |
| TGNN_Q_A | 5 | 86.39 ± 3.27 | 94.33 ± 0.40 | 183938 |
| CNN_H | 5 | 97.11 ± 1.16 | 96.49 ± 0.22 | 100226 |
| TGNN_C | 5 | 85.51 ± 4.19 | 92.52 ± 3.00 | 134274 |
| TGNN_Q_C | 5 | 87.07 ± 4.81 | 93.97 ± 0.77 | 233346 |
| TGNN_Q_A_C | 5 | 87.16 ± 6.41 | 93.15 ± 2.77 | 282498 |

Previously inspected test with 25 positives; new-capture confirmation required.

Full metrics, thresholds, losses, prediction and checkpoint hashes are in per-run JSONs. GPU telemetry and concurrency calibration are saved separately.
