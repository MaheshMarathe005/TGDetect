# Complete component results — 60 verified L4 runs

Completed 24 September; consolidated here 26 September 2026. Each model used
the same data/split, optimizer, 200 epochs, seeds 0–4, rehearsal and threshold rule.
All values below are percentages, mean ± sample SD. H denotes rehearsal;
TGNN = GNN + time + H. Attention means an **additional** attention readout.

| Model | Test F1 | Test PR-AP |
|---|---:|---:|
| GNN + H | **86.61 ± 4.84** | 88.18 ± 5.30 |
| TGNN | 84.52 ± 2.63 | 94.22 ± 0.49 |
| TGNN + GRU | 85.45 ± 1.38 | 93.64 ± 0.66 |
| TGNN + memory | 83.86 ± 4.01 | **94.77 ± 0.48** |
| TGNN + GRU + memory | 85.52 ± 2.57 | 94.67 ± 0.95 |
| TGNN + attention | 83.03 ± 3.08 | 92.88 ± 1.59 |
| TGNN + GRU + attention | **86.39 ± 3.27** | 94.33 ± 0.40 |
| TGNN + memory + attention | 76.72 ± 6.29 | 94.29 ± 0.98 |
| TGNN + GRU + memory + attention | 84.15 ± 4.03 | 94.27 ± 0.96 |
| TGNN + normalization | 82.26 ± 4.89 | 90.87 ± 4.03 |
| TGNN + weighted loss | 85.26 ± 2.94 | 94.64 ± 0.74 |
| GNN + H + attention | 80.90 ± 8.23 | 84.70 ± 4.39 |

The best observed TGNN improves plain TGNN mean F1 by 1.87 points and remains
0.23 points below GNN. This tiny observed gap does not establish equivalence,
superiority, or a cause such as sample size. Five random seeds use the same test
events; they do not provide five independent test datasets.

Nine configurations tie at 100% validation F1. The automatic summary selects the
first equal maximum, plain TGNN; it is not a unique validation winner. The best
test TGNN is an exploratory comparison, not independently confirmed selection.

GRU in this grid processes four historical neighbors with zero initial state
per target event. It is distinct from a GRU that updates persistent endpoint state.
Persistent memory is the optional fixed-EMA M factor. Rehearsal is training replay,
a third and separate mechanism. See [memory analysis](memory_and_data_scale.md).

All 60 checkpoints load with the registered architecture and finite weights;
source/data hashes, predictions, labels, thresholds, per-seed metrics, aggregate
statistics and paired differences passed independent verification. The ten
reference reruns reproduce original weights and probabilities bit for bit.

- [Full per-seed CSV](component_runs.csv)
- [Aggregate metrics and paired differences](component_summary.json)
- [Detailed original report](../../artifacts/components_200e_v1/TGNN_COMPONENT_REPORT.md)
- [Independent verification](../../artifacts/components_200e_v1/local_verification.json)
- [Resource measurements and interpretation](../../TGNN_COMPONENT_FINDINGS.md)
- [Frozen reference](../../REFERENCE_RESULTS.json)

Summary SHA256: `1be3420aae31cd555189f82d92456563e1bf8a04bed7e63202462381badc70db`.
