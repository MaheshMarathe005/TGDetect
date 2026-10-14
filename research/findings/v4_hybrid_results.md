# Completed V4/V3 hybrid results

Five epochs, seed 0, one NVIDIA L4. Run status is complete; the best checkpoint was epoch 5. Full invocation including GPU acceptance and final evaluation: 1,004.6 s (16.74 min). Training epochs sum to 881.34 s (14.69 min). No containers remained running at this check.

| Model | Precision (%) | Recall (%) | F1 (%) | AP (%) | TP | FP | Node accuracy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 | 0.4194 | 35.29 | 0.8289 | 0.7797 | 24 | 5,699 | 97.86 |
| V4 | 0.2151 | 45.59 | 0.4283 | 0.7661 | 31 | 14,378 | 94.62 |
| V4 hybrid | 0.2570 | 41.18 | 0.5107 | 0.7377 | 28 | 10,869 | 95.93 |
| ORTHRUS, 5 epochs | 0.1752 | 25.00 | 0.3479 | 0.2339 | 17 | 9,687 | 96.37 |
| ORTHRUS, 20 epochs | 0.2856 | 32.35 | 0.5662 | 0.1515 | 22 | 7,681 | 97.12 |

Against original V4, median epoch time improves 4.27x, false positives decrease 24.4%, and F1 increases 19.3%. Recall falls from 31/68 to 28/68; AP decreases 3.7%. The hybrid therefore improves speed and the selected threshold's precision/F1, but does not improve all detection metrics. V3 still has better F1/AP and fewer false positives.

| Model | Median epoch (s) | Peak CUDA allocated (GiB) | Parameters |
|---|---:|---:|---:|
| V3 | 182.4 | 3.47 | 785,327 |
| V4 | 755.9 | 12.54 | 560,897 |
| V4 hybrid | 176.8 | 12.47 | 802,003 |
| ORTHRUS, 5 epochs | 58.8 | 3.74 | 764,202 |
| ORTHRUS, 20 epochs | 57.3 | 3.74 | 764,202 |

V4 timing above uses only fully compiled epochs 2–5; epoch 1 includes the interrupted interpreted run and is excluded. These are recorded runs, not a simultaneous controlled benchmark. The hybrid has similar training time to V3, but its retained full 256D episodic rings leave memory near V4 levels. The parameter count is 802,003; the 764,202 ORTHRUS parameter parity target is not met. No separate hybrid inference-latency benchmark was run.

Same CADETS population: 268,153 test nodes, 68 malicious; four ground-truth nodes have no scored test events. V3 and hybrid share the exact prepared manifest and benign-only pretrained 64D table digests. The hybrid retains V4 temporal/episodic components and restores V3 encoder/affine node update and admission. Encoder and memory were changed together, so individual accuracy effects cannot be attributed.

ORTHRUS uses the matched calibrated node protocol, not its published top-30 alert rule. Its 20-epoch arm has a larger training budget. These single-seed, retrospective development comparisons do not establish superiority under the published ORTHRUS protocol. Node accuracy is dominated by benign examples; it is not sufficient evidence of detection quality.

Confusion-matrix arithmetic and population sizes were checked. AP/AUC are saved evaluation outputs, not recomputed from edge scores here. No new cloud training or GPU evaluation was launched for this result check.

Recommendation: keep V3 as the current stronger detection reference. For a future hybrid iteration, investigate episodic representation/admission and validation calibration with separate ablations. Do not assume replacing its full rings automatically improves accuracy. Raw source paths and computed changes are in `artifacts/sliding_v4/hybrid_completed_comparison.json`.
