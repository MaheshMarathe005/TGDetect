# Controlled TGNN pilot — completed 23 September 2026

> Follow-up: the previously missing matched `GNN+H` versus `GNN+T+H` comparison
> has now completed for 200 epochs × five seeds on one Modal L4. Its authoritative
> result is in `MATCHED_TIME_REHEARSAL_REPORT.md`. The three-epoch results below
> remain the broader exploratory component pilot.

All 40 neural runs (8 configurations × 5 seeds), 10 feature-control fits,
and 50 fixed-checkpoint interventions completed. This is an explicitly
adapted CTU-13 experiment, not a reproduction of the Wazuh paper or a
test of the complete legacy TG-Detect model.

## Data and protocol

Official CTU-13 scenario 6: 558,919 raw flows; 16,384 label-independently
sampled flows sorted by completion-time proxy. Train/validation/test:
9,830/3,277/3,277 records, with 91/25/25 Botnet labels. Background is
treated as an uncertain negative. Ten static features; train-only scaling;
same graph, labels, optimizer, three epochs and seeds 0–4. Thresholds
maximize validation F1 and are fixed for test. See [protocol](ADAPTED_PROTOCOL.md)
for operator, replay, source-selection amendments and state scheduling.

Scenario 11 and the smaller scenario 6 sample failed recorded class-support
gates before real-data training. The final pilot retains only 25 positive
test flows: one missed attack changes recall by four percentage points.

## Test results

Values are percentages, mean ± sample standard deviation across five
initialization seeds. PR-AP is average precision, not trapezoidal PR area.
Seed variability is not uncertainty over captures or independent attacks.
The deterministic feature controls repeat identical fits across seeds.

| Model | F1 | PR-AP | ROC-AUC | FPR | Parameters |
|---|---:|---:|---:|---:|---:|
| Adapted GNN | 83.80 ± 3.43 | 88.33 ± 3.07 | 98.68 ± 0.35 | 0.09 ± 0.03 | 35458 |
| GNN + time | 89.84 ± 3.99 | 93.96 ± 0.54 | 98.57 ± 0.13 | 0.04 ± 0.01 | 35714 |
| GNN + fixed memory | 77.78 ± 1.62 | 90.25 ± 1.95 | 99.77 ± 0.12 | 0.06 ± 0.03 | 35458 |
| GNN + GRU memory | 75.52 ± 3.13 | 86.11 ± 6.54 | 99.80 ± 0.14 | 0.06 ± 0.02 | 134530 |
| GNN + time + fixed memory | 86.91 ± 2.46 | 91.28 ± 2.72 | 99.30 ± 0.11 | 0.03 ± 0.00 | 35714 |
| Pilot TGNN (time + GRU memory) | 84.33 ± 4.07 | 90.34 ± 2.88 | 99.46 ± 0.12 | 0.03 ± 0.00 | 134786 |
| GNN + rehearsal | 81.05 ± 5.81 | 88.28 ± 3.46 | 98.60 ± 0.31 | 0.14 ± 0.06 | 35458 |
| Pilot TGNN + rehearsal | 84.37 ± 3.22 | 92.37 ± 1.57 | 99.43 ± 0.15 | 0.03 ± 0.00 | 134786 |
| Logistic regression | 83.64 ± 0.00 | 80.93 ± 0.00 | 96.00 ± 0.00 | 0.22 ± 0.00 | — |
| Gradient-boosted trees | 97.96 ± 0.00 | 96.06 ± 0.00 | 98.75 ± 0.00 | 0.00 ± 0.00 | — |

## Paired component effects

Differences in percentage points, paired by training seed. These are
pilot effect estimates; the full factorial study remains incomplete.

| Contrast | ΔF1 | ΔPR-AP |
|---|---:|---:|
| B_T minus B | 6.04 ± 3.60 | 5.64 ± 3.16 |
| B_M minus B | -6.02 ± 2.53 | 1.92 ± 3.74 |
| B_H minus B | -2.75 ± 3.02 | -0.04 ± 3.08 |
| B_TMR minus B | 0.53 ± 3.20 | 2.01 ± 5.40 |
| T_M interaction | 3.10 ± 3.31 | -4.61 ± 2.72 |
| conditional R given M | -2.26 ± 3.58 | -4.14 ± 7.25 |
| TGNN H interaction | 2.80 ± 7.68 | 2.08 ± 4.78 |
| remove T from pilot TGNN | -8.81 ± 6.34 | -4.23 ± 6.82 |
| remove R from pilot TGNN | 2.58 ± 2.58 | 0.94 ± 1.03 |

## Fixed-checkpoint falsification

Weights and original validation thresholds stay fixed. Reset/frozen/no-op
all produce zero state from a fresh start here, so they are related checks,
not three independent confirmations. Shuffled age preserves graph/order
and tests only the numeric-time channel.

| Parent | Intervention | F1 | PR-AP |
|---|---|---:|---:|
| B_TMR | reset_memory | 25.22 ± 35.27 | 93.56 ± 1.05 |
| B_TMR | frozen_memory | 25.22 ± 35.27 | 93.56 ± 1.05 |
| B_TMR | gru_noop | 25.22 ± 35.27 | 93.56 ± 1.05 |
| B_TMR | constant_time | 0.00 ± 0.00 | 53.83 ± 10.71 |
| B_TMR | shuffled_ages | 1.54 ± 3.44 | 76.96 ± 7.89 |
| B_TMRH | reset_memory | 60.17 ± 38.29 | 90.05 ± 3.87 |
| B_TMRH | frozen_memory | 60.17 ± 38.29 | 90.05 ± 3.87 |
| B_TMRH | gru_noop | 60.17 ± 38.29 | 90.05 ± 3.87 |
| B_TMRH | constant_time | 0.00 ± 0.00 | 52.30 ± 8.20 |
| B_TMRH | shuffled_ages | 1.54 ± 3.44 | 65.81 ± 4.51 |

## Resource measurements

One CPU thread in ~/main_env; no GPU. Inference includes rebuilding
history across all partitions. Latency is amortized batch throughput,
not a measured individual online request. Tensor state bytes exclude
Python objects. Process RSS is cumulative, not comparable per-model peak.

| Model | Train seconds | Inference events/s | State tensors KiB | Replay KiB | Checkpoint KiB |
|---|---:|---:|---:|---:|---:|
| B | 1.38 | 116468 | 0.0 | 0.0 | 143.2 |
| B_T | 1.47 | 101627 | 0.0 | 0.0 | 145.4 |
| B_M | 4.34 | 21999 | 2986.0 | 0.0 | 143.9 |
| B_MR | 15.82 | 6911 | 2986.0 | 0.0 | 532.0 |
| B_TM | 4.49 | 21655 | 2986.0 | 0.0 | 145.5 |
| B_TMR | 15.81 | 6877 | 2986.0 | 0.0 | 533.6 |
| B_H | 1.71 | 119040 | 0.0 | 499.2 | 143.9 |
| B_TMRH | 16.40 | 6840 | 2986.0 | 499.2 | 533.6 |

The live training budget is 29,490 event visits per run. Rehearsal adds
3,016 replay visits. Same epochs do not mean equal wall-clock compute.
The TGNN has more parameters; capacity-matched and compute-matched controls
remain outstanding. Full 200-epoch/multi-capture training belongs on Modal.

## Interpretation and limits

The time-only model improved F1 by 6.04 percentage points on average
(paired seed SD 3.60), with a positive difference in every seed. Adding
fixed memory reduced F1 by 6.02 points. The pilot TGNN changed F1 by
only +0.53 points (paired SD 3.20), with mixed signs across seeds.
The feature-only tree had the highest measured F1 and AP. These results
do not support selecting the larger TGNN for this task.

For the TGNN without rehearsal, resetting memory reduces F1 from
84.33% to 25.22% at its original threshold, yet AP increases from
90.34% to 93.56%. This is evidence of a score/threshold shift, not proof
that memory improves ranking. Constant or shuffled ages reduce AP too,
showing sensitivity to the numeric-time channel in this fitted model.
Interventions cause distribution shift and cannot replace retrained controls.

Persistent memory is not automatically useful merely because it learns
on a toy stream. Added time parameters and learned offsets still require
matched capacity/constant-time training controls. No result establishes
a general TGNN advantage, cross-family robustness, APT attribution, or
reduced catastrophic forgetting.

The synthetic learning check is separate: across five seeds, the static
model averaged 49.84% accuracy, GRU memory 100%, and resetting that memory
48.59%. Both toy graphs excluded history, and current query inputs were
identical. This verifies the implementation can learn the designed signal.

## Verification and artifacts

- Correctness suite: 21 passed; 4 legacy expected failures remain.
- All stored validation thresholds and test/intervention metrics were recomputed from saved predictions.
- Source, raw/processed data and all 40 checkpoint hashes match their manifests.
- Matrix: 40 completed seed rows; 200 explicitly unrun rows; all 28 initial-factor pairs represented.
- Checkpoints, predictions, logs, runtime details, module profiles and JSON summaries: artifacts/causal_pilot_v1/.
- Authoritative research state and next experiment: ASTRA_RESEARCH_LEDGER.md.
