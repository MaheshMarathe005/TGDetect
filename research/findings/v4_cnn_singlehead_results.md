# Completed causal CNN + single-head hybrid

Status complete, five epochs, seed 0, NVIDIA L4, selected checkpoint epoch 5. The durable Modal result exactly matches the local result. Invocation including GPU acceptance, training, evaluation and paired timing: 1,079.04 s (17.98 min). Training alone: 936.10 s (15.60 min).

| Model | Precision (%) | Recall (%) | F1 (%) | AP (%) | TP | FP | Node accuracy (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| V3 | 0.4194 | 35.29 | 0.8289 | 0.7797 | 24 | 5,699 | 97.86 |
| Previous hybrid | 0.2570 | 41.18 | 0.5107 | 0.7377 | 28 | 10,869 | 95.93 |
| CNN + single head | 0.3238 | 41.18 | 0.6426 | 1.3651 | 28 | 8,619 | 96.77 |
| ORTHRUS, 5 epochs | 0.1752 | 25.00 | 0.3479 | 0.2339 | 17 | 9,687 | 96.37 |

Against the previous hybrid: same 28 detected malicious nodes, 20.7% fewer false positives, 25.8% higher F1 and 85.0% higher AP. Against V3: 75.1% higher AP and higher recall, but 22.5% lower F1 and 51.2% more false positives. This is a ranking improvement and a different operating-point tradeoff, not superiority on every metric.

## Paired trained-checkpoint inference timing

Both arms ran in the same L4 container on identical chronological train-prefix batches, using frozen FP32 encoder tables. Three repetitions, alternating arm order; eight warmup and 32 measured batches per repetition, batch 1,024. Values below are medians of the three repetition statistics, not pooled percentiles.

| Model | Batch p50 (ms) | Batch p95 (ms) | Batch p99 (ms) | Throughput (events/s) |
|---|---:|---:|---:|---:|
| V3 | 22.193 | 26.681 | 29.684 | 45,214 |
| CNN_single_head | 21.362 | 23.966 | 27.204 | 46,617 |

Median batch latency is 3.74% lower than V3. The three paired p50 observations (V3/new, ms) were 23.738/20.964, 22.193/22.163 and 22.065/21.362. One pair is almost tied; the gain is modest and should not be represented as a large or statistically established acceleration. Tail estimates from 32 measured batches per repetition are provisional.

Timing includes prepared host-input transfer, forward and state updates, and excludes preprocessing, cache preparation and batch-formation wait. Prefix timing does not establish full-dataset mature-state latency or arrival-to-alert latency. Raw batches and checkpoint hashes are saved in `latency_vs_v3` in the result JSON.

## Training and memory

| Model | Median epoch (s) | Full-run peak CUDA allocated (GiB) | Parameters |
|---|---:|---:|---:|
| V3 | 182.4 | 3.47 | 785,327 |
| Previous hybrid | 176.8 | 12.47 | 802,003 |
| CNN + single head | 186.8 | 12.47 | 819,232 |
| ORTHRUS, 5 epochs | 58.8 | 3.74 | 764,202 |

New-model training is about 2.45% slower per median epoch than the recorded V3 run, and 5.65% slower than the preceding hybrid. These training times are separate-run observations. Full-run CUDA allocation is approximately 3.59x V3, because full 256D episodic rings remain. The paired short-prefix allocated peaks were approximately 0.276 GiB (V3) and 0.837 GiB (new); do not substitute these for full-run memory. Parameter count exceeds both V3 and the 764,202 ORTHRUS parity target.

## Interpretation and limits

This is the strongest observed AP among the variants listed here, with a modest short-prefix inference gain over V3. V3 still wins selected-threshold F1, false positives, training time and memory. Retain both as a measured tradeoff rather than declaring the new model universally better.

Same CADETS population: 268,153 nodes and 68 malicious; four truth nodes have no scored test events. All main rows are five-epoch seed-zero runs. ORTHRUS uses our matched calibrated-node protocol, not its released top-30 alert rule. These are retrospective single-seed development results; CNN and head count changed together, so their individual effects remain unisolated. AP/AUC are saved pipeline outputs, not recomputed from raw edge scores during this check. No new paid experiment was launched.

Next accuracy priority is validation-only calibration/false-positive analysis; changing a threshold cannot improve AP. Next memory priority is an explicit episodic-width ablation. Keep any additional experiments separately bounded and preserve the test set limitations.

Raw result: `artifacts/sliding_v4/V4_CNN_SingleHead_s0_e5.json`. Derived changes: `artifacts/sliding_v4/cnn_singlehead_summary.json`.
