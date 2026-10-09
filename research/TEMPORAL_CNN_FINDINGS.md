# Completed temporal-CNN study — 26 September 2026

All **40/40 runs completed** on one NVIDIA L4: eight model variants, five seeds,
200 epochs per seed. The Modal job used 16 independent workers after measured
throughput calibration. The latest full-run telemetry had 661 one-second samples:
mean GPU utilization **95.99%**, maximum 100%, mean device memory 4,903.9 MiB,
maximum 6,319 MiB, and mean power 57.22 W. VRAM was not artificially filled;
the GPU was kept busy through concurrent useful training work.

The completed study uses the same verified CTU scenario-6 sample, split, features,
causal four-neighbor windows, Adam settings, rehearsal rule, 200 epochs, five seeds
and validation-only threshold rule as the previous study. The new CNN is causal:
two 128-channel Conv1d layers, kernel 3, dilations 1 and 2, left padding only,
ReLU and dropout. The five available event positions are ordered from oldest to
newest, with the current event at the final position. The first native-Conv1d
calibration was excluded after a slow benchmark; the final study uses a mathematically
identical tap-gather/GEMM backend. CPU and L4 tests verified outputs and all gradients
against native Conv1d.

## Results

Percentages are mean ± sample SD over seeds 0–4. All models include rehearsal.
`Q` is the existing four-neighbor GRU, `A` is the existing additional attention,
and `C` is the new causal temporal CNN.

| Model | Test F1 | Test PR-AP | Parameters |
|---|---:|---:|---:|
| GNN + rehearsal | 86.61 ± 4.84 | 88.18 ± 5.30 | 35,458 |
| TGNN | 84.52 ± 2.63 | 94.22 ± 0.49 | 35,714 |
| TGNN + GRU | 85.45 ± 1.38 | 93.64 ± 0.66 | 134,786 |
| TGNN + GRU + attention | 86.39 ± 3.27 | 94.33 ± 0.40 | 183,938 |
| **CNN + rehearsal** | **97.11 ± 1.16** | **96.49 ± 0.22** | 100,226 |
| TGNN + CNN | 85.51 ± 4.19 | 92.52 ± 3.00 | 134,274 |
| **TGNN + GRU + CNN** | **87.07 ± 4.81** | 93.97 ± 0.77 | 233,346 |
| **TGNN + GRU + attention + CNN** | **87.16 ± 6.41** | 93.15 ± 2.77 | 282,498 |

The best mean F1 in this study is standalone CNN + rehearsal at **97.11 ± 1.16%**.
It is 10.49 points above the GNN reference and 12.59 points above plain TGNN.
The three CNN hybrids improve their corresponding prior models by:

- TGNN + CNN vs TGNN: **+0.99 F1 points**, paired SD 5.36;
- TGNN + GRU + CNN vs TGNN + GRU: **+1.62 points**, paired SD 6.00;
- TGNN + GRU + attention + CNN vs TGNN + GRU + attention: **+0.77 points**, paired SD 7.36.

These hybrid improvements are descriptive. Their paired seed differences are wide,
and the best hybrid is only about 0.55 points above the previous GNN reference.

## What the result means

The standalone CNN result is a real finding under this frozen protocol, but it does
not demonstrate that CNN is universally better than GNN. It says that, on this
capture and this five-position causal window, a small ordered convolution over the
available flow features learned a highly effective decision rule.

The CNN has more capacity than the GNN: 100,226 versus 35,458 parameters. It also
receives the current flow features and the same four historical graph-window rows.
The result therefore combines sequence processing with a different parameterization;
it is not a pure “CNN versus GNN with equal capacity” test. A matched-capacity static
MLP/1D-convolution control and a dense-data repeat are needed before claiming a
general architecture advantage.

The CNN hybrids do not show a dramatic additive gain. This suggests the CNN and the
existing GNN/GRU may be extracting overlapping local information. Combining modules
can also make optimization harder and increases parameter count; adding more modules
does not guarantee improvement.

The standalone CNN is not a persistent long-term memory model. It sees only the
five rows in each causal window. It does not carry a learned endpoint state from one
event to the next. Therefore its high score is evidence that local ordered context
is powerful here, not evidence that long-term memory is unnecessary for long streams.

The three hybrid CNN models have higher F1 than their non-CNN counterparts on mean,
but the gains are much smaller than standalone CNN’s gain. If the CNN result came
only from recovering local temporal order, that pattern is expected: the GNN already
contains graph attention and the GRU already summarizes the same four neighbors.
The CNN has less new information to add to those models.

## Important controls and limitations

The five CNN seeds are stable in a narrow range (95.83–97.96% F1), with zero test
false positives in every seed. That makes a single-seed accident unlikely, but the
test still has only 25 positive flows; it is not an independent generalization study.
The previous 60-run study and this 40-run study reuse the same inspected test
partition, so repeated comparisons do not create new test datasets.

The original native-Conv1d attempt was stopped before experimental results because
its one-worker calibration was 2.26 steps/s and eight-worker calibration 12.16.
Its partial artifacts are excluded. The final backend was verified against native
Conv1d; full v2 artifact hashes and source snapshots are preserved. All 40 v2
checkpoints, predictions, thresholds and aggregate metrics passed an independent
local audit; GNN and plain-TGNN control predictions are bit-identical to the prior
60-run study.

The test partition was inspected earlier and contains 25 positives. Validation F1
ties remain possible, and the test-best result is exploratory. The CNN may exploit
static feature shortcuts or capture-specific patterns. The next test must use an
untouched capture and compare equal-capacity models.

## Next experiment

Use full-resolution chronological data, while freezing evaluation events and labels.
Compare GNN, CNN, TGNN, TGNN+GRU+CNN and a persistent learned-memory model under:

1. sparse versus dense historical context;
2. short versus genuinely longer capture duration;
3. equal parameter budgets;
4. fixed compute and matched optimizer steps;
5. delayed-cue tests with controlled distractor events.

Report F1, PR-AP, precision, recall, false-positive rate, fixed-alert-budget recall,
latency, throughput, state memory and detection delay. This will tell us whether the
CNN advantage is local-window pattern extraction or a broader temporal-memory effect.

## Artifacts

- [Protocol](TEMPORAL_CNN_PROTOCOL.md)
- [Per-run summary](artifacts/temporal_cnn_200e_v2/summary.json)
- [Independent verification](artifacts/temporal_cnn_200e_v2/local_verification.json)
- [GPU telemetry](artifacts/temporal_cnn_200e_v2/gpu_utilization_summary.json)
- [All 40 checkpoints and predictions](artifacts/temporal_cnn_200e_v2/)
- [Previous component findings](TGNN_COMPONENT_FINDINGS.md)
