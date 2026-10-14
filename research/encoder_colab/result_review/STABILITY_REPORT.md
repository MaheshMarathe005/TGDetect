# Rarity stability results review

Completed: 20 training trajectories, each with 8/12/16-budget evaluations:
60 checkpoint result files and 300 condition score files. Actual GPU Tesla T4,
PyTorch 2.11.0+cu130. All raw-score AP, ROC-AUC and confusion counts at both
threshold targets match saved metrics. All checkpoint and bundled-source hashes
match. Original scores are exactly invariant across the count-corruption tests.
No archived code was executed or checkpoint deserialized to perform this review.

## Clean results

Means across five seeds and two splits. Percentages; F1 and recalls use benign-
calibrated 1% FPR target. AP is a ranking metric; actual test FPR differs.

| Encoder | Budget | AP | F1 | Test FPR | Non-Neptune recall | Back recall | Warezclient recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original | 8 | 98.257 | 95.577 | 0.769 | 54.929 | 0 | 10.390 |
| Frozen log rarity | 8 | 98.281 | 95.183 | 0.828 | 51.028 | 0.625 | 12.165 |
| Original | 12 | 98.526 | 95.791 | 0.713 | 56.544 | 6.250 | 14.372 |
| Frozen log rarity | 12 | 98.697 | 96.222 | 0.853 | 67.186 | 8.000 | 26.407 |
| Original | 16 | 98.954 | 96.009 | 0.834 | 63.132 | 21.000 | 21.688 |
| Frozen log rarity | 16 | 98.892 | 96.717 | 0.631 | 67.779 | 7.250 | 26.320 |

At budget 16, rarity gains 0.708 F1 points and 4.647 non-Neptune recall points,
while test FPR drops 0.203 points. AP loses 0.062 points; paired AP wins five
and loses five. Four per-run paired-bootstrap intervals favor rarity and four
favor original. No universal superiority is established. Record intervals
condition on each sample; split/seed values are not independent capture evidence.

Back is still weak: original detects mean 21%, rarity 7.25%. Warezclient's
aggregate rarity advantage reverses across splits: rarity/original recall is
38.10/14.29% in one split and 14.55/29.09% in the other. Rare-family counts are
small. A larger representative capture is necessary before claiming a solution.

Training length matters for both encoders. At the 16-budget endpoint, seven of
ten original checkpoints and five of ten rarity checkpoints select epoch 16.
Others select epochs 14–15. Some runs may still benefit from training, but this
does not justify automatically extending all runs indefinitely. Eight/12/16
evaluations share trajectories and are correlated. Budgets must not be selected
using attack test scores and then presented as untouched prospective choices.

## Frozen rarity robustness, budget 16

| Condition | AP | F1, fixed threshold | Test FPR, fixed threshold | Non-Neptune recall |
|---|---:|---:|---:|---:|
| Clean | 98.892 | 96.717 | 0.631 | 67.779 |
| Smaller CMS (collisions) | 98.850 | 96.679 | 0.644 | 67.565 |
| Duplicated reference updates | 98.823 | 96.694 | 0.656 | 67.976 |
| Half-reference stale proxy | 98.935 | 96.302 | 0.819 | 66.982 |
| 10% missing count fields | 98.161 | 93.643 | 2.050 | 69.944 |

Missing-count apparent recall gain comes with more false alerts: 20.5 rather than
6.31 false alerts per 1,000 benign records. Benign recalibration lowers FPR to
0.75%, with F1 94.967% and non-Neptune recall 59.647%. Ranking also worsens, so
recalibration alone does not restore clean behavior. This supports testing a
separate missing-count indicator plus retrained rarity dropout/fallback.
Zero log-count currently conflates missing with genuinely unseen values.

Collision and duplicate-update stress have modest effects on this sample.
These synthetic conditions are not real temporal drift/incident validation.

## Counting and encoder latency

Per-batch 128 values: mean across ten per-run p50s at budget 16, milliseconds.

| Rarity path | Time |
|---|---:|
| Python raw-field hash + lookup | 70.058 |
| NumPy prehashed lookup | 0.634 |
| GPU prehashed lookup | 0.161 |
| Cold fingerprint preparation | 10.652 |
| Host staging + GPU count + encoder benchmark | 2.652 |
| Repeated deployed rarity pipeline benchmark | 2.508 |
| Original deployed encoder benchmark | 1.903 |

The warm lookup bottleneck is substantially reduced, but the speedup relies on
prepared fingerprints. Cold hashing is not free. Online event preparation,
arrival wait, service scoring, GAT/CNN/memory and counter updates are excluded.
The two separate pipeline timing observations differ due to runtime variation;
do not sum unrelated timing medians into an asserted end-to-end latency.
Rarity is about 31.8% slower than original in the budget-16 deployed benchmark;
budget-12 observations have a smaller gap. Timing noise and limited repetitions
do not establish the exact production overhead. Peak recorded CUDA allocation
is approximately 343.3 MB, not full TGDetect deployment memory.

## Recommended next change

Retain original as the baseline and frozen log rarity as the operating-point
candidate. Test original versus rarity versus rarity trained with (a) explicit
count-availability indicator and (b) label-independent count dropout. Keep all
task updates matched and the feature flag/regularization ablations separate.
Compare clean and missing-count scores with both fixed and recalibrated thresholds.

For Back and Warezclient, inspect held-out development records and which input
fields/service predictions fail. Both architectures still miss many examples;
the benign service-reconstruction objective may omit useful attack evidence,
but these scores alone do not identify the cause. Test a separately controlled
attribute-reconstruction objective only after defining relevant available fields.
Do not use these test cases to tune the reported detector's threshold.

Before production integration, evaluate another dataset/capture and benchmark
cold/warm fingerprint preparation with realistic repeat rates. Cache static
fields before rarity modulation; final rarity-conditioned event vectors require
count-state validity. Masked recipe, precision and distillation remain separate
experiments with no gain claimed here.
