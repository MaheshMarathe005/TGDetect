# Colab encoder results review

All 30 runs (10 arms × 3 seeds) completed. Actual GPU: Tesla T4, 15,637,086,208
bytes total VRAM, not L4. The default 5,000/1,000/2,000/4,000 sample was used.
Test population: 3,200 benign and 800 attacks. 700 attacks are Neptune (87.5%).

Recomputed AP, ROC-AUC and 1%-target confusion counts from every raw scores.npz;
all match saved results. Every checkpoint digest matches its result JSON. Bundled
source hashes match source_snapshot. No failure files. GPU acceptance parity
errors are zero. No checkpoint deserialization or archived code execution was
needed for this review. Optional caching/precision/compile/distillation results
are absent: they were not run/exported.

## Results

Means across seeds; F1/recall/FPR use benign-calibrated 1% FPR target. AP and
F1 are percentages. Latency is mean of three per-seed batch-128 p50s in ms.

| Arm | AP % | F1 % | Recall % | Test FPR % | Encoder ms |
|---|---:|---:|---:|---:|---:|
| old | 85.56 | 48.40 | 41.13 | 0.66 | 1.940 |
| old_extra_updates | 91.97 | 72.31 | 64.33 | 0.90 | 1.808 |
| rarity | 90.82 | 66.72 | 64.04 | 1.03 | 2.150 |
| interaction | 87.39 | 42.60 | 32.83 | 0.62 | 2.561 |
| interaction_extra_updates | 93.80 | 70.50 | 66.21 | 0.85 | 2.209 |
| interaction_masked | 83.08 | 55.50 | 46.67 | 0.85 | 2.762 |
| rarity_interaction | 87.82 | 41.33 | 33.88 | 0.91 | 2.723 |
| rarity_interaction_extra_updates | 95.45 | 94.33 | 92.92 | 1.02 | 2.618 |
| full | 76.97 | 11.45 | 6.29 | 0.77 | 2.729 |
| full_contrastive | 89.38 | 66.94 | 63.92 | 1.11 | 2.532 |

## Interpretation

Rarity improved service-score AP in all seeds (+0.82, +3.08, +11.87 percentage
points), but seed-0 paired AP interval includes zero. Recall remains unstable:
93.0%, 6.375%, 92.75%. It is a promising isolated change, not a reliable final
operating point. Mean encoder latency increased approximately 10.8%; online
count construction is excluded. CMS audit found zero overestimation on sampled
training keys. A four-depth sketch is not a single literal table lookup.

Rarity+interaction+extra updates has the highest mean primary AP (95.45%) and
most consistent F1 (94.33% mean, 0.41 percentage-point SD). Its AP is +3.48 points
above update-matched old and encoder time is approximately 44.8% higher. Paired
AP differences against update-matched old are -1.03, +12.85, -1.38 points:
individual-seed superiority is not established. More task training benefits old
as well (AP 85.56% -> 91.97%), so raw comparison against old alone is misleading.

The raw rarity-only control has AP 95.08%, F1 92.92%, recall 91.88%, and test
FPR 1.469%. Learned rarity+interaction exceeds its AP by only 0.38 points; operating
points also differ. Establish added neural value before adding complexity.

Full masked pretraining loses primary AP in every seed versus its same-encoder
extra-task-training control. Full service AP is 76.97%; fused AP drops to 29.62%.
Interaction+masked fused AP is 32.59%, full+contrastive 35.49%. Masked scoring costs
roughly 12.5–18.6 ms/batch versus approximately 1.8–3.3 ms for ordinary scoring.
Contrastive is unstable and does not rescue masked-score fusion.

One limitation in the notebook I authored needs correction before judging the
masked idea itself: mask heads train during one pretraining epoch, then receive
no masked-loss gradients during three service-only epochs while the shared
encoder changes. This can leave the mask decoder mismatched to final features.
Every field gets only roughly 125 masked targets in pretraining. Future tests
should maintain an auxiliary masked objective or refit masked heads on benign
training with the encoder fixed, record per-field validation NLL, and compare
calibrated channels independently before fusion. These are hypotheses explaining
failure; the current data does not establish a cause.

Rank-CDF fusion caps every score above the benign reference maximum to the same
value. This also alters service-only ranking: compare raw primary scores rather
than treating fused values as interchangeable. Preserve tail evidence in the next
scoring experiment.

Non-Neptune recall for the stable candidate is 37%, 59%, 48%; old with extra
updates gets 56%, 39%, 51%. Strong total recall mostly reflects Neptune detection.
The stable candidate misses all Back and Teardrop cases in seed 0. Rare families
have very small counts. Only one data split and dated non-temporal telemetry were
used. Labels were used for sample stratification and conflict removal, although
not optimization/calibration. Exact deduplication does not remove near duplicates.
These results do not establish CADETS transfer or production precision.

## Next experiment

Start with rarity-only against old at equal update counts across at least five
seeds and additional fixed data splits. Add a frozen training-frequency variant
to separate prefix-count distribution shift from the representation. Compare raw
rarity, learned FiLM and validation-calibrated late score fusion. Record per-family
recall and more training epochs selected solely from benign validation. Keep
rarity+interaction+extra updates as the promising accuracy candidate, with its
latency cost explicit. Repair masked-head training before another masked trial.
Do not deploy the combined full recipe from this pilot.
