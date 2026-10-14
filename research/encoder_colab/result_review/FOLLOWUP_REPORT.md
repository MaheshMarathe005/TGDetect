# Rarity follow-up review

All 60 runs completed: six arms, five seeds, two data splits. GPU: Tesla T4,
PyTorch 2.11.0+cu130. Each arm received 4,000 optimizer records, 1,000 separate
benign reference records and eight task epochs. Each test has 4,000 records with
20% attacks. This is encoder-only KDD evaluation, not CADETS or full TGDetect.

AP/ROC-AUC and threshold confusion counts were recomputed from all 60 NPZ files.
They match saved results. Checkpoint SHA256 and bundled-source hashes match.
No failure record is present. Archived code was not executed or checkpoints
deserialized during this review.

## Main results

Means across ten seed/split runs. Percentages; F1 and recalls use the
validation-calibrated 1% FPR target. Encoder time is the mean per-run batch-128
p50, excluding count construction/ingestion. Not a pooled percentile.

| Variant | AP | F1 | Non-Neptune recall | Test FPR | Encoder ms |
|---|---:|---:|---:|---:|---:|
| Original | 98.150 | 95.341 | 53.732 | 0.834 | 1.938 |
| Prefix log-count FiLM | 98.121 | 95.783 | 62.000 | 0.881 | 2.152 |
| Frozen log-count FiLM | 98.218 | 95.725 | 61.722 | 0.881 | 2.172 |
| Frozen bounded rarity FiLM | 98.263 | 95.090 | 50.865 | 0.872 | 2.263 |
| Shuffled bounded rarity FiLM | 98.055 | 94.716 | 51.352 | 1.069 | 2.027 |
| Bounded rarity + interaction | 98.399 | 95.500 | 59.393 | 0.922 | 2.548 |

Frozen log-count FiLM is the strongest simple candidate for minority-family
recall: +7.99 percentage points relative to original, with improvements in eight
of ten paired runs. Split-specific gains are +10.69 and +5.29 points. AP improves
only six of ten paired runs; mean AP gain is just 0.067 percentage points. Test
FPR also increases 0.047 points. Thus this is a useful operating-point tradeoff,
not evidence of uniform ranking superiority. Encoder p50 increases about 12.1%.

Prefix log counts have similar minority-family improvement. Frozen log counts
beat prefix AP in eight of ten runs, but the average difference is only 0.097
points and the gains reverse in two runs. Count-regime mismatch alone did not
explain the first study's instability.

Bounded rarity did not improve minority-family recall: 50.86% vs original 53.73%,
and below frozen log-count 61.72%. Its AP wins five and loses five against frozen
log counts. It is not a justified replacement for log-count FiLM from this study.
True bounded rarity beats shuffled rarity AP in seven of ten runs (six positive
per-run record-bootstrap intervals), but average minority-family recall is
slightly lower. The association affects some ranking outcomes; it does not
establish reliable benefit at the selected threshold.

Interaction raises mean AP 0.249 points above original, at approximately 31.5%
higher encoder p50. It wins only five of ten AP comparisons against bounded
rarity without interaction. Retain as a research candidate; prefer simple frozen
log-count FiLM for the next minority-family test.

## Training and family interpretation

Original F1 is now consistent (mean 95.34%, SD 0.75 percentage points). This is
compatible with undertraining contributing to the first study, but changed data
splits and reference partitioning prevent attributing the difference solely to
epochs. 48/60 checkpoints select epoch eight, the budget boundary: convergence
is not established. Compare 8/12/16 epochs with benign-only checkpoint selection
before claiming architectural effects independent of optimization.

Frozen log counts improve mean Ipsweep recall (~75% -> 95%), Nmap (~85% -> 100%)
and Teardrop (~54% -> 85%). Back remains almost completely missed; Warezclient
recall remains low (~19%). Family counts are tiny and differ across splits, so
these means are descriptive. Majority Neptune detection remains around 99.5%.

## Direct scoring and reliance

Original + late rarity fusion produces mean AP 98.31/98.37/98.46% at predeclared
weights 0.25/0.5/1.0, versus 98.15% original. Minority recall changes little:
53.73/53.62/54.24%. Fusion helps ranking modestly, but does not reproduce the
minority recall benefit of learned log-count FiLM. Do not select a fusion weight
from these test results and claim untouched evaluation.

Frozen reference raw rarity AP is 96.18% and 96.73% in the two splits. Raw
inverse-count AP is 94.68% and 94.51%. Learned models add value over these controls
in this study, unlike the near-tie in the first pilot.

Zeroing rarity at inference drops frozen log-count AP to mean 24.01%. This is
an out-of-distribution intervention for a model trained on positive log counts;
it demonstrates dependence, not proof that the architecture is defective. Count
availability and cache validity matter. Test realistic noisy/stale/colliding
counts and missing-count fallback with retraining before deployment.

## Latency limitation

The Python CPU count-read prototype measures 0.436–0.581 ms per record, implying
about 55.8–74.4 ms of that measured average work for 128 records, compared with
~2.2 ms GPU encoder completion. These were preparation reads, not an integrated
online benchmark; the multiplication is a sizing warning, not measured end-to-end
batch latency. Online hashing/JSON loops would dominate if deployed unchanged.
Precompute stable field fingerprints and implement vectorized CMS reads before
claiming efficiency. A four-row count-min sketch needs four bucket reads plus
a minimum, not one literal lookup. Peak recorded CUDA allocation was 342.9–346.8
MB, including the resident bank, not whole-model deployment memory.

## Next controlled work

1. Carry original and frozen log-count FiLM into longer-training checks, comparing
   minority-family recall at comparable benign alert budgets, not AP alone.
2. Vectorize fingerprint/CMS lookup; include CPU staging and updates in a paired
   end-to-end encoder benchmark. Cache static field vectors before FiLM.
3. Test realistic stale/noisy counts and a trained missing-count fallback.
4. Separately compare exact numeric-value counts to benign-training-bin counts,
   and global to context-conditioned rarity with minimum-support fallback.
5. Re-test the same chosen configuration on a separate dataset/capture. Random
   KDD splits overlap and are not independent capture replication.

Masked pretraining, contrastive loss, int8/BF16, compile and distillation were
not exercised by this focused matrix. No new GPU training was launched here.
