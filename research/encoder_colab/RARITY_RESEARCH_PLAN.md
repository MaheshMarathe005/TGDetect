# Rarity: hypotheses, interventions, and expected consequences

The first pilot's frozen rarity+interaction candidate has strong aggregate F1,
but masked fusion failed, the raw rarity control was competitive, and most attacks
belonged to Neptune. New notebook: `TGDetect_Rarity_Followup.ipynb`.

## Priority 1: explain rarity behavior

| Observation/hypothesis | Intervention | Expected effect | Failure interpretation |
|---|---|---|---|
| Counts grow during training, freeze at evaluation | Disjoint frozen benign reference | Consistent rarity scale | No gain weakens the mismatch hypothesis |
| Synthetic event-type count encodes ingestion age | Set its rarity to zero | Remove a shortcut | Later compare age-on/off separately if needed |
| Large log-count drives FiLM more for common values | Bounded inverse-frequency input | Easier optimization; limited modulation scale | May add no information because transform is affine |
| Rarity gain could come from extra inputs/capacity | Field-wise shuffled rarity | Preserve marginals, destroy association | Similar results weaken evidence for semantic rarity |
| Extra training improved old | Same eight epochs/optimizer updates | Fairer capacity comparison | Curves may still show undertraining |
| Novelty may work best as direct evidence | Old + fixed late rarity fusion | Cheap accuracy path, cache-compatible base encoder | Requires separate recalibration and deployment counts |
| Neptune dominates test | Per-family/macro/non-Neptune metrics | Reveal missed families | Small family counts remain uncertain |

Default: 6 arms × 5 seeds × 2 new data splits. Both splits are same-dataset
sensitivity checks and may overlap; they are not independent capture evidence.
All arms share 4,000 optimizer records and 1,000 held-out benign reference
records. The reference is excluded from task optimization. Prefix counting uses
the same reference as an initial state and adds task records read-before-write;
frozen counting never adds them. Synthetic event-type rarity is zero for all arms.
This does not isolate every regime variable: final count sample size changes too.
Training corpus/label dictionaries may include reference records in the service
dictionary fitting, but task SGNS uses optimizer records only; all are benign.

Use benign-validation CE for checkpoints, calibration A for robust score scales,
and calibration B for thresholds. Service NLL is primary. Raw novelty and fixed
late-fusion weights are secondary, with no test-based weight selection. Record
zero-rarity counterfactual scores, sample-level paired AP intervals, and seed/split
spread. An inference intervention is a reliance diagnostic, not a causal proof
that the input improves learning.

## Priority 2: conditional rarity after the first matrix

Global rarity confuses a value unusual for one context with one normal for another.
Compare `(field,value)` counts with `(benign context,field,value)` counts, using
protocol/entity type and minimum support. A /tmp path may be normal for a temp
file but suspicious for a privileged executable. Conditioning can improve
specificity, but sparse contexts can increase false alerts. Keep global fallback,
cap state size, and test novel benign contexts. Do not feed service target or
attack labels into the count keys.

Numeric exact values need special care: durations/byte totals may be unique even
when behavior is normal. Compare raw-value counts with training-only quantile-bin
counts, retaining the original numeric input so outliers are not erased. Quantile
bins lose magnitude detail; test out-of-range indicators separately.

## Priority 3: preserve a useful masked decoder

The first notebook trains masked heads only during one pretraining epoch. Service
updates then move their shared features. Two controlled repairs: joint masked
auxiliary loss throughout training, or freeze final encoder and refit masked heads
on benign training. Compare to equal-update/compute controls, log per-field benign
validation NLL and no-context target-frequency baselines. Test one-field or small
selected-field scoring before paying for many masked passes. Conditional NLL minus
marginal target NLL is another candidate for reducing background-frequency
effects; it needs its own calibration and experiment.

This is motivated by the observation that likelihood can be dominated by
background statistics in [Ren et al., 2019](https://arxiv.org/abs/1906.02845).
Their experiments do not establish the effect for TGDetect. Key-conditioned
affine modulation uses the general mechanism described by
[Perez et al., FiLM](https://arxiv.org/abs/1709.07871), with security performance
still requiring our own evaluation.

## Execution and deployment

Rarity that changes over time makes the final event embedding state-dependent.
Cache static fields before FiLM, or keep the original deterministic encoder and
apply rarity in a separate scoring channel. Never reuse a final event vector for
different count states without invalidation. Benchmark CPU count reads separately
from GPU encoder timing; this notebook reports both but does not establish online
arrival-to-alert latency. BF16/int8/compile/distillation remain separate optional
studies and are not credited with unmeasured gains.

No local GPU is available. Local CPU tests validate functionality; the standalone
notebook runs the GPU experiment and exports raw scores. No paid cloud job was
launched by preparing this study.
