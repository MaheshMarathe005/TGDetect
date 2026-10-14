# Original versus frozen log-count rarity: next study

Use `TGDetect_Rarity_Stability.ipynb` in Colab. Select GPU and Run all. T4/L4
are accepted and recorded; no cloud GPU job was launched locally.

Two encoder architectures, five seeds, two new fixed split seeds. Train each
trajectory once to 16 epochs, evaluating best benign-validation checkpoints
available at 8, 12 and 16 epochs. Identical initialization, task batches and
optimizer updates between architectures. Eight-epoch selection is unaffected
by subsequent twelve/sixteen-epoch results. Budget measurements share a trajectory
and are correlated, not three independent experiments.

Each split has 1,000 benign reference, 4,000 task-training, 1,000 benign
validation, 2,000 benign calibration and 4,000 mixed test records. Reference
counts never ingest task/evaluation records. Synthetic event-type rarity is zero.
Use the original protocol's duplicate/conflict filtering; label use in sample
selection is disclosed. All task optimization and calibration remain benign-only.
Different random splits can overlap; KDD is not a new capture or a temporal test.

## Count engine

Compute all four BLAKE2 fingerprints once per distinct canonical field in each
preparation call. Vectorized NumPy scatter-add constructs the reference CMS.
NumPy gathers and a PyTorch device gather take the minimum across four buckets.
On clean inputs both engines must match the Python reference before training.
No narrower hashes or approximated numerical value keys are introduced.

Measure Python hash+lookup, NumPy prehashed lookup, device prehashed lookup,
prepared encoder, device count+encoder, and host-staging count+encoder separately.
Measure cold hashing separately. The original deployed path omits count lookup.
Inputs are prepared field identities, not production log ingestion. Count table
updates/stream causality are not benchmarked: the table is frozen.

## Predeclared stress conditions

- Half the reference: reduced benign count coverage, a stale-table proxy.
- Repeat a randomly chosen 30% of the benign reference: duplicated updates.
- Width 2,048 instead of 65,536: collision overcounts.
- Zero counts for 10% of fields: missing-count stress. The current feature cannot
  distinguish unavailable from never-seen values; no trained fallback is claimed.

These are synthetic hypotheses about realistic failure modes, not evidence of
actual production drift. No test labels construct the corruption. Each condition
uses the same trained checkpoint and reports clean-threshold and benign-
recalibrated metrics separately. Benign recalibration assumes representative
benign observations are available, which may not be true during an incident.

## Metrics and interpretation

AP, F1, benign FPR and false alerts/1,000, per-family recall with sample counts,
non-Neptune recall, and Back/Warezclient recall. Wilson intervals are included
for global operating-point metrics. Paired stratified AP bootstrap conditions
on a given test sample; report seed/split variation separately. Do not choose
the best tested budget or stress threshold with attack labels.

If rarity improves minority recall after adequate training, retain it with its
latency/false-alert cost. If rankings survive shifts but fixed thresholds fail,
the next issue is calibration. If both fail, test retrained count dropout and an
explicit missing-count flag separately. If both encoders still miss Back and
Warezclient, inspect service reconstruction and input sufficiency on development
records; rare attack families cannot be solved by claiming more layers help.

The next causal feature experiments, after this study, are numeric-value binning
for count keys and context-conditioned rarity with minimum-support fallback.
They are not added here, so the count-engine and training-length effects remain
interpretable. Masked objectives, interaction, precision and distillation stay
outside this two-encoder matrix.

## Validation

16 local CPU tests pass, including matching Python/NumPy/device count lookups,
collision overcount behavior, all robustness conditions, two checkpoint budgets,
original-encoder invariance to count corruption, and notebook cell compilation.
Default GPU timing/accuracy still requires Colab. No local CUDA GPU is available.

Both encoder trajectories also completed a real KDD CPU smoke run with 128
preparation-training records (32 reference), two short budgets (1/2), all five
count conditions, and both threshold policies. This produced 40 metric rows.
The sample is only a functional smoke test and was not used to select settings.
