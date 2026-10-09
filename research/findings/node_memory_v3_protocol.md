# Node memory V3 fine-tuning

Date: 2026-10-02. Dataset: existing `/vol/prepared/dedale_host_mem/prepared.npz`.
Previous datasets, models, and result directories are preserved.

## Changes

- Exact pre-event source/destination memory reads using segmented affine prefix
  scans; eval predictions are independent of encoder batch boundaries within
  floating-point tolerance. Event order is the prepared dataset's stable order;
  equal timestamps are ordered records, not simultaneous observations.
- Separate endpoint write spaces and read projections with context-dependent
  tanh read gates initialized to zero. The initial predictor equals its QAC seed.
- 64-dimensional fast and slow states for each endpoint. Write strengths are
  admission times 0.5 and 0.005. Half-life depends on actual admission, not a fixed
  promised time horizon. No wall-time decay was introduced.
- Novelty compares endpoint-specific write vectors to a detached fixed-rate
  reference EMA in the same space. This reference supplies batch-independent
  novelty without a sequential adaptive-gate dependency. Novelty is a proxy,
  not an estimate of future detection utility.
- Fixed learned importance threshold replaces batch quantiles. Novel candidates
  still enter the candidate pool. Importance receives task gradients through
  retention even when novelty overrides candidacy.
- Correct admission budget/redundancy count candidate membership once.
- Memory-free replay remains isolated from runtime endpoint state.
- Inference scatters update only touched rows; training uses functional writes
  for autograd correctness. Gradient accumulation is averaged over each window.

## Training

Three seeds, up to 80 additional epochs each, one L4, overall 600-minute budget.
Main/controller learning rates: 1e-4. Weight decay: 1e-5. Budget/redundancy
weights: 0.05/0.01. Batch: 512; TBPTT: three batches. Exact within-batch reads
provide task gradients for recurrent endpoints before a batch boundary.

Warm start: same-seed QAC checkpoints in `/vol/results/dedale_host_mem/`.
Only matching encoder/head/normalization weights transfer; new memory and
controller weights initialize fresh. Initial QAC-equivalent validation AP is
included as a selectable checkpoint. This uses additional optimization and is
not a matched-budget comparison against earlier QAC results.

Model selection uses validation PR-AP; primary operating threshold maximizes
validation F1 and is frozen for test. Test-oracle thresholds, if reported by the
inherited runner, are descriptive upper bounds and never select models.
All runs record warm-start SHA256, epochs, optimizer steps, gradients, admission,
read-gate magnitude and effective write weights. Results:
`/vol/results/dedale_host_mem_v3/` on volume `tg-detect-dedale-v1`.

## Validation and limits

Eight local acceptance tests pass, with the same tests required on the actual
Modal GPU before training. They cover scan values/gradients against sequential
updates, self-edges, arbitrary chunk boundaries, future perturbations, full
encoder eval batching, task credit, zero-gate fallback, replay isolation and
the corrected admission penalty.

The existing endpoint identity and reservoir-subsampled data are retained to
reuse comparable preprocessing and checkpoints. PID/provider fallback identity,
continuous-history data, longer-gradient-horizon ablations, and a matched-budget
from-scratch comparison remain separate experiments. Improved accuracy or
throughput is not assumed; evaluate saved test predictions and latency results.
