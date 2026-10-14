# KAIROS and TGNN on shared DEDALE provenance

Status: implementation and CPU acceptance checks; real training requires the
complete raw DEDALE ZIP and compute access. Synthetic smoke results are not
DEDALE performance results.

## Registered comparison

- Arms: KAIROS, unchanged TGNN_Node_Mem_V3, experimental TGNN_Node_Mem_V4.
- Seeds 0, 1, 2; 50 epochs from scratch; batch 1024. No pretrained checkpoint
  imports, no reuse of the previously inspected 72-feature test predictions.
- Shared source events, graph endpoints, ordering, 41-dimensional messages,
  train/validation/test boundaries, and held-out IDs. No benign-event subsampling.
- Modal preparation uses the established class-1 active interval from the prior
  study: 2025-01-06 10:01:05.579 UTC through 2025-01-13 11:00:35.516 UTC,
  on CLIENT1/CLIENT2. This label-defined historical study scope is an inherited
  selection limitation; it is not a prospective deployment evaluation.
- Each message is `[source hierarchical hash (16), relation one-hot (9),
  destination hierarchical hash (16)]`. No label, MITRE, user, command line,
  timestamp-derived category, or old 72-dimensional feature is included.
- KAIROS predicts the current relation using only prior messages and memories.
  Its current relation is a target and only enters memory after scoring. TGNN
  observes the current message to classify its maliciousness. The feature
  schema is shared; prediction targets and permitted use of the current relation
  differ by design, as authorized in the conversation.
- KAIROS optimizes cross entropy on wholly benign training windows, using Adam
  at 5e-5, epsilon 1e-8, weight decay .01; final epoch checkpoint (released code).
- TGNN optimizes supervised, class-weighted attack cross entropy with reservoir
  replay and admission penalties; Adam 1e-4, weight decay 1e-5; three-batch
  truncated backpropagation. Select checkpoint by validation event PR-AP only.
- Same epoch count is not equal compute, label access, optimizer steps, or
  training-example count. All these differences are recorded. Do not interpret
  this as an isolated architecture ablation between KAIROS and TGNN.
- Train/validation/test: chronological 60/20/20 event quantiles, advanced to
  complete 15-minute boundaries. No overlap. Refuse supervised performance runs
  if any partition lacks a class. If this happens, register explicit date bounds
  before another run; do not move boundaries to optimize observed performance.
- Primary common unit: 15-minute windows. TGNN window score is maximum event
  probability; its window threshold maximizes validation window F1. KAIROS uses
  the paper's anomalous-edge means, rare-node correlation, and queue products.
  Its threshold is the maximum benign-validation queue score, with strict `>`.
- KAIROS IDF uses entity descriptions in benign training windows only; alpha is
  `log(.9*N)`, following the released CADETS threshold rule. No test keyword
  exclusions from the original dataset are imported.
- Report online-at-window-close scores separately from retrospective queue
  scores. A later queue alert may flag earlier windows only retrospectively.
- Event metrics for KAIROS are extra reconstruction-error diagnostics, not its
  original end-to-end detection result. Do not compare them to window F1.
- Inference resets state and warms on the chronological prefix without labels.
  Paper/released-code minibatch semantics are retained for KAIROS; V3/V4 have
  exact ordered-event memory reads. Equal timestamps have stable record order.
- No claim of an untouched dataset: earlier DEDALE experiments have already been
  inspected. A future external holdout remains necessary for generalization.

## Paper and released-code trace

Paper: `research/2024-sp.pdf`, Cheng et al., IEEE S&P 2024,
*KAIROS: Practical Intrusion Detection and Investigation using Whole-system
Provenance*.

Official source: https://github.com/ProvenanceAnalytics/kairos
commit `0e0b633beb46a1117c0a6d63be5d2481b59ac0dc`.

| Component | Implementation basis |
|---|---|
| Hierarchical signed hashing, width 16 | Paper §4.1; released `embedding.py` character-hash behavior |
| TGN memory, 100 dimensions, GRU | Paper §4.2; `CADETS_E3/train.py`, IdentityMessage + LastAggregator |
| Time encoding, width 100 | Released `CADETS_E3/config.py` |
| UniMP graph encoder | `CADETS_E3/model.py`: TransformerConv, 8 heads then 1, ReLU, no convolution dropout |
| Endpoint embedding width 100 | Released config; source/destination pair has 200 values before decoder projections |
| Decoder | Separate 100→200 endpoint projections, concatenation, 400→800→200→50→9 MLP, Tanh and dropout .5 |
| Nine relations | Paper Table 1. Released CADETS uses seven; OpTC uses ten |
| 20 historical neighbors | Paper §5 and released config |
| 15-minute windows, mean+1.5 SD | Paper §4.3 |
| Queue product and rare-node intersection | Paper §4.3; log space prevents numerical overflow |
| Weighted Louvain summary graphs | Paper §4.4; merge identical directed typed edges with maximum RE, retain multiplicities |

The paper alone does not specify every MLP layer. Released configurations vary:
OpTC's decoder has BatchNorm and trains 10 epochs; CADETS's released standalone
decoder has no BatchNorm and trains 50. This implementation consistently follows
the standalone CADETS encoder/decoder configuration with the paper's nine output
relations. It is not claimed to be a bit-identical reproduction of every paper
experiment. PyTorch/PyG versions also differ from the original release. The
paper's `|z|=200` and the release's endpoint width 100 are documented explicitly.

## DEDALE adaptation and label limits

DEDALE Sysmon is not a complete kernel syscall trace. The adapter includes only
events with defensible endpoint extraction and reports all exclusion counts:

| Sysmon event | Relation | Endpoint mapping / limitation |
|---|---|---|
| 1, process creation | START | Parent process → new process |
| 5, process termination | CLOSE | Process self-edge; no invented second process |
| 7, image loaded | READ | Process → image file; image loading is a read proxy |
| 11, file creation | WRITE | Process → created file; not all write syscalls |
| 3, network connection | SEND/RECEIVE | Process/socket, explicit Initiated direction; connection proxy, not packet/syscall capture |

Unsupported codes, unknown network direction, and missing endpoint attributes
are excluded. CLONE, OPEN, EXEC may be absent; no events are fabricated to fill
the nine output classes. Therefore this is a **KAIROS architecture comparison
adapted to the supported DEDALE subset**, not an exact data/pipeline reproduction.

GUID resolution reads `process.entity_id`, parent ECS fields, event_data, and
explicit Sysmon message fields. A PID is reused only with a process start time;
otherwise an event-local identity is used and counted. Provider names never
stand in for process identities. Host identity is used only in graph keys.
File identity is host plus normalized path; file generations cannot be recovered
reliably from this telemetry and remain a limitation.

Labels join class-1 ground truth by host, channel, record ID and timestamp.
Class-2-only records are excluded rather than treated as benign. Unmatched
events are background under the dataset labeling assumption. Positive coverage
and omitted events are recorded; recall is only over included events.

## V4 changes

`node_memory_v4.py` adds endpoint-specific importance/admission and learned
wall-clock decay to fast/slow state. Half-lives initialize to 1 hour and 1 day,
bounded to 1 second..365 days. For each endpoint:

`pre = exp(-log(2)*elapsed/half_life) * old`

`updated = (1-write_weight)*pre + write_weight*content`

Reads use `pre`; current writes become visible only to later ordered events.
The affine prefix scan remains differentiable and independent of batch splits.
Self-edges write once; time reversal is rejected. Runtime memory/timestamps are
excluded from model checkpoints. V3 source/checkpoints are preserved.
V4's benefit is unproven until the matched experiment completes.

## Run

From the repository root with the existing Python environment:

```bash
PYTHONPATH=research /home/mahesh/main_env/bin/python -m pytest -q research/test_kairos_comparison.py research/test_node_memory_v3.py

PYTHONPATH=research /home/mahesh/main_env/bin/python -m kairos_comparison.data \
  --raw data/real/dedale/system_logs_winlogbeat.zip \
  --labels data/real/dedale/system_logs_labels.zip \
  --output artifacts/kairos_dedale_shared41/prepared \
  --start 1736157665.579 --end 1736766035.516001 --workers 8

PYTHONPATH=research /home/mahesh/main_env/bin/python -m kairos_comparison.run \
  --prepared artifacts/kairos_dedale_shared41/prepared/prepared.npz \
  --output artifacts/kairos_dedale_shared41/results --epochs 50 --seeds 0 1 2 --device cuda
```

For a durable Modal run (survives the launching shell/session):

```bash
PYTHONPATH=research /home/mahesh/main_env/bin/python -m modal deploy research/modal_kairos_comparison.py
PYTHONPATH=research /home/mahesh/main_env/bin/python research/modal_kairos_comparison.py --spawn --epochs 50 --seeds 3
```

Modal uses existing `tg-detect-dedale-v1` raw archives and writes new directories
under `/prepared/dedale_kairos_shared41_v1` and `/results/dedale_kairos_shared41_v1`.
One L4 per seed runs its three models sequentially; three seeds may run together.
CPU preprocessing uses 64 GB memory. This is not yet a measured cost estimate.

Preparation resumes at completed ZIP members via SQLite transactions. Existing
output with incompatible source hashes/configuration is rejected. Checkpoints,
full prediction arrays, labels/IDs, metrics, queues and graph summaries are saved.
Predictions must be compared only when dataset, feature and ID hashes match.
Training writes resumable optimizer/RNG/replay checkpoints every five epochs;
completed arms are reused only when their source/config/data signatures match.
The deployed coordinator records `pipeline_status.json` and automatically runs
the full saved-prediction audit after all seeds finish. Its call ID is recorded
locally in `artifacts/kairos_dedale_shared41/launch.json`.

Read progress or download and independently audit completed results:

```bash
PYTHONPATH=research /home/mahesh/main_env/bin/python research/sync_kairos_comparison.py
PYTHONPATH=research /home/mahesh/main_env/bin/python research/sync_kairos_comparison.py --download
```

### Benign-only TGNN comparison

`TGNN_Node_Mem_V3_Benign` and `TGNN_Node_Mem_V4_Benign` train only on wholly benign chronological training windows. They retain the TGNN memory and admission mechanisms, but optimize a one-class binary objective with benign targets and score the attack-probability output at test time. This gives a label-free TGNN comparison against KAIROS while keeping the shared 41-dimensional messages and split fixed.
