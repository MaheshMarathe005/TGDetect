# POMSATHIT_ADAPTED pilot v1 — registered before training

Date: 2026-09-23. This protocol resolves implementation choices for a small CTU-13
pilot. The original Wazuh experiment remains unreproducible from available material.
Results here concern sampled botnet flows, not APT detection on endpoint telemetry.

## Data and observation time

Download the complete scenario 11 text flow file from the authoritative CTU server:
https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow
Record HTTPS source, retrieval time, size and SHA-256. This establishes retrieval
provenance; no independently published checksum is claimed. Scenario 11 is chosen
for its small download size before inspecting model scores.

Observe each record at `StartTime + Dur`, an **offline flow-completion proxy**.
Actual exporter arrival times are unavailable. Duration and totals are available
under this proxy, not at flow start. Stable-sort on completion time and raw row ID.
For this CPU pilot choose 4096 evenly spaced records across that ordered capture,
independently of labels. All graph/state history refers to this sampled stream.
This thins traffic and limits temporal conclusions. Keep Background as class 0
(not a verified benign label); Botnet is 1, Normal is 0. Fail on unknown labels.
Use chronological 60/20/20 train/validation/test; place equal timestamps in one
partition. Abort performance comparison if any partition lacks either class.

STATIC_FAIR: log1p duration, packets, bytes, source bytes; source-byte fraction;
TCP/UDP/ICMP indicators; destination port / 65535 and missing-port indicator.
Exclude IP strings, node IDs as features, labels, scenario/family identifiers,
absolute time and all history counters. Fit feature mean/std on training only.
Only TCP/UDP have a port feature; all other protocols get the missing-port flag.
This was clarified before training after an ESP record supplied a 32-bit hex field.
TEMPORAL_ONLY_ADDITION supplies log1p elapsed seconds of graph neighbors to T.
FULL equals STATIC_FAIR plus this separately controlled time channel for this pilot.
LEAKAGE_STRESS is reserved for unit-test future perturbations, never training.

## Explicit baseline adaptations

Events become graph nodes. Each target has self plus its last four prior events
sharing either endpoint; keep only strictly earlier completion times. Edges point
from history to target; history nodes have self loops only. No cross-example edges
exist in a minibatch. This bounded causal ego graph is a task adaptation, not the
paper's process/file/socket graph. The static baseline already receives recency
through neighbor selection; the test measures extra timing/state beyond this.

Use three single-head additive GAT layers, width 128: z=W h;
e_ij=LeakyReLU(a_src^T z_j + a_dst^T z_i, .2); alpha=softmax over allowed j;
h'_i=Dropout(ReLU(sum_j alpha_ij z_j), .2). Biases in W; Xavier weights,
zero biases. No attention dropout, residual, identity embedding or LayerNorm.
Linear 128→2 target-node head; softmax and unweighted cross entropy.
These operator/head/loss/initialization/dropout choices resolve missing details
explicitly. Keep the exact paper-spec class blocked.

## Temporal additions and rehearsal

T: add cos(w * log1p(age_seconds) + b), with 128 trainable frequencies/phases,
to the first layer's projected input. Mask padding. Only T consumes numeric time.
M: after scoring event i, update each endpoint's persistent 128-vector using
`m_new=.5*m_old+.5*h_i`; score with `h_i+(m_src+m_dst)/2`. State starts at zero.
R: replace M's fixed update with GRUCell(h_i,m_old). R requires M; it is a
replacement update, not an independent memory. Same self-endpoint is updated once.
Temporal message passing beyond the common ego graph is deferred, not claimed.
The pilot TGNN is B+T+M+R, not the complete legacy TG-Detect architecture.

Within a chronological 256-event training block, state remains differentiable
until the block loss has backpropagated; detach only between optimizer steps.
Reset state every epoch. Frozen-weight evaluation starts from zero and rebuilds
train history without labels, then validation and test sequentially. No evaluation
labels enter model inputs/state. Validation/test stream observations update state.

H: Algorithm R uniform reservoir of individual TRAIN events, capacity floor(0.1 *
N_train). Before admitting the current block, sample up to 26 stored events without
replacement. Optimize current mean CE plus replay mean CE (lambda=1). Reservoir
stores event ID and detached pre-event memory context from its original training
visit. Replay recomputes ego embeddings at current weights but reads this stored
context and never updates live state. This stale-context approximation is explicit;
GRU learns from live chronology, not replay. Buffer persists across epochs, counts
stream visits (duplicates possible), and never includes validation/test events.
No VAE. Memory/context bytes and extra processed examples must be reported.
Single-scenario replay is an engineering check, not a forgetting experiment.

## Registered pilot and reporting

Arms: B, B+T, B+M, B+M+R, B+T+M, B+T+M+R, B+H, B+T+M+R+H.
Seeds 0,1,2,3,4. Three epochs, no early stopping or score-based arm selection.
Adam lr=.001, weight_decay=.00001, betas=(.9,.999), eps=1e-8; float32; CPU
one thread. Batch size 256; same chronological blocks and epochs for all arms.
This is a deliberately small CPU pilot, not 200-epoch reproduction or GPU-heavy
training. Parameter counts/runtime differ; do not claim compute-matched superiority.
No cloud run is needed for this bounded pilot. Larger training belongs on Modal.

Fit LogisticRegression and HistGradientBoosting feature-only controls on the same
split. Select each threshold by maximal validation F1 (largest threshold on ties),
freeze it before test scoring. Report F1/precision/recall/FPR, ROC-AUC and PR average
precision (AP; explicitly not trapezoidal PR area), five-seed mean/sample SD,
paired differences, parameters, runtime, event throughput, state/replay bytes and
checkpoint size. Save per-event probabilities and checkpoints for neural runs.

After training, evaluate fixed TGNN checkpoints with memory reset each event,
frozen memory, GRU no-op, constant time, and shuffled age channels. Keep validation
thresholds fixed. These are interventions, not retrained ablations. Timestamp-age
shuffle preserves graph/order and destroys only the numeric-time channel; full
order destruction is deferred because it changes causal topology and memory too.
Add a toy task whose label depends only on the preceding event; verify memory can
learn and the gain vanishes when memory resets. Keep synthetic results separate.

Acceptance gates: hand-computed attention, padding, future-prefix invariance,
batch partition invariance in eval, repeatable eval from fresh state, connected GRU
gradients, label-free state, train-only scaling, reservoir train-only/uniformity,
zero/new-node handling, no-op T/M switches. Fail loudly on unsupported factors.
Generate all requested singles/pairs/full/reverse rows, marking unimplemented
factors and dependent R-only arms explicitly. No unrun row can be reported as done.

## Amendment A1 — class-support gate, before any model training

The scenario 11 sample has train/validation/test positives 1/3/314. Its late attack
onset makes validation-based model comparison too poorly supported. Preserve its
manifest as an excluded candidate; no performance scores were computed.
Use scenario 6 (Menti), another modest download, with 8192 evenly spaced records:
https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/detailed-bidirectional-flow-labels/capture20110816.binetflow
Require at least 20 positive and negative training examples and at least 10 of
each in validation and test. This choice uses class-support metadata, not model
scores. It introduces dataset-selection bias and must be disclosed. All other
protocol settings stay as registered. If this also fails, report the gate instead
of choosing models or thresholds from test performance.

## Amendment A2 — expand coverage, before real-data training

Scenario 6 at 8192 rows yields 41/14/9 positives, just below the registered test
support minimum. Record that gate failure; retain the capture and double the
label-independent sample to 16384 records once. Keep the 60/20/20 split rule and
all model/threshold choices unchanged. This adds coverage rather than reducing
the acceptance threshold. If 16384 also fails, stop this real-data pilot. The
synthetic learning test is separate and cannot justify a CTU performance claim.
