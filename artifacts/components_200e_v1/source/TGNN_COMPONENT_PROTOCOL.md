# Controlled TGNN component study — 24 September 2026

## Frozen reference

The completed five-seed L4 reference is GNN + rehearsal F1 **86.61 ± 4.84%**
and TGNN = GNN + time + rehearsal F1 **84.52 ± 2.63%**. These are means and
sample standard deviations, not confidence intervals. The complete metrics,
per-seed predictions, checkpoints and hashes remain unchanged in
`MATCHED_TIME_REHEARSAL_REPORT.md` and `artifacts/modal_matched_time_rehearsal_200e/`.
TGNN PR-AP was 94.22 ± 0.49%, versus GNN 88.18 ± 5.30%.

## Registered grid

All additions are hypotheses, not assumed improvements. `H` always means the
same 10% Algorithm-R rehearsal. Interpret the requested “rel” as rehearsal.

| Symbol | Exact addition |
|---|---|
| T | Existing learned cosine encoding of log neighbor age, 256 parameters |
| Q (GRU) | GRUCell(128,128) over the same four strictly earlier neighbor embeddings, oldest to newest, skipping padding; zero state per ego; residual addition to target embedding |
| M (memory) | Persistent endpoint EMA: m ← 0.5m + 0.5h; pre-event context (m_src+m_dst)/2; score before update; all same-time events use strictly prior memory |
| A (extra attention) | Separate 128-dimensional query/key/value projections; scaled dot-product attention over valid earlier neighbors; residual readout; no history contributes zero |
| N | LayerNorm(128) immediately before the classification head |
| W | Training-only class weights [1, sqrt(N_negative/N_positive)] in mean cross entropy, applied identically to live and replay losses |

Run GNN+H, GNN+H+A, all eight TGNN combinations of Q/M/A (including TGNN
alone), TGNN+N, and TGNN+W: **12 arms × seeds 0–4 = 60 runs**.
The inherited GNN already uses additive graph attention. A tests an *additional*
attention readout, not the first attention layer. Q is a neighbor-sequence GRU,
not the earlier pilot's R that replaced the persistent memory update with a GRU.
Q and M are independent, making TGNN+GRU, TGNN+memory and TGNN+GRU+memory
separately interpretable. This grid does not test every possible neural component.

## Fixed training and information

One Modal invocation, one NVIDIA L4, profile `prathampansare25`, main_env client.
PyTorch 2.7.1, NumPy 2.2.6, scikit-learn 1.7.2; deterministic algorithms.
Same frozen scenario06_pilot16384.npz, SHA256
`cf4c0d3c9eba674db8297923f97b664d3de5c5dea412b7ebca444410949a80c6`.
Train/validation/test: 9830/3277/3277 flows, positive counts 91/25/25.
Same 10 static features, train-only scaling, strictly earlier four-neighbor graph.
Same 3×128 GAT, ReLU, dropout .2, head, Adam lr .001, decay .00001,
batch 256, **200 epochs for every arm and seed**, final-epoch checkpoint.
No test-driven early stopping, epoch choice, threshold change or grid expansion.
Reset memory at epoch start; detach only after an optimizer step. Evaluate a
chronological train→validation→test pass from zero memory without updating weights
or using labels in state. Rehearsal capacity 983, up to 26 samples per batch,
weight 1, dedicated seed+937 RNG, persist buffer across epochs, admit after replay.
Replay uses a detached stored pre-event context and cannot mutate live memory.
Shared GNN/time tensors retain seed-matched initialization; dropout RNG starts at
seed+10000 after module construction. Extra modules change computation/capacity.

The EMA execution is vectorized using constant causal recurrence coefficients;
it is checked against the original sequential model's outputs and gradients,
including timestamps tied across batch boundaries. Reservoir optimization skips
cloning rejected snapshots without changing sampling decisions or RNG order.
Original source files and original results are preserved. A durable Modal Volume
is committed after each completed run; resume only when source/data/protocol hashes
match. Complete checkpoints, predictions, per-epoch losses and verification remain.

## Gates and reporting

Before cloud training: old causal tests plus component reference equivalence,
gradient flow, updates, future/prefix and batch invariance, padding, ties, new nodes,
replay isolation, memory reset, ordering guard and exact reservoir RNG tests.
Repeat component tests on the actual L4 before training. Stop on a failed gate.
GPU name and data checksums must pass. Each saved result is verified by recomputing
validation-selected threshold and test metrics from saved predictions; all artifact
hashes recorded. A new baseline rerun is reported separately if numerical differences
occur; never overwrite the frozen 86.61/84.52 reference.

Primary metric F1; secondary PR-AP, precision, recall, FPR, ROC-AUC, accuracy,
balanced accuracy; parameters, training time, inference time and peak allocated GPU
memory. Threshold maximizes validation F1, highest threshold breaks ties. Report all
five seeds, mean±sample SD, paired differences against both rerun anchors, and complete
grid. Validation mean F1 determines the preferred configuration; separately identify
highest observed test mean, explicitly exploratory. Test has only 25 positives and
has been inspected in previous work; 60 runs do not constitute an untouched test or
60 independent datasets. Final claims need confirmation on a new capture.

Related primary sources motivating, not specifying, these adapted components:
[TGN](https://arxiv.org/abs/2006.10637) (memory and neural operators),
[TGAT](https://arxiv.org/abs/2002.07962) (temporal attention).
Persistence follows [Modal Volumes](https://modal.com/docs/guide/volumes).
