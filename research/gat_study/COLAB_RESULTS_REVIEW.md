# Old versus role-factored GAT: verified prefix results

All six runs completed (two operators, three seeds, 16 epochs) on Tesla T4 with
PyTorch 2.11.0+cu130. Recomputed AP, ROC-AUC and confusion counts from raw node
scores match the saved metrics. Every checkpoint SHA256 and bundled source hash
matches. Both arms score the same 25,008 nodes, with 13 malicious. Ground-truth
coverage is 13 of the source dataset's 72 IDs; this prefix is not full CADETS.
No archived model code was executed or checkpoint deserialized during review.

## Accuracy and cost

Means across three seeds. Percentages; false-positive counts are mean node counts.
Latency is the mean of nine paired repetition medians, batch 256, full V3 forward
plus prepared transfer and state updates, not arrival-to-alert latency.

| Metric | Old two-layer GAT | New one-layer role4 |
|---|---:|---:|
| AP | 1.426% | 1.610% |
| Recall | 7.692% | 7.692% |
| F1 | 1.669% | 1.700% |
| Mean false positives | 122.67 | 137.33 |
| Node FPR | 0.491% | 0.549% |
| Full-model batch p50 | 18.576 ms | 18.047 ms |
| Training incl. selection | 94.25 s | 89.65 s |
| Model parameters | 785,327 | 817,319 |
| Peak CUDA allocation | ~296.2 MB | ~296.9 MB |

The new operator's average batch latency is 2.85% lower and it is faster in
eight of nine paired repetition medians. This is a modest prefix observation,
not evidence that four role heads are intrinsically faster than two GAT layers:
both depth and operator change. Parameters increase by 31,992 (~4.1%). Training
times are separate sequential observations; do not claim statistically proven
training acceleration.

AP per seed (old/new, percent): 1.690/2.072; 2.038/1.667; 0.549/1.091.
Each paired stratified-node-bootstrap AP interval crosses zero. Only 13 positive
nodes make ranking uncertainty large; node scores in a graph are dependent, so
these conditional node intervals do not establish capture-level significance.

Both models detect the SAME compact node ID 16918 and miss the same other 12
in every seed at their separately benign-calibrated thresholds. Recall has no
observed gain. False positives improve only seed 0 (67 -> 53) and worsen in
seeds 1 (184 -> 209) and 2 (117 -> 150). Mean F1 hides mixed precision tradeoffs.
Do not promote the new operator to TGNN Final based on this trial.

## Degree diagnostic: structural support for roles

Every run reports the same structural histograms on the first 4,096 training
queries (an untimed audit, not the whole train/test stream):

- Query degree nine including self: 4,038/4,096 = 98.58%.
- Query self-loop-only: 6/4,096 = 0.146%.
- Query degree <=2: 13/4,096 = 0.317%.
- History-row self-only: 12.97%; degree <=2: 25.84%.
- sd has any nonself neighbor: 14/4,096 = 0.342%.
- ds has any nonself neighbor: 38/4,096 = 0.928%.
- dd has eight nonself neighbors: 3,867/4,096 = 94.41%.

Thus query neighborhoods are not tiny in aggregate: they usually connect to
all retained incident events. However, dependency-like sd/ds role groups are
rare in this audited prefix, while dd dominates. Two role heads predominantly
receive only self messages. Splitting the 128D aggregate into four 32D subspaces
may allocate capacity to seldom-used roles; this is a hypothesis, not a measured
cause of failure. Broader train/test degree measurements and endpoint types are
needed. Shared endpoint roles alone do not prove causal information transfer.

## Why the study is inconclusive for role-content accuracy

All six best checkpoints are epoch 16, and benign selection CE is still falling
at the budget boundary (~0.54–0.61 at the end). Training convergence is not
established. The graph/model learns only a 32,768-event benign prefix, though
the frozen encoder learned from the full original benign training corpus.
Restricted task coverage and relation distribution may limit detector quality.
No test family/attack-chain annotations are supplied. This trial cannot test
minority-family recall or specifically download→chmod→exec chains.

Both arms share CNN, memory, pair/compression objectives and node scoring. Those
components can dominate alert behavior. There is no no-GAT/one-original-layer
control here, and depth plus attention type change together. Data remapping
can also alter count-min collision patterns, shared across both arms. Frozen
encoder, identical non-GAT initialization, source hashes and causal integration
make the paired pilot useful, but do not resolve these scope limitations.

Calibration supports the configured 1% per-node-window budget with 663 benign
calibration windows. That is not a guarantee of 1% node-level test FPR. All 13
prefix truth nodes are scored; 59 source truth IDs fall outside the prefix.

## Recommended next experiment

Keep old GAT as the reference; role4 remains an experimental speed/accuracy
candidate. First audit degree/role/type distributions across the full train,
validation and test slices. Identify dependency-chain coverage from external
annotations before selecting a prospective evaluation slice. Do not mine
misclassified test nodes for threshold tuning.

Then compare no GAT, one original GAT, two original GATs, one role4, and two
role4 layers with matched longer training and unchanged downstream components.
Use a larger representative chronological provenance subset and predefined
minority/chain annotations. This isolates whether the 2.85% speed difference
mainly comes from removing a layer and whether role-content messages improve
chain recall. Fixed architecture updates and same alert budget matter more
than small AP changes on 13 positives. Role4-deviation, dropout and rarity
retention should stay separate until this operator/depth comparison is clear.

TGNN Final remains encoder-only. No new GPU experiment or model change was
launched during this results review.
