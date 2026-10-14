# Next component: causal provenance graph attention

Decision: investigate GAT next, after accepting the original encoder. No GAT is
implemented in TGNN Final yet. Current V3 has two additive single-head causal
provenance GAT layers operating on a nine-event window.

Updated priority: use the [role-message experiment](../gat_study/README.md).
It implements the revised eight-arm matrix (including role-factored values,
deviation and mean aggregation) and an untimed degree audit. GATv2 below is a
lower-priority future candidate, superseded in the first matrix by role-aware
content. Existing components in TGNN Final remain unchanged.

Reason: the encoder represents individual attributes; provenance graph attention
can connect source/destination activity. This is the next transformation after
the encoder's 64→128 projection and time features in the current V3. A failed
event-only classification does not prove that graph attention will solve it.

Small experiment matrix:

| Arm | Question |
|---|---|
| No message passing, same projection/readout | Does graph propagation add useful evidence? |
| One existing provenance GAT layer | Is the cheapest graph layer enough? |
| Two existing layers | Does additional propagation justify cost? |
| Two tied-weight layers | Can depth help without duplicating parameters? |
| One role-aware dynamic-attention layer | Does query-dependent ranking help? |

Use the original encoder for every arm. First freeze the same trained encoder
to isolate graph learning; if promising, repeat with matched joint fine-tuning.
Hold the existing nine-slot causal window, endpoint quota, time features, readout,
decoder and benign calibration protocol fixed. Changes needed to supply these
fixtures are experiment scaffolding, not accepted components in TGNN Final.
Use real endpoint IDs and chronological events from a small provenance subset.
KDD lacks the required provenance graph and is unsuitable for a definitive GAT
accuracy comparison. Synthetic link fixtures test causality, not attack accuracy.

Masks permit only past/current shared-entity events with directed endpoint-role
features; never current target relation or future context. Verify equal-timestamp
ordering, self-edges, padded windows, disconnected histories, future independence,
gradients and batch/chunk invariance of window construction before GPU training.

The current GAT uses scalar additive query/key scores plus an endpoint-role term.
Within a fixed role/mask, the monotonic activation limits query-dependent neighbor
ranking. GATv2-style vector interaction before the scalar score is a plausible
candidate. The role-aware implementation is an adaptation, not an unmodified
published reproduction. Source: [Brody et al., How Attentive are Graph Attention
Networks?](https://arxiv.org/abs/2105.14491). Its benchmarks do not establish a
TGDetect gain.

Measure AP, precision/recall at benign-calibrated alert budgets, per-attack-family
coverage, parameters, encoder-frozen setup cost, train time, GPU peak memory and
paired batch latency. Use five seeds if affordable, and a matched-data baseline.
Parameter count/FLOPs do not predict short-window kernel-launch cost. Bigger GAT
depth is not automatically better: the nine-slot history bounds evidence and
can exclude useful activity before any graph layer sees it.

Choose a candidate only after validating causal data flow and measured tradeoffs.
Keep CNN, short/long memory and rarity out of this isolated GAT contribution test.
