# Temporal 1D CNN study — 26 September 2026

User requested CNN alone, TGNN+CNN and TGNN+GRU+CNN on Modal, with better use
of one L4. Run eight arms × seeds 0–4 = **40 runs**, all 200 epochs:
GNN+H, TGNN, TGNN+Q, TGNN+Q+A, CNN+H, TGNN+C, TGNN+Q+C, TGNN+Q+A+C.
H is the same rehearsal policy; TGNN=T+H+GNN; Q is the prior local history GRU;
A is the prior additional attention; C is the new temporal convolution branch.

## Architecture and input

Preserve the frozen sample SHA256
`cf4c0d3c9eba674db8297923f97b664d3de5c5dea412b7ebca444410949a80c6`,
10 features, chronological splits 9830/3277/3277, causal four-neighbor graph,
positive support 91/25/25. This is an architecture comparison on the existing
sample, not a larger-data or long-term-memory experiment.

C is Conv1d 128→128, kernel 3, dilations 1 then 2, ReLU and dropout .2 after
each layer, **left padding only**. Mask invalid slots before and after each layer.
Convolution runs along event order, oldest neighbor to newest, then current event.
It never convolves across different batch samples or along the feature axis.
Nominal receptive field is seven positions; only five event slots are available.
Read the final position. Current observed flow features are permitted, as in
all prior models; no later event or same-time peer enters the history.

Standalone CNN+H uses Linear(10,128)+ReLU, these convolutions and a two-class
linear head. It has no GNN, numeric-age encoding or persistent memory. Hybrids
apply the CNN to the three-layer GNN's node embeddings, adding its final output
to the existing target representation after optional Q/A. Each old history node
has self-only graph edges; the current node can read past valid nodes. Shared
GNN/time/GRU/attention initialization is identical to the paired control seed.
This is a small adapted causal CNN, not a reproduction of a full published TCN.
Reference motivation: [Bai et al.](https://arxiv.org/abs/1803.01271).

## Matched optimization

Adam lr .001, decay .00001, batch 256, unweighted CE, final epoch 200, five seeds.
Rehearsal capacity 983, up to 26 examples per live batch, weight 1, dedicated
seed+937 Algorithm-R RNG, persistent reservoir across epochs, sample before admission.
All arms have zero persistent-memory contexts, so the exact rehearsal event-ID
schedule can be precomputed. Verify it against the original reservoir; never use
validation/test event IDs for training. Dropout seed starts at seed+10000.
No change to batch size, precision, epochs or dataset merely to fill GPU memory.
Threshold maximizes validation F1 (highest threshold on ties); test untouched
during training. Report all runs, mean±sample SD, paired differences, validation
ties and prior-anchor reproduction. Existing test is exploratory, with 25 positives.

## Useful L4 utilization

Use independent spawned training processes sharing **one NVIDIA L4**, each with
its own model, optimizer and RNG. Preload features/windows and replay schedules
on GPU; precompute replay scheduling to remove per-event Python work.
Benchmark 1/2/4/8/16 concurrent workers on training-only steps, selecting the best
aggregate training-step throughput within memory limits; throughput ties within
5% favor fewer workers. Failed benchmark workers do not become experimental runs.
Stop increasing concurrency after two consecutive material throughput regressions
or an out-of-memory failure. No artificial allocation to make VRAM appear busy.

Record nvidia-smi utilization, memory use and power every second during calibration
and production; report actual utilization and selected concurrency. Spare VRAM is
possible because these models are small. Concurrent wall times include contention
and cannot be compared directly with earlier isolated-run speed measurements.

## Gates, artifacts and resume

Local and actual-L4 gates: convolution causality, zero future gradients, chronological
input order, padding isolation, batch independence, all CNN/GRU gradients and weight
updates, shared initialization, zero-CNN control equivalence, exact replay schedule.
Four control arms use unchanged frozen implementations. Each result saves the final
checkpoint, probabilities, threshold, validation/test metrics, 200 epoch losses,
source/data/artifact hashes and resource measurements. Verify saved metrics and
checkpoint hashes before marking complete. Persist each result to a Modal Volume.
Resume completed results only with identical source/data/protocol hashes.
No promise that adding CNN improves F1; all variants and failures are reported.
