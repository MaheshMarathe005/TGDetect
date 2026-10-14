# Remaining latency audit, 2026-10-11

The existing detached training job was left running without a restart, source
upload, optimizer change, GPU profile or extra GPU allocation. One active
container was verified for app `ap-bAcaTlek8IpxUczLTmsHIR`. The changes below
are local, for a future run. The container retains its original source snapshot.

## Measured status and version distinction

The active model is V4 (RoPE temporal attention, full 256D episodic vectors,
nonlinear surprise-gated state), although the request calls it V3. V3 uses
CNN/optional causal attention, compressed 64D episodic vectors and associative
affine state updates. Their latency profiles cannot be interchanged.

The earlier compiled-recurrence recovery measured 3.70x L4 training-step speedup
on a short real prefix. Subsequent complete epochs took 745.8 s and 753.7 s for
3,526,541 training events, approximately 4,729 and 4,679 events/s respectively.
Epoch 3 validation CE was 0.0646; this is relation-prediction loss, not attack
accuracy, node F1 or AP. At this audit's final check epoch 4 had passed 500k
events. No final V4 attack metrics are claimed here.

## Causes, remedies and implementation status

| Code path | Why it still costs time | Remedy and status |
|---|---|---|
| `array_memory.run` and `compiled_memory.forward/backward` | Every event runs a small nonlinear decoder and gate; the compiled loops remain sequential. Outputs affect subsequent entity state. | Keep the compiled implementation already deployed. Consider fused kernels that parallelize matrix/vector arithmetic inside each event; benchmark before replacing it. Not implemented here. |
| `model.forward_batch` CPU memory branch | Fused GPU features and touched entity rows move to CPU; logits/state return to GPU. Backward crosses the same boundary. The dependency prevents treating all transfers as independent asynchronous work. | A fused device-resident recurrence could remove the boundary, but needs output/gradient/unroll tests and a new L4 comparison. Existing Python per-wave GPU execution is not automatically faster. |
| Temporal attention followed by `h[:, -1]` | All nine query outputs, 81 pairwise elapsed-time biases and nine FFN outputs were computed, though only the last output was consumed. | **Implemented locally:** `TemporalAttention.forward_last`, used by V4. Compute one query readout, nine time biases, and one output projection/FFN. All history K/V and the shared QKV projection are retained. Same parameter shapes and objective. |
| Synchronous `read_batches` CPU producer | Memmap column copies and tensor construction happen between model calls. | **Implemented locally, opt-in:** `--prefetch-batches` overlaps one CPU batch read with training. One worker, bounded lookahead, same order and resume cursor, no model/RNG access. L4 benefit unmeasured; can be negligible or negative when CPU contention dominates. |
| Ring planning, `unique`, sorting, `searchsorted`, dynamic short-read `nonzero` | Many launches and indexing allocations, including decisions involving dynamic sizes. | Measure the mature-state GPU path before fusing or using static short reads. Earlier V3 static reads were slower than lookup alone, and partial CUDA graphs had large tail outliers. No blanket graph/static conversion applied. |
| Full episodic rings and state allocator | Sixteen 256D FP32 vector slots use 16,384 bytes/entity before metadata and persistent state. Growing capacity copies existing buffers; reset fills capacity; checkpoint snapshots copy used state. | The existing asynchronous checkpoint fix remains in place. Future paged growth/lazy reset must be checked for stale-node resurrection and checkpoint parity. Compression/FP16 would require a separate accuracy study. |
| `dependency_waves` constructed for array execution | The array backend does not consume the schedule for its recurrence; the count is still returned as a diagnostic. | Low priority. A future optional diagnostic can avoid constructing lists. No fabricated wave count or silently removed diagnostic. |

## Parallelism: measured limits on actual data

Read-only CPU metadata analysis sampled four real training batches, without
launching a new GPU worker or running GPU kernels in the training container:

| Batch start | Events | Dependency waves | Mean events/wave | Maximum events/wave |
|---:|---:|---:|---:|---:|
| 0 | 1,024 | 818 | 1.25 | 6 |
| 100,352 | 1,024 | 223 | 4.59 | 10 |
| 1,003,520 | 1,024 | 270 | 3.79 | 11 |
| 3,000,320 | 1,024 | 1,024 | 1.00 | 1 |

An event's wave level is one plus the latest level of either endpoint. Events
within a wave have disjoint endpoints. The final sample shares a hub in every
event, so exact inter-event parallelism is one there. These four samples are
diagnostics, not a dataset-wide distribution. Raw counts:
`artifacts/sliding_v4/dependency_wave_samples.json`.

GPU GAT, encoding and window attention already operate on batches. Safe further
parallelism is CPU data preparation, intra-event matrix arithmetic, and disjoint
endpoint groups when the workload exposes them. A fused kernel can avoid hundreds
of Python launches even when the state dependency is sequential. Its latency
gain remains a hypothesis until measured. Naive multi-GPU/shuffled-event training
would require synchronizing shared entities, and stale pre-batch state changes
the model. V3's affine scan parallelizes hub updates because its transition has a
different algebraic form; adopting that transition in V4 is an architecture
ablation requiring retraining and held-out AP/F1 comparison.

## Validation and impact of the new local changes

Sixteen local V4 tests pass. New checks compare final-position outputs, input
gradients and every attention parameter gradient against the full attention
reference for both manual and SDPA backends. They cover all-padding, invalid
final queries, noncontiguous masks and small/equal epoch-scale timestamps. An
end-to-end two-batch training unroll compares all model gradients with dropout.
CPU prefetch tests check exact batch columns/order/start cursor, propagated
errors and early-close cleanup. Checkpoint/resume testing compares prefetched
training with an uninterrupted synchronous reference, including final metrics
and model weights within explicit floating-point tolerances.

Local CPU temporal-module forward+backward benchmark, batch 256, window 9,
one thread, alternating arm order and 12 measured observations per arm:

| Execution | Median |
|---|---:|
| Full sequence, consume last position | 41.87 ms |
| Last-position read | 18.34 ms |

This is **2.28x for that CPU module benchmark only**. It is not an L4 speedup,
whole-training speedup, or measured accuracy/F1 improvement. Script:
`research/profile_v4_last_read_cpu.py`; raw observations:
`artifacts/sliding_v4/last_read_cpu.json`. CPU Torch version is 2.11.0; the
running Modal image uses Torch 2.4.1. Newly added changes have not had CUDA
validation. The earlier 11-test CUDA acceptance applies to the running snapshot.

## Next steps after the existing run

1. Preserve its source signatures, checkpoints and final metrics before any
   execution change. Current strict migration guards reject the changed model
   and added module; do not bypass them for an unattended resume.
2. With a separately bounded allowance, run CUDA output/gradient tests and a
   paired L4 full training-step comparison of full vs last-position attention.
   Measure prefetch on/off separately, with identical weights, RNG, state and
   batches. Report mature-state p50/p95, transfer spans and peak memory.
3. Profile recurrence forward/backward and CPU/GPU idle gaps on that disposable
   benchmark. Only then prioritize fused recurrence vs ring/kernel work.
4. If evaluating V3's affine scan as an alternative, use a separate architecture
   arm with retraining, same split and calibrated AP/F1. No accuracy-preserving
   claim can be made from an execution benchmark alone.

The existing V4 timeout reservation is fully allocated, so this audit launched
no additional paid GPU experiments. The original training continues.
