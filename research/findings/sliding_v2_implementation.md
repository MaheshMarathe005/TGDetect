# Sliding V2 implementation and latency validation

10 October 2026. Implementation: `research/sliding_node_mem_v2/`.
Usage and limitations: [V2 README](../sliding_node_mem_v2/README.md).
Original diagnosis: [weakness audit](sliding_model_weakness_audit.md).

## Code changes

| Identified issue | Implemented change | Scope |
|---|---|---|
| Repeated Python-to-CUDA index construction | Bulk NumPy metadata packing and CPU-planned affine endpoint scan | Both profiles |
| Repeated compressed-history gathers | Gather shared backing blocks once, with sparse fallback | Both profiles |
| Re-encoding unchanged attributes during inference | Bounded, weight-versioned frozen embedding cache | Both profiles; disabled with gradients |
| Underoccupied token references pin whole batches | Compact sparse detached blocks; scheduled repacking limited to eight blocks every 32 detaches | Both profiles |
| Small CPU convolution launch overhead | Algebraically equivalent linear implementation for batches up to 64; CUDA retains dense convolutions | Both profiles |
| Busy endpoint crowds out quiet history | Reserve half the active-history budget per endpoint, deduplicate and fill | Corrected profile |
| Sticky admission and rare-event crowding | One observation-time decision; separate regular/exception quotas, age-decayed priority | Corrected profile |
| Value-order collisions | Position-tagged adjacent subtoken hashes | Corrected profile; hashing remains lossy |
| Forced repetitive writes and event-count-only forgetting | Zero default admission floor, contextual writes and elapsed-time fast/slow decay | Corrected profile |
| Mismatched compression target scales | Normalized contextual compression objective | Corrected profile |
| Detaching every batch cuts memory-write credit | Two-batch chronological unroll, one backward/update, sparse pending memory | V2 runner |
| Final-epoch-only selection | Best benign validation-loss checkpoint | V2 runner |
| Dilution of isolated anomalies and top-30 recall cap | Preserve isolated and repeated evidence; disjoint calibration; uncapped threshold alerts | V2 runner |
| Per-event score serialization overhead | Buffered binary edge batches, memory-mapped calibration inputs | V2 runner |

Both default profiles have 764,202 parameters. Compatible mode accepts the upgrade
model's weight dictionary with matching dimensions, and retains its prediction
semantics. V2 runtime snapshots carry a checked configuration. The corrected
profile changes semantics and requires retraining. The old model directories and
old experiment outputs have not been rewritten.

## Why latency was high

The L4 baseline profile showed roughly 0.49 seconds across four real 1,024-event
batches. Large contributors were Python forward processing, 78 `torch.tensor`
calls (about 84 ms), and compressed-token gathering (about 65 ms). An individual
PyTorch operator profile recorded about 21 ms self CPU time and 2.6 ms self CUDA
time. These nested timings are not additive and do not represent a full training
run, but identify substantial host overhead around a relatively small GPU workload.

CUDA convolution was faster than its equivalent shifted linear implementation,
so the optimized CUDA path keeps all convolution channels and weights. A longer
benchmark revealed a compaction p99 regression that the short benchmark missed;
scheduled repacking was subsequently bounded. Eligibility discovery still scans
live state, so tail latency at much larger entity counts remains an open limit.

## Validation

The combined suite passed 65 tests across V2, sliding, upgraded sliding, node
memory and ORTHRUS. One additional partial-compaction regression test passed
afterward: **66 passing tests in total**. One legacy test was deselected because
`artifacts/sliding_node_mem_e3/performance_before/model.py` is missing.

Coverage includes compatible logits, gradients and optimizer updates; exact scan
and CNN gradients; causal/batch-invariant inference; quiet-endpoint history;
cache eviction and invalidation; shared-row compaction; snapshot resume;
cross-batch gradients; elapsed-time decay; evidence preservation; uncapped
scoring; and a synthetic interrupted/resumed training run. L4 prediction parity
also passed on four chronological real batches with float32 tolerances.

## Final L4 forward measurements

Real CADETS prepared training prefixes, eight warmup batches plus 64 measured
batches per size, including scheduled compaction at detach counts 32 and 64.
Float32, one PyTorch CPU thread, pinned remote PyTorch 2.4.1. Models use seeded
initial weights; compatible mode copies the reference weights. These are not
benchmarks of a newly trained detector. Retention and cache workloads can differ
with other weights, input prefixes and longer streams.

| Events/batch | Reference median ms | Compatible median ms | Corrected median ms |
|---:|---:|---:|---:|
| 1 | 10.309 | 8.200 | 9.131 |
| 32 | 13.426 | 10.888 | 11.859 |
| 256 | 29.229 | 23.764 | 28.261 |
| 1024 | 86.122 | 61.573 | 78.655 |

At batch 1,024, compatible throughput increased from 11,554 to 15,084 events/s
(30.5%). Median latency fell 28.5%, and p95 fell from 92.578 to 79.861 ms.
However, **p99 increased from 94.952 to 124.436 ms**. Bounded repacking reduced
the preceding run's maintenance spike, but it did not resolve the tail regression;
run-to-run variation also prevents isolating that change's effect from these
aggregate timings alone. This implementation is not a hard-latency guarantee.
Measured peak allocated CUDA memory was 99,347,968 bytes for reference and
103,089,664 bytes for compatible on this short prefix (about 3.8% higher).

Corrected mode reached 12,230 events/s at batch 1,024, but had worse p95/p99
(99.926/134.069 ms). At batch 1, compatible p99 was also worse; at batch 32 it
improved; at batch 256 it worsened. Do not extrapolate median improvements to
all latency percentiles or workloads. Investigating incremental maintenance
without full live-state scans remains the next tail-latency task.

## Interpretation and remaining work

There is no measured real-data F1/accuracy improvement yet. Existing five-epoch
results describe the older models. The new runner must retrain and evaluate the
corrected profile before adopting its detection changes. Lower benign relation
loss alone does not prove better anomaly discrimination.

The CPU unique-attribute stress test found the corrected encoder/model could be
about 1.6 times slower than the old model at batch 1,024, while compatible mode
improved throughput. These measurements preceded only the final maintenance
budget adjustment (the short test never reached scheduled compaction). This is
a material tradeoff, not an unconditional efficiency improvement.

Default memory decay, retention quotas and the auxiliary objective need separate
ablations. Two-batch training increases activation memory and reduces optimizer
updates per epoch. Global entity state remains proportional to distinct observed
entities. The empirical alert budget is per node-window; whole-stream peak
selection and distribution shift prevent interpreting it as a guaranteed
whole-run false-positive rate. Hash collisions, truncation, proxy-task mismatch
and missing ground-truth coverage remain limitations.

Benchmark artifacts are under `artifacts/sliding_v2/`. `gpu_latency.json` contains
the final measured source hashes and detailed results; older snapshots document
the rejected/unbounded-compaction measurements. These are prepared-input forward
benchmarks including state update and maintenance. They exclude ingestion,
scoring, file I/O, optimizer steps and batch-arrival waiting. Small prefix memory
peaks must not be extrapolated to a full multi-day training run.
