# TGDetect V4

User-requested redesign of V3-AttnShort. New runs only; existing baseline
checkpoints/results are reused. No performance or detection improvement is
assumed before measurement.

## Architecture

- Unchanged offline token IDs, dual hash salts and 5,303 buckets; learned width
  16. Typed field projection 48→16; four-head field pooling; event projection
  16→128. Regenerate SGNS initialization at width 16 on benign training paths.
- Unchanged balanced eight-past-plus-current window, directed endpoint-role
  features, and two independent causal 128D single-head GAT layers.
- One exact temporal attention block: four 32D heads, RoPE on Q/K using valid
  position ranks, learned 1→16→4 bias of log1p(pairwise seconds), residual/LN,
  128→256→128 GELU feed-forward residual/LN. Current-position output replaces
  both temporal CNN and subsequent local pooling. Existing input age features
  remain. Manual and PyTorch SDPA implementations have parity tests. SDPA does
  not imply that a FlashAttention kernel was selected.
- Full 256D [local context; observed event] episodic records. Eight active and
  eight short slots per endpoint; unchanged novelty/rarity admission and causal
  eviction. Attention reads at most nine deduplicated short records. No learned
  256→64 compression and no compression loss.
- Single 128D entity state. Learnable nonnegative per-dimension time decay,
  role-specific message projections, log-gap message features, 128D write gate.
  Score first, then use clipped detached CE/log(number of relations) as surprise.
  A learned scalar surprise trace per entity carries recent surprise into the
  write gate (a disclosed addition to the core 128D state). Initial decay is one
  day; all dimensions can learn different rates. No fast/slow partition.
- The 384→130→10 relation decoder and its endpoint read projections remain.
  The 64D bilinear pair head remains with a new shared 16→64 endpoint adapter;
  retaining the old head without this adapter would be dimensionally invalid.
- Loss = relation CE + 0.1 pair loss. No additional associative/compression loss.

## Research basis and exact scope

Primary sources inspected 2026-10-11:

1. [Titans, Behrouz et al.](https://arxiv.org/html/2501.00663v1), §3.1, equations
   12–15: neural associative memory updated through inner-loop gradients,
   momentum and adaptive forgetting. V4 uses prediction-surprise gating,
   learned decay and a recent-surprise trace in an entity vector. This is an
   adaptation inspired by those ideas, **not an implementation of Titans' neural
   weight memory or its gradient momentum**. Adding literal inner-loop neural
   memory is a distinct experiment with different state/computation costs.
2. [RoFormer](https://arxiv.org/abs/2104.09864): rotary query/key positions.
   Pairwise elapsed-time bias is this proposal's additional mechanism.
3. [Temporal Graph Networks](https://arxiv.org/abs/2006.10637): chronological
   node memory and graph context. No claim that these papers establish V4's
   intrusion-detection accuracy or novelty.

Memory is nonlinear and depends on the relation prediction. V3's affine scan
cannot implement it exactly. A CPU metadata pass builds dependency waves;
events in a wave have disjoint endpoints and execute together. Waves preserve
all endpoint dependencies; self-edges write once. This can be slower for hot
entities, and is measured before launching full training. No stale pre-batch
state is substituted for causal state.

## Evidence archive

`archive.EventArchive` stores every supplied original byte record independently
of neural admission, with SHA-256 and entity/time indexes. Repeated identical
IDs are idempotent; conflicting evidence is rejected. Retrieval requires an
ingestion cursor and time bound. It is an ingestion API, not a claim that hashed
training arrays recover original telemetry. Existing prepared CADETS data must
be identified as prepared evidence. Archive retrieval is separate from this
first fixed-capacity neural-cache experiment; extra retrieval is not silently
added to model inputs. Full vectors are lossless relative to those learned
vectors, not the original attributes. Persisted vectors become stale as model
weights change; training detaches state at the same optimizer boundaries as V3.

## Matched protocol

Existing CADETS E3 manifest:
`/study/v3_vs_orthrus_20261010/prepared_v3_local/manifest.json`.
Keep original checksums and all chronological splits. Five epochs, seeds 0/1/2,
batch 1,024, two-batch unroll, Adam (embedding lr 1e-4, other lr 1e-5,
weight decay 1e-5), norm clipping 1, cold split state, TF32 disabled.
Checkpoint selection: minimum relation CE on the first third of benign
validation. Remaining validation calibrates combined CE + 0.1 pair evidence
with the same 0.001 false-alert setting. Test labels only evaluate predictions.
Frozen encoder lookup at evaluation uses the same prepared identities as V3.

The user explicitly excluded baseline retraining and encoder ablation runs.
`modal_sliding_v4.py` permits only V4 seeds 0–2, at five epochs. GPU acceptance
is bounded to 900 seconds; each full job is bounded to 7,200 seconds with no
automatic duplicate launch. Results and failures both remain in the ledger.
The 28,800-second total timeout guard includes development probes and is an
execution ceiling, not an invoice. Failed probes remain counted in the ledger.

```bash
~/main_env/bin/python -m pytest -q research/test_sliding_v4.py
~/main_env/bin/python -m modal run research/modal_sliding_v4.py --action inspect
~/main_env/bin/python -m modal run research/modal_sliding_v4.py --action acceptance
~/main_env/bin/python -m modal run --detach research/modal_sliding_v4.py --action train --seed 0
```

Validation covers exact attention backends, padded windows, future/target
causality, repeated endpoints and self-edges, sequential/batch agreement,
stream save/restore, trainable-module gradients, repeated optimizer steps,
interrupted training continuation, and exact archive bytes/cursor filtering.

## Fast execution and interrupted-run recovery

`array_memory.py` now defaults to the compiled recurrence in
`compiled_memory.py`, requiring `numba==0.61.2`. Set
`TG_V4_COMPILED_MEMORY=0` for the interpreted reference. Compilation has a
one-time warmup cost; warmed training timing is reported separately. The
nonlinear surprise memory and its gradients are preserved within tested
floating-point tolerances. See `../findings/v4_training_latency_fix.md` for
measurements and the checkpoint I/O changes.

`../modal_v4_fast_restart.py` recovers the existing seed-zero checkpoint into a
separate output directory, uses one L4, checks CUDA parity and speed before
training, and overlaps immutable checkpoint uploads with computation. The
resume invocation must use detached mode. Its reservations share the existing
V4 ledger; repeated paid launches are rejected.
