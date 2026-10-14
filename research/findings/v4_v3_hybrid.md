# V4 hybrid: restored V3 encoder and node memory

Requested change: replace V4's encoder and persistent node memory with V3's,
then start a fresh five-epoch run. Existing V3 and V4 outputs are preserved.

| Component | Completed V4 | New hybrid |
|---|---|---|
| Attribute encoder | Typed KV, two hashes/ordered subtokens, 16D | V3 `PackedEncoder`, same token bank, 64D |
| Feature projection | 16 to 128 | 64 to 128 |
| Pair head input | Learned 16-to-64 adapter | Direct 64D encoder output, as in V3 |
| Persistent node update | Sequential surprise-gated nonlinear state | V3 `endpoint_scan`: associative affine updates, 128D fast/slow state |
| State time decay | Learned per-dimension decay | V3 fixed 1-hour/1-day half-lives |
| Write admission | Surprise-based updater | V3 learned admission from temporal importance, novelty and rarity |
| Device boundary | CPU array recurrence and decoder | Encoder, scan and decoder on GPU |
| Temporal module | Exact RoPE + elapsed-time attention | Retained |
| Event-line GAT | Two causal layers | Retained |
| Episodic rings | Full 256D, eight history + eight short slots | Retained |
| Loss | Relation CE + 0.1 pair loss, no compression loss | Retained |

The hybrid uses the full temporal output to calculate V3's importance-based
admission signal, while retaining V4's final-position temporal context. It does
not add V3's importance-pooled residual to the context or restore V3's CNN.
The later last-position-only execution cannot supply the admission signal's
earlier temporal outputs, so it is used only for the ordinary V4 variant.

The full episodic state is still large. Restoring the affine node updater removes
nonlinear event dependence and CPU transfers; it does not remove episodic storage,
ring sorting/gathers, or every kernel launch. Speed and accuracy gains require
the new measured results.

## Validation and restart protocol

18 local tests pass. Hybrid-specific checks cover sequential versus batch
execution, self-edges/hubs/equal timestamps, target/future causality, stream
checkpoint continuation, rejecting an ordinary V4 checkpoint, and gradients
through the encoder, time attention, both affine writers, admission, importance
and episodic reads across two batches. Existing V4 tests remain passing.

Training starts from fresh model/optimizer/stream state, seed 0, batch 1,024,
two-batch unroll, five epochs, the same prepared CADETS split and calibration.
The original V3 benign-only 64D pretrained table is reused, with its exact digest
checked. This is an initialization reuse, not a resume from trained V3/V4 weights.
The incompatible completed V4 checkpoint is preserved.

The corrected launch passed **all 18 CUDA tests** on the L4. The disposable
prefix measured 2,048-event optimizer steps of 1.8171 s (cold), 0.1162 s,
0.1140 s and 0.1208 s. Its approximately 1,000-second estimate covers five
training epochs only; it does not establish mature-state runtime or accuracy.
The run then started epoch 1 from scratch and passed 200,704 events in 11 s.
Raw remote acceptance is saved beside the output, and copied locally to
`artifacts/sliding_v4/hybrid_acceptance.json`.

Active detached app:
<https://modal.com/apps/fukatkaamsathi/main/ap-OS9eZHxU1KkS4C7uYT6PTn>.
Local log: `/tmp/v4-v3-hybrid-attempt2.log`.

Driver: `research/modal_v4_v3_hybrid.py`. One L4, four CPU cores, 32 GiB requested
RAM, 3,600-second timeout including GPU acceptance/smoke checks, no automatic
function retries. CUDA tests run before training and a short-prefix estimate
must fit the allowance with a margin. CPU prefetch is enabled; final evaluation
uses a frozen encoder lookup. No baseline retraining is launched.

Durable output:
`/study/v4_v3_hybrid_20261011_v1/V4_V3EncoderMemory_s0_e5`.
Local final result will be
`artifacts/sliding_v4/V4_V3EncoderMemory_s0_e5.json`.

The initial launcher failed during container import before tests/training because
its helper module was not mounted. The failed app was stopped and the driver was
made self-contained. Its entire 3,600-second reservation remains counted.

## Existing budget accounting

The interrupted original V4 app was verified stopped. Modal's app listing gives
creation 2026-10-11 12:23:37 IST and stop 13:14:13 IST. Its old 18,000-second
reservation is retained in the ledger, with a settled conservative bound of
3,156 seconds (full app lifetime plus 120-second teardown margin). Only its
provably unused reservation is released. Every other old reservation, including
failures, remains counted at its full timeout.

Including the failed hybrid startup and corrected 3,600-second attempt,
the accounted upper bound is 21,156 seconds of the original 28,800-second cap.
This is GPU-time reservation accounting, not an invoice or a statement of the
current Modal dollar balance. At the user's quoted $0.80/L4-hour rate, one new
one-hour worker is at most $0.80 for GPU time; CPU, RAM and storage cost extra.
