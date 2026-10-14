# V4 training slowdown and recovery (2026-10-11)

The interrupted run is V4 with `memory_device='array'`, not the earlier V3
affine-scan model. Its checkpoint remains at
`/study/v4_20261011_v1/V4_s0_e5/resume.pt`.

## Evidence

- User log: approximately 73–77 seconds per 100,352 events between checkpoints,
  about 1,350 events/s. Growing checkpoint stalls add additional time.
- CPU inspection: checkpoint is 7,336,430,399 bytes; saved event cursor
  3,184,640, epoch index 0, 417,128 allocated/used entities; saved elapsed epoch
  time 2,803.54 seconds. Later printed events were not in the durable checkpoint.
- A synthetic two-batch CPU profile of the old implementation found about
  569,000 function calls in 1.60 seconds; recurrence forward/backward accounted
  for approximately 1.03 seconds. This is local profile evidence, not L4 timing.
- Warmed local full training-step comparison, 2,048 events per step:
  interpreted 1.81/1.79/1.92 seconds; compiled 1.03/1.05/1.03 seconds.
  Raw observations: `artifacts/sliding_v4/cpu_speed_comparison.json`.

## Causes and fixes

1. **Sequential nonlinear surprise memory.** Current prediction produces the
   detached surprise used in the update; the next event on that entity depends
   on the new state. V3's associative affine scan is not an equivalent execution
   of this model. The CPU fallback performs Python/NumPy work for every event,
   including a tiny decoder, gates, and reverse-mode derivatives.
   `compiled_memory.py` compiles the recurrence and its existing first-order
   derivative with Numba, using typed caches and allocation-free gradient outer
   accumulation. No fast-math, stale pre-batch state, precision reduction, or
   architecture substitution is used. Floating-point results are checked within
   explicit tolerances; bitwise identity is not claimed.
2. **Large synchronous checkpoints.** Full 256D episodic vectors in sixteen slots
   per observed entity dominate the multi-gigabyte stream state. Saving and
   committing that state to a remote volume every five minutes blocks training.
   The restart serializes to local scratch, pins immutable checkpoint file
   versions with hard links, and uploads an atomic replacement in a background
   worker. An upload in progress prevents a backlog. Checkpoint cadence becomes
   fifteen minutes plus epoch boundaries. This trades a longer possible
   interruption replay interval for lower I/O overhead. Original checkpoint
   files are preserved; uploads retain the previous complete remote version
   until replacement.
3. **Local client disconnect.** The previous non-detached invocation was stopped
   when its client disconnected. Recovery uses `modal run --detach`, one remote
   training call, and one L4. It resumes weights, Adam state, RNG, stream state,
   and the saved event cursor rather than rerunning pretraining or epoch 1.
4. **Strict source signatures.** Compiled execution changes source hashes.
   Recovery accepts only reviewed execution-file changes with identical model,
   data, seed, batch, objectives, optimizer configuration, and initialization.
   All other source/configuration changes remain rejected. Both signatures are
   recorded in `execution_migration.json` in the new output directory.

## Validation and scope

Eleven local V4 tests pass, including output/gradient equivalence, dropout with
two-batch unrolling, self-edges, causal ordering, checkpoint continuation, and
rejection of incompatible source/configuration migration. The initial CUDA
probe exposed a test-only device-placement bug after switching memory backends;
the test now reapplies the model's device partition. The failed probe remains
counted in the budget. CUDA validation and a same-container real-prefix timing
comparison precede recovery; speedup below 1.2 refuses training.

The restart is in `research/modal_v4_fast_restart.py`; durable output is
`/study/v4_20261011_v1/V4_s0_e5_fast`. The existing V4 ledger includes a
600-second probe reservation and a 4,800-second recovery reservation, reaching
its original 28,800-second total timeout ceiling. No baseline retraining or
additional seeds are launched. The recovery timeout is a spending bound, not a
promise that all remaining epochs will finish. GPU startup, CPU/RAM charges,
and actual account balance are separate from the timeout ledger.

New final accuracy/F1 results remain pending until evaluation completes.

## L4 verification and live recovery

The corrected CUDA validation passed all **11 tests**. The same-container,
real-prefix training benchmark (each observation is a 2,048-event optimizer
step; first observation excluded for warmup) measured:

| Execution | Warm step seconds | Median |
|---|---|---|
| Interpreted reference | 1.4988, 1.4914, 1.5213 | 1.4988 s |
| Compiled recurrence | 0.4088, 0.4049, 0.3941 | 0.4049 s |

Measured median step speedup is **3.70×** on NVIDIA L4, approximately 1,366 to
5,058 events/s on this short prefix. This does not establish the same speedup
across every capture, mature state size, evaluation, or checkpoint upload.
Raw result: `artifacts/sliding_v4/fast_acceptance.json`.

The detached recovery app is
<https://modal.com/apps/fukatkaamsathi/main/ap-bAcaTlek8IpxUczLTmsHIR>.
Its log confirms `resume=True`, restored count 3,184,640, and advancement to
3,186,688 events with finite loss 1.0775. Only one L4 worker is active. The
restart has a 4,800-second timeout including its inline verification and setup.
