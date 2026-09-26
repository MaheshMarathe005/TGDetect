# Does more data make temporal memory more useful?

Research date: 26 September 2026. Status: hypothesis review and measured
implementation/data audit; no new training or GPU experiment performed.

## Assessment of the hypothesis

The user's hypothesis is plausible: a small, heavily thinned dataset can remove
useful event sequences and give a larger temporal model too few examples to learn
from. A short observation period also cannot establish performance over days or
weeks. However, the experiment does not show that sample size caused the 0.23-point
F1 gap, or that more data will produce a large memory advantage. Larger data can
benefit the GNN too, and effective memory depends on retention and task structure.

The defensible hypothesis is: **dense, continuous histories with predictive
long-range dependencies may benefit a suitably trained persistent-memory model
more than a short-context model.** This is testable; it is not a completed finding.

## What we actually tested

| Mechanism | What it does in this project | Can it retain long-running endpoint state? |
|---|---|---|
| Rehearsal H | Replays stored training examples and detached pre-event contexts during optimization | Helps train weights on past examples; not an inference-time endpoint-state mechanism |
| Four-neighbor graph | Gives the current event the four most recent strictly earlier incident events | Explicit small history; available even to GNN+H |
| History GRU Q | Reads those four neighbor embeddings oldest-first, with zero state for each target | No state carried from one target evaluation to the next |
| Persistent memory M | Stores a 128-dimensional EMA state per endpoint and uses prior state to score | Yes, but old direct contributions shrink quickly with endpoint activity |
| Pilot R, not Q | Replaces the persistent EMA updater with a GRUCell | Potential learned retention; only the earlier three-epoch pilot and toy test used it |

Therefore the 86.39% TGNN+GRU+attention result is **not** a result for learned
long-term persistent memory. Its factors are T+H+Q+A; M is absent. The 60-run grid
did not test a persistent GRU updater at 200 epochs. Its Q+M arm combines a local
history GRU with a separate fixed-EMA persistent state. The distinction was in the
registered protocol, but is essential when reasoning about long-term detection.

Memory is reset at each training epoch's start to replay the chronology from zero.
An epoch is another pass through the same events, not another day of observation.
Memory values persist across 256-event training chunks; `detach_state` removes
cross-chunk gradient history, not stored values. Thus state can carry information
longer than the gradient-training horizon. At evaluation, state is reset once,
then warmed chronologically through train, validation and test with fixed weights
and without feeding labels into the state. There is no state reset at test start.

Evidence: [model](../../tgnn_components.py), [runner](../../run_tgnn_components.py),
[state handling](../../causal_tgn.py), [protocol](../../TGNN_COMPONENT_PROTOCOL.md).

## Measured data limitations

The checksum-verified dataset contains **16,384 of 558,919 raw flows (2.9314%)**:
about one retained flow per 34 raw flows. The full file has about **34.11 times** as
many events. Selection was evenly spaced in completion-time order, not random
Bernoulli sampling and not label-selected.

The retained sample includes the first and last chronological raw events and spans
**7,749.87 seconds = 2.15274 hours**, approximately 2 h 9 min. The sample is sparse
across the entire capture; it is not simply an initial short contiguous excerpt.
Using all raw flows increases event density, but does **not** extend the capture
into days or weeks.

| Partition | Events | Positive flows | Observed span |
|---|---:|---:|---:|
| Training | 9,830 | 91 | 78.61 min |
| Validation | 3,277 | 25 | 27.02 min |
| Test | 3,277 | 25 | 23.52 min |

Only 91 positive training events and 25 positive test events give limited evidence
about diverse attack sequences. One more detected positive changes test recall
by four percentage points. A flow is not necessarily an independent attack.

**Memory was not deprived of all past activity.** Before prediction, 3,265 of 3,277
test events (99.63%) had prior activity for at least one endpoint. 3,240 test events
already had four valid graph-history neighbors. Median test neighbor age was
3.19 seconds; the 90th and 99th percentiles were 88.22 and 2,362.70 seconds.
These are descriptive event ages, not evidence of predictive long-range causality.

Thinning can remove intermediate clues in an attack chain. That is a plausible
mechanism for lost temporal signal, not something proven by the current F1 scores.
Conversely, longer gaps between retained events can make four sampled neighbors
cover more clock time than four neighbors in the dense stream. Restoring all data
changes both available evidence and the context window's effective time span.

Diagnostics: [JSON](memory_horizon_audit.json), reproducible with
`~/main_env/bin/python audit_memory_horizon.py`.

## Why our fixed memory is a weak long-term-retention test

The update is `new_state = 0.5 * old_state + 0.5 * current_embedding`.
After k further updates of that endpoint, the direct old-state contribution is
multiplied by `0.5**k`:

| Further endpoint updates | Old-state contribution remaining |
|---|---:|
| 1 | 50% |
| 4 | 6.25% |
| 5 | 3.125% |
| 7 | 0.78125% |
| 10 | 0.09765625% |

Expanding the recurrence gives 96.875% of total coefficient mass to the five most
recent update embeddings. This concerns direct linear weights, not necessarily
96.875% of useful information: embeddings themselves can already contain history,
and their magnitudes and redundancy vary. It is not a hard five-event cutoff.

Decay is by **endpoint update count**, not wall-clock age. An idle endpoint can
retain its state for hours; a busy endpoint can replace nearly all of its direct
old-state contribution in seconds. Dense traffic can therefore shrink this
memory's effective clock-time horizon. More data alone does not repair retention.

A learned persistent GRU updater, or separate fast and slow memory states, could
retain selected clues for longer. Both are hypotheses requiring delayed-dependency
tests and matched real-data controls. A GRU architecture does not guarantee that
training actually teaches it to preserve the relevant old signal.

## What the existing memory comparisons say

These pairs add only M within the completed grid. Differences are mean percentage
points; they are descriptive and not significance claims.

| Fixed model before adding M | F1 change from M | PR-AP change from M |
|---|---:|---:|
| TGNN | −0.66 | +0.55 |
| TGNN + attention | −6.31 | +1.41 |
| TGNN + GRU | +0.07 | +1.02 |
| TGNN + GRU + attention | −2.24 | −0.05 |

Current EMA memory does not show a consistent F1 advantage. It sometimes improves
ranking while lowering recall at the validation-selected threshold. This neither
proves memory is useless nor proves more data will reverse the result.

## What primary research supports

**TGN:** The original framework compresses node history in persistent state, uses
learned recurrent memory updates, and reports benefits from memory in matched
ablations. Its results support investigating retained historical information, but
do not predict our CTU intrusion-classification F1 or guarantee a scaling benefit.
[Rossi et al., Sections 3.1 and 5.2](https://arxiv.org/pdf/2006.10637).

**Temporal Graph Benchmark:** Benchmarks spanning long periods and several domains
show substantial dataset dependence, including strong simple-model results on
dynamic node prediction. Size and duration alone do not determine the winner.
[Huang et al.](https://arxiv.org/abs/2307.01026).

**ORTHRUS:** This security study uses temporal history sampling with a stateless
encoder. Its reported encoder ablation favors that design over a state-memory
alternative. It is relevant counterevidence to a universal claim that persistent
state must improve detection, not a direct reproduction of our task.
[Section 4.3 and encoding ablation](https://www.usenix.org/system/files/usenixsecurity25-jiang-baoxiang.pdf).

**METANOIA:** Its lifelong IDS setting explicitly addresses concept drift,
catastrophic forgetting and avoiding adaptation to malicious behavior. This
illustrates why a multi-day online-learning evaluation is a different question
from classifying a held-out slice with fixed weights.
[Version 2, 17 March 2026](https://arxiv.org/abs/2501.00438v2).

**CTU-13:** The data owner describes captured botnet network traffic. These flows
are useful for controlled studies, but do not by themselves establish persistent
APT detection across endpoint process/file/network histories.
[Owner documentation](https://www.stratosphereips.org/datasets-ctu13).

## A controlled way to test the hypothesis next

This is a proposed design, not a launched or fully preregistered experiment.
Exact captures, budgets, cutoff times and candidate counts must be frozen before
looking at new model results.

1. **Separate training volume from history density.** Freeze chronological time
   boundaries and scored validation/test event IDs. Vary training examples within
   the training prefix, and separately vary how many unlabeled prior events are
   available to update histories. A small-versus-large training set crossed with
   sparse-versus-dense causal context separates these two effects. Apply each
   information policy to every model. Keep labels unavailable to state updates.
2. **Separate density from elapsed duration.** Full scenario 6 restores events
   across the same 2.15 hours. A days/weeks hypothesis requires a suitable longer
   continuous capture with valid endpoint identity and timestamps. Do not create
   artificial continuity by joining unrelated captures or carrying memory between
   unrelated machines that happen to reuse an IP address.
3. **Use a small matched model set.** Include GNN+H, plain TGNN, current TGNN+GRU+A,
   the EMA-memory control and a learned persistent-GRU memory model. To study
   memory on top of GRU+A, compare that exact backbone with and without state.
   Keep replay, features, labels, optimizer and threshold selection matched.
   Include a larger stateless context or matched-capacity control where feasible
   so access to more history or more parameters is not mislabeled as a state effect.
4. **Test actual retention.** First use delayed-cue synthetic sequences where a
   known early clue determines a later label, with 4/32/256/1024 distractor updates.
   This tests mechanism and gradient reach, not real-security accuracy. On real
   streams, compare intact history with registered reset intervals, missing-history
   conditions and history lengths. Show fixed-threshold checkpoint interventions
   separately from retrained models, and include PR-AP to reduce threshold confounding.
5. **Separate compute from data.** Primary scaling comparisons should control total
   optimizer steps; also report a common-pass/epoch comparison and actual runtime.
   Applying 200 epochs to 34 times as many events adds roughly 34 times as many
   live training visits, so it cannot isolate a pure data-volume effect.
6. **Evaluate deployment behavior.** Predict before updating with an event; only
   update weights when its label would really be available. Measure F1, PR-AP,
   recall at a fixed false-alert budget, alerts/hour, detection delay, throughput,
   latency and state size. Measure forgetting separately across time periods if
   weights are adapted online. Hold an untouched capture out for final confirmation.

Evidence for the hypothesis would be a reproducible **increasing memory advantage**
as relevant history is restored or lengthened, with that advantage reduced by
appropriate history-removal controls. A higher score for every model after adding
data would show a data benefit, not specifically a memory benefit.
