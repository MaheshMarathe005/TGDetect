# Role-message GAT study

Experimental operators only. TGNN Final still contains only the accepted original
KV encoder. This package implements the user's revised eight-arm matrix by
substituting only corrected V3's GAT region:

| Arm | Operator |
|---|---|
| none | No message passing |
| gat1 | One original provenance GAT |
| gat2 | Two original GATs |
| gat2_tied | Two applications of shared original GAT weights |
| dot_role_bias | One masked dot-product head, scalar role bias |
| role4 | Four role-masked 32D heads; separate value projections |
| role4_deviation | role4 plus learned projection of `[x,m,x-m]` |
| mean | Masked uniform mean with residual/projection/normalization |

Total operator output width is always 128. Each role head has its own softmax
and value subspace. Multi-role edges participate in every matching head, and
self-loops are included in all heads. Output projection mixes role aggregates.
Heads still average within a role; they do not guarantee preservation of rare
evidence. The deviation projection adds parameters and compute: count both.
With a single linear projection, `W[x,m,x-m]` reduces algebraically to `Ax+Bm`:
the signed deviation adds no information or expressiveness over `[x,m]` at that
point. Its parameterization/optimization can still differ. A separately tested
`abs(x-m)` or other nonlinear comparison would add a nonlinear feature.

The role operator has a standard four-head attention layout, but is not FLOP or
parameter matched to the original scalar additive GAT. It avoids constructing
a `[B,9,9,128]` pairwise-feature tensor. All implementations use explicit dense
matrix multiplies; no short-window SDPA speedup is assumed. Endpoint-role masks
are directed and causal. Time and original endpoint features are unchanged.

## Free-first diagnostic

```sh
PYTHONPATH=research python -m gat_study.run \
  --prepared /path/to/corrected_v3_prepared --output /path/to/new_audit \
  --device cuda --audit-only
```

Degree auditing is a separate **untimed** chronological forward pass. Its CPU
histogram collection is disabled during training and timing. `degree_audit.json`
reports query-row and history-row neighbor histograms, self-loop-only fractions,
degree<=2 fractions, valid-window sizes and per-role query nonself degrees.

The current V3 planner selects incident events from both query endpoints, so a
populated query window can have **nine** neighbors, not necessarily 1–4. The
current query ordinarily connects to every retained valid event. History rows
are potentially much sparser, and role-specific groups can also be sparse.
Measure these separately before attributing a tie to operator expressiveness.

Two layers can propagate through ordered event dependency chains already inside
the window; they cannot retrieve events the planner omitted. ss/sd/ds/dd roles
identify endpoint overlap, not guaranteed causal information transfer: endpoint
type and observed operation determine whether a shared-entity link is a real
dependency. The current event's target relation remains masked.

## Matched full-model study

Requires existing corrected V3 prepared **real chronological provenance data**,
a common original-V3 SGNS initialization, and a predefined minority ground-truth
node list. KDD lacks endpoint/provenance structure and is not silently converted
into a feature-similarity graph. No new dataset download/cloud training is launched.

```sh
PYTHONPATH=research python -m gat_study.run \
  --prepared /path/to/corrected_v3_prepared --output /path/to/new_gat_study \
  --initialization /path/to/common_initial_embeddings.pt \
  --minority-nodes /path/to/predefined_minority_nodes.json \
  --device cuda --epochs 16 --batch 256 --seeds 0 1 2 3 4
```

Sixteen epochs is a matched screening budget, not an assumed convergence result.
Common V3 parameters initialize identically per paired seed. The encoder/CNN/
memories/decoder train jointly under unchanged relation/pair/compression losses,
and identical batches/unroll lengths and benign checkpoint selection. New graph
operators have different parameter counts and are reported explicitly. The
original two-GAT arm is exactly the V3 architecture and initial weights.

The trainer is a local copy of original V3's runner with only imports changed
and experimental-source hashes added. TRAINER_SOURCE.json records the changes.
Original trainer/production model files are not modified. Resume requires exact
model/source/config signatures. A source change invalidates old resume artifacts.

Primary: minority recall with one predeclared benign alert budget, paired seeds.
`minority_metrics.json` reports scored and unscored minority IDs and coverage-
adjusted recall. Define this subset by an external dataset annotation taxonomy
before seeing scores; do not mine false negatives and declare them a prospective
minority test. On datasets without minority annotations, record the missing
information rather than relabeling arbitrary nodes. AP, F1, false alerts,
parameters, train time and memory are secondary. This runner retains the existing
node-window threshold calibration, which is not a guarantee of whole-stream FPR.

After training, the runner rotates operator order and replays the same train
prefix with fresh state for three timing repetitions per seed. Frozen FP32
encoder lookup is enabled for every arm. Full-model timing includes prepared
host-input transfer, V3 forward, and state updates/detach; it excludes raw log
preparation, lookup build (reported separately), alert scoring and batch wait.
There are 8 warmup and 64 measured batches. This is a prefix/mature-state-limited
measurement, not a production latency guarantee. Histograms are off while timed.

Raw edge scores, checkpoint/source signatures, minority results, and full-model
timing samples are saved. No accuracy/speed advantage is claimed before GPU tests.

## Next isolated interventions

1. History-slot dropout at p=0.1 and p=0.2 after choosing an operator. Drop only
   prior slots, never current query, and apply the same resulting mask to the
   entire active-window processing path. Decide whether dropped history remains
   in future state; local observation masking and actual missing-event ingestion
   are different experiments. Probe benign-history flooding separately.
2. Rarity retention at the fixed eight-slot budget: preserve an endpoint quota
   and a recent-history quota, then fill remaining slots by frozen reference
   surprise. Compare exact numeric keys versus training-derived bins and track
   benign novelty rate, retained ages, endpoint coverage and missed dependencies.
   Novel does not mean malicious; pure top-surprise retention risks saturation
   by novel benign traffic and can discard recent causal steps.

Time-bias and GATv2 are lower priority. For current-query edges, the historical
age feature is already the pairwise time gap; the missing explicit gap mainly
affects history-to-history edges. Role-conditioned message content is the
distinct hypothesis being tested now.

## Validation

Run `python -m pytest -q research/test_gat_study.py research/test_tgnn_final.py`
from repo root with the project environment. Tests cover original-mask parity,
future independence, finite padding/gradients, role-head message isolation,
degree audit, common-weight initialization, full V3 baseline equality, every
operator's chunked stream equivalence and preservation of the encoder-only
TGNN Final stage. Local acceptance does not establish GPU accuracy or speed.
