# Technique research: what can raise F1 on the current CTU-13 task

Author: Claude · Date: 2026-09-26 · Scope: paper-mode theory screen *before* any new GPU run.
Framing chosen by project owner: **general accuracy on the current data** (beat the
`GNN + rehearsal` reference of **86.61 ± 4.84%** F1 on CTU-13 scenario-6, 16,384-flow sample).

This document does the "does it mathematically work?" screen you asked for. Each candidate
is judged on two things before it earns a practical test:

1. **Bias match** — does the technique's inductive bias correspond to a structure that is
   actually present in this data?
2. **Signal-vs-noise** — is the expected effect larger than the measurement noise floor on
   this dataset? A technique can be theoretically sound and still be untestable here.

## TL;DR verdict

On this dataset the bottleneck is **not architecture**. Two hard numbers drive everything below:

- **The noise floor is ±3–5 F1 points.** Test has 25 positive flows, so one flow = 4 recall
  points ≈ 2–3 F1 points. The reported gap between the GNN (86.61) and the best TGNN (86.39)
  is 0.23 points — about **one twentieth of the noise**. No architecture change can be *proven*
  to beat the GNN at this sample size; you would be fitting noise.
- **The temporal models already rank better than the GNN.** I swept the decision threshold on
  the saved test predictions (all 12 arms × 5 seeds). At the *best* threshold the temporal
  models reach **92–95% F1** while the GNN reaches only **88%**. The GNN wins the headline table
  only because its validation-selected threshold happens to transfer to test; the temporal
  models throw away 7–16 F1 points purely at the thresholding step (analysis in §2).

So the highest-value work is not a new sequence encoder. It is (a) getting enough positive
labels that a comparison means anything, and (b) fixing the operating-point/calibration step
that is currently hiding the temporal models' real ranking advantage. CNN/RNN/transformer
additions are ranked at the bottom, with reasons.

---

## 1. The measurement reality (read this before proposing any architecture)

The scored partitions (from `audit_memory_horizon.py`):

| Partition | Events | Positive flows | Span |
|---|---:|---:|---:|
| Train | 9,830 | 91 (0.93%) | 78.6 min |
| Validation | 3,277 | 25 | 27.0 min |
| Test | 3,277 | 25 | 23.5 min |

Consequences that constrain every technique:

- **F1 is quantised by the positives.** With 25 test positives, recall moves in 4-point steps.
  Precision moves in similar jumps because false positives are also rare. The achievable F1
  grid is coarse, and the seed SD (±3–5 pts) already reflects this. Any claimed improvement
  below ~5 points is inside the noise.
- **Extreme class imbalance (≈1:107 in training).** Unweighted cross-entropy is dominated by
  the ~9,700 negatives. The gradient signal from the rare positive class is small — this is a
  loss/sampling problem, and it is the *correct* place to spend effort for "more accuracy".
- **Validation is too easy to calibrate a threshold.** For the temporal models the 25
  validation positives are *perfectly* separated from negatives (I checked: min positive score
  > max negative score on validation). "Maximise validation F1" therefore places the threshold
  at an extreme value (0.986–0.999). Extreme thresholds are fragile: on test a few positives
  score just below, recall collapses, F1 drops. The GNN's validation is *not* perfectly
  separable, so it lands on a calmer threshold (~0.95) that generalises. That is the entire
  source of the GNN's 0.23-point "win".

**Implication:** before testing architectures, restore more data (full scenario 6 gives ~34×
the events → far more positives) so the noise floor shrinks below the effect you're hunting.
Otherwise every result you get is a coin flip dressed as a finding.

---

## 2. New analysis — where the F1 is actually being lost (threshold headroom)

I re-scored the frozen predictions in `artifacts/components_200e_v1/` at three thresholds:
the validation-selected one (as reported), the val-F1-maximising one, and the *oracle*
threshold that maximises F1 on test (an upper bound, not deployable). Mean over seeds 0–4:

| Arm | Reported F1 | Oracle-threshold F1 | Headroom left on the table |
|---|---:|---:|---:|
| GNN + rehearsal | 86.61 | 88.11 | **1.50** |
| TGNN | 84.52 | 92.29 | 7.78 |
| TGNN + GRU | 85.45 | 91.54 | 6.09 |
| TGNN + GRU + attn | 86.39 | 93.43 | 7.04 |
| TGNN + memory | 83.86 | 93.30 | 9.44 |
| TGNN + memory + attn | 76.72 | 92.75 | 16.03 |
| TGNN + weighted loss | 85.26 | 94.76 | 9.50 |

Reading this table:

- The GNN has almost **no headroom (1.5)** — its ranking is already fully exploited by its
  threshold. It is near its own ceiling.
- Every temporal model has a **much better ranking** (oracle F1 91–95%) that current
  thresholding wastes. `TGNN + weighted loss` ranks best of all (oracle 94.76%) yet reports
  85.26% because of the operating point. This matches its high PR-AP.
- This is the quantitative proof that the problem is **calibration/threshold transfer under
  label scarcity, not model capacity.** Fixing the operating point is worth ~7–16 points to the
  temporal models and ~1.5 to the GNN — i.e. it would flip the ranking in the temporal models'
  favour without adding a single parameter.

The oracle numbers are an upper bound (they peek at test), so they are not a claim of
deployable F1. They are a *diagnostic*: the signal is present in the scores; the pipeline
loses it at the last step.

---

## 3. Candidate techniques, ranked by mathematical fair-chance

Notation for the current model: 3-layer single-head GAT over a star ego-graph (node 0 = the
flow being scored, plus up to 4 most-recent strictly-earlier incident flows), 10 log-scaled
flow features, hidden 128, binary head. Time encoding `T = cos(Linear(1→128)(age))`. Memory
`M` = per-endpoint EMA `s ← 0.5·s + 0.5·h`. `Q` = GRU over the 4 neighbours (resets per event).
`A` = extra attention readout. `W` = class-weighted loss.

### Tier 1 — cheap, and the math points straight at the real bottleneck

**1.1 Probability calibration + a robust threshold rule.** *Bias match: exact.* §2 shows the
temporal models rank at 91–95% oracle F1 but report 84–86% because the val-selected threshold
sits at an extreme (0.986–0.999) where perfect-but-tiny validation separation put it. Fixes,
all near-zero cost: temperature scaling / Platt scaling on validation logits; select the
threshold at a **target precision or recall** (or the val point that maximises a *smoothed* F1
with Laplace-style counts) rather than raw max-val-F1; or average the operating point across
seeds. Expected effect: several F1 points on the temporal models, i.e. larger than the noise
floor. **Verdict: do this first. Highest expected payoff per GPU-second (it needs none).**

**1.2 Focal loss (γ≈2) instead of unweighted / linearly-weighted CE.** *Bias match: strong.*
The training set is ≈0.9% positive, so CE gradient is swamped by easy negatives. Focal loss
`-(1-p_t)^γ log p_t` reshapes gradient toward hard/rare examples — mathematically the right
tool for 1:107 imbalance, and different from the `W` linear weighting you already tried
(85.26). Cheap (loss swap). **Verdict: test. Fair chance: medium-high.** Pair with §1.1 so a
recall gain isn't hidden by a bad threshold.

**1.3 Seed ensembling.** *Bias match: variance reduction.* With ±4-pt seed noise, averaging the
5 seed probability outputs reduces variance ~√5 and typically lifts and stabilises F1 more than
any single architecture tweak. Free (models already trained). **Verdict: do it — it also gives
a less noisy yardstick for judging everything else.**

**1.4 Restore the full scenario-6 data (~34× events, ~34× positives).** *Not a technique, a
precondition.* Shrinks the noise floor so a 2–3 point architectural effect becomes detectable.
Note it does **not** extend the 2.15-hour window, so it does not test long-range/APT memory —
it only makes the current comparison statistically meaningful. **Verdict: essential before any
architecture verdict is trustworthy.**

### Tier 2 — principled architecture with a genuine fair chance

**2.1 Better time encoding (Time2Vec / Fourier features).** *Bias match: strong; best
architectural bet.* Time was your single biggest proven signal (the pilot jumped 83.8 → 89.8
when time encoding was added). But the current encoder is a degenerate form: a single linear
map through `cos(·)` gives one bank of phase-less cosines. Time2Vec `[ωτ+φ₀, sin(ω_iτ+φ_i)]`
adds a linear term and learned per-frequency phase — strictly more expressive and the standard
choice in TGAT/TGN. Because you are improving the mechanism that already works rather than
bolting on a new one, this has the cleanest theoretical justification of any architecture
change here. Cheap (swap the encoder). **Verdict: test. Fair chance: medium-high.**

**2.2 Learned memory updater (the `R` GRUCell that already exists but was never run at 200e).**
*Bias match: correct fix for a known defect.* The EMA `0.5·s + 0.5·h` gives 96.9% of its weight
to the last 5 endpoint updates and decays by *update count, not clock time* — a busy endpoint
erases its state in seconds. That is why `M` never helped F1. A GRUCell updater learns
input/forget gates and can retain selectively (this is the TGN design, already coded as factor
`R`). **Verdict: test the `R` memory at 200 epochs** — it is the scientifically correct memory
experiment you have not yet run. Fair chance *on current data*: low-medium (few positives, short
window), but it is the right mechanism and the one worth reporting.

**2.3 Time-decay memory (decay by Δt, not update count).** *Bias match: good.* Replace the
fixed 0.5 with `exp(-λΔt)`: `s ← exp(-λΔt)·s + (1-exp(-λΔt))·h`, λ learned. Directly repairs the
"dense traffic destroys memory" defect and aligns retention with real time. One extra scalar
parameter. **Verdict: cheap, sound — test alongside 2.2 as the EMA control's principled cousin.**

### Tier 3 — low mathematical fair-chance here (test only for completeness)

This is where **CNN** and the other sequence encoders land, and it is worth being precise about
*why*, since CNN was your explicit question.

**3.1 Temporal CNN / TCN (1-D dilated convolution over the neighbour sequence).** *Bias match:
weak on this data.* A TCN's whole reason to exist is a long receptive field built cheaply from
dilated convolutions over **hundreds+** of timesteps. Your sequence is **4 neighbours**. Over 4
items a TCN collapses to a tiny convolution ≈ a small MLP on the concatenated window — the
dilation machinery has nothing to bite on, so it cannot express anything the GRU (`Q`) does not
already capture, and `Q` is already your best temporal arm. Its one real advantage is latency:
convolutions are parallel, so a TCN is the *fastest* of the sequence encoders (RNN is sequential
and slowest; attention is quadratic in length). But at 4 steps and ~35k params, all three are
sub-millisecond — well under the paper's 6.7 ms budget — so latency is not a deciding factor
either. **Verdict: expected gain ≈ 0 on the current 4-neighbour window; would only matter if you
widen the window to tens/hundreds of events (see 3.4). Test once for due diligence, not as a
likely winner.** Mathematically it earns a fair chance *only* after the window is long.

**3.2 LSTM / vanilla RNN.** *Bias match: redundant.* Your `Q` GRU is already the gated RNN that
matches or beats LSTM and plain RNN on short sequences, and it is your strongest temporal
component. Swapping in an LSTM shuffles within the same family; expected difference is inside
the ±4-pt noise. **Verdict: skip — no new inductive bias, no fair chance of a real effect.**

**3.3 More self-attention / a Transformer block over neighbours.** *Bias match: already
saturated.* You have graph attention in every GAT layer *and* the extra attention readout `A`.
Adding `A` alone *lowered* F1 (83.03). Over 4 items softmax attention is close to a learned
weighted mean — little capacity headroom, and the empirical result already went the wrong way.
**Verdict: skip a bare Transformer; a full TGAT-style block is only interesting bundled with the
better time encoding of 2.1, not on its own.**

**3.4 Widen the context window (k = 8/16/32 neighbours).** *Bias match: enabling condition.*
This is the change that would actually *give CNN/attention a reason to exist*. More neighbours =
a real sequence, where dilated conv or attention can model order and gaps. It also helps the GNN
and GRU, so it is a fair, model-agnostic lever. Cost: moderate (bigger ego tensors). **Verdict:
worth a test — and it is the prerequisite that would move 3.1/3.3 up a tier.**

**3.5 Richer features / edge attributes.** *Bias match: raises the Bayes ceiling directly.*
Only 10 flow features are used. Architecture can only refit existing features; new discriminative
signal (port entropy, per-endpoint rolling byte-rate, connection-count-in-window, periodicity)
raises the achievable ceiling for *every* model. Out of the "add a component" framing, but
mathematically the highest ceiling-raiser of all. **Verdict: flagged as the biggest untapped
lever if you are willing to touch the feature pipeline.**

---

## 4. Recommended order (cheapest and most mathematically justified first)

1. **Fix the operating point** (§1.1) + **seed-ensemble** (§1.3) — no training; re-scores
   existing runs. This alone likely reorders the leaderboard in the temporal models' favour.
2. **Restore full scenario-6 data** (§1.4) — precondition so later effects clear the noise floor.
3. **Focal loss** (§1.2) — cheap loss swap targeting the true 1:107 imbalance bottleneck.
4. **Better time encoding / Time2Vec** (§2.1) — best-justified architecture change; improves the
   one mechanism already proven to help.
5. **Learned + time-decay memory** (§2.2, §2.3) — the correct memory experiment you have not run.
6. **Widen the window** (§3.4), then and only then **TCN / attention** (§3.1, §3.3) — so the
   convolutional/attention bias finally has a long enough sequence to matter.
7. Skip LSTM/plain-RNN and bare extra attention (§3.2, §3.3).

## 5. One honest caveat

Every "fair chance" verdict above is about *mechanism fit*, not a promise of a win. On the
current 25-positive test set almost nothing can be *proven*, because the noise floor (±3–5 pts)
swallows the effects. That is exactly why steps 1–2 (calibration + more labels) come before any
architecture: they are what turn the later experiments from coin flips into evidence. If the
goal shifts back to APT specifically — long dwell, low-and-slow, day/week horizons — then the
learned/time-decay memory (§2.2–2.3) and a genuinely longer capture become the headline, because
that regime is where persistent state can finally pay off; the current 2.15-hour window cannot
show it either way.


