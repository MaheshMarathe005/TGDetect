#!/usr/bin/env python3
"""
TG-Detect LAB — a *deployed*, warm Modal app you trigger repeatedly.

Why this file exists
--------------------
`modal run` builds/attaches, uploads, runs, tears down. Iterating that way costs
minutes of container churn per experiment. This app is deployed ONCE:

    modal deploy modal_lab.py

and then every experiment is a cheap trigger from the laptop (see trigger.py),
with no image build, no re-upload, and no data reload:

  * the image comes from modal_common.IMAGE, which is FROZEN — the pip layer is
    content-addressed, so it is built once and never again
  * @modal.enter() loads the 13 CTU-13 npz files and the causal context ONCE per
    container, into instance state
  * scaledown_window keeps that container alive between triggers, so the 2nd..Nth
    experiment starts computing immediately
  * the expensive EWMA context pass is additionally cached to the Volume, so even
    a genuinely cold container skips recomputing it

Everything heavy runs here; the laptop only sends arguments and prints results.
"""

import os

import modal
from modal_common import (
    APP_LAB, DATA_DIR, IMAGE, volume, FAMILY, FEAT, CUMULATIVE,
    FEAT_IN, HID, MEM, TDIM, CTX_BASE, CTX_DIM, FEAT_Z, Z_SCALES, Z_REF,
    PAIR_BUCKETS, PMEM, BEACON_ALPHA, BEACON_HARMONICS, TAU_LOGIT_ADJ,
    FEATURE_GROUPS, FEATURE_SETS, feature_mask,
    FEATURE_GROUPS, FEATURE_SETS, feature_mask,
    DROPOUT, WEIGHT_DECAY,
    pair_hash, build_context, prior_adapt, get_models,
)

# Any change to the emitted context widths must invalidate the Volume cache.
CTX_TAG = "z" + "-".join(str(s) for s in Z_SCALES) + f"_w{FEAT_Z}x{CTX_DIM}"

app = modal.App(APP_LAB, image=IMAGE)

# ─────────────────────────── log tee ───────────────────────────
# Calling a DEPLOYED function's .remote() does not stream container stdout back to
# the client — it only lands in the Modal dashboard. Every entry point therefore
# runs under tee(), which keeps writing to real stdout (so the dashboard still has
# it) while also capturing the text into the returned dict under "log", letting
# trigger.py print the formatted tables locally.


class _Tee:
    def __init__(self, buf, real):
        self.buf, self.real = buf, real

    def write(self, s):
        self.buf.write(s)
        self.real.write(s)
        return len(s)

    def flush(self):
        self.real.flush()

    def isatty(self):
        return False


def tee():
    import contextlib, io, sys

    @contextlib.contextmanager
    def _cm():
        buf, real = io.StringIO(), sys.stdout
        sys.stdout = _Tee(buf, real)
        try:
            yield buf
        finally:
            sys.stdout = real
    return _cm()


STATIONARY = [i for i in range(22) if i not in CUMULATIVE]
BATCH, LR, POS_W = 2048, 1e-3, 20.0
# Step size for the exponentiated-gradient update on the Group-DRO group weights.
# 0.01 is Sagawa et al.'s default. It must be small: q is updated once per BATCH
# here, not once per epoch, so a large eta lets one unlucky batch capture all the
# weight and the run degenerates into training on a single capture.
DRO_ETA = 0.01
# A12: how many (positive, negative) pairs to draw per batch for the pairwise
# ranking surrogate. Capped rather than all-pairs: a batch of 2048 with 2%
# prevalence has ~40 positives and ~2000 negatives, so all-pairs is 80k terms whose
# gradient is dominated by the easy ones. 256 sampled pairs is an unbiased estimate
# of the same expectation at 1/300th the cost.
RANK_PAIRS = 256
# A14: the columns the score is decorrelated FROM. 13 log_src_prior,
# 14 log_dst_prior, 17 log_src_fanout -- the three host-activity counters, i.e. the
# FEATURE_GROUPS["history"] block. They are averaged into one composite activity
# index rather than penalised separately, because they are near-collinear (all
# three are monotone in "how much has this host done so far") and penalising a
# rank-1 direction three times just triples the effective lambda.
ACT_COLS = (13, 14, 17)
# MEASURED floor on run-to-run MACRO AUC-PR noise for a memory model on the
# notime/family=Neris exam. NOT a guess: the first real `suite robust` run trained
# TGDetect_TGN twice at seed 0 under a byte-identical configuration (step 0 and
# step 1 of the same plan, same folds, same scalers, same feature mask) and got
#     MACRO 0.7319 / pooled 0.7613 / s09 AUC-PR 0.8795
#     MACRO 0.5074 / pooled 0.1790 / s09 AUC-PR 0.2702
# The cause is that the TRAINING path is not bit-reproducible on CUDA: the backward
# of index_add / index_select / index_copy accumulates in nondeterministic order,
# and a 480k-event recurrent memory chain amplifies a 1e-7 perturbation into a
# different basin. Any control that demands EXACT equality of two trained models is
# therefore unsatisfiable, and any verdict smaller than this floor is unsupported.
NOISE_MACRO = 0.22
# ── PLAN A6: what NOISE_MACRO actually measures ──────────────────────────────
# The two runs above are not two draws from a Gaussian. Across all 15 recorded
# family-holdout runs in results/bench-*.json the held-out capture's AUC-PR is
# BIMODAL -- LOW n=9 spanning 0.119..0.372, HIGH n=6 spanning 0.511..0.780, with an
# empty interval between them. Decomposing the variance:
#     between-basin  p(1-p)(mu_hi - mu_lo)^2 = 0.05687   87.3%
#     within-basin   E[var | basin]          = 0.00830   12.7%
# So 0.22 is the distance BETWEEN TWO MODES, not a standard deviation, and the pair
# in the comment above is simply one run per mode. Treating it as a sigma (a "noise
# floor" every effect must clear) is a category error in the conservative direction:
# it hides real effects behind a threshold built from a mixture gap.
#
# The estimand that IS well defined for a mixture is p = P(high basin). Any threshold
# inside the empty interval labels all 15 recorded runs identically; 0.45 is its
# midpoint and is therefore not a tuned parameter.
BASIN_THR = 0.45
NODE_BUCKETS = 1 << 17
# Ceiling on per-window node-memory rows. Node ids are remapped to be contiguous
# inside the window (see load_streams), so n_nodes is normally just "how many hosts
# this window mentions" and this never fires. It exists so a pathological scan
# capture cannot try to allocate an arbitrarily large mem table: 1<<20 rows at
# MEM=128 float32 is 512 MiB, which is the most we are willing to spend.
NODE_MAX = 1 << 20
REHEARSAL_FRAC = 0.10

# Ablation verdict calibration.
#   FLOOR_MIN  a noise floor may never be 0. Training nondeterminism is sporadic,
#              so a measured floor can legitimately read 0.00e+00 on a lucky draw,
#              and every threshold derived from it would then collapse.
#   MIN_EFFECT smallest AUC-PR change worth calling an effect. The 13-scenario run
#              put TGDetect_TGN at 0.1750 +- 0.0178, so seed-to-seed sd is ~0.018;
#              0.002 is ~11% of that - below it, a "win" is real but pointless.
FLOOR_MIN, MIN_EFFECT = 1e-6, 0.002

# Which contiguous slice to keep when a capture exceeds `cap`.
#   "tail"   last cap events  - label-free, default
#   "attack" centred on the labelled span - retains most positives, reads y
#   "head"   first cap events - the ORIGINAL behaviour, kept only to reproduce
#            the broken 13-scenario run
WINDOW = os.environ.get("TGD_WINDOW", "tail")

# Warm-container lifetimes. Long enough to iterate against, short enough not to
# burn credit overnight if a session is abandoned.
KEEP_CPU, KEEP_GPU = 1800, 900


# ────────────────────────── shared data loading ──────────────────────────

def load_streams(cap: int, halflife: int, want_ctx: bool):
    """Load the 13 capped streams, with the causal context attached.

    The context cache is keyed by (scenario, cap, CTX_TAG) and lives on the
    Volume, so the O(n) EWMA pass over ~6M events is paid once ever rather than
    once per container.

    CTX_TAG encodes the z-scales and the emitted widths, so widening
    build_context automatically invalidates every stale cache file instead of
    silently loading a 22-wide XZ into a 66-wide model.

    `halflife` is retained in the signature because it is a modal.parameter() on
    both Lab classes, but the z-scales are now the module-level Z_SCALES tuple; a
    single EWMA could not be both responsive and stable, which is what motivated
    the multi-scale channel in the first place.
    """
    import os, numpy as np
    streams, built = [], 0
    for k in sorted(FAMILY):
        p = f"{DATA_DIR}/{k}.npz"
        if not os.path.exists(p):
            print(f"  ! missing {p}", flush=True)
            continue
        z = np.load(p)
        n_full, pos_full = len(z["y"]), int(z["y"].sum())
        # ── window selection ────────────────────────────────────────────────
        # WAS slice(0, cap) -- the FIRST cap events. CTU-13 botnets ramp up late,
        # so that destroyed 73.3% of every positive in the corpus (444,699 -> 118,796),
        # including 100% of s01's 40,960 and 184,916 of s09's 184,987, while
        # AMPLIFYING s10 (8.12% -> 12.47%). The survivors were then dominated by the
        # two Rbot flood captures s10+s11, which hold 86.1% of test positives and are
        # near-perfectly ANTI-ranked (ROC 0.0091 / 0.0139) because the botnet is the
        # traffic MAJORITY there. Pooling those against 7 captures at mean ROC 0.9805
        # produced the "model is broken" verdict. It was the loader.
        #
        # "tail" is the default because it is LABEL-FREE: it uses only the documented
        # fact that CTU-13 attacks run late, so it cannot be accused of selecting on
        # the target. "attack" centres the window on the labelled span and retains
        # more positives, but it does read y, so it is opt-in and reported as such.
        if cap and n_full > cap:
            if WINDOW == "head":
                lo = 0
            elif WINDOW == "attack":
                nz = np.nonzero(z["y"])[0]
                mid = int((nz[0] + nz[-1]) // 2) if len(nz) else n_full // 2
                lo = int(np.clip(mid - cap // 2, 0, n_full - cap))
            else:                                    # "tail"
                lo = n_full - cap
            sl = slice(lo, lo + cap)
        else:
            sl = slice(None)
        X = np.nan_to_num(z["X"][sl])
        ts = z["ts"][sl].astype(np.float64)
        ts = (ts - ts[0]).astype(np.float32)
        # src/dst are written as int32 by modal_data_real.py. Cast to int64 HERE,
        # at the single load boundary, because Tensor.index_copy_ / index_add_
        # require a Long index and will raise
        #   "index_copy_(): Expected a long tensor for index, but got Int"
        # on real data. The offline fixture generated int64 and so was structurally
        # incapable of catching this - the fixture now matches the real dtype.
        # ── node ids: REMAP WITHIN THE WINDOW ───────────────────────────────
        # modal_data_real.py assigns node ids per capture in ORDER OF FIRST
        # APPEARANCE (`node_id.setdefault(ip, len(node_id))`), so ids grow with
        # stream position and run to the capture's full node count -- ~1.47M across
        # the corpus, vs a mem/last table this file sized at NODE_BUCKETS = 131,072
        # and then indexed with the RAW id. That is an 11x out-of-bounds write,
        # reported by CUDA as the useless "device-side assert triggered".
        #
        # WINDOW="head" hid it for free: the first `cap` events can only mention ids
        # 0..(distinct nodes in those events), which stayed under 131,072. Switching
        # the default to "tail" exposed it, because a tail window's ids are biased
        # HIGH. So this is a latent bug the window fix uncovered, not one it caused.
        #
        # Remapping inside the window is better than hashing: it is EXACT, so two
        # unrelated hosts can never share a memory row, and it bounds the table at
        # (distinct nodes in window) instead of a fixed oversubscribed size. It is a
        # bijective relabel, so build_context's fanout/prior-count features are
        # unchanged, and grouping by it in seg_softmax is collision-free.
        both = np.concatenate([z["src"][sl], z["dst"][sl]]).astype(np.int64)
        uniq_nodes, inv = np.unique(both, return_inverse=True)
        ne = len(both) // 2
        src = np.ascontiguousarray(inv[:ne]).astype(np.int64)
        dst = np.ascontiguousarray(inv[ne:]).astype(np.int64)
        n_nodes = int(len(uniq_nodes))
        if n_nodes > NODE_MAX:                    # guard, not expected to fire
            src, dst = src % NODE_MAX, dst % NODE_MAX
            print(f"  ! {k}: {n_nodes:,d} nodes in window > NODE_MAX="
                  f"{NODE_MAX:,d}; hashing, so memory rows now COLLIDE", flush=True)
            n_nodes = NODE_MAX
        y = z["y"][sl].astype(np.float32)
        # hsrc/hdst are the SAME array as src/dst on purpose. They used to be a
        # second, differently-scaled identity space, and mixing the two is exactly
        # what caused the out-of-bounds read above. One convention, one array, so
        # the bug cannot be written again.
        s = {"key": k, "family": FAMILY[k], "X": X, "ts": ts, "src": src, "dst": dst,
             "y": y, "n_full": n_full, "pos_full": pos_full,
             "hsrc": src, "hdst": dst,
             "hpair": pair_hash(src, dst),
             "n_nodes": n_nodes}
        for kk in ("src", "dst", "hsrc", "hdst", "hpair"):
            assert s[kk].dtype == np.int64, f"{k}.{kk} is {s[kk].dtype}, need int64"
        # The invariant the CUDA assert was really complaining about. Check it HERE,
        # on the host, where the message can name the capture and the number.
        for kk in ("src", "dst"):
            hi = int(s[kk].max()) if len(s[kk]) else -1
            assert hi < s["n_nodes"], (
                f"{k}.{kk} max id {hi:,d} >= n_nodes {s['n_nodes']:,d} -- mem/last "
                f"would be indexed out of bounds")
        assert int(s["hpair"].max() if len(s["hpair"]) else 0) < PAIR_BUCKETS
        kept = int(s["y"].sum())
        s["pos_kept"], s["retention"] = kept, kept / max(pos_full, 1)
        if want_ctx:
            c = f"{DATA_DIR}/ctx_{k}_{cap}_{WINDOW}_{CTX_TAG}.npz"
            if os.path.exists(c):
                zc = np.load(c)
                s["XZ"], s["C"] = zc["XZ"], zc["C"]
            else:
                s["XZ"], s["C"] = build_context(X, ts, src, dst)
                np.savez(c, XZ=s["XZ"], C=s["C"])
                built += 1
            assert s["XZ"].shape[1] == FEAT_Z and s["C"].shape[1] == CTX_DIM, (
                f"{k}: cached context is {s['XZ'].shape[1]}/{s['C'].shape[1]}, "
                f"expected {FEAT_Z}/{CTX_DIM} -- CTX_TAG failed to invalidate")
        streams.append(s)
    if built:
        volume.commit()
        print(f"  built + cached context for {built} scenario(s)", flush=True)
    tot = sum(len(s["y"]) for s in streams)
    pos = sum(int(s["y"].sum()) for s in streams)
    print(f"  loaded {len(streams)} scenarios | {tot:,d} events | {pos:,d} pos "
          f"({pos/max(tot,1)*100:.4f}%)", flush=True)
    # Node counts are printed because they size the per-segment memory bank, and
    # because silently oversubscribing that bank is what produced the CUDA
    # device-side assert. Biggest window x MEM x 4B is the peak extra GPU cost.
    nn = [s["n_nodes"] for s in streams]
    print(f"  nodes per window: min {min(nn):,d} max {max(nn):,d} | node memory "
          f"peak {max(nn) * MEM * 4 / 2**20:.0f} MiB", flush=True)
    pf = sum(s["pos_full"] for s in streams)
    print(f"  WINDOW={WINDOW} | positive retention {pos:,d}/{pf:,d} = "
          f"{pos/max(pf,1)*100:.1f}%  (head/600k retained only 26.7%)", flush=True)
    lost = [s["key"] for s in streams if s["pos_full"] > 100 and s["retention"] < 0.10]
    if lost:
        print(f"  ! captures still losing >90% of positives: {lost}", flush=True)
    return streams


# ── Experiment suites: a whole plan in ONE warm container ────────────────────────
# Why this exists. Each bench invocation costs a container spin-up plus a full
# load_streams (13 captures, 6M events) before the first model trains. Running three
# configurations as three triggers pays that three times and, worse, invites the
# configurations to drift apart between runs. A suite runs them back to back on ONE
# load, swapping folds server-side, and prints a single cross-step comparison at the
# end -- which is the thing you actually want to read.
#
# Timings are measured from the real Neris-holdout run: Pomsathit ~213 s per seed,
# TGN/R_* ~255 s, Drift/RD_* ~270 s. The default plan is sized to land near 2 h 10 m
# against the 4 h class timeout, so a slow GPU or an extra epoch cannot truncate it.
# Every step also writes its own JSON and commits the Volume before the next one
# starts, so a timeout loses at most the step in flight.
SUITE_COST = {"Pomsathit_GCN": 215, "PlainMLP": 60, "TGDetect_TGN": 255,
              "TGDetect_Drift": 270}
SUITES = {
    # The default. Answers, in order: (a) are the two new parameters inert when off?
    # (b) do the memory fixes repair the exam we lost? (c) what is the headline
    # full-feature number?
    "robust": [
        {"step": "0 CONTROL",
         "why": "R_Off must be IDENTICAL to TGDetect_TGN and RD_Off to "
                "TGDetect_Drift. If not, the 2 new params leak and NOTHING below "
                "means anything. 1 seed is enough: this is an identity check, not "
                "an estimate.",
         "models": "TGDetect_TGN,R_Off,TGDetect_Drift,RD_Off",
         "seeds": 1, "features": "notime", "protocol": "family",
         "holdout": "Neris", "dro": False},
        {"step": "1 MEMORY GRID",
         "why": "The exam we LOST by 3.3 SE (Pomsathit 0.6162 vs ours 0.5019 MACRO), "
                "so it is the cleanest test of the memory fixes. Watch the s09 row: "
                "the whole loss was there, and if ||m|| really was the leak then s09 "
                "must improve more than s01/s02.",
         "models": "Pomsathit_GCN,TGDetect_TGN,R_MemNorm,R_MemDrop,R_Both",
         "seeds": 0, "features": "notime", "protocol": "family",
         "holdout": "Neris", "dro": False},
        {"step": "2 HEADLINE",
         "why": "Full features, family holdout, Group DRO. This is the SYSTEM "
                "comparison and the presentation number. TGDetect_TGN is included so "
                "DRO's effect on the anchor is attributable against the already "
                "recorded ERM run (MACRO 0.5592 on identical folds and features).",
         "models": "Pomsathit_GCN,TGDetect_TGN,R_Both,RD_Both",
         "seeds": 0, "features": "all", "protocol": "family",
         "holdout": "Neris", "dro": True},
    ],
    # Same three questions, one seed each and 3 epochs. ~35 min. Use this first if
    # you want to know the pipeline works before committing two hours.
    "quick": [
        {"step": "0 CONTROL", "why": "identity check only",
         "models": "TGDetect_TGN,R_Off", "seeds": 1, "features": "notime",
         "protocol": "family", "holdout": "Neris", "dro": False},
        {"step": "1 MEMORY", "why": "does memnorm move s09 at all",
         "models": "Pomsathit_GCN,TGDetect_TGN,R_Both", "seeds": 1,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False},
        {"step": "2 HEADLINE", "why": "full features + DRO",
         "models": "Pomsathit_GCN,R_Both", "seeds": 1, "features": "all",
         "protocol": "family", "holdout": "Neris", "dro": True},
    ],
    # Replication. One holdout is one sample; this is the same headline on three
    # different families. Run it only AFTER `robust` says the fixes work.
    "families": [
        {"step": f"holdout={fam}",
         "why": f"headline configuration, {fam} held out",
         "models": "Pomsathit_GCN,TGDetect_TGN,R_Both,RD_Both", "seeds": 0,
         "features": "all", "protocol": "family", "holdout": fam, "dro": True}
        for fam in ("Neris", "Rbot", "Virut")
    ],
    # ══ THE PLAN THAT REPLACED `robust` ═════════════════════════════════════════
    # `robust` ran, and its own control step refuted its premise. R_Off is
    # arithmetically identical to TGDetect_TGN (mem_read is the identity when both
    # flags are off; the two new tensors are zero/one-initialised and consume no
    # RNG) yet it scored 0.1609 MACRO below it. The explanation is not a leak: the
    # SAME model at the SAME seed under a byte-identical configuration appeared
    # twice in that run -- TGDetect_TGN in step 0 and again in step 1 -- and scored
    #     step 0   MACRO 0.7319   pooled 0.7613   s09 AUC-PR 0.8795  ROC 0.9626
    #     step 1   MACRO 0.5074   pooled 0.1790   s09 AUC-PR 0.2702  ROC 0.7417
    # So run-to-run noise on this exam is ~0.22 MACRO, and every effect the suite
    # tried to measure (0.08 to 0.20) is smaller than its own noise. Nothing in
    # `robust` was measurable, in either direction.
    #
    # That reframes the problem. Our best run (0.7319) BEATS Pomsathit's mean
    # (0.6887); our mean does not. The defect is not capacity, it is VARIANCE plus
    # the absence of any selection rule -- we train N epochs and score whatever the
    # last batch left behind. So this plan attacks variance first and architecture
    # second, in the order that the evidence supports.
    "stability": [
        {"step": "0 NOISE",
         "why": "Measure the thing that invalidated the last suite. ONE model, ONE "
                "seed, THREE byte-identical runs. The spread of those three IS the "
                "resolution limit of every other number in this file. Cheap and "
                "non-negotiable: without it we cannot tell a fix from a coin flip.",
         "models": "TGDetect_TGN", "seeds": 1, "repeats": 3,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False},
        {"step": "1 SELECT+ENS",
         "why": "The two variance fixes, on the UNCHANGED architecture. A10 keeps "
                "the epoch with the best WORST-capture val AUC-PR instead of the "
                "last epoch; A11 rank-averages the seeds we already paid for. "
                "probe_oracle prints the best-epoch-by-test bound so we can see how "
                "much of the headroom selection actually captures. Pomsathit gets "
                "the identical treatment -- a fix that only we are allowed to use "
                "is not a comparison.",
         "models": "Pomsathit_GCN,TGDetect_TGN", "seeds": 0,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False, "select": "worst", "ensemble": True, "probe_oracle": True},
        {"step": "2 OBJECTIVE",
         "why": "Now the objective. A12 optimises the ranking we actually grade "
                "(pairwise, within capture, an upper bound on 1-AUC) and A14 "
                "penalises the correlation between the score and host activity -- "
                "the specific quantity whose sign flips on s09. Both on top of "
                "selection, so any gain is attributable to the objective and not to "
                "no longer picking a random epoch.",
         "models": "TGDetect_TGN", "seeds": 0,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False, "select": "worst", "ensemble": True,
         "rankloss": 1.0, "decorr": 1.0},
        {"step": "3 MEMHARD",
         "why": "The architecture correction. R_MemNorm LOST (0.4921 vs anchor "
                "0.5666) because its gate is learned from zero by ERM, and ERM has "
                "no reason to close a shortcut that lowers TRAINING loss. R_MemHard "
                "imposes alpha=1, making LN(a*m+b)=LN(m) a property of the model "
                "class rather than an optimiser preference. The mem_alpha readout in "
                "step 2's log is the direct check on that reasoning.",
         "models": "TGDetect_TGN,R_MemHard,R_HardDrop", "seeds": 0,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False, "select": "worst", "ensemble": True,
         "rankloss": 1.0, "decorr": 1.0},
    ],
    # ══ THE NOVEL PLAN ═══════════════════════════════════════════════════════════
    # Everything in `stability` is a known technique (DRO, ranking loss,
    # decorrelation, best-epoch selection). This plan tests the two contributions
    # that are NOT in the literature-synthesis and were each verified on a toy first
    # (.labtest/research/e9_vrex_vs_dro.py, 15 seeds):
    #   (A) V-REx risk-variance across captures. DRO bounds the WORST capture but
    #       only inside the convex hull of training captures; s09 is a 22x
    #       prevalence EXTRAPOLATION outside it, where equal-risk (VREx) is the
    #       property preserved under affine env shifts. Toy: ROC 0.396->0.775.
    #   (B) dt-only memory decay. The default couples the retention time-constant to
    #       the model's own risk, a forward-dynamics feedback loop that AMPLIFIES
    #       the host-activity shortcut (toy H-B: +0.1318 correlation, t=+26). This
    #       removes it.
    # Ordered so each addition is attributable: control, then each factor alone,
    # then both, then both stacked on the architectural block they are meant to
    # complement (memhard).
    "novel": [
        {"step": "0 CONTROL",
         "why": "vrex=0 and plain_decay off MUST reproduce the plain-ERM anchor. "
                "The offline rig proved byte-inertness on fixtures; this confirms it "
                "on the real folds. 1 seed, identity check.",
         "models": "TGDetect_TGN", "seeds": 1,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False, "select": "worst"},
        {"step": "1 VREX",
         "why": "V-REx alone vs DRO alone vs plain, all with the same selection. "
                "The prediction from the toy: on the sign-flipped capture VREx "
                "un-inverts harder than DRO because s09 is outside the training "
                "hull. Watch the s09 ROC row specifically.",
         "models": "TGDetect_TGN", "seeds": 0,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False, "select": "worst", "vrex": 1.0},
        {"step": "2 DECAY",
         "why": "dt-only decay alone. If the risk-modulated decay really amplifies "
                "the volume shortcut, removing it should lift the low-ROC captures "
                "with no cost to the healthy ones. Pair with the memprobe readout "
                "to confirm ||m||-vs-k flattens.",
         "models": "TGDetect_TGN", "seeds": 0,
         "features": "notime", "protocol": "family", "holdout": "Neris",
         "dro": False, "select": "worst", "plain_decay": True, "memprobe": True},
        {"step": "3 BOTH+MEMHARD",
         "why": "The full novel stack: VREx (objective) + dt-only decay (forward "
                "dynamics) + memhard (architecture). These attack the shortcut at "
                "three different levels and the toy said VREx and the architectural "
                "block are COMPLEMENTS, not substitutes -- VREx cannot dislodge a "
                "constant-strength shortcut, memhard blocks it structurally. This is "
                "the headline novel configuration.",
         "models": "TGDetect_TGN,R_MemHard", "seeds": 0,
         "features": "all", "protocol": "family", "holdout": "Neris",
         "dro": False, "select": "worst", "ensemble": True,
         "vrex": 1.0, "plain_decay": True, "decorr": 1.0},
    ],
}


def suite_eta(plan, seeds, epochs):
    """Rough wall-clock estimate, in seconds, printed BEFORE anything runs.

    Costs are per model-seed at 6 epochs on an L4 over the Neris holdout, scaled
    linearly in epochs. Unknown model names fall back to the TGN cost, which is the
    expensive end, so the estimate errs long rather than short.

    Two multipliers on top of that, both from added no_grad passes rather than
    added training:
      select        one VAL pass per epoch. Val is ~0.96M events against train's
                    ~3.84M, and a forward-only pass costs roughly a third of a
                    training step per event, so ~8%.
      probe_oracle  one TEST pass per epoch as well, ~1.15M events, so ~10% more.
    """
    tot = 0.0
    for st in SUITES[plan]:
        ns = st["seeds"] or seeds
        mult = 1.0
        if st.get("select", "none") != "none":
            mult += 0.08
        if st.get("probe_oracle"):
            mult += 0.10
        for m in st["models"].split(","):
            tot += (SUITE_COST.get(m.strip(), 255) * ns * max(st.get("repeats", 1), 1)
                    * (epochs / 6.0) * mult)
    return tot


def make_folds(streams, tr_frac=0.60, va_frac=0.75, protocol="temporal", holdout=""):
    """Build (train, val, test) segment lists.

    A segment is (stream, warm_start, eval_start, eval_end): the runner replays
    [warm_start, eval_start) under no_grad so memory and last-seen clocks reach the
    state a live detector would be in, without scoring those events.

    protocol="temporal"  60/15/25 chronological split inside EACH capture.
        This is the ORIGINAL and it is the easy one. The same capture -- and
        therefore the same botnet host IP -- appears in train, val and test. Since
        the feature set contains log_src_prior_count / log_dst_prior_count /
        log_src_fanout, a model can memorise "the busiest host in this capture is
        the bot" during training and simply recognise it at test time. That is host
        fingerprinting, not threat detection, and it needs neither node memory nor
        Δt -- which is precisely why a plain 3-layer GCN matched a TGN on it
        (0.9828 vs 0.9341 AUC-PR).

    protocol="capture"   whole captures held out. Test hosts are never seen in
        training, so the fingerprint shortcut is gone.

    protocol="family"    whole MALWARE FAMILIES held out. The test botnet family
        was never trained on. This is the protocol the base paper's own claim
        requires: they claim robustness "against evolving APT campaigns", and an
        evolving campaign is by definition one you have not trained on.

    For held-out captures the first 20% is replayed as warm-up (not scored), which
    is what deploying a streaming detector on a new network actually looks like.
    """
    if protocol == "temporal":
        tr, va, te = [], [], []
        for s in streams:
            n = len(s["y"]); a, b = int(n * tr_frac), int(n * va_frac)
            tr.append((s, 0, 0, a)); va.append((s, 0, a, b)); te.append((s, 0, b, n))
        return tr, va, te

    if protocol == "family":
        fams = sorted({s["family"] for s in streams})
        pick = holdout or fams[0]
        te_keys = {s["key"] for s in streams if s["family"] == pick}
        label = f"family={pick}"
    elif protocol == "capture":
        pick = holdout or streams[-1]["key"]
        want = {k.strip() for k in pick.split(",") if k.strip()}
        te_keys = {s["key"] for s in streams if s["key"] in want}
        label = f"captures={sorted(te_keys)}"
    else:
        raise ValueError(f"unknown protocol {protocol!r}")
    if not te_keys:
        raise ValueError(f"protocol={protocol} holdout={holdout!r} selected nothing")
    if len(te_keys) == len(streams):
        raise ValueError("holdout selected every capture; nothing left to train on")

    tr, va, te = [], [], []
    for s in streams:
        n = len(s["y"])
        if s["key"] in te_keys:
            te.append((s, 0, int(n * 0.20), n))       # warm 20%, score the rest
        else:
            a = int(n * 0.80)                          # val = last 20% of train caps
            tr.append((s, 0, 0, a)); va.append((s, 0, a, n))
    print(f"  protocol={protocol} | HELD OUT {label} | "
          f"{len(tr)} train caps, {len(te)} test caps", flush=True)
    return tr, va, te


def concat(segs, field):
    import numpy as np
    return np.concatenate([s[field][i:j] for s, _, i, j in segs])


def last_write(idx, *vals):
    """Deduplicate `idx`, keeping the LAST occurrence of each index, and gather
    `vals` at those positions. Returns (unique_idx, val0_last, val1_last, ...).

    Why this exists: `Tensor.index_copy` with DUPLICATE indices does not specify
    which write wins, and on CPU with 8 threads it genuinely races. Measured on
    2048 rows of width 128:

        op                          distinct results over 12 identical runs
        index_add                    1   (safe)
        index_reduce(amax)           1   (safe; max is order-independent)
        index_copy, duplicate idx   12   (RACES)

    The node-memory writeback is `mem.index_copy(0, sb, nm)` and a busy source
    appears many times per batch, so the whole streaming runner was
    non-deterministic: the SAME model at the SAME seed, evaluated twice, differed
    by 2.0e-02 in predicted probability. That silently inflated every seed-to-seed
    standard deviation ever reported by this project and would have made the V2_Off
    control impossible to interpret.

    Deduplicating first fixes both problems at once. It is deterministic, because
    torch.unique returns sorted output and an integer amax is exact and
    order-independent. And it makes the semantics match what the code always
    claimed: the most recent event for that node wins, which is the correct
    streaming state.
    """
    import torch
    uniq, inv = torch.unique(idx, return_inverse=True)
    pos = torch.zeros(uniq.numel(), dtype=torch.long, device=idx.device)
    pos.scatter_reduce_(0, inv, torch.arange(idx.numel(), device=idx.device),
                        reduce="amax", include_self=False)
    # torch.unique preserves the input dtype, and every caller feeds the result
    # straight into index_copy_/index_add_, which demand Long. .long() is a no-op
    # when it is already Long, so this is free insurance for a utility whose whole
    # purpose is to produce scatter indices.
    return (uniq.long(),) + tuple(v[pos] for v in vals)


def mem_leak_probe(M, k, seed=0, min_k=2, max_nodes=20000):
    """Is a node's EVENT COUNT linearly decodable from its final node memory?

    This measures the quantity that PART XIII asserted and PART XIV built
    `R_MemHard` on top of, and it distinguishes two hypotheses that the algebra
    `LN(a*m + b*1) == LN(m)` conflates:

      H_scale     the count rides the memory's MAGNITUDE. Then LayerNorm removes it
                  and r2_ln collapses to ~0. `memhard` works.
      H_rotate    the GRU's retention gates z_j differ across dimensions, so
                  E[m_k] = (1 - z**k) * n_bar is a k-dependent DIAGONAL reweighting.
                  The memory DIRECTION rotates with k. LayerNorm is invariant to
                  rescaling, not to rotation, so r2_ln stays ~= r2_raw and `memhard`
                  cannot work no matter how it is tuned.

    Offline rigs (.labtest/research/e1_memory_leak.py on the analytic form, and
    e1b_real_gru.py on a real torch.nn.GRUCell) came down hard on H_rotate: at
    retention 0.847 LayerNorm removed 1.5% of the leak, at 0.974 it removed 2.1%.
    Those are SYNTHETIC. This function is the same question asked of the trained
    weights, and it is the measurement that decides whether R_MemHard / RD_MemHard
    are worth any more GPU minutes.

    r2_resid fits the rank-1 projection on one half of the nodes and evaluates it on
    the other half, so it answers the DEPLOYABLE question -- "does removing a single
    direction estimated from seen hosts also de-leak unseen hosts?" -- rather than
    the tautology "does removing the direction that predicts t stop predicting t".

    Crucially, k is the host's event count: OBSERVABLE at test time with no labels.
    That is what separates this from the score-residualisation family, which needs
    the SIGN of corr(activity, label) and therefore needs labels (see
    .labtest/research/e7_identifiability.py, where that sign is proved
    unidentifiable from unlabeled data).

    Returns None rather than raising when there are too few eligible nodes, because
    a warm-start segment can legitimately have almost no repeat sources and a probe
    must never be able to fail a benchmark run.
    """
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    keep = k >= min_k
    if int(keep.sum()) < 400:
        return None
    M, k = M[keep].astype(np.float64), k[keep].astype(np.float64)
    rng = np.random.default_rng(seed)
    if len(k) > max_nodes:
        sel = rng.choice(len(k), max_nodes, replace=False)
        M, k = M[sel], k[sel]
    t = np.log1p(k)
    if t.std() < 1e-9:
        return None
    n = len(k)
    perm = rng.permutation(n)
    tr, te = perm[: n // 2], perm[n // 2:]

    def ln(A):
        return (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-5)

    L = ln(M)
    # rank-1 direction fitted on TRAIN nodes only, then applied to every node
    Lc = L - L[tr].mean(0)
    ac = (t[tr] - t[tr].mean()) / (t[tr].std() + 1e-9)
    w = Lc[tr].T @ ac / len(ac)
    w = w / (np.linalg.norm(w) + 1e-9)
    R = Lc - np.outer(Lc @ w, w)

    out = {"n_nodes": int(n), "k_median": float(np.median(k)),
           "k_max": float(k.max())}
    for nm_, A in (("raw", M), ("ln", L), ("resid", R)):
        mdl = Ridge(alpha=1.0).fit(A[tr], t[tr])
        out["r2_" + nm_] = float(r2_score(t[te], mdl.predict(A[te])))
    # The empirical saturation curve. If ||m|| is already flat by the second bin the
    # memory has effectively no horizon and there is no leak to remove -- a different
    # and worse finding than either hypothesis above, so it must be visible.
    nrm = np.linalg.norm(M, axis=1)
    qs = np.quantile(k, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    out["norm_vs_k"] = [[float(np.median(k[s])), float(nrm[s].mean())]
                        for i in range(5)
                        for s in [(k >= qs[i]) & (k <= qs[i + 1])]
                        if int(s.sum()) > 5]
    return out


def gru_gate_stats(cell):
    """Distribution of the GRU's RETENTION gate from the trained biases.

    torch packs GRUCell gates as [reset, update, new] and computes
        h' = (1 - z) * n + z * h
    so `z` (slice [MEM:2*MEM]) retains the OLD state and the memory horizon is
    1/(1-z), NOT 1/z. Getting this backwards inverted the first reading of the
    offline sweep entirely, so it is spelled out here.

    Bias-only, i.e. the input-independent part of the gate. It is a proxy for the
    operating regime, not the realised gate, and is reported as such: if z_mean is
    small the memory is near-memoryless and the whole leak argument is moot.
    """
    import torch
    with torch.no_grad():
        h = cell.bias_hh.shape[0] // 3
        z = torch.sigmoid(cell.bias_ih[h:2 * h] + cell.bias_hh[h:2 * h])
        z = z.detach().float().cpu()
        return {"z_mean": float(z.mean()), "z_sd": float(z.std()),
                "z_min": float(z.min()), "z_max": float(z.max()),
                "horizon_mean": float(1.0 / max(1.0 - z.mean().item(), 1e-9)),
                # spread of z is what makes the saturation ANISOTROPIC, which is the
                # whole mechanism behind H_rotate. z_sd ~ 0 => memhard should work.
                "z_cv": float(z.std() / (z.mean() + 1e-9))}


# ══════════════════════════════ CPU lab ══════════════════════════════

@app.cls(volumes={"/data": volume}, cpu=8.0, memory=65536,
         scaledown_window=KEEP_CPU, timeout=7200)
class LabCPU:
    """Diagnostics and tabular controls. No GPU, so it is cheap to keep warm."""
    cap: int = modal.parameter(default=600000)
    halflife: int = modal.parameter(default=20000)

    @modal.enter()
    def setup(self):
        import numpy as np
        print(f"[enter] LabCPU cap={self.cap} halflife={self.halflife}", flush=True)
        self.streams = load_streams(self.cap, self.halflife, want_ctx=True)
        self.tr, self.va, self.te = make_folds(self.streams)
        self.Xtr, self.ytr = concat(self.tr, "X"), concat(self.tr, "y")
        self.Xva, self.yva = concat(self.va, "X"), concat(self.va, "y")
        self.Xte, self.yte = concat(self.te, "X"), concat(self.te, "y")
        self.Ztr, self.Zva, self.Zte = (concat(g, "XZ") for g in (self.tr, self.va, self.te))
        self.Ctr, self.Cva, self.Cte = (concat(g, "C") for g in (self.tr, self.va, self.te))
        self.base_te = float(self.yte.mean())
        self.rng = np.random.default_rng(0)
        print(f"[enter] ready | train {len(self.ytr):,d} val {len(self.yva):,d} "
              f"test {len(self.yte):,d} | test prevalence {self.base_te*100:.4f}%",
              flush=True)

    # ---- helpers ----
    def _evaluate(self, tag, pva, pte, adapt=False):
        import numpy as np
        from sklearn.metrics import (average_precision_score, roc_auc_score,
                                     f1_score, precision_recall_curve)
        yva, yte = self.yva, self.yte
        ap = float(average_precision_score(yte, pte))
        roc = float(roc_auc_score(yte, pte))
        pr, rc, th = precision_recall_curve(yva, pva)
        f = 2 * pr * rc / (pr + rc + 1e-12)
        thr = float(th[int(np.argmax(f[:-1]))]) if len(th) else 0.5
        f1_va = float(f1_score(yte, (pte >= thr).astype(int), zero_division=0))
        pr2, rc2, th2 = precision_recall_curve(yte, pte)
        f2 = 2 * pr2 * rc2 / (pr2 + rc2 + 1e-12)
        f1_or = float(np.max(f2[:-1])) if len(th2) else 0.0
        rate = float((pva >= thr).mean())
        thr_q = float(np.quantile(pte, 1 - max(rate, 1e-9)))
        f1_q = float(f1_score(yte, (pte >= thr_q).astype(int), zero_division=0))
        row = {"tag": tag, "auc_pr": ap, "lift": ap / max(self.base_te, 1e-12),
               "roc": roc, "f1_val_thr": f1_va, "f1_qmatch": f1_q, "f1_oracle": f1_or}
        print(f"  {tag:34s} AUC-PR {ap:.4f} ({row['lift']:5.2f}x rnd) | ROC {roc:.4f} "
              f"(flip {1-roc:.4f}) | F1 val-thr {f1_va:.4f} | F1 q-match {f1_q:.4f} "
              f"| F1 oracle {f1_or:.4f}", flush=True)
        if adapt:
            # prior_adapt REQUIRES calibrated input, and the reference prior must be
            # the prevalence of the set the CALIBRATOR WAS FIT ON. Both were got
            # wrong here first and the guard caught it:
            #   raw balanced-LogReg scores  -> mean 0.474 vs pi 0.0049 -> pi_hat 0.5 (BOUND)
            #   isotonic + pi_train (0.0049) -> still diverged to the bound
            #   isotonic + pi_val   (0.0233) -> pi_hat 0.086 vs true 0.054, no bound
            # Isotonic fitted on val calibrates to the VAL prior, so pass that.
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pva, yva)
            pc = np.clip(iso.predict(pte), 1e-6, 1 - 1e-6)
            pi_ref = float(yva.mean())
            f1_cal = float(f1_score(yte, (pc >= 0.5).astype(int), zero_division=0))
            pa, pi_hat, hit = prior_adapt(pc, pi_ref)
            f1_pa = float(f1_score(yte, (pa >= 0.5).astype(int), zero_division=0))
            row.update({"pi_ref": pi_ref, "pi_hat": pi_hat, "pi_true": self.base_te,
                        "f1_calib": f1_cal, "f1_prior_adapt": f1_pa,
                        "pa_hit_bound": hit})
            print(f"  {'':34s} calib F1@0.5 {f1_cal:.4f} | prior-adapt pi_hat "
                  f"{pi_hat:.4f} (ref {pi_ref:.4f}, true {self.base_te:.4f}) "
                  f"F1@0.5 {f1_pa:.4f}{'  !! HIT BOUND' if hit else ''}", flush=True)
        return row

    def _fit_eval(self, tag, A, B, C, fit_sample, adapt=False):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        mu, sd = A.mean(0), A.std(0); sd[sd < 1e-6] = 1.0
        As, Bs, Cs = (A - mu) / sd, (B - mu) / sd, (C - mu) / sd
        idx = self.rng.choice(len(self.ytr), size=min(fit_sample, len(self.ytr)),
                              replace=False)
        out = []
        lr = LogisticRegression(max_iter=300, class_weight="balanced")
        lr.fit(As[idx], self.ytr[idx])
        out.append(self._evaluate(f"{tag} / LogReg", lr.predict_proba(Bs)[:, 1],
                                  lr.predict_proba(Cs)[:, 1], adapt))
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                            class_weight="balanced", random_state=0)
        gb.fit(As[idx], self.ytr[idx])
        out.append(self._evaluate(f"{tag} / HistGB", gb.predict_proba(Bs)[:, 1],
                                  gb.predict_proba(Cs)[:, 1], adapt))
        return out

    # ---- entry points ----
    @modal.method()
    def info(self):
        """Cheapest possible trigger: proves the container is warm and describes it."""
        return {"scenarios": [s["key"] for s in self.streams],
                "n_train": int(len(self.ytr)), "n_val": int(len(self.yva)),
                "n_test": int(len(self.yte)), "prev_test": self.base_te,
                "prev_train": float(self.ytr.mean()),
                "prev_ratio": float(self.yte.mean() / max(self.ytr.mean(), 1e-12)),
                "ctx_dim": CTX_DIM, "cap": self.cap, "halflife": self.halflife}

    @modal.method()
    def diag(self, fit_sample: int = 800000):
        """H1-H4 in one pass. Hypotheses are stated in RESEARCH_AND_FIXES.md PART VI."""
        with tee() as buf:
            out = self._diag(fit_sample)
        out["log"] = buf.getvalue()
        return out

    def _diag(self, fit_sample: int = 800000):
        import json, numpy as np
        from scipy import stats as sps
        from sklearn.metrics import average_precision_score, roc_auc_score
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xtr, ytr, Xva, yva, Xte, yte = (self.Xtr, self.ytr, self.Xva, self.yva,
                                        self.Xte, self.yte)
        rng = self.rng

        print("\n" + "=" * 100)
        print("H4 — does the first-N cap bias train toward the pre-infection period?")
        print("=" * 100)
        print(f"  {'scen':5s} {'family':9s} {'n_full':>11s} {'prev_full':>10s} "
              f"{'n_cap':>10s} {'prev_cap':>10s} {'ratio':>7s}   prevalence by quintile")
        h4 = {}
        for s in self.streams:
            pf = s["pos_full"] / s["n_full"] * 100
            pc = float(s["y"].mean()) * 100
            q = [float(a.mean()) * 100 for a in np.array_split(s["y"], 5)]
            h4[s["key"]] = {"prev_full": pf, "prev_cap": pc, "quintiles": q}
            r = pc / pf if pf > 1e-9 else float("nan")
            print(f"  {s['key']:5s} {s['family']:9s} {s['n_full']:11,d} {pf:9.4f}% "
                  f"{len(s['y']):10,d} {pc:9.4f}% {r:7.2f}   "
                  + " ".join(f"{v:6.3f}" for v in q))
        print(f"\n  train prevalence {ytr.mean()*100:7.4f}%  val {yva.mean()*100:7.4f}%"
              f"  test {yte.mean()*100:7.4f}%")
        print(f"  RANDOM-BASELINE AUC-PR on test = {self.base_te:.4f}")
        print(f"  train->test prevalence ratio   = {yte.mean()/ytr.mean():.2f}x")

        print("\n" + "=" * 100)
        print("H2a — per-feature train vs test shift (KS) and within-capture drift")
        print("=" * 100)
        print(f"  {'idx':>3s} {'feature':20s} {'KS':>7s} {'mu_tr':>9s} {'mu_te':>9s} "
              f"{'d/sd_tr':>8s} {'rho_pos':>8s}  flag")

        # Drift must be measured WITHIN each capture. `train` is 13 captures spliced
        # end to end, each restarting its ramp from zero, so a Spearman against the
        # global concatenated index sees a sawtooth and washes the drift out
        # (observed on a fixture: rho 0.08 on a feature whose true value was 0.845).
        def drift_per_scenario(col):
            rs = []
            for s, _, lo, hi in self.tr:
                v = s["X"][lo:hi, col]
                if len(v) < 1000 or float(np.ptp(v)) < 1e-9:
                    continue
                v = v[::max(len(v) // 20000, 1)].astype(np.float64)
                r = sps.spearmanr(np.arange(len(v), dtype=np.float64), v).statistic
                if np.isfinite(r):
                    rs.append(float(r))
            return float(np.mean(rs)) if rs else float("nan")

        # keep the subsamples SORTED: an unsorted draw destroys stream order
        sub_tr = np.sort(rng.choice(len(ytr), size=min(200000, len(ytr)), replace=False))
        sub_te = np.sort(rng.choice(len(yte), size=min(200000, len(yte)), replace=False))
        h2a = {}
        for i, name in enumerate(FEAT):
            a, b = Xtr[sub_tr, i], Xte[sub_te, i]
            ks = float(sps.ks_2samp(a, b, method="asymp").statistic)
            mu_tr, mu_te, sd_tr = float(a.mean()), float(b.mean()), float(a.std() + 1e-9)
            d = (mu_te - mu_tr) / sd_tr
            rho = drift_per_scenario(i)
            flag = ""
            if ks > 0.20: flag += " KS-SHIFT"
            if abs(d) > 0.50: flag += " MEAN-SHIFT"
            if np.isfinite(rho) and abs(rho) > 0.20: flag += " DRIFTS"
            if i in CUMULATIVE: flag += " [suspect]"
            h2a[name] = {"ks": ks, "mu_tr": mu_tr, "mu_te": mu_te, "d": d, "rho": rho}
            print(f"  {i:3d} {name:20s} {ks:7.4f} {mu_tr:9.3f} {mu_te:9.3f} "
                  f"{d:8.3f} {rho:8.3f} {flag}")

        print("\n" + "=" * 100)
        print("H2c — does any feature FLIP the sign of its label correlation?")
        print("       drift alone does NOT invert a model; a sign flip does")
        print("=" * 100)
        print(f"  {'idx':>3s} {'feature':20s} {'r_train':>9s} {'r_val':>9s} "
              f"{'r_test':>9s} {'|dr|':>7s}  flag")

        def pbis(Xs, ys, col):
            v = Xs[:, col].astype(np.float64)
            if v.std() < 1e-12 or ys.min() == ys.max():
                return float("nan")
            return float(np.corrcoef(v, ys.astype(np.float64))[0, 1])

        h2c, flips = {}, []
        for i, name in enumerate(FEAT):
            r_tr, r_va, r_te = (pbis(Xtr, ytr, i), pbis(Xva, yva, i), pbis(Xte, yte, i))
            flag = ""
            if np.isfinite(r_tr) and np.isfinite(r_te):
                if r_tr * r_te < 0 and min(abs(r_tr), abs(r_te)) > 0.02:
                    flag += " SIGN-FLIP"; flips.append((name, r_tr, r_te))
                if abs(r_te - r_tr) > 0.10:
                    flag += " WEAKENS" if abs(r_te) < abs(r_tr) else " STRENGTHENS"
            if i in CUMULATIVE: flag += " [suspect]"
            h2c[name] = {"r_train": r_tr, "r_val": r_va, "r_test": r_te}
            print(f"  {i:3d} {name:20s} {r_tr:9.4f} {r_va:9.4f} {r_te:9.4f} "
                  f"{abs(r_te - r_tr):7.4f} {flag}")
        if flips:
            print(f"\n  {len(flips)} SIGN-FLIPPING feature(s) — these invert the model:")
            for nm, a_, b_ in sorted(flips, key=lambda t: -abs(t[1] - t[2])):
                print(f"    {nm:20s} train {a_:+.4f} -> test {b_:+.4f}")
        else:
            print("\n  no sign flips: H2c REJECTED, inversion comes from elsewhere")

        print("\n" + "=" * 100)
        print("H1 + H2b — feature-set ablation, all on identical folds.")
        print("  A/B/C test whether the cumulative columns cause the inversion.")
        print("  E/F test the DEPLOYABLE fix: keep every feature, add its causal")
        print("  z-score and the regime vector. D is a non-causal probe, not a bound.")
        print("=" * 100)
        res = []
        res += self._fit_eval("A all-22", Xtr, Xva, Xte, fit_sample, adapt=True)
        res += self._fit_eval("B minus-cumulative", Xtr[:, STATIONARY],
                              Xva[:, STATIONARY], Xte[:, STATIONARY], fit_sample)
        res += self._fit_eval("C cumulative-ONLY", Xtr[:, CUMULATIVE],
                              Xva[:, CUMULATIVE], Xte[:, CUMULATIVE], fit_sample)

        # Rank-normalise within each split slice. NOT deployable and NOT an upper
        # bound: it reads the test marginal (no labels, so not label leakage, but
        # not causal either) AND it discards absolute magnitude, so it can score
        # BELOW A whenever raw magnitude carries real signal — observed on a
        # fixture, A 0.306 vs D 0.228. Read it only as "how much of the gap is
        # attributable to the marginal shift in these columns".
        def rk(M):
            out = np.empty_like(M)
            for j in range(M.shape[1]):
                o = np.argsort(M[:, j], kind="stable")
                r = np.empty(len(o), dtype=np.float32)
                r[o] = np.linspace(0.0, 1.0, len(o), dtype=np.float32)
                out[:, j] = r
            return out

        res += self._fit_eval("D rank-norm (non-causal shift probe)",
                              rk(Xtr), rk(Xva), rk(Xte), fit_sample)
        res += self._fit_eval("E raw + causal z", np.hstack([Xtr, self.Ztr]),
                              np.hstack([Xva, self.Zva]),
                              np.hstack([Xte, self.Zte]), fit_sample)
        res += self._fit_eval("F raw + z + context", np.hstack([Xtr, self.Ztr, self.Ctr]),
                              np.hstack([Xva, self.Zva, self.Cva]),
                              np.hstack([Xte, self.Zte, self.Cte]), fit_sample, adapt=True)

        print("\n" + "=" * 100)
        print("H3 — is the pooled test metric an artifact of a few captures?")
        print("=" * 100)
        A = Xtr[:, STATIONARY]; mu, sd = A.mean(0), A.std(0); sd[sd < 1e-6] = 1.0
        idx = rng.choice(len(ytr), size=min(fit_sample, len(ytr)), replace=False)
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                            class_weight="balanced", random_state=0)
        gb.fit((A[idx] - mu) / sd, ytr[idx])
        print(f"  {'scen':5s} {'family':9s} {'n_test':>10s} {'pos':>8s} {'prev':>9s} "
              f"{'share':>8s} {'AUC-PR':>8s} {'lift':>7s} {'ROC':>8s}")
        h3, tot_pos = {}, int(yte.sum())
        for s, _, i, j in self.te:
            ys = s["y"][i:j]
            npos = int(ys.sum())
            if npos == 0 or npos == len(ys):
                print(f"  {s['key']:5s} {s['family']:9s} {len(ys):10,d} {npos:8,d}"
                      f"   (degenerate)")
                continue
            p = gb.predict_proba((s["X"][i:j][:, STATIONARY] - mu) / sd)[:, 1]
            ap = float(average_precision_score(ys, p)); roc = float(roc_auc_score(ys, p))
            prev = float(ys.mean())
            h3[s["key"]] = {"n": len(ys), "pos": npos, "prev": prev,
                            "auc_pr": ap, "roc": roc}
            print(f"  {s['key']:5s} {s['family']:9s} {len(ys):10,d} {npos:8,d} "
                  f"{prev*100:8.4f}% {npos/max(tot_pos,1)*100:7.2f}% "
                  f"{ap:8.4f} {ap/max(prev,1e-9):7.2f} {roc:8.4f}")
        if h3:
            aps = np.array([v["auc_pr"] for v in h3.values()])
            share = sorted((v["pos"] / max(tot_pos, 1) for v in h3.values()), reverse=True)
            print(f"\n  per-scenario AUC-PR  min {aps.min():.4f}  max {aps.max():.4f}"
                  f"  spread {aps.max()/max(aps.min(),1e-9):.1f}x")
            print(f"  top-3 scenarios hold {sum(share[:3])*100:.1f}% of test positives")

        out = {"h4": h4, "h2a": h2a, "h2c": h2c, "results": res, "h3": h3,
               "base_te": self.base_te,
               "prev_ratio": float(yte.mean() / max(ytr.mean(), 1e-12))}
        with open(f"{DATA_DIR}/diag.json", "w") as fh:
            json.dump(out, fh, indent=1)
        volume.commit()
        print(f"\nwrote {DATA_DIR}/diag.json")
        return out


# ══════════════════════════════ GPU lab ══════════════════════════════

@app.cls(gpu="L4", volumes={"/data": volume}, cpu=8.0, memory=65536,
         # 4 h, raised from 3 h for `suite`: the default plan estimates ~2 h 10 m and
         # a slow GPU or an extra epoch must not be able to truncate the last step.
         # Each step commits its own JSON anyway, so a timeout costs one step, not all.
         scaledown_window=KEEP_GPU, timeout=14400)
class LabGPU:
    """Graph-model training. Same folds, same features, same budget for every
    model, so any delta is attributable to the architecture alone."""
    cap: int = modal.parameter(default=600000)
    halflife: int = modal.parameter(default=20000)

    @modal.enter()
    def setup(self):
        import torch, numpy as np
        print(f"[enter] LabGPU cap={self.cap} halflife={self.halflife}", flush=True)
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.streams = load_streams(self.cap, self.halflife, want_ctx=True)
        self.tr, self.va, self.te = make_folds(self.streams)
        self._refit_scalers()

    def _refit_scalers(self):
        """Recompute label stats and train-only standardisation for the CURRENT
        folds. Split out of setup() so `bench(protocol=...)` can swap folds in a
        warm container: leaving the old scalers in place would standardise the new
        train set with the old one's mean, which is a silent leak."""
        ytr = concat(self.tr, "y")
        self.yva, self.yte = concat(self.va, "y"), concat(self.te, "y")
        self.pi_train = float(ytr.mean())
        self.base_te = float(self.yte.mean())
        # train-only standardisation, for X and for C alike. C is deliberately NOT
        # re-centred per split: the whole point of C[:,0] (events seen) is that its
        # test values sit above the train range — that IS the regime signal.
        Xtr = concat(self.tr, "X")
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0)
        self.sd[self.sd < 1e-6] = 1.0
        Ctr = concat(self.tr, "C")
        self.cmu, self.csd = Ctr.mean(0), Ctr.std(0)
        self.csd[self.csd < 1e-6] = 1.0
        del Xtr, Ctr
        print(f"[enter] ready on {self.dev} | train prevalence {self.pi_train*100:.4f}% "
              f"-> test {self.base_te*100:.4f}% "
              f"({self.base_te/max(self.pi_train,1e-12):.2f}x) | "
              f"{len(self.tr)} train / {len(self.te)} test segments", flush=True)

    # ---- streaming runner ----
    def _dro_state(self, segs):
        """Fresh Group-DRO state: one weight per TRAINING CAPTURE, uniform at start.

        Why captures and not families: DRO needs groups that are (a) numerous enough
        that the max is informative and (b) actually different from one another. With
        Neris held out there are 6 remaining families but 10 captures, and two of the
        captures (the Rbot floods s10/s11) are wildly unlike the rest in prevalence --
        which is precisely the heterogeneity we want the max to see. Grouping by family
        would average those two together with the quiet Rbot captures and hide it.

        Returns a plain dict, and it is rebuilt per model+seed inside _one so that
        weights never leak across runs in a warm container -- the same class of bug as
        the stale scalers that _refit_scalers was split out to prevent.
        """
        import numpy as np
        keys = sorted({sg[0]["key"] for sg in segs})
        return {"idx": {k: i for i, k in enumerate(keys)},
                "keys": keys,
                "q": np.ones(len(keys), dtype=np.float64) / len(keys),
                "seen": np.zeros(len(keys), dtype=np.int64)}

    def _vrex_state(self, segs):
        """Fresh VREx state: one EMA of per-capture risk, plus the running mean of
        those EMAs. See _run_epoch for the objective.

        VREx (Krueger et al., ICML 2021) minimises  mean_e R_e + beta*Var_e(R_e).
        DRO (already here) minimises max_e R_e; its guarantee holds only when the
        test env is a CONVEX MIXTURE of training envs, which s09 (16.6% prevalence
        vs 0.749% train mean) provably is not. VREx equalises risks, and equal
        risks are stable under AFFINE-with-negative-coefficient combinations of
        envs -- i.e. extrapolation, which is the regime s09 actually sits in.

        Measured on the sign-flip toy (.labtest/research/e9_vrex_vs_dro.py, 15
        seeds): when the shortcut's strength VARIES across training envs, test ROC
        was ERM 0.396 / DRO 0.707 / VREx 0.775 -- VREx un-inverts hardest and
        shrinks the shortcut weight most. When it does NOT vary, all three stay at
        0.13: no objective can fix a consistently-signed shortcut, which is why
        memhard (an architectural block) and VREx (an objective) are complements,
        not alternatives.
        """
        import numpy as np
        keys = sorted({sg[0]["key"] for sg in segs})
        return {"idx": {k: i for i, k in enumerate(keys)},
                "ema": np.zeros(len(keys), dtype=np.float64),
                "seen": np.zeros(len(keys), dtype=np.int64)}

    def _run_epoch(self, model, segs, train, opt=None, buf=None,
                   is_tgn=True, uses_ctx=False, uses_ex=False, collect=False,
                   logit_adj=False, dro=None, rankloss=0.0, decorr=0.0,
                   decorr_scope="batch", vrex=None, vrex_beta=0.0,
                   plain_decay=False):
        """One chronological pass. Every piece of state is READ BEFORE IT IS
        WRITTEN for the current batch, which is what makes the whole thing causal.

        `uses_ex` additionally maintains PER-PAIR streaming state:
            plast  last timestamp on this directed pair
            pmu    EWMA mean inter-arrival on this pair
            pm2    EWMA mean square, for the sd
            pn     events seen on this pair
            pmem   GRU pair memory (only touched when the model returns pm_new)
        Cost is 5 extra hash-bucketed tensors; PAIR_BUCKETS*PMEM floats dominates
        at 2^21 * 32 * 4B = 268 MiB, allocated once per segment like node memory.

        `logit_adj` replaces POS_W reweighting with Menon et al. (2021) logit
        adjustment: train on logit + tau*log(pi1/pi0), predict on the RAW logit.
        The learned posterior is then balanced and INVARIANT TO THE TRAIN PRIOR,
        which is the principled version of what POS_W=20 was hacking at -- and it
        is the one intervention that can act on the *ranking*, unlike prior_adapt.

        `dro` switches ERM for online GROUP DRO over captures (Sagawa et al. 2020),
        with each capture as one group. See _dro_state for the argument.
        """
        import torch, torch.nn.functional as F, numpy as np, random, math
        dev = self.dev
        # tau * log(pi1/pi0), a CONSTANT offset added to the logit during training
        adj = (TAU_LOGIT_ADJ * math.log(max(self.pi_train, 1e-9)
                                        / max(1.0 - self.pi_train, 1e-9))
               if logit_adj else 0.0)
        scores, labels = [], []
        for s, w0, i0, i1 in segs:
            mem = torch.zeros(s["n_nodes"], MEM, device=dev)
            last = torch.full((s["n_nodes"],), -1.0, device=dev)   # -1 => never seen
            X = torch.from_numpy(((s["X"][w0:i1] - self.mu) / self.sd)
                                 .astype(np.float32)).to(dev)
            # Feature ablation. Multiplying AFTER standardisation makes the masked
            # columns identically zero, i.e. genuinely uninformative, while keeping
            # FEAT_IN at 22 so no layer shape and no superset proof changes.
            fm = getattr(self, "fmask_t", None)
            if fm is not None:
                X = X * fm
            if uses_ctx:
                XZ = torch.from_numpy(s["XZ"][w0:i1]).to(dev)
                fmz = getattr(self, "fmask_zt", None)
                if fmz is not None:
                    XZ = XZ * fmz
                C = torch.from_numpy(((s["C"][w0:i1] - self.cmu) / self.csd)
                                     .astype(np.float32)).to(dev)
            else:
                XZ = C = None
            ts = torch.from_numpy(s["ts"][w0:i1]).float().to(dev)
            sr = torch.from_numpy(s["src"][w0:i1]).to(dev)
            ds = torch.from_numpy(s["dst"][w0:i1]).to(dev)
            hsr = torch.from_numpy(s["hsrc"][w0:i1]).to(dev)
            hds = torch.from_numpy(s["hdst"][w0:i1]).to(dev)
            yy = torch.from_numpy(s["y"][w0:i1]).to(dev)
            if uses_ex:
                hpr = torch.from_numpy(s["hpair"][w0:i1]).to(dev)
                plast = torch.full((PAIR_BUCKETS,), -1.0, device=dev)
                pmu = torch.zeros(PAIR_BUCKETS, device=dev)
                pm2 = torch.zeros(PAIR_BUCKETS, device=dev)
                pn = torch.zeros(PAIR_BUCKETS, device=dev)
                pmem = torch.zeros(PAIR_BUCKETS, PMEM, device=dev)
            n, warm = X.shape[0], i0 - w0
            # Running moments of (activity, score) over THIS capture, for
            # decorr_scope="capture". Reset per capture because a capture is the
            # environment whose between-batch structure the penalty must see.
            dc = {"n": 0.0, "a": 0.0, "a2": 0.0, "s": 0.0, "s2": 0.0}
            bounds = [(a, min(a + BATCH, warm), True) for a in range(0, warm, BATCH)]
            bounds += [(a, min(a + BATCH, n), False) for a in range(warm, n, BATCH)]
            for b0, b1, is_warm in bounds:
                x, sb, db = X[b0:b1], sr[b0:b1], ds[b0:b1]
                xz = XZ[b0:b1] if uses_ctx else x
                cc = C[b0:b1] if uses_ctx else x[:, :0]
                hsb, hdb, tb, yb = hsr[b0:b1], hds[b0:b1], ts[b0:b1], yy[b0:b1]
                ls = last[sb]
                # dt == 0 marks "no prior contact"; the causal gate keys off it, so a
                # never-before-seen source must not receive a huge spurious dt.
                dt = torch.where(ls >= 0, torch.clamp(tb - ls, min=0.0),
                                 torch.zeros_like(tb))
                ms, md = mem[sb], mem[db]
                ex = None
                if uses_ex:
                    pb = hpr[b0:b1]
                    pl = plast[pb]
                    dtp = torch.where(pl >= 0, torch.clamp(tb - pl, min=0.0),
                                      torch.zeros_like(tb))
                    pmu_b, pm2_b, pn_b = pmu[pb], pm2[pb], pn[pb]
                    psd = torch.sqrt(torch.clamp(pm2_b - pmu_b * pmu_b, min=0.0))
                    ex = {"dtp": dtp, "piat": pmu_b, "psd": psd, "pn": pn_b,
                          "pm": pmem[pb]}
                if train and not is_warm:
                    logits, nm = model(x, xz, cc, ms, md, hsb, hdb, dt, ex)
                    # Capture pair memory NOW. model.pm_new is an attribute set on
                    # every forward, and the rehearsal replay below calls forward
                    # again with a DIFFERENT batch size, so reading it after the
                    # replay scattered the replayed batch's rows into this batch's
                    # buckets (IndexError: index 1665 out of bounds for size 1616).
                    pnew = getattr(model, "pm_new", None)
                    if logit_adj:
                        loss = F.binary_cross_entropy_with_logits(logits + adj, yb)
                    else:
                        w = torch.where(yb > 0, POS_W, 1.0)
                        loss = F.binary_cross_entropy_with_logits(logits, yb, weight=w)
                    if rankloss > 0.0:
                        # ── A12: pairwise ranking surrogate ──────────────────────
                        # We REPORT AUC-PR and we OPTIMISE weighted cross-entropy.
                        # Those are different functionals. BCE's minimiser is the
                        # posterior p(y|x), which depends on the prevalence; the
                        # minimiser of a ranking loss is any monotone function of the
                        # likelihood ratio, which does not. Train prevalence is
                        # 0.749% and test is 6.54% (8.7x), so we are grading on the
                        # one property BCE was not asked to deliver.
                        #
                        # AUC = P(s+ > s-), so with z = s+ - s-,
                        #     1 - AUC = P(z < 0) = E[1{z<0}] <= E[softplus(-z)]/log 2
                        # because softplus(-z) >= log2 * 1{z<0} pointwise (at z=0
                        # softplus = log 2, and softplus is decreasing in z). So this
                        # term is a real upper bound on 1 - AUC, not an analogy.
                        #
                        # Pairs are drawn WITHIN the batch, and a batch never
                        # straddles captures, so the surrogate is a within-capture
                        # ranking loss -- exactly the quantity MACRO AUC-PR averages.
                        # A cross-capture pair would instead ask the model to make
                        # one capture's benign traffic outrank another's, which is
                        # not a question the metric ever asks.
                        pi = (yb > 0.5).nonzero(as_tuple=True)[0]
                        ni = (yb <= 0.5).nonzero(as_tuple=True)[0]
                        if len(pi) > 0 and len(ni) > 0:
                            k = min(RANK_PAIRS, max(len(pi), len(ni)))
                            a = pi[torch.randint(len(pi), (k,), device=dev)]
                            b = ni[torch.randint(len(ni), (k,), device=dev)]
                            loss = loss + rankloss * F.softplus(
                                -(logits[a] - logits[b])).mean()
                    if decorr > 0.0:
                        # ── A14: activity decorrelation ──────────────────────────
                        # The measured failure is a SIGN INVERSION on s09 (ROC 0.2216
                        # -- confidently wrong, not uncertain). The mechanism: in the
                        # training families the busiest hosts are servers and are
                        # benign-heavy, in s09 (16.6% prevalence, 22x train) the
                        # busiest hosts ARE the bots. Any score with a positive
                        # activity component therefore flips.
                        #
                        # So penalise that component directly:
                        #     L += lambda * rho(s, a)^2,  a = mean of the 3 host
                        #                                     activity counters
                        # Unlike a learnable gate, ERM cannot decline to pay this:
                        # it is a term in the objective, not a parameter the
                        # optimiser is free to leave at zero.
                        #
                        # What it does and does not buy: if rho(s,a) = 0 within every
                        # training capture then s carries no LINEAR activity
                        # component, so reversing the sign of corr(y,a) at test time
                        # cannot reverse s. A monotone NONLINEAR dependence can still
                        # survive -- HSIC would close that at O(B^2); this is the
                        # O(B) version and it is honest about being first-order.
                        av = x[:, list(ACT_COLS)].mean(1)
                        if decorr_scope == "capture":
                            # ── the law-of-total-covariance fix ──────────────────
                            #   Cov(s,a) = E_b[Cov(s,a|b)] + Cov_b(E[s|b],E[a|b])
                            #              \___ batch scope sees ___/  \__ FREE __/
                            # ACT_COLS are CUMULATIVE counters (see
                            # modal_common.CUMULATIVE), so activity RAMPS with time
                            # and batches are contiguous time windows: the ramp is
                            # nearly constant inside a batch, so batch-centering
                            # subtracts precisely the between-batch term. A
                            # time-ordered offline rig
                            # (.labtest/research/e8_decorr_scope.py) measured the
                            # split at +0.824 within vs +0.570 between => 40.9% of
                            # the association is structurally invisible to the
                            # batch-scope penalty.
                            #
                            # Centering on the RUNNING CAPTURE mean restores it.
                            # Activity carries no gradient (it is input data), so
                            # its moments are exact running sums; the score's
                            # history is detached and only the current batch's
                            # deviation carries grad.
                            with torch.no_grad():
                                dc["n"] += float(av.numel())
                                dc["a"] += float(av.sum())
                                dc["a2"] += float((av * av).sum())
                                dc["s"] += float(logits.sum())
                                dc["s2"] += float((logits * logits).sum())
                            cnt = max(dc["n"], 1.0)
                            mu_a, mu_s = dc["a"] / cnt, dc["s"] / cnt
                            va = max(dc["a2"] / cnt - mu_a * mu_a, 0.0)
                            vs = max(dc["s2"] / cnt - mu_s * mu_s, 0.0)
                            sa_, ss_ = math.sqrt(va), math.sqrt(vs)
                            if dc["n"] < 2.0 * BATCH or sa_ < 1e-6 or ss_ < 1e-6:
                                pass          # not enough history to be meaningful
                            else:
                                # Denominator DETACHED on purpose. rho is invariant
                                # to rescaling the score, which leaves a flat
                                # direction the optimiser can drift along -- the
                                # offline rig saw ||w|| inflate 30-100x with a fully
                                # differentiable rho. Freezing the scale makes this
                                # a well-conditioned covariance penalty.
                                rho = (((av - mu_a) * (logits - mu_s)).mean()
                                       / (sa_ * ss_ + 1e-8))
                                loss = loss + decorr * rho.pow(2)
                        else:
                            sa, ss = av.std(), logits.std()
                            # .detach() in the GUARD only. The guard is a python
                            # branch and cannot be differentiated through; without
                            # the detach torch warns that a grad-tracking tensor is
                            # being read as a scalar, which is a real smell even
                            # when it is harmless.
                            if float(sa.detach()) > 1e-6 and float(ss.detach()) > 1e-6:
                                rho = (((av - av.mean())
                                        * (logits - logits.mean())).mean()
                                       / (sa * ss + 1e-8))
                                loss = loss + decorr * rho.pow(2)
                    if vrex is not None:
                        # ── online V-REx: variance-across-captures penalty ───────
                        # DRO (above) minimises max_g R_g; its risk bound holds only
                        # when the test env is a CONVEX MIXTURE of train envs. s09 is
                        # not: 16.6% prevalence vs 0.749% train mean is a 22x
                        # EXTRAPOLATION, outside the training hull, where the max-risk
                        # bound is silent. V-REx (Krueger 2021) instead minimises
                        #     mean_g R_g + beta * Var_g(R_g),
                        # and equal risks across envs are preserved under AFFINE (incl.
                        # negative-coefficient) combinations, i.e. extrapolation.
                        #
                        # Batches never straddle captures, so only one env g is seen per
                        # step. The tractable online surrogate keeps a detached EMA of
                        # each env's risk, and penalises the current batch's deviation
                        # from the running cross-env mean:
                        #     L <- L + beta * (R_g - mean_e EMA_e)^2 .
                        # d/dR_g of the true Var equals (2/E)(1-1/E)(R_g - Rbar), the
                        # same direction, so this pushes every env's risk toward the
                        # common mean -- which is what un-inverts a sign-flipped env.
                        #
                        # Measured on the sign-flip toy (e9_vrex_vs_dro.py, 15 seeds,
                        # VARIED shortcut strength): test ROC ERM 0.396 / DRO 0.707 /
                        # VREx 0.775. When shortcut strength is CONSTANT across envs the
                        # variance signal vanishes and all three sit at 0.13 -- there
                        # VREx needs an architectural block (memhard) to lean on, not a
                        # replacement for it.
                        gi = vrex["idx"][s["key"]]
                        with torch.no_grad():
                            cur = float(loss.detach())
                            if vrex["seen"][gi] == 0:
                                vrex["ema"][gi] = cur
                            else:
                                vrex["ema"][gi] += 0.1 * (cur - vrex["ema"][gi])
                            vrex["seen"][gi] += 1
                            m = vrex["seen"] > 0
                            rbar = float(vrex["ema"][m].mean())
                        loss = loss + vrex_beta * (loss - rbar).pow(2)
                    if dro is not None:
                        # ── online Group DRO, exponentiated gradient on q ────────
                        # ERM minimises sum_g (n_g/n) R_g, so a shortcut that helps the
                        # majority of the training mass and INVERTS on a minority group
                        # still lowers the objective. Nothing in ERM punishes that. It is
                        # exactly what we measured: per-capture ROC below 0.5, which is a
                        # learned relationship with the wrong sign, not underfitting.
                        #
                        # DRO minimises max_g R_g instead, so raising ANY group's risk is
                        # paid for at full weight regardless of that group's size. And if
                        # the test family lies in the convex hull of the training
                        # families, max_g R_g upper-bounds the test risk -- which is the
                        # guarantee we actually want from a leave-one-family-out claim.
                        #
                        # One group is observed per batch (batches never straddle
                        # captures), so the update touches a single coordinate:
                        #     q_g <- q_g * exp(eta * L_g);  q <- q / sum(q)
                        #     loss <- G * q_g * L_g
                        # The factor G keeps the gradient scale at L when q is uniform,
                        # so DRO on/off is comparable at a fixed learning rate.
                        gi = dro["idx"][s["key"]]
                        with torch.no_grad():
                            dro["q"][gi] *= math.exp(DRO_ETA * float(loss.detach()))
                            dro["q"] /= dro["q"].sum()
                            dro["seen"][gi] += 1
                        loss = loss * (len(dro["q"]) * dro["q"][gi])
                    if buf is not None and buf.buf:
                        r = buf.sample()
                        rl, _ = model(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
                                      r[9])
                        if logit_adj:
                            loss = loss + F.binary_cross_entropy_with_logits(
                                rl + adj, r[8])
                        else:
                            rw = torch.where(r[8] > 0, POS_W, 1.0)
                            loss = loss + F.binary_cross_entropy_with_logits(
                                rl, r[8], weight=rw)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if buf is not None and (yb.sum() > 0 or random.random() < 0.05):
                        rex = ({k: v.detach() for k, v in ex.items()}
                               if ex is not None else None)
                        buf.add((x, xz, cc, ms.detach(), md.detach(), hsb, hdb, dt,
                                 yb, rex))
                else:
                    with torch.no_grad():
                        logits, nm = model(x, xz, cc, ms, md, hsb, hdb, dt, ex)
                        pnew = getattr(model, "pm_new", None)
                with torch.no_grad():
                    if is_tgn:
                        risk = torch.sigmoid(logits)
                        ld = last[db]
                        dtd = torch.where(ld >= 0, torch.clamp(tb - ld, min=0.0),
                                          torch.zeros_like(tb))
                        if plain_decay:
                            # ── H-B fix: dt-only decay (risk removed) ────────────
                            # The default couples the retention time-constant to the
                            # model's OWN score: risky endpoints keep their memory
                            # 10x longer, so memory norm grows with predicted risk,
                            # and memory norm is itself an activity-like signal the
                            # scorer reads back next step -- a positive feedback loop.
                            # This is a FORWARD-dynamics leak (the block is under
                            # no_grad, so it was never a gradient effect); .detach()
                            # would not touch it, only removing the risk term does.
                            #
                            # Measured on the memory-loop toy (e9, H-B, 15 seeds,
                            # PAIRED): coupling the decay to risk raised the
                            # score-vs-shortcut correlation by +0.1318 +- 0.0050
                            # (t = +26.1, 20/20 seeds) -- i.e. the repo's own decay
                            # rule AMPLIFIES the exact shortcut the rest of the file
                            # is fighting. dt-only decay removes that channel while
                            # leaving genuine recency intact.
                            dec = torch.exp(-dtd / 3600.0)
                        else:
                            # risk-modulated decay: memory of risky endpoints persists longer
                            dec = torch.exp(-dtd / (3600.0 * (1.0 + 9.0 * risk)))
                        # Deduplicate to the last event per node before writing:
                        # index_copy with duplicate indices RACES (see last_write).
                        # Call order is preserved -- the src write must still
                        # overwrite the dst decay for a node that is both.
                        ud, mdd = last_write(db, md * dec.unsqueeze(-1))
                        mem = mem.index_copy(0, ud, mdd)
                        us, nms = last_write(sb, nm.detach())
                        mem = mem.index_copy(0, us, nms)
                    ua, ta = last_write(sb, tb)
                    last = last.index_copy(0, ua, ta)
                    ub, tb2 = last_write(db, tb)
                    last = last.index_copy(0, ub, tb2)
                    if uses_ex:
                        # Fold this event in AFTER it was scored, keeping the last
                        # event per pair. Within-batch staleness (two events of one
                        # pair both reading the pre-batch state) is the same
                        # approximation the node memory already makes: stale, never
                        # non-causal.
                        d = torch.clamp(dtp, min=0.0)
                        al = BEACON_ALPHA
                        # Seed the EWMA with the FIRST measured interval instead of
                        # from zero. Starting at zero biased mu low for the first
                        # ~1/alpha events, which inflates r = dt/mu and would have made
                        # every young pair look like a missed-interval beacon.
                        init = pn_b <= 1.0
                        nmu = torch.where(init, d, (1 - al) * pmu_b + al * d)
                        nm2 = torch.where(init, d * d, (1 - al) * pm2_b + al * d * d)
                        if pnew is not None and pnew.shape[0] != pb.shape[0]:
                            raise RuntimeError(
                                f"pm_new has {pnew.shape[0]} rows but the batch has "
                                f"{pb.shape[0]}: something forwarded after the scored "
                                f"batch and clobbered it")
                        gather = [nmu, nm2, tb] + ([pnew.detach()] if pnew is not None
                                                   else [])
                        up, *gl = last_write(pb, *gather)
                        pmu = pmu.index_copy(0, up, gl[0])
                        pm2 = pm2.index_copy(0, up, gl[1])
                        plast = plast.index_copy(0, up, gl[2])
                        if pnew is not None:
                            pmem = pmem.index_copy(0, up, gl[3])
                        # The COUNT must stay exact even with duplicates, so index_add
                        # rather than a deduplicated copy: three events on one pair in
                        # one batch is three events, not one.
                        pn = pn.index_add(0, pb, torch.ones_like(pn_b))
                if collect and not is_warm:
                    scores.append(torch.sigmoid(logits).detach().float().cpu().numpy())
                    labels.append(yb.detach().cpu().numpy())
            # ── memory-leak probe: stash final memory + per-node event count ──────
            # Counted over the WHOLE window including warm-up, because the memory
            # write-back above is unconditional on is_warm: a node's saturation
            # exponent k is every GRU application it ever received, not just the
            # scored ones. Counting only scored events would understate k for
            # exactly the busiest hosts and bias the probe toward "no leak".
            #
            # Only the source count is used. dst nodes receive a decay
            # (md * dec), not a GRU update, so they do not saturate.
            if is_tgn and collect and getattr(self, "_memprobe", False):
                with torch.no_grad():
                    kc = torch.bincount(sr, minlength=s["n_nodes"])
                    self._mem_stash.append(
                        (mem.detach().float().cpu().numpy(),
                         kc.detach().cpu().numpy(), s["key"]))
        if collect:
            return np.concatenate(scores), np.concatenate(labels)
        return None, None

    @modal.method()
    def smoke(self):
        """Cheap on-GPU sanity check. Verifies every model forwards with the shared
        signature, and that TGDetect_Drift is EXACTLY TGDetect_TGN at init — the
        superset property, measured rather than asserted."""
        with tee() as buf:
            out = self._smoke()
        out["log"] = buf.getvalue()
        return out

    def _smoke(self):
        import torch, numpy as np, math
        M = get_models()
        dev, out = self.dev, {}

        def fake(n=512, seed=1):
            torch.manual_seed(seed)
            return dict(x=torch.randn(n, FEAT_IN, device=dev),
                        xz=torch.randn(n, FEAT_Z, device=dev),
                        c=torch.randn(n, CTX_DIM, device=dev),
                        ms=torch.randn(n, MEM, device=dev),
                        md=torch.randn(n, MEM, device=dev),
                        hs=torch.randint(0, 64, (n,), device=dev),
                        hd=torch.randint(0, 64, (n,), device=dev),
                        dt=torch.rand(n, device=dev) * 3600,
                        ex={"dtp": torch.rand(n, device=dev) * 3600,
                            "piat": torch.rand(n, device=dev) * 3600 + 1.0,
                            "psd": torch.rand(n, device=dev) * 100,
                            "pn": torch.randint(0, 50, (n,), device=dev).float(),
                            "pm": torch.randn(n, PMEM, device=dev)})

        b_ = fake()
        args = (b_["x"], b_["xz"], b_["c"], b_["ms"], b_["md"],
                b_["hs"], b_["hd"], b_["dt"], b_["ex"])
        for name, cls in M.items():
            torch.manual_seed(0)
            m = cls().to(dev).eval()
            with torch.no_grad():
                lo, nm = m(*args)
            out[name] = {"params": sum(p.numel() for p in m.parameters()),
                         "logit_std": float(lo.std()), "has_mem": nm is not None,
                         "has_pairmem": getattr(m, "pm_new", None) is not None}
            print(f"  {name:16s} params {out[name]['params']:>9,d}  "
                  f"logit_std {out[name]['logit_std']:.4f}  "
                  f"mem {'yes' if nm is not None else 'no ':3s} "
                  f"pairmem {'yes' if out[name]['has_pairmem'] else 'no'}", flush=True)

        # ── superset proof, TGDetect_Drift ──────────────────────────────────
        # The paper's argument rests on TGDetect_Drift's function class CONTAINING
        # TGDetect_TGN's, so any measured delta is attributable to the drift
        # machinery and not to extra capacity being compared against a weaker
        # model. That is proved by explicit construction, not by sharing a seed:
        #   feat[0]  : zero the z-score input block, copy the raw block  -> ef same
        #   film[-1] : zero weight AND bias -> gamma=0, beta=0 -> (1+0)*ef+0 = ef
        #   head[0]  : zero the context input block, copy the rest      -> head same
        # Anything with an identical shape is copied verbatim. If this prints a
        # nonzero difference the experimental design is invalid, so it is asserted
        # on every run rather than trusted.
        torch.manual_seed(0); a = M["TGDetect_TGN"]().to(dev).eval()
        torch.manual_seed(7); b = M["TGDetect_Drift"]().to(dev).eval()   # different seed on purpose
        sa, sb = a.state_dict(), b.state_dict()
        b.load_state_dict({k: v for k, v in sa.items()
                           if k in sb and sb[k].shape == v.shape}, strict=False)
        with torch.no_grad():
            b.feat[0].weight.zero_()
            b.feat[0].weight[:, :FEAT_IN].copy_(a.feat[0].weight)
            b.feat[0].bias.copy_(a.feat[0].bias)
            b.film[-1].weight.zero_(); b.film[-1].bias.zero_()
            k = HID * 3 + MEM
            b.head[0].weight.zero_()
            b.head[0].weight[:, :k].copy_(a.head[0].weight)
            b.head[0].bias.copy_(a.head[0].bias)
        with torch.no_grad():
            la, ma = a(*args)
            lb, mb = b(*args)
        d = float((la - lb).abs().max())
        dm = float((ma - mb).abs().max())
        out["superset_max_logit_diff"] = d
        out["superset_max_mem_diff"] = dm
        ok = d < 1e-4 and dm < 1e-4
        print(f"\n  SUPERSET PROOF  max|logit(TGN) - logit(Drift_embedded)| = {d:.3e}"
              f"   mem diff {dm:.3e}   {'PASS' if ok else 'FAIL'}", flush=True)

        # ── superset proof, every TGDetectV2 variant ────────────────────────
        # V2 was deliberately built so that EVERY extra input arrives through a
        # zero-initialised projection ADDED to the base embedding, and so that no
        # base parameter is renamed or resized. Both properties are checked here:
        # a plain strict=False load of TGDetect_TGN's own state_dict must leave the
        # V2 model functionally identical, for EVERY flag combination. That is what
        # makes flags legitimately combinable — an interaction measured later is an
        # interaction of mechanisms, not of input layouts.
        out["v2_superset"] = {}
        worst = 0.0
        for name, cls in M.items():
            if not name.startswith("V2_"):
                continue
            torch.manual_seed(11)            # different seed on purpose
            v = cls().to(dev).eval()
            sv = v.state_dict()
            missing = [k for k in sa if k not in sv or sv[k].shape != sa[k].shape]
            v.load_state_dict({k: t for k, t in sa.items()
                               if k in sv and sv[k].shape == t.shape}, strict=False)
            with torch.no_grad():
                lv, mv = v(*args)
            dl = float((la - lv).abs().max())
            dmv = float((ma - mv).abs().max())
            worst = max(worst, dl, dmv)
            out["v2_superset"][name] = {"logit": dl, "mem": dmv,
                                        "unmapped_base_keys": missing}
            print(f"  {name:16s} d_logit {dl:.3e}  d_mem {dmv:.3e}  "
                  f"unmapped base keys {missing if missing else 'none'}  "
                  f"{'PASS' if dl < 1e-4 and dmv < 1e-4 and not missing else 'FAIL'}",
                  flush=True)
        out["v2_superset_worst"] = worst
        print(f"  V2 SUPERSET worst deviation over all variants = {worst:.3e} "
              f"{'PASS' if worst < 1e-4 else 'FAIL'}", flush=True)

        # ── N1 mechanism test: is periodicity information actually PRESENT? ──
        # Falsifiable WITHOUT training, on the CPU, in milliseconds: BeaconEnc is a
        # fixed function, so if it cannot separate a jittered beacon from Poisson
        # traffic at the SAME mean interval, N1 cannot help and must be dropped
        # before spending GPU time on it.
        #
        # This simulates the runner's EWMA recursion verbatim, so mu/sd are the same
        # noisy ~1/alpha-sample estimates the model actually receives. Three earlier
        # versions of this test fed the TRUE dispersion and reported d' = 1e9 for cv,
        # which was pure fixture artefact -- with mu and sd constant within a group
        # the within-group variance is zero and d' divides by nothing. The estimation
        # noise IS the phenomenon here.
        cols = ([f"cos{k+1}" for k in range(BEACON_HARMONICS)]
                + [f"sin{k+1}" for k in range(BEACON_HARMONICS)]
                + ["comb", "log_mu", "cv", "log_n"])
        bec = M["V2_N1_Beacon"]().to(dev).eval().bec
        rng = np.random.default_rng(3)
        NP, NE, alpha = 20000, 24, BEACON_ALPHA

        def sim(kind, frac):
            T = np.exp(rng.uniform(np.log(5), np.log(7200), NP))[:, None]
            def draw(shape):
                if kind == "poisson":
                    return rng.exponential(1.0, shape) * T
                k = 1.0
                if kind == "beacon_miss":
                    k = 1 + (rng.random(shape) < 0.2).astype(float)
                return k * T * (1.0 + frac * rng.uniform(-1, 1, shape))
            d = np.maximum(draw((NP, NE)), 1e-3)
            mu = np.zeros(NP); m2 = np.zeros(NP); n = np.zeros(NP)
            for j in range(NE):                      # the runner's exact recursion
                dj = d[:, j]
                init = n <= 1
                mu = np.where(init, dj, (1 - alpha) * mu + alpha * dj)
                m2 = np.where(init, dj * dj, (1 - alpha) * m2 + alpha * dj * dj)
                n = n + 1.0
            dt = np.maximum(draw((NP, 1))[:, 0], 1e-3)   # a FRESH unseen interval
            sd = np.sqrt(np.maximum(m2 - mu * mu, 0.0))
            t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)
            with torch.no_grad():
                return bec(t(dt), t(mu), t(sd), t(n)).float().cpu().numpy()

        def dprime(a, b):
            s = np.sqrt(0.5 * (a.var(0) + b.var(0))) + 1e-9
            return np.abs(a.mean(0) - b.mean(0)) / s

        Fp = sim("poisson", 0.0)
        print(f"\n  N1 BEACON d' vs Poisson (alpha={alpha}, {NP:,d} pairs)")
        print("    " + f"{'case':12s}" + " ".join(f"{c:>6s}" for c in cols))
        n1 = {}
        for lab, kind, frac in (("clean", "beacon", 0.0), ("jit10", "beacon", 0.10),
                                ("jit25", "beacon", 0.25), ("jit37", "beacon", 0.37),
                                ("jit50", "beacon", 0.50),
                                ("miss20+j25", "beacon_miss", 0.25)):
            dp = dprime(sim(kind, frac), Fp)
            n1[lab] = {c: float(v) for c, v in zip(cols, dp)}
            print("    " + f"{lab:12s}" + " ".join(f"{v:6.3f}" for v in dp))
        out["beacon_dprime"] = n1
        # keyed off cv and cos1, NOT the comb: the comb is a diluted summary that
        # collapses past ~10% jitter and would make a working N1 look dead.
        cv50, c1_25 = n1["jit50"]["cv"], n1["jit25"]["cos1"]
        out["beacon_ok"] = bool(cv50 > 1.0 and c1_25 > 0.5)
        print(f"    VERDICT cv@50%jitter {cv50:.2f} (need >1.0), "
              f"cos1@25%jitter {c1_25:.2f} (need >0.5) -> "
              f"{'N1 VIABLE' if out['beacon_ok'] else 'N1 NOT VIABLE — drop it'}",
              flush=True)
        print(f"    log_mu d' {n1['clean']['log_mu']:.3f} ~ 0 confirms scale-free: "
              f"not just re-detecting short intervals", flush=True)

        # ── N3 mechanism test: logit adjustment changes the learned function ──
        # tau*log(pi1/pi0) is a constant, so it CANNOT change the ranking of a fixed
        # scorer. It only matters because it changes the gradients during training.
        # Verify that: same seed, same data, one with the offset, one without.
        def train_toy(use_adj):
            torch.manual_seed(5)
            m = M["TGDetect_TGN"]().to(dev)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            y = (torch.rand(512, device=dev, generator=None) < 0.02).float()
            off = (TAU_LOGIT_ADJ * math.log(0.02 / 0.98)) if use_adj else 0.0
            for _ in range(40):
                lo, _ = m(*args)
                if use_adj:
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        lo + off, y)
                else:
                    w = torch.where(y > 0, torch.tensor(POS_W, device=dev),
                                    torch.tensor(1.0, device=dev))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        lo, y, weight=w)
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                return m(*args)[0]
        torch.manual_seed(5); l_pw = train_toy(False)
        torch.manual_seed(5); l_la = train_toy(True)
        dla = float((l_pw - l_la).abs().mean())
        out["logit_adj_delta"] = dla
        print(f"  N3 LOGIT ADJ    mean|logit(POS_W) - logit(adj)| = {dla:.4f} "
              f"({'different function, testable' if dla > 1e-3 else 'NO EFFECT -- bug'})",
              flush=True)

        # ── runner determinism ──────────────────────────────────────────────
        # Guards a bug that was live in this project until it was measured: the node
        # memory writeback used index_copy with duplicate indices, which races under
        # multithreading, so the SAME model at the SAME seed scored differently on
        # every run (max |dp| 2.0e-02). Every seed-to-seed standard deviation printed
        # before that fix was inflated by run-to-run noise, and the V2_Off control
        # would have been uninterpretable. last_write() dedupes to last-event-wins,
        # which is both deterministic and the semantics the code always claimed.
        seg = [self.te[0]]
        torch.manual_seed(0)
        det = M["TGDetect_TGN"]().to(dev).eval()
        d1 = self._run_epoch(det, seg, False, is_tgn=True, collect=True)[0]
        d2 = self._run_epoch(det, seg, False, is_tgn=True, collect=True)[0]
        dd = float(np.abs(d1 - d2).max())
        out["runner_nondeterminism"] = dd
        print(f"\n  RUNNER DETERMINISM  same model, same seed, two passes over "
              f"{len(d1):,d} events -> max|dp| {dd:.3e} "
              f"{'PASS' if dd == 0.0 else 'FAIL — variance estimates are invalid'}",
              flush=True)

        # ── V2 control through the REAL runner ──────────────────────────────
        # The forward-pass check above shares one batch; this one shares the whole
        # streaming pipeline (memory, clocks, pair state, batching). V2_Off must be
        # bit-identical to TGDetect_TGN, and so must every variant at init, because
        # the side paths are zero-initialised.
        out["v2_runner_control"] = {}
        for name in [k for k in M if k.startswith("V2_")]:
            cls = M[name]
            torch.manual_seed(0)
            pv = self._run_epoch(cls().to(dev).eval(), seg, False,
                                 is_tgn=bool(getattr(cls, "uses_mem", False)),
                                 uses_ctx=bool(getattr(cls, "uses_ctx", False)),
                                 uses_ex=bool(getattr(cls, "uses_ex", False)),
                                 collect=True)[0]
            dv = float(np.abs(d1 - pv).max())
            out["v2_runner_control"][name] = dv
            print(f"    {name:16s} max|dp| vs TGN through the runner {dv:.3e} "
                  f"{'PASS' if dv == 0.0 else 'FAIL'}", flush=True)

        # and confirm the drift machinery is genuinely live once trained: a nonzero
        # FiLM response is what makes the model regime-aware in the first place
        torch.manual_seed(0)
        drift = M["TGDetect_Drift"]().to(dev)
        g0, b0 = drift.film(drift.cnorm(b_["c"][:, :CTX_BASE])).chunk(2, -1)
        opt = torch.optim.Adam(drift.parameters(), lr=1e-3)
        y = (torch.rand(512, device=dev) < 0.2).float()
        for _ in range(30):
            lo, _ = drift(*args)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lo, y)
            opt.zero_grad(); loss.backward(); opt.step()
        g1, b1 = drift.film(drift.cnorm(b_["c"][:, :CTX_BASE])).chunk(2, -1)
        out["film_gamma_init"] = float(g0.abs().mean())
        out["film_gamma_trained"] = float(g1.abs().mean())
        print(f"  FiLM RESPONSE   |gamma| init {float(g0.abs().mean()):.2e} -> "
              f"trained {float(g1.abs().mean()):.4f} | |beta| init "
              f"{float(b0.abs().mean()):.2e} -> trained {float(b1.abs().mean()):.4f}",
              flush=True)
        return out

    @modal.method()
    def bench(self, models: str = "Pomsathit_GCN,TGDetect_TGN,TGDetect_Drift",
              seeds: int = 3, epochs: int = 6, calibrate: bool = True,
              logit_adj: bool = False, features: str = "all",
              protocol: str = "temporal", holdout: str = "", dro: bool = False,
              select: str = "none", rankloss: float = 0.0, decorr: float = 0.0,
              ensemble: bool = False, probe_oracle: bool = False,
              repeats: int = 1, memprobe: bool = False,
              decorr_scope: str = "batch", vrex: float = 0.0,
              plain_decay: bool = False):
        """Train each named model on identical folds and report matched metrics."""
        with tee() as buf:
            if protocol != "temporal" or holdout:
                self.tr, self.va, self.te = make_folds(
                    self.streams, protocol=protocol, holdout=holdout)
                self._refit_scalers()
            out = self._bench(models, seeds, epochs, calibrate, logit_adj,
                              features, dro, select=select, rankloss=rankloss,
                              decorr=decorr, ensemble=ensemble,
                              probe_oracle=probe_oracle, repeats=repeats,
                              memprobe=memprobe, decorr_scope=decorr_scope,
                              vrex=vrex, plain_decay=plain_decay)
        out["log"] = buf.getvalue()
        return out

    @modal.method()
    def suite(self, plan: str = "robust", seeds: int = 3, epochs: int = 6,
              calibrate: bool = True):
        """Run a whole experiment plan back to back on ONE stream load.

        Per-step `seeds: 0` means "use the caller's --seeds"; a non-zero value in the
        plan overrides it, which is how the control step stays at 1 seed (it is an
        identity check, so extra seeds buy nothing but GPU minutes).

        Folds and scalers are rebuilt whenever protocol/holdout changes, and the
        rebuild ALWAYS calls _refit_scalers: standardising a new train set with the
        previous split's mean is a silent leak, and in a warm container that leak
        would be invisible.
        """
        import json, time
        if plan not in SUITES:
            raise ValueError(f"unknown plan {plan!r}; have {sorted(SUITES)}")
        steps = SUITES[plan]
        with tee() as buf:
            eta = suite_eta(plan, seeds, epochs)
            print(f"\n{'#'*100}\n# SUITE '{plan}' — {len(steps)} steps, "
                  f"estimated {eta/60:.0f} min ({eta/3600:.1f} h)\n{'#'*100}",
                  flush=True)
            for i, st in enumerate(steps, 1):
                ns = st["seeds"] or seeds
                nm = len(st["models"].split(","))
                knobs = " ".join(
                    k for k, v in (("dro", st["dro"]),
                                   (f"select={st.get('select')}",
                                    st.get("select", "none") != "none"),
                                   (f"rank={st.get('rankloss')}",
                                    st.get("rankloss", 0.0)),
                                   (f"decorr={st.get('decorr')}",
                                    st.get("decorr", 0.0)),
                                   ("ensemble", st.get("ensemble")),
                                   ("oracle-probe", st.get("probe_oracle")),
                                   (f"x{st.get('repeats')}reps",
                                    st.get("repeats", 1) > 1)) if v) or "plain ERM"
                print(f"  step {i}/{len(steps)}  {st['step']:<16} "
                      f"{nm} models x {ns} seeds x {st.get('repeats',1)} reps | "
                      f"features={st['features']} "
                      f"holdout={st['holdout'] or '-'} | {knobs}", flush=True)
            print(flush=True)

            results, cur = [], None
            t_all = time.perf_counter()
            for i, st in enumerate(steps, 1):
                key = (st["protocol"], st["holdout"])
                if key != cur:
                    self.tr, self.va, self.te = make_folds(
                        self.streams, protocol=st["protocol"],
                        holdout=st["holdout"])
                    self._refit_scalers()
                    cur = key
                print(f"\n{'='*100}\n=== STEP {i}/{len(steps)}: {st['step']}\n"
                      f"=== {st['why']}\n{'='*100}", flush=True)
                t0 = time.perf_counter()
                out = self._bench(st["models"], st["seeds"] or seeds, epochs,
                                  calibrate, False, st["features"], st["dro"],
                                  select=st.get("select", "none"),
                                  rankloss=st.get("rankloss", 0.0),
                                  decorr=st.get("decorr", 0.0),
                                  ensemble=st.get("ensemble", False),
                                  probe_oracle=st.get("probe_oracle", False),
                                  repeats=st.get("repeats", 1))
                out["_step"] = st["step"]; out["_cfg"] = st
                out["_secs"] = time.perf_counter() - t0
                results.append(out)
                # Commit after EVERY step. A timeout in step 3 must not cost steps 1-2.
                with open(f"{DATA_DIR}/suite_{plan}.json", "w") as fh:
                    json.dump({"plan": plan, "steps": results,
                               "complete": len(results) == len(steps)}, fh, indent=1)
                volume.commit()
                print(f"  [step {i} done in {out['_secs']/60:.1f} min | "
                      f"elapsed {(time.perf_counter()-t_all)/60:.1f} min | "
                      f"partial results committed]", flush=True)
            self._suite_table(results)
            print(f"\nwrote {DATA_DIR}/suite_{plan}.json "
                  f"| total {(time.perf_counter()-t_all)/60:.1f} min", flush=True)
        return {"plan": plan, "steps": results, "log": buf.getvalue()}

    @staticmethod
    def _suite_table(results):
        """One table across every step. This is the point of running a suite.

        MACRO is the column to read (the pooled number is whatever the
        highest-prevalence capture says), and the +-SE is what decides whether a gap
        is a result: the loss this suite is trying to reverse was 3.3 SE, and an
        earlier win of ours was only 1.9 SE and therefore merely directional.
        """
        import math
        print(f"\n\n{'#'*100}\n# SUITE SUMMARY — MACRO AUC-PR (mean +- SE over "
              f"seeds)\n{'#'*100}")
        for r in results:
            cfg = r["_cfg"]
            print(f"\n  {r['_step']}  |  features={cfg['features']} "
                  f"protocol={cfg['protocol']} holdout={cfg['holdout'] or '-'} "
                  f"dro={cfg['dro']}  ({r['_secs']/60:.0f} min)")
            per = {}
            for row in r["rows"]:
                # Repeats (#r0/#r1/...) are byte-identical configurations, so they
                # belong in ONE bucket -- their spread is the noise floor, and
                # splitting them into three single-run rows would hide exactly the
                # quantity the step exists to measure. The #ens row is a different
                # estimator and stays separate.
                nm = row["model"].split("#r")[0]
                per.setdefault(nm, []).append(row["auc_pr_macro"])

            def stat(v):
                mu = sum(v) / len(v)
                se = ((sum((x - mu) ** 2 for x in v) / (len(v) - 1)) ** .5
                      / math.sqrt(len(v))) if len(v) > 1 else 0.0
                return mu, se

            # Reference = Pomsathit when it is in the step (that is the comparison
            # everyone wants). A step WITHOUT Pomsathit is a diagnostic step, and
            # there the reference is the anchor the variant was built from.
            #
            # The verdict used to be "identical or the suite is void". That was
            # wrong, and the first real suite proved it wrong: two byte-identical
            # trainings of TGDetect_TGN differed by 0.2245 MACRO, so demanding exact
            # equality of two TRAINED models is demanding something CUDA will not
            # give. Exactness is a property of the EVAL path (smoke asserts it) and
            # never of the training path. The comparison is therefore against the
            # measured noise floor, and the honest verdict for a difference below it
            # is "cannot be distinguished from run-to-run noise" -- which is not the
            # same as "the same" and not the same as "different".
            ANCHOR_OF = {"R_Off": "TGDetect_TGN", "R_MemNorm": "TGDetect_TGN",
                         "R_MemDrop": "TGDetect_TGN", "R_Both": "TGDetect_TGN",
                         "R_MemHard": "TGDetect_TGN", "R_HardDrop": "TGDetect_TGN",
                         "RD_Off": "TGDetect_Drift", "RD_Both": "TGDetect_Drift",
                         "RD_MemHard": "TGDetect_Drift"}
            control = "Pomsathit_GCN" not in per
            ref = None if control else "Pomsathit_GCN"
            hdr = "vs anchor" if control else "vs Pomsathit"
            print(f"    {'model':<16} {'MACRO AUC-PR':>16} {'spread':>9} {hdr:>12} "
                  f"{'SE units':>9}  verdict")
            rm, rse = stat(per[ref]) if ref else (None, 0.0)
            for m, v in per.items():
                mu, se = stat(v)
                cell = f"{mu:.4f}+-{se:.4f}" if se else f"{mu:.4f}"
                spread = f"{max(v)-min(v):.4f}" if len(v) > 1 else "--"
                if control:
                    a = ANCHOR_OF.get(m)
                    if a is None or a not in per:
                        print(f"    {m:<16} {cell:>16} {spread:>9} {'--':>12} "
                              f"{'--':>9}  anchor"
                              + (f"  [{len(v)} runs, range {max(v)-min(v):.4f} "
                                 f"= THE NOISE FLOOR]" if len(v) > 2 else ""))
                        continue
                    am, _ = stat(per[a])
                    d = mu - am
                    print(f"    {m:<16} {cell:>16} {spread:>9} {d:>+12.4f} "
                          f"{'--':>9}  "
                          + ("within noise floor (|d| < %.2f) - NOT a result "
                             "either way" % NOISE_MACRO if abs(d) < NOISE_MACRO
                             else "EXCEEDS the %.2f noise floor - worth a "
                                  "replication" % NOISE_MACRO))
                    continue
                if m == ref:
                    print(f"    {m:<16} {cell:>16} {spread:>9} {'--':>12} "
                          f"{'--':>9}  baseline")
                    continue
                d = mu - rm
                sed = math.sqrt(se ** 2 + rse ** 2)
                nse = abs(d) / sed if sed > 0 else float("inf")
                verdict = ("BEATS Pomsathit" if d > 0 and nse >= 2 else
                           "beats, directional only" if d > 0 else
                           "LOSES to Pomsathit" if nse >= 2 else
                           "loses, directional only")
                print(f"    {m:<16} {cell:>16} {spread:>9} {d:>+12.4f} "
                      f"{nse:>9.2f}  {verdict}")
        print("\n  Read MACRO, not pooled. A gap under ~2 SE is directional only,")
        print(f"  and the run-to-run noise floor on this exam is ~{NOISE_MACRO:.2f}")
        print("  MACRO (measured: the same model at the same seed, trained twice,")
        print("  scored 0.7319 and 0.5074). The SPREAD column is max-min over runs;")
        print("  when it is larger than the gap you are reading, there is no result\n"
              "  there yet -- add seeds or reduce variance, do not add architecture.\n")

    # ---- one (model, seed) run, shared by bench and ablate ----
    class _Reservoir:
        """Fixed-capacity uniform reservoir over training batches (Vitter's R).
        This is the base paper's 10% rehearsal buffer, kept identical for every
        model so continual-learning budget is never a confound."""
        def __init__(self, cap):
            import random
            self.cap, self.buf, self.n, self._r = cap, [], 0, random
        def add(self, item):
            self.n += 1
            if len(self.buf) < self.cap:
                self.buf.append(item)
            else:
                j = self._r.randint(0, self.n - 1)
                if j < self.cap:
                    self.buf[j] = item
        def sample(self):
            return self._r.choice(self.buf) if self.buf else None

    @staticmethod
    def _macro_ap(p, y, segs):
        """Mean AUC-PR over the captures in `segs`, skipping degenerate ones.

        Split out of _one because best-epoch selection and the oracle probe both
        need it mid-training, and duplicating the offset walk is how the pooled and
        macro numbers drift apart.
        """
        import numpy as np
        from sklearn.metrics import average_precision_score
        vals, off = [], 0
        for sg in segs:
            n = sg[3] - sg[2]
            yc, pc = y[off:off + n], p[off:off + n]
            off += n
            k = int((yc > 0.5).sum())
            if k == 0 or k == len(yc):
                continue
            vals.append(float(average_precision_score(yc, pc)))
        return float(np.mean(vals)) if vals else float("nan")

    def _val_select_score(self, model, is_tgn, uses_ctx, uses_ex, how,
                          plain_decay=False):
        """The scalar that best-epoch selection maximises. Never touches test."""
        import numpy as np
        from sklearn.metrics import average_precision_score
        p, y = self._run_epoch(model, self.va, False, is_tgn=is_tgn,
                               uses_ctx=uses_ctx, uses_ex=uses_ex, collect=True,
                               plain_decay=plain_decay)
        vals, off = [], 0
        for sg in self.va:
            n = sg[3] - sg[2]
            yc, pc = y[off:off + n], p[off:off + n]
            off += n
            k = int((yc > 0.5).sum())
            if k == 0 or k == len(yc):
                continue
            vals.append(float(average_precision_score(yc, pc)))
        if not vals:
            return float("nan")
        return float(min(vals)) if how == "worst" else float(np.mean(vals))

    def _one(self, name, cls, seed, epochs, calibrate, logit_adj=False,
             features="all", dro=False, select="none", rankloss=0.0, decorr=0.0,
             probe_oracle=False, tag="", memprobe=False, decorr_scope="batch",
             vrex=0.0, plain_decay=False):
        """Train one model at one seed on the shared folds and score it.

        Every knob that could become a confound is read off the class, never
        guessed from the name: an earlier version inferred `is_tgn` from
        name.startswith("TGDetect"), which would have silently disabled node
        memory for every V2_* variant and made the whole ablation meaningless.
        """
        import torch, numpy as np, random, time
        from sklearn.metrics import (average_precision_score, roc_auc_score,
                                     f1_score, precision_recall_curve)
        from sklearn.isotonic import IsotonicRegression
        uses_ctx = bool(getattr(cls, "uses_ctx", False))
        uses_ex = bool(getattr(cls, "uses_ex", False))
        is_tgn = bool(getattr(cls, "uses_mem", False))
        # Build the feature mask BEFORE the model and cache it on self, so
        # _run_epoch picks it up without threading another argument through.
        fmv = feature_mask(features)
        self.fmask_t = torch.from_numpy(fmv).to(self.dev)
        self.fmask_zt = torch.from_numpy(
            np.tile(fmv, len(Z_SCALES)).astype(np.float32)).to(self.dev)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = cls().to(self.dev)
        # weight_decay 1e-5 is Pomsathit III.G.1c. It was absent, so every number
        # on record was produced with NO weight decay and NO dropout.
        opt = torch.optim.Adam(model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
        nb = sum(j - i for _, _, i, j in self.tr) // BATCH
        buf = self._Reservoir(max(int(nb * REHEARSAL_FRAC), 1))
        t0 = time.perf_counter()
        # Fresh per (model, seed). Carrying group weights over from a previous run in a
        # warm container would make results depend on call ORDER -- the same failure
        # mode as the stale scalers _refit_scalers was split out to prevent.
        dstate = self._dro_state(self.tr) if dro else None
        vstate = self._vrex_state(self.tr) if vrex > 0.0 else None
        best, hist = None, []
        for ep in range(epochs):
            model.train()
            self._run_epoch(model, self.tr, True, opt, buf, is_tgn, uses_ctx,
                            uses_ex=uses_ex, logit_adj=logit_adj, dro=dstate,
                            rankloss=rankloss, decorr=decorr,
                            decorr_scope=decorr_scope, vrex=vstate,
                            vrex_beta=vrex, plain_decay=plain_decay)
            if select == "none" and not probe_oracle:
                continue
            # ── A10: per-epoch validation and BEST-EPOCH selection ───────────────
            # Until now nothing selected anything: the runner trained `epochs`
            # epochs and scored whatever weights the last batch happened to leave
            # behind. With a training process whose outcome is bimodal (the same
            # config at the same seed produced MACRO 0.7319 and 0.5074 -- see
            # NOISE_MACRO) that is a lottery ticket, not an estimate.
            #
            # Selection criterion is per-capture on the VAL split, which under a
            # family holdout is the last 20% of each TRAINING capture:
            #   "macro"  mean AUC-PR over val captures
            #   "worst"  MIN over val captures. This is the DRO-flavoured criterion
            #            and it is the one to prefer: the held-out family is by
            #            construction the hardest group, so an epoch that is good on
            #            average but has already collapsed on one training capture is
            #            not the epoch you want to ship.
            # The generalisation cost of selecting over E epochs is
            # O(sqrt(log E / n_val)); n_val here is ~10^6 and E is 6, so it is ~1e-3,
            # three orders below the 0.22 spread it is there to remove.
            model.eval()
            vs = self._val_select_score(model, is_tgn, uses_ctx, uses_ex, select,
                                        plain_decay=plain_decay)
            rec = {"ep": ep, "val": vs}
            if probe_oracle:
                # ORACLE PROBE. Reads the TEST labels, so it is NOT a result and can
                # never be reported as one -- it is the headroom measurement that
                # says whether best-epoch selection is worth having at all. If the
                # oracle epoch and the val-selected epoch score the same, selection
                # is already extracting everything available.
                pt, yt = self._run_epoch(model, self.te, False, is_tgn=is_tgn,
                                         uses_ctx=uses_ctx, uses_ex=uses_ex,
                                         collect=True, plain_decay=plain_decay)
                rec["test_macro"] = self._macro_ap(pt, yt, self.te)
            hist.append(rec)
            if select != "none" and (best is None or vs > best[0]):
                best = (vs, {k: v.detach().clone()
                             for k, v in model.state_dict().items()}, ep)
        if best is not None:
            model.load_state_dict(best[1])
            print(f"  {'':16s} SELECT[{select}] epoch {best[2]} of {epochs} "
                  f"(val {best[0]:.4f}) | val trajectory "
                  + " ".join(f"{h['val']:.3f}" for h in hist), flush=True)
        model.eval()
        pva, yva = self._run_epoch(model, self.va, False, is_tgn=is_tgn,
                                   uses_ctx=uses_ctx, uses_ex=uses_ex, collect=True,
                                   plain_decay=plain_decay)
        # Armed ONLY for the test pass. The probe asks whether a held-out capture's
        # host event counts are readable out of memory, so it must be measured on
        # held-out captures; arming it for val too would just double the transfer.
        self._memprobe, self._mem_stash = bool(memprobe) and is_tgn, []
        pte, yte = self._run_epoch(model, self.te, False, is_tgn=is_tgn,
                                   uses_ctx=uses_ctx, uses_ex=uses_ex, collect=True,
                                   plain_decay=plain_decay)
        self._memprobe = False
        if dstate is not None:
            # The final q IS the finding. A near-uniform q means DRO found no group
            # conflict and should behave like ERM; a q concentrated on one or two
            # captures names exactly which ones the shortcut was failing on.
            ordq = sorted(zip(dstate["keys"], dstate["q"]), key=lambda t: -t[1])
            print(f"  {'':16s} DRO q (uniform = {1/len(dstate['q']):.3f}): "
                  + " ".join(f"{k}:{v:.3f}" for k, v in ordq[:5])
                  + f" ... min {ordq[-1][0]}:{ordq[-1][1]:.3f}", flush=True)
        ap = float(average_precision_score(yte, pte))
        roc = float(roc_auc_score(yte, pte))
        # per-capture MACRO metrics. The pooled number is dominated by whichever
        # capture holds the positives: on the 13-scenario run s10+s11 held 86.1% of
        # test positives at mean ROC 0.0115, while 7 captures sat at mean ROC 0.9805
        # holding only 13.6%. Pooled read 0.4076, macro 0.6988. Report BOTH, always.
        per_cap, off = {}, 0
        for sg in self.te:
            st, n = sg[0], sg[3] - sg[2]
            yc, pc_ = yte[off:off + n], pte[off:off + n]
            off += n
            npos = int((yc > 0.5).sum())
            if npos == 0 or npos == len(yc):
                per_cap[st["key"]] = None            # degenerate: no valid metric
                continue
            per_cap[st["key"]] = {
                "n": int(len(yc)), "pos": npos, "prev": float(npos / len(yc)),
                "auc_pr": float(average_precision_score(yc, pc_)),
                "roc": float(roc_auc_score(yc, pc_))}
            # ── the inversion diagnostic, per capture ────────────────────────────
            # r_ya  corr(label, activity). Its SIGN is the property of the capture
            #       that our model is accused of not surviving: positive in the
            #       training families (busy host -> more scored traffic), and if it
            #       goes negative on a held-out capture then any score that rides
            #       activity must invert there.
            # r_sa  corr(score, activity). How much the model IS riding it.
            # Together they are a falsifiable mechanism test, not a narrative: a
            # sub-0.5 ROC accompanied by r_sa > 0 and r_ya < 0 IS the shortcut
            # inverting, and a sub-0.5 ROC with r_sa ~ 0 is something else entirely
            # and A14 will not fix it.
            av = st["X"][sg[2]:sg[3], list(ACT_COLS)].mean(1).astype(np.float64)
            if av.std() > 1e-9:
                per_cap[st["key"]]["r_sa"] = float(
                    np.corrcoef(av, pc_.astype(np.float64))[0, 1])
                per_cap[st["key"]]["r_ya"] = float(
                    np.corrcoef(av, yc.astype(np.float64))[0, 1])
        ok = [v for v in per_cap.values() if v]
        ap_macro = float(np.mean([v["auc_pr"] for v in ok])) if ok else float("nan")
        roc_macro = float(np.mean([v["roc"] for v in ok])) if ok else float("nan")
        # ── PLAN A3/A4/A5: the macro estimator is degenerate, so report the
        #    diagnostics NEXT TO it rather than silently changing it ──────────────
        # auc_pr_macro above is UNCHANGED on purpose: every number on record was
        # computed with it, and redefining it retroactively would make the results/
        # archive incomparable. These are extra columns.
        #
        #   1. MIN_POS. s03 carries 2 positives in 150k events (prev 1.3e-05) and
        #      scores AP 9.6e-05. Averaging that in measures the sampler, not the
        #      model. A capture needs enough positives for AP to be estimable at all.
        #   2. SATURATION. s02 scores 0.9830-0.9999 for EVERY model and seed on the
        #      Neris holdout, contributing a near-constant +1/3 with ~zero variance,
        #      so macro3 is really a 2-capture comparison compressed by 1/3 while the
        #      SE is computed as if n=3. Reporting macro-without-saturated shows how
        #      much of a gap is carried by the captures that actually discriminate.
        #   3. LIFT. AP/base is dominated by the RAREST capture (s02 at 0.97%
        #      prevalence weighs ~100x), so macro-averaging it is meaningless.
        #      (AP - pi) / (1 - pi) is the prevalence-normalised version.
        MIN_POS, SAT_AP = 50, 0.99
        est = [v for v in ok if v.get("pos", 0) >= MIN_POS]
        r_extra = {
            "n_captures_estimable": len(est),
            "auc_pr_macro_estimable": (float(np.mean([v["auc_pr"] for v in est]))
                                       if est else float("nan")),
            "macro_excluded": sorted(k for k, v in per_cap.items()
                                     if v and v.get("pos", 0) < MIN_POS),
        }
        uns = [v for v in est if v["auc_pr"] < SAT_AP]
        r_extra["n_captures_unsaturated"] = len(uns)
        r_extra["auc_pr_macro_unsaturated"] = (
            float(np.mean([v["auc_pr"] for v in uns])) if uns else float("nan"))
        r_extra["saturated"] = sorted(k for k, v in per_cap.items()
                                      if v and v.get("pos", 0) >= MIN_POS
                                      and v["auc_pr"] >= SAT_AP)
        nl = [(v["auc_pr"] - v["prev"]) / max(1.0 - v["prev"], 1e-12) for v in est]
        r_extra["lift_norm_macro"] = float(np.mean(nl)) if nl else float("nan")
        pr, rc, th = precision_recall_curve(yva, pva)
        f = 2 * pr * rc / (pr + rc + 1e-12)
        thr = float(th[int(np.argmax(f[:-1]))]) if len(th) else 0.5
        f1v = float(f1_score(yte, (pte >= thr).astype(int), zero_division=0))
        pr2, rc2, th2 = precision_recall_curve(yte, pte)
        f2 = 2 * pr2 * rc2 / (pr2 + rc2 + 1e-12)
        f1o = float(np.max(f2[:-1])) if len(th2) else 0.0
        r = {"model": name + tag, "seed": seed, "auc_pr": ap,
             "lift": ap / max(self.base_te, 1e-12), "roc": roc,
             "features": features, "n_features": int(fmv.sum()),
             "auc_pr_macro": ap_macro, "roc_macro": roc_macro,
             "per_capture": per_cap, "n_captures": len(ok),
             "f1_val_thr": f1v, "f1_oracle": f1o, "logit_adj": bool(logit_adj),
             "params": sum(p.numel() for p in model.parameters()),
             "select": select, "rankloss": float(rankloss),
             "decorr": float(decorr), "epoch_hist": hist,
             "sel_epoch": (best[2] if best else epochs - 1),
             "secs": time.perf_counter() - t0}
        r.update(r_extra)
        if probe_oracle and hist:
            # Reported as a BOUND, next to the deployable number, never instead of
            # it. Same convention as f1_oracle and the rank-norm probe in _diag.
            r["macro_oracle_epoch"] = max(h.get("test_macro", float("-inf"))
                                          for h in hist)
        av_p = getattr(model, "mem_alpha", None)
        if av_p is not None:
            # The falsifiable half of A13's argument. The claim is that ERM has no
            # reason to open a gate that suppresses a shortcut which LOWERS training
            # loss, so alpha should be found near its zero initialisation. If this
            # comes back large, the claim is wrong and memnorm failed for some other
            # reason.
            r["mem_alpha_absmean"] = float(av_p.detach().abs().mean())
        # ── memory-leak probe: the measurement that decides memhard's fate ────────
        if memprobe and getattr(self, "_mem_stash", None):
            cell = getattr(model, "cell", None)
            if cell is not None:
                r["gru_gate"] = gru_gate_stats(cell)
            leaks = {}
            for Mv, kc, key in self._mem_stash:
                pr_ = mem_leak_probe(Mv, kc, seed=seed)
                if pr_:
                    leaks[key] = pr_
            if leaks:
                r["mem_leak"] = leaks
                r["mem_leak_mean"] = {
                    z: float(np.mean([v[z] for v in leaks.values()]))
                    for z in ("r2_raw", "r2_ln", "r2_resid")}
            self._mem_stash = []
        # prior adaptation ONLY on isotonically recalibrated scores: POS_W=20
        # deliberately distorts calibration and the EM diverges on raw logits.
        if calibrate:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(pva, yva)
            pc = np.clip(iso.predict(pte), 1e-6, 1 - 1e-6)
            # reference prior = prevalence of the calibrator's fit set (val),
            # NOT the train prevalence; passing pi_train diverges to the bound
            pi_ref = float(yva.mean())
            pa, pi_hat, hit = prior_adapt(pc, pi_ref)
            r.update({"pi_ref": pi_ref, "pi_hat": pi_hat,
                      "pi_true": self.base_te, "pa_hit_bound": hit,
                      "f1_calib": float(f1_score(yte, (pc >= 0.5).astype(int),
                                                 zero_division=0)),
                      "f1_prior_adapt": float(f1_score(yte, (pa >= 0.5).astype(int),
                                                       zero_division=0))})
        # Handed to the caller OUT OF BAND rather than inside r, because r is
        # json.dump'ed to the Volume and a 1.15M-element array does not belong in a
        # results file. _bench picks this up immediately after the call to build the
        # seed ensemble (A11).
        self._last_scores = (pte, yte)
        return r

    def _say(self, r, calibrate):
        msg = (f"  {r['model']:16s} seed {r['seed']}  "
               f"pooled AP {r['auc_pr']:.4f} ROC {r['roc']:.4f} | "
               f"MACRO AP {r.get('auc_pr_macro', float('nan')):.4f} "
               f"ROC {r.get('roc_macro', float('nan')):.4f} "
               f"({r.get('n_captures', 0)} cap) | "
               f"F1 val {r['f1_val_thr']:.4f} oracle {r['f1_oracle']:.4f}")
        if calibrate and "pi_hat" in r:
            # SLD is unstable for weakly-discriminative / poorly-calibrated
            # models: on a 1-epoch fixture it hit the upper bound for the GCN
            # and collapsed to zero for the MLP. Never print an adapted F1
            # without its pi_hat and bound flag next to it.
            msg += (f" | calib {r['f1_calib']:.4f} -> pa {r['f1_prior_adapt']:.4f}"
                    f" (pi_hat {r['pi_hat']:.4f} vs true {self.base_te:.4f}"
                    f"{', BOUND HIT — DISCARD' if r['pa_hit_bound'] else ''})")
        print(msg + f"  [{r['secs']:.0f}s]", flush=True)
        ml = r.get("mem_leak_mean")
        if ml:
            g = r.get("gru_gate") or {}
            # Read this as: r2_ln is what the CLASSIFIER can still decode about a
            # host's event count after `memhard` has done its worst. If r2_ln is
            # close to r2_raw the leak is a ROTATION and memhard is a dead end; if
            # r2_ln collapses it is a SCALE and memhard is sound. If BOTH are ~0,
            # check horizon_mean before celebrating -- a near-memoryless GRU has no
            # leak because it has no memory, which is a worse finding.
            killed = (1.0 - ml["r2_ln"] / ml["r2_raw"]) * 100 if ml["r2_raw"] > 0.05 \
                else float("nan")
            print(f"    mem-leak R2(log k):  raw {ml['r2_raw']:+.4f}  "
                  f"LN {ml['r2_ln']:+.4f}  rank1-resid {ml['r2_resid']:+.4f}"
                  f"   | LN removes {killed:.1f}% of it", flush=True)
            if g:
                print(f"    GRU retention z: mean {g['z_mean']:.4f} sd {g['z_sd']:.4f}"
                      f" cv {g['z_cv']:.3f} -> horizon 1/(1-z) = "
                      f"{g['horizon_mean']:.1f} events"
                      f"   (z spread is what makes saturation anisotropic)",
                      flush=True)

    def _ensemble_row(self, name, per_seed, scores, yte):
        """A11: one extra row per model, from the seeds ALREADY trained. Free.

        Ranks are averaged WITHIN each capture, not raw probabilities. Two reasons,
        and both matter here:
          * AUC-PR and ROC are invariant to any strictly increasing transform of the
            score, so a seed's calibration is irrelevant information. Averaging raw
            probabilities lets the most over-confident seed dominate the mean for a
            reason the metric does not care about.
          * The seeds differ wildly in scale (F1-optimal thresholds ranged 0.10-0.62
            on the first suite), which is exactly the situation where rank averaging
            and probability averaging disagree.

        Why it should work, and when it should not: write seed k's score as
        s_k = f(x) + e_k with e_k the run-to-run noise that NOISE_MACRO measures. If
        the e_k are even partially independent, the mean of K has noise variance
        ~sigma^2/K, so the signal-to-noise ratio rises by up to sqrt(K); for
        equal-variance Gaussian class-conditionals AUC = Phi(d/sqrt(2)) with
        d = Delta/sigma, and Phi is increasing, so the ensemble's AUC is >= the mean
        seed's. The failure mode is honest and visible: if one seed INVERTED (ROC
        below 0.5) its ranks fight the others, and the ensemble lands between them
        instead of above them. Read the ensemble row against the seed spread, not
        against the best seed.
        """
        import numpy as np
        from sklearn.metrics import average_precision_score, roc_auc_score
        if len(scores) < 2:
            return None
        agg = np.zeros(len(yte), dtype=np.float64)
        off = 0
        for sg in self.te:
            n = sg[3] - sg[2]
            for s in scores:
                o = np.argsort(s[off:off + n], kind="stable")
                rk = np.empty(n, dtype=np.float64)
                rk[o] = np.linspace(0.0, 1.0, n)
                agg[off:off + n] += rk / len(scores)
            off += n
        per_cap, off = {}, 0
        for sg in self.te:
            st, n = sg[0], sg[3] - sg[2]
            yc, pc_ = yte[off:off + n], agg[off:off + n]
            off += n
            k = int((yc > 0.5).sum())
            if k == 0 or k == len(yc):
                per_cap[st["key"]] = None
                continue
            per_cap[st["key"]] = {
                "n": int(len(yc)), "pos": k, "prev": float(k / len(yc)),
                "auc_pr": float(average_precision_score(yc, pc_)),
                "roc": float(roc_auc_score(yc, pc_))}
        ok = [v for v in per_cap.values() if v]
        base = dict(per_seed[0])
        # Anything that describes ONE member run would be a lie on the ensemble row.
        # F1 in particular needs a threshold, and a threshold on an average of ranks
        # is not the same object as a threshold on a probability, so it is reported
        # as nan rather than inherited from seed 0.
        for k in ("mem_alpha_absmean", "macro_oracle_epoch", "sel_epoch",
                  "f1_calib", "f1_prior_adapt", "pi_hat", "pi_ref", "pi_true",
                  "pa_hit_bound"):
            base.pop(k, None)
        base.update({
            "model": name + "#ens", "seed": -1,
            "f1_val_thr": float("nan"), "f1_oracle": float("nan"),
            # The POOLED columns are computed on the same within-capture ranks, so
            # they are comparable ACROSS captures only in the weak sense that every
            # capture contributes a uniform [0,1] grid. Read MACRO for this row.
            "auc_pr": float(average_precision_score(yte, agg)),
            "roc": float(roc_auc_score(yte, agg)),
            "lift": float(average_precision_score(yte, agg))
                    / max(self.base_te, 1e-12),
            "auc_pr_macro": float(np.mean([v["auc_pr"] for v in ok])) if ok
                            else float("nan"),
            "roc_macro": float(np.mean([v["roc"] for v in ok])) if ok
                         else float("nan"),
            "per_capture": per_cap, "n_captures": len(ok),
            "n_members": len(scores), "secs": 0.0, "epoch_hist": []})
        return base

    def _bench(self, models: str, seeds: int, epochs: int, calibrate: bool,
               logit_adj: bool = False, features: str = "all", dro: bool = False,
               select: str = "none", rankloss: float = 0.0, decorr: float = 0.0,
               ensemble: bool = False, probe_oracle: bool = False,
               repeats: int = 1, memprobe: bool = False,
               decorr_scope: str = "batch", vrex: float = 0.0,
               plain_decay: bool = False):
        import numpy as np, json
        M = get_models()
        dropped = FEATURE_SETS.get(features, [])
        extra = ""
        if dro: extra += " | OBJECTIVE=GroupDRO over captures"
        if vrex: extra += f" | OBJECTIVE=+{vrex:g}*V-REx risk-variance over captures"
        if plain_decay: extra += " | dt-only memory decay (risk decoupled)"
        if select != "none": extra += f" | SELECT best epoch by {select} val AUC-PR"
        if rankloss: extra += f" | +{rankloss:g}*pairwise-rank loss"
        if decorr: extra += (f" | +{decorr:g}*activity-decorrelation"
                            f" @ {decorr_scope} scope")
        if memprobe: extra += " | +memory-leak probe"
        if ensemble: extra += " | + seed ensemble row"
        if repeats > 1: extra += f" | {repeats} REPEATS per seed (noise measurement)"
        print(f"\n  FEATURES={features} | keeping "
              f"{int(feature_mask(features).sum())}/{FEAT_IN} | dropped groups: "
              f"{dropped or 'none'}{extra}", flush=True)
        by = {}
        names = [n.strip() for n in models.split(",") if n.strip()]
        rows, summary = [], {}
        for name in names:
            if name not in M:
                print(f"  ! unknown model {name}"); continue
            per_seed, scores, yref = [], [], None
            for seed in range(seeds):
                for rep in range(max(repeats, 1)):
                    # A repeat is a byte-identical configuration run again. Its ONLY
                    # purpose is to measure the training path's own nondeterminism,
                    # which the first real suite showed is the largest term in every
                    # comparison this file makes (see NOISE_MACRO).
                    tag = f"#r{rep}" if repeats > 1 else ""
                    r = self._one(name, M[name], seed, epochs, calibrate, logit_adj,
                                  features=features, dro=dro, select=select,
                                  rankloss=rankloss, decorr=decorr,
                                  probe_oracle=probe_oracle, tag=tag,
                                  memprobe=memprobe, decorr_scope=decorr_scope,
                                  vrex=vrex, plain_decay=plain_decay)
                    self._say(r, calibrate)
                    per_seed.append(r); rows.append(r)
                    if ensemble:
                        pte, yte = self._last_scores
                        scores.append(pte); yref = yte
            if per_seed:
                agg = {k: (float(np.mean([p[k] for p in per_seed])),
                           float(np.std([p[k] for p in per_seed])))
                       for k in per_seed[0] if isinstance(per_seed[0][k], (int, float))
                       and k != "seed"}
                summary[name] = agg
                by[name] = per_seed
                sp = [p["auc_pr_macro"] for p in per_seed]
                print(f"  {name:16s} MEAN     AUC-PR {agg['auc_pr'][0]:.4f}"
                      f"+-{agg['auc_pr'][1]:.4f}  ROC {agg['roc'][0]:.4f}"
                      f"+-{agg['roc'][1]:.4f} | MACRO spread "
                      f"{min(sp):.4f}..{max(sp):.4f} (range {max(sp)-min(sp):.4f})",
                      flush=True)
                if "mem_alpha_absmean" in per_seed[0]:
                    print(f"  {'':16s} mem_alpha |.|mean "
                          + " ".join(f"{p['mem_alpha_absmean']:.2e}"
                                     for p in per_seed)
                          + "   (zero-init; large => ERM DID open the gate)",
                          flush=True)
                if probe_oracle and "macro_oracle_epoch" in per_seed[0]:
                    ob = [p["macro_oracle_epoch"] for p in per_seed]
                    print(f"  {'':16s} ORACLE-EPOCH MACRO (upper bound, reads test "
                          f"labels, NOT a result): "
                          + " ".join(f"{v:.4f}" for v in ob), flush=True)
                if ensemble and len(scores) > 1:
                    er = self._ensemble_row(name, per_seed, scores, yref)
                    if er:
                        rows.append(er)
                        by[name + "#ens"] = [er]
                        summary[name + "#ens"] = {
                            k: (float(v), 0.0) for k, v in er.items()
                            if isinstance(v, (int, float))}
                        print(f"  {name+'#ens':16s} rank-avg of {len(scores)} runs  "
                              f"pooled AP {er['auc_pr']:.4f} | MACRO AP "
                              f"{er['auc_pr_macro']:.4f} ROC {er['roc_macro']:.4f}",
                              flush=True)
                print("", flush=True)

        # per-capture breakdown for seed 0 of each model. This is what exposed the
        # pooled metric as an artifact (s10+s11 held 86.1% of positives at ROC 0.0115
        # while 7 captures sat at 0.9805), so it is now standard output, not a debug aid.
        for name, per in by.items():
            pc = (per[0].get("per_capture") or {}) if per else {}
            if not pc:
                continue
            print(f"\n  {name} - per-capture test (seed {per[0]['seed']})")
            print(f"    {'cap':5s} {'n':>9s} {'pos':>8s} {'prev':>8s} "
                  f"{'AUC-PR':>8s} {'ROC':>8s} {'r_sa':>7s} {'r_ya':>7s}  flag")
            for k in sorted(pc):
                v = pc[k]
                if v is None:
                    print(f"    {k:5s}   degenerate (0 positives in test) - excluded")
                    continue
                rsa, rya = v.get("r_sa"), v.get("r_ya")
                # INVERTED   the model is confidently wrong here, which is a learned
                #            relationship with the wrong sign, never underfitting
                #            (underfitting reads exactly 0.5).
                # SHORTCUT-FLIP  the specific explanation A14 targets: the model's
                #            score tracks host activity while the LABEL's relation to
                #            host activity has reversed in this capture.
                flag = ""
                if v["roc"] < 0.5:
                    flag += " INVERTED"
                if (rsa is not None and rya is not None
                        and rsa * rya < 0 and abs(rsa) > 0.05):
                    flag += " SHORTCUT-FLIP"
                print(f"    {k:5s} {v['n']:>9,d} {v['pos']:>8,d} "
                      f"{v['prev']*100:>7.3f}% {v['auc_pr']:>8.4f} {v['roc']:>8.4f} "
                      f"{'' if rsa is None else f'{rsa:>+7.3f}'} "
                      f"{'' if rya is None else f'{rya:>+7.3f}'} {flag}")

        print("\n" + "=" * 118)
        print(f"{'model':18s} {'POOLED AUC-PR':>16s} {'xrnd':>7s} {'POOLED ROC':>16s} "
              f"{'MACRO AUC-PR':>14s} {'MACRO ROC':>12s} {'F1 val':>8s} {'F1 orc':>8s}")
        for name, a in summary.items():
            print(f"{name:18s} {a['auc_pr'][0]:.4f}+-{a['auc_pr'][1]:.4f} "
                  f"{a['lift'][0]:6.2f}x {a['roc'][0]:.4f}+-{a['roc'][1]:.4f} "
                  f"{a['auc_pr_macro'][0]:9.4f}+-{a['auc_pr_macro'][1]:.4f} "
                  f"{a['roc_macro'][0]:7.4f}+-{a['roc_macro'][1]:.4f} "
                  f"{a['f1_val_thr'][0]:8.4f} {a['f1_oracle'][0]:8.4f}")
        print(f"\nrandom-baseline AUC-PR = test prevalence = {self.base_te:.4f}")
        print("MACRO = mean over captures with at least one positive. Prefer it: the\n"
              "pooled column is whatever the biggest-prevalence capture says.")
        self._basin_report(by)
        out = {"rows": rows, "summary": summary, "base_te": self.base_te,
               "pi_train": self.pi_train, "cap": self.cap, "halflife": self.halflife,
               "epochs": epochs, "seeds": seeds, "logit_adj": logit_adj,
               "select": select, "rankloss": rankloss, "decorr": decorr,
               "ensemble": ensemble, "repeats": repeats, "dro": dro,
               "features": features}
        with open(f"{DATA_DIR}/bench_lab.json", "w") as fh:
            json.dump(out, fh, indent=1)
        volume.commit()
        print(f"wrote {DATA_DIR}/bench_lab.json")
        return out

    # ══════════════════ hypothesis ablation ══════════════════

    def _basin_report(self, by, thr=BASIN_THR):
        """Report P(high basin) per model -- see PLAN.md PART 0.

        WHY THIS EXISTS. Across all 15 recorded family-holdout runs the held-out
        capture's AUC-PR is BIMODAL, not Gaussian:

            LOW  n=9  0.119 .. 0.372        HIGH n=6  0.511 .. 0.780

        and a variance decomposition puts 87.3% of the spread BETWEEN the two modes
        (0.05687) versus 12.7% within them (0.00830). corr(macro, held-out cap) =
        +0.964, so the macro number is largely a readout of which mode the run landed
        in. The two runs behind NOISE_MACRO = 0.22 are one per mode under a
        byte-identical config -- that 0.22 is the BASIN GAP, not a sigma.

        The consequence is the whole point: averaging more seeds does NOT shrink a
        discrete mixture, it only estimates p = P(high basin) better. So the mean is
        the wrong estimand and every MDE computed for it is answering the wrong
        question. Fisher-exact power on p (0.17 -> 0.83) is 0.90 at n=12/arm, while
        the mean at d=0.10, sd=0.199 needs n~300.

        Reported for the WORST capture (min AP), which on family=Neris is the
        held-out-family capture that actually discriminates -- s02 saturates at
        0.98-1.00 for every model and carries no signal about the basin.
        """
        if not by:
            return
        caps = []
        for per_seed in by.values():
            for p in per_seed:
                for k, c in (p.get("per_capture") or {}).items():
                    if c and c.get("pos", 0) >= 50 and k not in caps:
                        caps.append(k)
        caps.sort()
        if not caps or max(len(v) for v in by.values()) < 2:
            return
        print(f"\n{'=' * 118}\nBASIN RATE per capture (AP >= {thr:.2f})"
              "   -- the ENDPOINT to compare, not the mean; see PLAN.md PART 0")
        print("Reported for EVERY estimable capture on purpose: which capture carries\n"
              "the bimodality is an empirical question, and picking one after seeing\n"
              "the numbers would be the forking path this report exists to avoid.")
        print(f"\n{'model':22s} {'capture':>8s} {'high/n':>8s} {'p_hat':>7s} "
              f"{'95% CI':>14s} {'gap':>6s}  seed APs")
        nmax = 0
        for name, per_seed in by.items():
            for cap in caps:
                v = sorted((p.get("per_capture") or {}).get(cap, {}).get("auc_pr")
                           for p in per_seed
                           if (p.get("per_capture") or {}).get(cap)
                           and (p["per_capture"][cap].get("pos", 0) >= 50))
                v = [x for x in v if x is not None]
                if len(v) < 2:
                    continue
                nmax = max(nmax, len(v))
                hi, n = sum(x >= thr for x in v), len(v)
                p_ = hi / n
                # Wilson: the normal approximation is worthless at n=3-12 and
                # degenerate (width 0) at p=0 or 1, which is exactly where we sit.
                z = 1.959964
                den = 1.0 + z * z / n
                ctr = (p_ + z * z / (2 * n)) / den
                hw = z * math.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / den
                # largest jump between consecutive sorted values: the signature of a
                # mixture rather than a spread-out unimodal sample.
                gap = max((b - a for a, b in zip(v, v[1:])), default=0.0)
                print(f"{name:18s} {cap:>8s} {f'{hi}/{n}':>8s} {p_:7.2f} "
                      f"{f'[{max(ctr - hw, 0):.2f}, {min(ctr + hw, 1):.2f}]':>14s} "
                      f"{gap:6.3f}  " + " ".join(f"{x:.3f}" for x in v))
        print("\ngap = largest jump between consecutive sorted seed APs. A gap that is\n"
              "large relative to the within-group spread is the mixture signature: on\n"
              "the 15 recorded family runs s09 splits 0.372 | 0.511 (gap 0.139) while\n"
              "s01 and s02 do not split at all.")
        if nmax and nmax < 12:
            print(f"n <= {nmax}/arm: Fisher-exact power for 0.17->0.83 is "
                  f"{0.370 if nmax <= 6 else 0.76:.2f}. "
                  "Read this as a MECHANISM probe, not a comparison.")

    @modal.method()
    def ablate(self, variants: str = "", seeds: int = 3, epochs: int = 4,
               logit_adj: bool = False, calibrate: bool = False):
        """Test N1..N5 individually AND combined against the same anchor.

        Reported PAIRED: variant and anchor share every seed, so the per-seed
        difference cancels the initialisation noise that swamped the 13-scenario
        run (TGDetect_TGN's ROC-AUC had sd 0.2622 — larger than any effect we are
        hunting). An unpaired comparison of two such means could not resolve a
        0.02 AUC-PR effect at 3 seeds; the paired one can.
        """
        with tee() as buf:
            out = self._ablate(variants, seeds, epochs, logit_adj, calibrate)
        out["log"] = buf.getvalue()
        return out

    def _ablate(self, variants: str, seeds: int, epochs: int, logit_adj: bool,
                calibrate: bool):
        import numpy as np, json
        M = get_models()
        ANCHOR = "TGDetect_TGN"
        if variants.strip():
            names = [n.strip() for n in variants.split(",") if n.strip()]
        else:
            names = ["V2_Off", "V2_N1_Beacon", "V2_N2_Pair", "V2_N4_Z3",
                     "V2_N5_Sync", "V2_N1N2", "V2_N4N5", "V2_All"]
        names = [n for n in names if n in M]
        print(f"ANCHOR {ANCHOR} | variants {names} | seeds {seeds} epochs {epochs} "
              f"| logit_adj {logit_adj}\n", flush=True)

        rows, by = [], {}
        for name in [ANCHOR, "Pomsathit_GCN"] + names:
            if name in by or name not in M:
                continue
            per = []
            for seed in range(seeds):
                r = self._one(name, M[name], seed, epochs, calibrate, logit_adj)
                self._say(r, calibrate)
                per.append(r); rows.append(r)
            by[name] = per
            print("", flush=True)

        anc0 = {p["seed"]: p for p in by[ANCHOR]}

        # ── measured noise floor ────────────────────────────────────────────
        # The EVAL path is bit-exact (smoke asserts it). TRAINING is not: the
        # backward passes of index_add / index_reduce reduce in nondeterministic
        # order. So the control cannot be asserted at exactly 0 and no verdict may
        # be issued below this floor. It is MEASURED, not hard-coded, by training
        # the anchor a SECOND time at every seed and taking the paired spread.
        #
        # Why every seed and why max() rather than one pair and mean(): this noise
        # is SPORADIC, not per-run. Measured on a fixture, four runs of an
        # identical configuration gave
        #     TGN    #1 0.002902565640   #2 0.002902565640
        #     V2_Off #1 0.002902638955   #2 0.002902565640
        # i.e. three of four runs were bit-identical and the third differed by
        # 7.3e-08 AUC-PR / 1.1e-05 ROC. A single twin pair therefore has a high
        # chance of drawing 0.00e+00 and declaring the floor to be zero, which
        # collapses every threshold below and flagged a mathematically identical
        # control as CONTROL BROKEN while promoting a 7e-08 difference to HELPS.
        # A noise floor must be an UPPER bound on noise, so take the worst pair.
        tw = []
        for seed in range(seeds):
            t = self._one(ANCHOR + "_twin", M[ANCHOR], seed, epochs, calibrate,
                          logit_adj)
            rows.append(t)
            tw.append(abs(t["auc_pr"] - anc0[seed]["auc_pr"]))
        floor = max(max(tw), FLOOR_MIN)
        print(f"NOISE FLOOR  anchor trained twice at each of {seeds} seeds -> "
              f"|dAUC-PR| " + " ".join(f"{v:.1e}" for v in tw))
        print(f"             floor = max(worst, FLOOR_MIN {FLOOR_MIN:.0e}) = "
              f"{floor:.2e}  (nothing below this is a result)\n", flush=True)

        anc = anc0
        print("=" * 108)
        print(f"{'variant':16s} {'params':>10s} {'AUC-PR':>16s} {'ROC-AUC':>16s} "
              f"{'dAUC-PR paired':>20s} {'wins':>6s} {'verdict':>12s}")
        table = {}
        for name, per in by.items():
            ap = np.array([p["auc_pr"] for p in per])
            rc = np.array([p["roc"] for p in per])
            d = np.array([p["auc_pr"] - anc[p["seed"]]["auc_pr"] for p in per])
            wins = int((d > 0).sum())
            # sd of the paired mean; with 3 seeds this is indicative, not a p-value.
            se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")
            # A verdict needs THREE things, and dropping any one of them produced a
            # wrong call on the fixture:
            #   numerical  |d| > 3*floor      - above the sporadic training noise
            #   statistical |d| > 2*SE        - consistent across seeds
            #   practical  |d| > MIN_EFFECT   - big enough to matter at all
            # The practical test is not decoration. When every d collapsed toward
            # zero, SE collapsed with it, so 2*SE approached 0 and a 7e-08 AUC-PR
            # difference cleared the statistical bar and was printed as HELPS.
            # An effect that survives statistics but not MIN_EFFECT is real and
            # uninteresting, which is what "negligible" says.
            thresh = max(2 * se if np.isfinite(se) else 0.0, 3 * floor)
            if name == ANCHOR:
                verdict = "anchor"
            elif name == "V2_Off":
                # must be the anchor up to training noise; a real difference here
                # means the runner feeds the variants something different and NO
                # other row in this table can be trusted.
                verdict = "CONTROL ok" if abs(d.mean()) <= max(3 * floor, MIN_EFFECT) \
                    else "CONTROL BROKEN"
            elif abs(d.mean()) < 3 * floor:
                verdict = "noise"
            elif abs(d.mean()) < MIN_EFFECT:
                verdict = "negligible"
            elif d.mean() > thresh:
                verdict = "HELPS"
            elif d.mean() < -thresh:
                verdict = "HURTS"
            else:
                verdict = "noise"
            table[name] = {"auc_pr": [float(ap.mean()), float(ap.std())],
                           "roc": [float(rc.mean()), float(rc.std())],
                           "d_auc_pr": float(d.mean()), "d_se": se,
                           "wins": wins, "n": len(d), "verdict": verdict,
                           "params": per[0]["params"]}
            print(f"{name:16s} {per[0]['params']:>10,d} "
                  f"{ap.mean():.4f}+-{ap.std():.4f} {rc.mean():.4f}+-{rc.std():.4f} "
                  f"{d.mean():+.4f}+-{se if np.isfinite(se) else 0:.4f} "
                  f"{wins:>3d}/{len(d):<2d} {verdict:>12s}")
        print(f"\nrandom-baseline AUC-PR = test prevalence = {self.base_te:.4f}")
        print(f"noise floor {floor:.2e} | min effect {MIN_EFFECT:.3f} | a verdict needs"
              f" |d| > max(2*SE, 3*floor) AND |d| > min effect")
        print("V2_Off is TGDetect_TGN with dead zero-init side paths, so its row must\n"
              "read CONTROL ok. If it does not, the runner is feeding the variants\n"
              "something different and no other row means anything.")
        out = {"rows": rows, "table": table, "anchor": ANCHOR,
               "noise_floor": floor,
               "base_te": self.base_te, "pi_train": self.pi_train,
               "cap": self.cap, "epochs": epochs, "seeds": seeds,
               "logit_adj": logit_adj}
        with open(f"{DATA_DIR}/ablate_lab.json", "w") as fh:
            json.dump(out, fh, indent=1)
        volume.commit()
        print(f"wrote {DATA_DIR}/ablate_lab.json")
        return out
