#!/usr/bin/env python3
"""
Shared definitions for every TG-Detect Modal app.

Two reasons this file exists:

1. ONE image definition. The pip layer is content-addressed, so as long as this
   package list does not change, no app ever rebuilds it again. Every other
   module imports IMAGE from here instead of declaring its own.

2. The drift machinery. We do NOT drop features whose meaning changes over the
   stream — that throws away real signal (a host that has contacted 5,000
   distinct destinations IS suspicious). Instead we give the model the means to
   understand that the meaning is regime-dependent:

     * a causal EWMA z-score channel  -> "how unusual is this value RIGHT NOW"
     * a causal context vector        -> "what regime are we in"
     * FiLM conditioning              -> lets the network re-interpret every
                                         feature dimension as a function of the
                                         regime, which is literally the
                                         statement 'the meaning can change'
     * online prior adaptation        -> absorbs the prevalence shift at
                                         decision time instead of at feature time

The raw feature is always kept alongside its contextual view. Nothing is discarded.
"""

import math

import modal

# ─────────────────────────── infrastructure ───────────────────────────

APP_LAB = "tg-detect-lab"
DATA_DIR = "/data/real/ctu13"

volume = modal.Volume.from_name("tg-detect-data", create_if_missing=True)

# Keep this list FROZEN. Changing it forces every app to rebuild its image.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "numpy==2.1.3",
                 "scikit-learn==1.5.2", "scipy==1.14.1")
    .add_local_python_source("modal_common")
)

FAMILY = {
    "s01": "Neris", "s02": "Neris", "s03": "Rbot", "s04": "Rbot", "s05": "Virut",
    "s06": "Menti", "s07": "Sogou", "s08": "Murlo", "s09": "Neris", "s10": "Rbot",
    "s11": "Rbot", "s12": "NSIS.ay", "s13": "Virut",
}

FEAT = [
    "log_dur", "log_pkts", "log_totb", "log_srcb", "srcb_ratio",
    "log_bytes_per_pkt", "log_pkts_per_sec", "is_tcp", "is_udp", "is_icmp",
    "dport_wellknown", "dport_ephemeral", "dport_scaled",
    "log_src_prior", "log_dst_prior", "log_dt_src", "log_dt_pair",
    "log_src_fanout", "log_iat_mean", "iat_cv", "is_bidir", "is_established",
]
# Columns whose absolute value is regime-dependent (cumulative lifetime counters
# and lifetime running statistics). NOT dropped — their current reference frame
# is handed to the model as context instead.
CUMULATIVE = [13, 14, 17, 18, 19]

# model dims (single source of truth; the legacy copies in modal_bench_real.py
# must be kept in step until that file is migrated to import from here)
FEAT_IN, HID, MEM, TDIM = 22, 128, 128, 32
GCN_LAYERS, TGN_LAYERS = 3, 3
# Pomsathit III.G.1 specifies "ReLU with dropout = 0.2" and "Adam ... weight
# decay = 1e-5". Both were MISSING from this project entirely - no model had any
# regulariser. That made Pomsathit_GCN an unfaithful, under-regularised version of
# their model, and it is also a plausible contributor to our own val->test
# collapse (val AP 0.955 -> test AP 0.175). Applied to ALL models so the
# comparison stays like-for-like.
DROPOUT, WEIGHT_DECAY = 0.2, 1e-5

# ── Feature groups, by index into modal_data_real.FEATURE_NAMES ──────────────────
# Why this exists. The first real head-to-head handed BOTH architectures the same
# 22-dim vector -- which contains four features we engineered ourselves:
#     log_dt_src, log_dt_pair, log_pair_iat_mean, pair_iat_cv
# Those are inter-arrival statistics. They ARE temporal modelling, done in the
# feature layer. Pomsathit's node features are "command lines, hashes, and IP
# addresses" (their Eq. 2) -- nothing temporal. So feeding their time-blind
# architecture our precomputed timing features makes it artificially temporal, and
# makes the question "does architectural time modelling help?" unanswerable: the
# answer is already in x.
#
# "history" is separated for the same reason. log_src_prior_count / log_dst_prior_count
# / log_src_fanout FINGERPRINT a host by how busy it is. Under a within-capture
# split the botnet host appears in train AND test, so those three features let any
# model identify it by activity volume alone -- no memory and no Δt required.
FEATURE_GROUPS = {
    "volume":   [0, 1, 2, 3, 4, 5, 6],       # log_dur .. log_pkts_per_sec
    "protocol": [7, 8, 9],                    # proto_tcp/udp/icmp
    "port":     [10, 11, 12],                 # dport_wellknown/ephemeral/norm
    "history":  [13, 14, 17],                 # src/dst prior count, src fanout
    "time":     [15, 16, 18, 19],             # dt_src, dt_pair, iat_mean, iat_cv
    "flags":    [20, 21],                     # dir_bidirectional, state_established
}

# Named feature configurations for the bench.
#   all      every feature. What the first head-to-head used.
#   notime   drop the 4 engineered timing features. THE ARCHITECTURE EXPERIMENT:
#            the only remaining route to temporal information is Δt, which the
#            runner passes to the model directly -- so Pomsathit_GCN becomes truly
#            time-blind (faithful to their paper) and TGDetect_TGN must earn its
#            temporal advantage from its own machinery.
#   noid     drop the 3 host-fingerprint features.
#   minimal  drop both. Pure per-flow evidence plus graph structure.
FEATURE_SETS = {
    "all":     [],
    "notime":  ["time"],
    "noid":    ["history"],
    "minimal": ["time", "history"],
}


def feature_mask(spec="all"):
    """A (FEAT_IN,) float32 vector of 1.0 (keep) / 0.0 (drop).

    Masking by MULTIPLYING AFTER STANDARDISATION, rather than by slicing the
    tensor, is deliberate: FEAT_IN stays 22, every layer shape is unchanged, the
    context cache stays valid, and all eight V2 superset proofs keep working. A
    column that is identically zero after standardisation carries exactly zero
    information, so this is a true ablation and not an approximation.
    """
    import numpy as np
    if spec not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {spec!r}; have {sorted(FEATURE_SETS)}")
    m = np.ones(FEAT_IN, dtype=np.float32)
    for g in FEATURE_SETS[spec]:
        m[FEATURE_GROUPS[g]] = 0.0
    return m

# Multi-scale causal z-scoring. One scale lagged the drift badly (a cumulative
# column shifted 1.367 sd raw and still 0.647 sd as a single-scale z), because an
# EWMA cannot be both responsive and stable. Three scales let the model pick:
# 2k = "unusual this minute", 200k = "unusual this capture".
Z_SCALES = (2000, 20000, 200000)
Z_REF = 1                              # index of the scale used for the reference frame
FEAT_Z = FEAT_IN * len(Z_SCALES)       # width of the z-channel block

CTX_BASE = 3 + 2 + len(CUMULATIVE)     # maturity(3) + rates(2) + reference frame(5) = 10
CTX_DIM = CTX_BASE + 3                 # + synchrony(3) = 13

# Pair-level state. Beaconing is a property of a (src,dst) PAIR, but the model has
# only ever had node slots, which average the pair structure away.
# Hash-bucketed, like the existing NODE_BUCKETS = 1<<17 for 1.47M nodes. Collisions
# mix two pairs' state; cost grows as m^2/(2b), so 500k distinct pairs into 2M
# buckets loses ~12%. Documented as a limitation, not hidden: the node memory the
# project already ships is over-subscribed 11x, so this is strictly less lossy.
PAIR_BUCKETS = 1 << 21
PMEM = 32
BEACON_HARMONICS = 4
# EWMA rate for per-pair inter-arrival stats. 0.10, NOT the 0.25 first guessed:
# alpha sets the effective estimation window (~1/alpha intervals), and the strongest
# channel in the encoder turns out to be the dispersion cv = sd/mu, whose estimate
# is noise-limited. Measured class separation d' (periodic vs Poisson, same mean
# interval, simulating the runner's exact recursion) on the cv channel:
#     alpha 0.25 (~4 intervals)  -> d' 4.92 clean, 3.33 at 50% jitter
#     alpha 0.10 (~10 intervals) -> d' 6.91 clean, 4.73 at 50% jitter   <-- best
#     alpha 0.05 (~20 intervals) -> d' 5.23 clean, 3.63 at 50% jitter
# 0.05 is worse than 0.10 despite the longer window because a 20-interval memory
# spans real regime changes, so the estimate is biased rather than merely noisy.
BEACON_ALPHA = 0.10
TAU_LOGIT_ADJ = 1.0


# ─────────────────────────── causal context ───────────────────────────

def _ewma(x, alpha):
    """Bias-corrected EWMA along axis 0, vectorised as a first-order IIR filter.

    y[n] = a*x[n] + (1-a)*y[n-1]. Done with lfilter so 6M events cost a C pass
    rather than a Python loop. An EWMA is O(1) state per feature, so everything
    here is implementable in a live streaming detector.
    """
    import numpy as np
    from scipy.signal import lfilter
    y = lfilter([alpha], [1.0, -(1.0 - alpha)], x, axis=0)
    n = np.arange(1, x.shape[0] + 1, dtype=np.float64)
    n = n.reshape((-1,) + (1,) * (x.ndim - 1))
    return y / np.maximum(1.0 - (1.0 - alpha) ** n, 1e-12)


def _first_occurrence(a):
    """Boolean mask, in stream order, of the first appearance of each value."""
    import numpy as np
    m = np.zeros(len(a), dtype=bool)
    _, idx = np.unique(a, return_index=True)
    m[idx] = True
    return m


def pair_hash(src, dst):
    """Directed (src,dst) -> bucket. 64-bit multiply-xor-shift mix so that
    sequential IP integers do not collide in patterns; ordered so that A->B and
    B->A are different pairs, because a beacon is directional."""
    import numpy as np
    a = src.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    b = dst.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
    h = a ^ b
    h ^= h >> np.uint64(29)
    h *= np.uint64(0xBF58476D1CE4E5B9)
    h ^= h >> np.uint64(32)
    return (h % np.uint64(PAIR_BUCKETS)).astype(np.int64)


def _group_cumcount(key):
    """For each element, how many EARLIER elements shared its key. Vectorised and
    strictly causal: a stable sort preserves stream order inside each group."""
    import numpy as np
    o = np.argsort(key, kind="stable")
    ks = key[o]
    idx = np.arange(len(ks))
    newgrp = np.empty(len(ks), dtype=bool)
    newgrp[0] = True
    np.not_equal(ks[1:], ks[:-1], out=newgrp[1:])
    start = np.maximum.accumulate(np.where(newgrp, idx, 0))
    out = np.empty(len(ks), dtype=np.float64)
    out[o] = (idx - start).astype(np.float64)
    return out


def _group_cumsum_excl(key, w):
    """Per-key cumulative sum of w over STRICTLY EARLIER elements."""
    import numpy as np
    o = np.argsort(key, kind="stable")
    ks, ws = key[o], w[o].astype(np.float64)
    idx = np.arange(len(ks))
    newgrp = np.empty(len(ks), dtype=bool)
    newgrp[0] = True
    np.not_equal(ks[1:], ks[:-1], out=newgrp[1:])
    start = np.maximum.accumulate(np.where(newgrp, idx, 0))
    cs = np.cumsum(ws)
    before = np.where(start > 0, cs[np.maximum(start - 1, 0)], 0.0)
    out = np.empty(len(ks), dtype=np.float64)
    out[o] = cs - ws - before
    return out


def build_context(X, ts, src, dst, scales=Z_SCALES, window=2048):
    """Return (XZ, C), both strictly causal — every value at event i is derived
    from events < i only. No leakage, and no peeking at the test marginal (unlike
    a batch rank transform, which is why variant D is not deployable).

      XZ : (n, 22*len(scales)) causal EWMA z-score of every feature at every
           timescale. Stationary by construction: "unusual relative to now",
           not "large". Scales are concatenated in the order given.
      C  : (n, 13) the regime descriptor
             0  log1p(events seen so far)
             1  log1p(seconds elapsed)
             2  log1p(distinct nodes seen so far)
             3  log trailing event rate   (events/sec over the last `window`)
             4  trailing new-node rate    (fraction of events introducing a node)
             5..9  EWMA mean of each CUMULATIVE column at scale Z_REF = the
                   current reference frame, i.e. what "high" means right now
            10  log1p(distinct sources that have contacted THIS dst so far)
                  -> a C2 server accumulates many distinct callers; this is the
                     temporal build-up of the star that a snapshot GCN sees only
                     as a static shape
            11  log1p(prior contacts on THIS exact directed pair)
                  -> beacon persistence
            12  trailing new-pair rate (network churn / scanning)
    """
    import numpy as np
    n, d = X.shape
    Xd = X.astype(np.float64)

    zblocks, ref_mu = [], None
    for si, hl in enumerate(scales):
        alpha = 1.0 - 0.5 ** (1.0 / max(hl, 1))
        mu = _ewma(Xd, alpha)
        m2 = _ewma(Xd * Xd, alpha)
        sd = np.sqrt(np.maximum(m2 - mu * mu, 0.0)) + 1e-3
        # shift by one event: the state used for event i excludes event i itself
        mu_c = np.vstack([np.zeros((1, d)), mu[:-1]])
        sd_c = np.vstack([np.ones((1, d)), sd[:-1]])
        zblocks.append(np.clip((Xd - mu_c) / sd_c, -12.0, 12.0).astype(np.float32))
        if si == Z_REF:
            ref_mu = mu_c
    if ref_mu is None:
        ref_mu = np.zeros((n, d))
    XZ = np.concatenate(zblocks, axis=1) if len(zblocks) > 1 else zblocks[0]

    tsd = ts.astype(np.float64)
    idx = np.arange(n, dtype=np.float64)

    # distinct nodes seen so far, counting src and dst in event order
    nodes = np.empty(2 * n, dtype=np.int64)
    nodes[0::2], nodes[1::2] = src, dst
    newmask = _first_occurrence(nodes)
    new_per_event = newmask[0::2].astype(np.float64) + newmask[1::2].astype(np.float64)
    distinct = np.cumsum(new_per_event)

    # trailing rates over a fixed event window
    w = min(window, max(n - 1, 1))
    dt_w = np.empty(n, dtype=np.float64)
    dt_w[:w] = np.maximum(tsd[:w] - tsd[0], 1e-3)
    dt_w[w:] = np.maximum(tsd[w:] - tsd[:-w], 1e-3)
    ev_rate = np.log1p(np.minimum(w / dt_w, 1e6))
    cs = np.concatenate([[0.0], np.cumsum(new_per_event)])
    nn_rate = np.empty(n, dtype=np.float64)
    nn_rate[:w] = cs[1:w + 1] / np.arange(1, w + 1)
    nn_rate[w:] = (cs[w + 1:] - cs[1:n - w + 1]) / w

    # ── synchrony block ────────────────────────────────────────────────────
    ph = pair_hash(src, dst)
    pair_prior = _group_cumcount(ph)                  # prior contacts on this pair
    first_contact = (pair_prior == 0).astype(np.float64)
    fanin = _group_cumsum_excl(dst, first_contact)    # distinct srcs seen by this dst
    fc_cs = np.concatenate([[0.0], np.cumsum(first_contact)])
    np_rate = np.empty(n, dtype=np.float64)
    np_rate[:w] = fc_cs[1:w + 1] / np.arange(1, w + 1)
    np_rate[w:] = (fc_cs[w + 1:] - fc_cs[1:n - w + 1]) / w

    C = np.empty((n, CTX_DIM), dtype=np.float32)
    C[:, 0] = np.log1p(idx)
    C[:, 1] = np.log1p(np.maximum(tsd - tsd[0], 0.0))
    C[:, 2] = np.log1p(distinct)
    C[:, 3] = ev_rate
    C[:, 4] = nn_rate / 2.0
    for k, col in enumerate(CUMULATIVE):
        C[:, 5 + k] = ref_mu[:, col]
    C[:, 10] = np.log1p(fanin)
    C[:, 11] = np.log1p(pair_prior)
    C[:, 12] = np_rate
    return XZ, C


# ─────────────────────── decision-time prior shift ───────────────────────

def prior_adapt(p, pi_ref, pi_max=0.5, damping=0.5, iters=500, tol=1e-9):
    """Saerens-Latinne-Decaestecker EM prior correction.

    Attacks the 14x train->test prevalence jump at DECISION time. Uses only the
    model's own predicted probabilities, never test labels — but it is
    transductive (it reads the batch of test scores), so it must always be
    reported as a separate column, never folded into the headline number.

    NOTE the transform is strictly MONOTONE in p, so it cannot change AUC-PR or
    ROC-AUC at all. It fixes thresholds, calibration, recall and F1. It cannot fix
    an inverted ranking — that needs the architecture side (z-channel + FiLM).

    !! TWO PRECONDITIONS, BOTH LEARNED THE HARD WAY !!

    1. p MUST BE CALIBRATED. Fed overconfident scores whose mean sat far above
       pi_ref, the EM ran away to pi = 1.0, collapsing every score to exactly 1.0
       and DESTROYING the ranking. Our graph models train with POS_W = 20 and the
       tabular controls use class_weight="balanced"; both deliberately distort
       calibration, so isotonic/Platt recalibration on validation comes FIRST.

    2. pi_ref MUST BE THE PREVALENCE OF THE SET THE CALIBRATOR WAS FIT ON — the
       validation prevalence — NOT the training prevalence. Isotonic fitted on val
       returns P(y=1|s) under the val prior, so that is the prior the EM must be
       told to correct away from. Measured on a fixture (true test pi = 0.0536):
           raw scores,        pi_ref = pi_train 0.0049 -> pi_hat 0.5000  BOUND HIT
           isotonic on val,   pi_ref = pi_train 0.0049 -> pi_hat 0.5000  BOUND HIT
           isotonic on val,   pi_ref = pi_val   0.0233 -> pi_hat 0.0860  ok
       SLD still overshoots (0.086 vs 0.054) when the classifier is imperfect;
       report pi_hat next to the true value rather than claiming it is exact.

    Guarded accordingly: pi is clamped to (0, pi_max] each step, updates are
    damped, and `hit_bound` is returned so a divergence can never be reported as
    a result by accident.

    Returns (p_adjusted, pi_estimated, hit_bound).
    """
    import numpy as np
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    pi_ref = float(np.clip(pi_ref, 1e-6, 1 - 1e-6))
    lo, hi = 1e-6, float(min(max(pi_max, pi_ref), 1 - 1e-6))

    def adjust(pi):
        a = (pi / pi_ref) * p
        b = ((1.0 - pi) / (1.0 - pi_ref)) * (1.0 - p)
        return a / (a + b + 1e-300)

    pi, hit = pi_ref, False
    for _ in range(iters):
        target = float(adjust(pi).mean())
        pi_new = float(np.clip((1.0 - damping) * pi + damping * target, lo, hi))
        if pi_new >= hi - 1e-12 or pi_new <= lo + 1e-12:
            hit = True
        if abs(pi_new - pi) < tol:
            pi = pi_new
            break
        pi = pi_new
    return adjust(pi), pi, hit


# ─────────────────────────────── models ───────────────────────────────

def get_models():
    """Torch is imported lazily so this module stays importable during
    `modal deploy` on a machine without torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TimeEnc(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.w = nn.Parameter(torch.logspace(-4, 2, d))   # log-spaced: 1s..1h beacons
            self.b = nn.Parameter(torch.zeros(d))
        def forward(self, dt):
            x = torch.log1p(torch.clamp(dt, min=0.0)).unsqueeze(-1)
            return torch.cos(x * self.w + self.b)

    def seg_softmax(scores, index, n_seg):
        mx = torch.full((n_seg,) + scores.shape[1:], -1e30, device=scores.device)
        mx = mx.index_reduce(0, index, scores, "amax", include_self=True)
        e = torch.exp(scores - mx[index])
        den = torch.zeros((n_seg,) + scores.shape[1:], device=scores.device)
        den = den.index_add(0, index, e)
        return e / (den[index] + 1e-12)

    class PomsathitGCN(nn.Module):
        """3-layer GCN, hidden 128, attention aggregation, softmax head, on window
        snapshots. Featureless init (mean of incident edge features)."""
        uses_ctx = False
        uses_ex = False
        uses_mem = False
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(FEAT_IN, HID), nn.ReLU(),
                                      nn.Dropout(DROPOUT), nn.Linear(HID, HID))
            self.W = nn.ModuleList([nn.Linear(HID * 2, HID) for _ in range(GCN_LAYERS)])
            self.att = nn.ModuleList([nn.Linear(HID * 2, 1) for _ in range(GCN_LAYERS)])
            self.head = nn.Sequential(nn.Linear(HID * 3, HID), nn.ReLU(),
                                      nn.Dropout(DROPOUT), nn.Linear(HID, 2))
            self.drop = nn.Dropout(DROPOUT)
        def forward(self, x, xz, c, ms, md, hsrc, hdst, dt, ex=None):
            uniq, inv = torch.unique(torch.cat([hsrc, hdst]), return_inverse=True)
            E = hsrc.numel(); ls, ld = inv[:E], inv[E:]; N = uniq.numel()
            ef = self.feat(x)
            one = torch.ones(E, 1, device=ef.device)
            deg = torch.zeros(N, 1, device=ef.device).index_add(0, ls, one).index_add(0, ld, one)
            h = torch.zeros(N, HID, device=ef.device)
            h = h.index_add(0, ls, ef).index_add(0, ld, ef) / deg.clamp(min=1.0)
            for W, A in zip(self.W, self.att):
                hs_i, hd_i = h[ls], h[ld]
                a_in = seg_softmax(A(torch.cat([hd_i, hs_i], -1)), ld, N)
                agg = torch.zeros_like(h).index_add(
                    0, ld, a_in * torch.relu(W(torch.cat([hs_i, ef], -1))))
                a_out = seg_softmax(A(torch.cat([hs_i, hd_i], -1)), ls, N)
                agg = agg.index_add(
                    0, ls, a_out * torch.relu(W(torch.cat([hd_i, ef], -1))))
                h = F.relu(h + agg)
            l2 = self.head(torch.cat([h[ls], h[ld], ef], -1))
            return l2[:, 1] - l2[:, 0], None

    class TGDetectTGN(nn.Module):
        """Baseline aggregation PLUS multi-resolution time encoding and a per-node
        GRU memory with risk-modulated decay. Strict superset of the GCN."""
        uses_ctx = False
        uses_ex = False
        uses_mem = True     # runner must scatter node memory back; do NOT infer
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(FEAT_IN, HID), nn.ReLU(),
                                      nn.Dropout(DROPOUT), nn.Linear(HID, HID))
            self.tenc = TimeEnc(TDIM)
            self.mproj = nn.Linear(MEM, HID)
            self.W = nn.ModuleList([nn.Linear(HID * 2 + TDIM, HID) for _ in range(TGN_LAYERS)])
            self.att = nn.ModuleList([nn.Linear(HID * 2 + TDIM, 1) for _ in range(TGN_LAYERS)])
            self.ln = nn.ModuleList([nn.LayerNorm(HID) for _ in range(TGN_LAYERS)])
            self.cell = nn.GRUCell(HID + HID + TDIM, MEM)
            self.head = nn.Sequential(nn.Linear(HID * 3 + MEM, HID), nn.ReLU(),
                                      nn.Dropout(DROPOUT), nn.Linear(HID, 1))
            self.drop = nn.Dropout(DROPOUT)

        def encode(self, x, xz, c, ex=None):
            return self.feat(x)

        def forward(self, x, xz, c, ms, md, hsrc, hdst, dt, ex=None):
            uniq, inv = torch.unique(torch.cat([hsrc, hdst]), return_inverse=True)
            E = hsrc.numel(); ls, ld = inv[:E], inv[E:]; N = uniq.numel()
            ef = self.encode(x, xz, c, ex)
            te = self.tenc(dt)
            causal = (dt > 0).float()
            one = torch.ones(E, 1, device=ef.device)
            deg = torch.zeros(N, 1, device=ef.device).index_add(0, ls, one).index_add(0, ld, one)
            h = torch.zeros(N, HID, device=ef.device)
            h = h.index_add(0, ls, ef).index_add(0, ld, ef) / deg.clamp(min=1.0)
            mp = torch.zeros(N, HID, device=ef.device)
            # Memory is READ through mem_read() and WRITTEN from raw ms. In the anchor
            # mem_read is the identity, so this refactor adds no parameters and changes
            # no arithmetic -- TGDetect_TGN stays bit-identical. Subclasses use the hook
            # to normalise or drop the memory read (see TGDetectRobust).
            mr_s, mr_d = self.mem_read(ms), self.mem_read(md)
            mp = mp.index_add(0, ls, self.mproj(mr_s) * causal.unsqueeze(-1))
            mp = mp.index_add(0, ld, self.mproj(mr_d))
            h = h + mp / deg.clamp(min=1.0)
            for W, A, LN in zip(self.W, self.att, self.ln):
                hs_i, hd_i = h[ls], h[ld]
                a_in = seg_softmax(A(torch.cat([hd_i, hs_i, te], -1)), ld, N)
                agg = torch.zeros_like(h).index_add(
                    0, ld, a_in * torch.relu(W(torch.cat([hs_i, ef, te], -1))))
                a_out = seg_softmax(A(torch.cat([hs_i, hd_i, te], -1)), ls, N)
                agg = agg.index_add(
                    0, ls, a_out * torch.relu(W(torch.cat([hd_i, ef, te], -1))))
                h = LN(h + agg)
            hs, hd = h[ls], h[ld]
            logits = self.score(self.head_in(hs, hd, ef, mr_s, c, ex), c, ex)
            # RAW ms, deliberately. The GRU recurrence is the state itself; feeding it a
            # normalised or dropped-out read would corrupt the memory chain for every
            # later event, which is a different (and much worse) intervention than
            # changing what the CLASSIFIER is allowed to see. Read-path only.
            new_mem = self.cell(torch.cat([hs, ef, te], -1), ms)
            self.pm_new = self.pair_update(hs, ef, te, ex)
            return logits, new_mem

        def mem_read(self, m):
            """What the classifier is allowed to see of the node memory.

            Identity here, on purpose: TGDetectTGN is the anchor that every superset
            proof is measured against, so it must not gain a parameter or an operation.
            TGDetectRobust overrides this."""
            return m

        def head_in(self, hs, hd, ef, ms, c, ex=None):
            return torch.cat([hs, hd, ef, ms], -1)

        def head_hidden(self, hin):
            """Everything in self.head EXCEPT the final projection.

            Subclasses that need to inject into the head's hidden layer used to index
            self.head[0]/[1]/[2] directly. Adding nn.Dropout shifted those indices and
            silently turned the final projection into a dropout call, so the split is
            now expressed structurally and is immune to head layout changes."""
            h = hin
            for m in list(self.head)[:-1]:
                h = m(h)
            return h

        def head_out(self, h):
            return list(self.head)[-1](h)

        def score(self, hin, c, ex=None):
            return self.head(hin).squeeze(-1)

        def pair_update(self, hs, ef, te, ex):
            """Overridden by TGDetectV2(pair=True). Returning None tells the runner
            there is no pair state to scatter back."""
            return None

    class TGDetectDrift(TGDetectTGN):
        """TG-Detect + explicit drift semantics.

        Three additions, and nothing removed:

          1. z-channel  - every feature is presented BOTH as its raw value and as
             a causal EWMA z-score against its own recent history. The raw value
             keeps the absolute information ("5,000 distinct destinations"); the
             z-score supplies the stationary view ("unusual for right now").

          2. FiLM conditioning - the regime vector c generates a per-dimension
             scale and shift applied to the feature embedding. This is the direct
             mechanism by which the network can learn that a feature MEANS
             something different early in a capture than it does late: high
             fanout is a busy server at t=0 and a scanning bot at t=T.

          3. c is also concatenated into the classification head, so the decision
             boundary itself is regime-aware.

        The final FiLM layer is zero-initialised and applied as (1+gamma), so at
        initialisation this model is EXACTLY TGDetectTGN. The superset property is
        therefore provable, not asserted, and any measured difference is
        attributable to the drift machinery alone.

        FROZEN AT ONE Z-SCALE AND THE BASE CONTEXT. build_context now emits
        len(Z_SCALES) z-blocks and 3 extra synchrony dims, but this model
        deliberately slices back to what it was benchmarked with, so the 13-scenario
        numbers already on record stay comparable. The wider inputs are consumed by
        TGDetectV2, where they are individually ablatable.
        """
        uses_ctx = True
        uses_ex = False
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(FEAT_IN * 2, HID), nn.ReLU(),
                                      nn.Dropout(DROPOUT), nn.Linear(HID, HID))
            self.film = nn.Sequential(nn.Linear(CTX_BASE, HID), nn.ReLU(),
                                      nn.Linear(HID, 2 * HID))
            nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)
            self.head = nn.Sequential(nn.Linear(HID * 3 + MEM + CTX_BASE, HID),
                                      nn.ReLU(), nn.Dropout(DROPOUT),
                                      nn.Linear(HID, 1))
            self.cnorm = nn.LayerNorm(CTX_BASE)

        def encode(self, x, xz, c, ex=None):
            z = xz[:, FEAT_IN * Z_REF:FEAT_IN * (Z_REF + 1)]
            ef = self.feat(torch.cat([x, z], -1))
            g, b = self.film(self.cnorm(c[:, :CTX_BASE])).chunk(2, -1)
            return (1.0 + g) * ef + b

        def head_in(self, hs, hd, ef, ms, c, ex=None):
            return torch.cat([hs, hd, ef, ms, self.cnorm(c[:, :CTX_BASE])], -1)

    class BeaconEnc(nn.Module):
        """Scale-free periodicity encoder for one directed pair. (Novelty N1.)

        Inputs are per-(src,dst) streaming state, all read BEFORE the current event
        is folded in, so this is strictly causal:
            dtp : seconds since this pair's previous event
            mu  : EWMA mean inter-arrival for this pair
            sd  : EWMA sd of inter-arrival for this pair
            n   : events seen on this pair so far

        r = dtp / mu is the PHASE RATIO. A beacon of period T has r ~ 1, or r ~ k
        when k-1 intervals were missed or netflow-aggregated -- and crucially this
        holds REGARDLESS OF T. The existing `log_dt_pair` feature cannot express
        that: a 30 s beacon and a 3600 s beacon land in different places, so the
        model has to learn each period separately and none of them transfer to a new
        family. r transfers.

        cos(2*pi*k*r) for k = 1..H is a comb (Dirichlet) filter: every harmonic
        equals +1 at integer r and they cancel elsewhere. This is phase folding, the
        standard periodicity test in signal processing, applied per pair instead of
        per capture.

        MEASURED channel separation d' (periodic vs Poisson at the same mean
        interval, running the runner's exact EWMA recursion, so the mu/sd handed in
        are noisy estimates rather than ground truth):

            channel   clean  jit10  jit25  jit37  jit50  miss20%+jit25
            cv         6.91   6.49   5.85   5.32   4.73      4.14
            cos1       1.93   1.79   1.04   0.40   0.03      0.32
            sin1       0.32   0.28   0.22   0.23   0.27      1.00
            comb       3.85   1.66   0.24   0.10   0.00      0.12
            log_mu     0.01   0.00   0.00   0.00   0.01      0.09

        Three things that table settles:

          * cv (dispersion) is the workhorse and it survives 50% jitter, where all
            phase information is gone. Real C2 jitters heavily -- Cobalt Strike's
            sleep jitter is configurable to 0-99% -- so this is the channel that
            makes N1 viable rather than a clean-lab trick.
          * cos1 is the jitter-robust PHASE channel. The comb collapses past ~10%
            jitter because high harmonics sinc-null out at particular jitter widths
            (cos2 is 0.014 at 25% jitter while cos3 is 0.177). The comb is kept as
            an interpretable summary ONLY; the raw harmonics are what the network
            should use, and it gets them.
          * sin1 is the channel that lights up for MISSED intervals (0.32 -> 1.00),
            because dropped beacons inflate mu and shift the phase. Keeping both
            quadratures is therefore not decoration.
          * log_mu is flat at ~0.00, which PROVES the encoder is scale-free: it is
            not secretly re-detecting "short intervals", which log_dt_pair already
            gives the model.
        """
        OUT = 2 * BEACON_HARMONICS + 4

        def __init__(self):
            super().__init__()
            self.register_buffer(
                "k", torch.arange(1, BEACON_HARMONICS + 1, dtype=torch.float32))

        def forward(self, dtp, mu, sd, n):
            mus = mu.clamp(min=1e-3)
            r = (dtp.clamp(min=0.0) / mus).clamp(max=64.0).unsqueeze(-1)
            ang = 2.0 * math.pi * r * self.k
            cos, sin = torch.cos(ang), torch.sin(ang)
            # phase is meaningless until the pair has a few intervals behind it
            warm = (n > 2).float().unsqueeze(-1)
            comb = cos.mean(-1, keepdim=True)
            cv = (sd / mus).clamp(max=8.0).unsqueeze(-1)
            return torch.cat([cos * warm, sin * warm, comb * warm,
                              torch.log1p(mus).unsqueeze(-1),
                              cv * warm, torch.log1p(n).unsqueeze(-1)], -1)

    class TGDetectV2(TGDetectTGN):
        """Composable upgrades over TGDetect_TGN, one flag per hypothesis.

            zs     : 0 | 1 | 3   width of the causal z-channel (N4)
            film   : bool        regime conditioning + regime-aware head (N4)
            sync   : bool        include the 3 synchrony/fan-in ctx dims (N5)
            pair   : bool        per-(src,dst) GRU memory (N2)
            beacon : bool        scale-free phase encoding (N1)

        EVERY extra input enters through a ZERO-INITIALISED projection ADDED to the
        base embedding, and no base parameter is renamed or resized. Two consequences
        that matter more than they look:

          1. For ANY flag combination this model is EXACTLY TGDetect_TGN at
             initialisation. The superset claim is a one-line test
             (load the base state_dict with strict=False, compare logits) instead of
             the manual weight surgery TGDetect_Drift needs.

          2. Flags are therefore genuinely COMBINABLE. Turning two on cannot
             interact through a changed input layout, so a combined run measures
             the interaction of the mechanisms, not of the plumbing.
        """
        FLAGS = dict(zs=0, film=False, sync=False, pair=False, beacon=False)
        uses_ctx = True
        uses_ex = False

        def __init__(self):
            super().__init__()
            f = self.FLAGS
            self.zs = int(f.get("zs", 0))
            self.use_film = bool(f.get("film", False))
            self.use_sync = bool(f.get("sync", False))
            self.use_pair = bool(f.get("pair", False))
            self.use_beacon = bool(f.get("beacon", False))
            self.cdim = CTX_DIM if self.use_sync else CTX_BASE

            side_in = FEAT_IN * self.zs
            if self.use_beacon:
                self.bec = BeaconEnc()
                side_in += BeaconEnc.OUT
            if self.use_pair:
                side_in += PMEM
                self.pcell = nn.GRUCell(HID + HID + TDIM, PMEM)
            self.side_in = side_in
            if side_in:
                self.side = nn.Sequential(nn.Linear(side_in, HID), nn.ReLU(),
                                          nn.Linear(HID, HID))
                nn.init.zeros_(self.side[-1].weight)
                nn.init.zeros_(self.side[-1].bias)
            if self.use_film:
                self.cnorm = nn.LayerNorm(self.cdim)
                self.film = nn.Sequential(nn.Linear(self.cdim, HID), nn.ReLU(),
                                          nn.Linear(HID, 2 * HID))
                nn.init.zeros_(self.film[-1].weight)
                nn.init.zeros_(self.film[-1].bias)
                # injected AFTER head[0]'s ReLU so the decision boundary is
                # regime-dependent without widening head[0]
                self.chead = nn.Linear(self.cdim, HID)
                nn.init.zeros_(self.chead.weight); nn.init.zeros_(self.chead.bias)

        def ctx(self, c):
            return c if self.use_sync else c[:, :CTX_BASE]

        def zsel(self, xz):
            if self.zs == 1:                      # the reference scale, not scale 0
                return xz[:, FEAT_IN * Z_REF:FEAT_IN * (Z_REF + 1)]
            return xz[:, :FEAT_IN * self.zs]

        def encode(self, x, xz, c, ex=None):
            ef = self.feat(x)
            parts = []
            if self.zs:
                parts.append(self.zsel(xz))
            if self.use_beacon:
                parts.append(self.bec(ex["dtp"], ex["piat"], ex["psd"], ex["pn"]))
            if self.use_pair:
                parts.append(ex["pm"])
            if parts:
                ef = ef + self.side(torch.cat(parts, -1))
            if self.use_film:
                g, b = self.film(self.cnorm(self.ctx(c))).chunk(2, -1)
                ef = (1.0 + g) * ef + b
            return ef

        def score(self, hin, c, ex=None):
            h1 = self.head_hidden(hin)
            if self.use_film:
                h1 = h1 + self.chead(self.cnorm(self.ctx(c)))
            return self.head_out(h1).squeeze(-1)

        def pair_update(self, hs, ef, te, ex):
            if not self.use_pair:
                return None
            return self.pcell(torch.cat([hs, ef, te], -1), ex["pm"])

    class PlainMLP(nn.Module):
        """No graph, no memory, no time. Isolates neural-vs-tree from graph value."""
        uses_ctx = False
        uses_ex = False
        uses_mem = False
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(FEAT_IN, HID), nn.ReLU(), nn.Linear(HID, HID))
            self.head = nn.Sequential(nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))
        def forward(self, x, xz, c, ms, md, hsrc, hdst, dt, ex=None):
            return self.head(self.feat(x)).squeeze(-1), None

    class TGDetectRobust(TGDetectTGN):
        """TG-Detect + two changes aimed at ONE measured failure: cross-family transfer.

        THE OBSERVATION. On leave-one-family-out (hold out Neris) TGDetect_TGN scored
        MACRO AUC-PR 0.5019 against Pomsathit_GCN's 0.6162 -- a 3.3-SE loss -- and the
        whole gap came from ONE capture, s09 (Neris, 16.6% prevalence): 0.2554 vs
        0.7799. s01 and s02 were a tie. Several per-capture ROCs came in BELOW 0.5
        (TGN on s09: 0.4056; Drift on s01: 0.4850). Below 0.5 is not underfitting --
        underfitting gives exactly 0.5. Below 0.5 means the model learned a real
        relationship whose SIGN is inverted in the new family.

        WHAT INVERTS. The GRU recurrence is m_t = (1-z)*m_{t-1} + z*tanh(.) with
        m_0 = 0. A fresh host sits at the origin; a much-visited host has been dragged
        toward an attractor. So ||m|| increases monotonically with the host's event
        count: the memory NORM is an activity-volume proxy. That is the same
        non-transferable quantity as log_src_prior_count -- except it is smuggled in
        through MEM=128 continuous dimensions, where no feature mask can reach it.
        Train prevalence 0.749% vs s09's 16.6% means every host in s09 is driven far
        past any magnitude regime seen in training, so a decision rule keyed on
        magnitude does not merely weaken, it flips. Pomsathit_GCN has no memory, hence
        nothing to invert -- which is exactly why the simpler model won.

        FIX 1, memnorm: read the memory's DIRECTION, not its magnitude.
            Write m = ||m|| * m_hat. The direction says WHAT KIND of behaviour the host
            has shown; the magnitude says HOW MUCH. Only the second is
            environment-specific, so drop it from the read path.
            LayerNorm does this exactly. For any a > 0 and any b,
                LN(a*m + b*1) = ((a*m + b*1) - (a*mu + b)*1) / (a*sigma) * g + beta
                              = (m - mu*1)/sigma * g + beta = LN(m).
            This is an algebraic identity, not an empirical hope: ANY per-host affine
            rescaling of memory intensity -- precisely what changes when the traffic
            level or the malicious fraction changes -- leaves the memory CONTENT
            untouched. The magnitude is not deleted from the system; TGDetectDrift
            already carries explicit maturity/rate terms in its context vector c, which
            is the right place for "how much", because c is regime-normalised.

            Applied as a GATED RESIDUAL with a zero-initialised gate:
                m_read = m + alpha * (LN(m) - m),   alpha init 0
            so at initialisation this model is BIT-IDENTICAL to TGDetect_TGN and the
            superset proof still holds. alpha is per-dimension, so the optimiser can
            keep magnitude in the dimensions where it is genuinely informative and
            discard it elsewhere -- it is not forced to choose globally.

        FIX 2, memdrop: give the architecture a FLOOR.
            With probability p, zero the memory read for the whole batch (rescaling by
            1/(1-p) so the expectation is unchanged). Two consequences:

            (a) Regularisation, and a targeted one. Dropout on a pathway is equivalent
                in the linearised regime to an L2 penalty on that pathway's weights
                with coefficient proportional to p/(1-p) times the pathway's input
                second moment. The pathway here is EXACTLY the domain-fragile one, so
                the shrinkage lands on the weights whose inputs go out of distribution
                at test time, and on no others.

            (b) The structural point, which matters more. It forces the MEMORYLESS
                sub-model to be a competent classifier in its own right. TGDetect_TGN
                is a superset of Pomsathit_GCN in CAPACITY but had no guarantee of
                being one in PERFORMANCE: nothing in the ERM objective required the
                no-memory pathway to work, so under family shift the model could and
                did fall BELOW a strictly weaker architecture. With memdrop, the worst
                case is "memory is useless on this family", and the model degrades TO
                the time-blind baseline rather than through it.

            Dropped per batch, not per row, so every event in a batch sees a consistent
            world -- per-row dropout would make the graph aggregation mix
                memory-aware and memory-blind neighbours, which is a different
                intervention and not the one being tested.
        """
        FLAGS = {"memnorm": False, "memdrop": 0.0, "memhard": False}

        def __init__(self):
            # super() FIRST so every inherited parameter is drawn from the same RNG
            # stream in the same order as TGDetect_TGN. The new parameters are drawn
            # afterwards, so with alpha=0 and memdrop=0 this model is not merely
            # equivalent to the anchor, it is numerically identical.
            super().__init__()
            self.mem_ln = nn.LayerNorm(MEM)
            self.mem_alpha = nn.Parameter(torch.zeros(MEM))

        def mem_read(self, m):
            if self.FLAGS.get("memhard"):
                # alpha == 1, IMPOSED, not learned. This is the correction the
                # first real-data suite forced (see RESEARCH_AND_FIXES PART XIV).
                #
                # memnorm's gate starts at alpha=0 and is trained by ERM. But the
                # thing the gate is supposed to suppress -- the activity-volume
                # component of ||m|| -- LOWERS the training loss, because within the
                # training families the busiest hosts really are the likeliest to be
                # scored. So
                #     d L_train / d alpha_j = <dL/dm_j , LN(m)_j - m_j>
                # points toward keeping alpha near 0 for exactly the coordinates
                # where the shortcut pays. A gate that is optimised on the
                # distribution where the shortcut WORKS cannot be expected to close
                # it. Invariance has to be a constraint on the hypothesis class, not
                # a parameter inside it.
                #
                # With alpha == 1 the classifier's view of the memory is
                # LN(m), and LN(a*m + b*1) = LN(m) exactly for every a>0, b. So the
                # per-host activity SCALE is algebraically unreadable from the
                # classifier's memory input; only the memory's PATTERN survives. The
                # magnitude is still in the system (the GRU recurrence receives raw
                # ms, and the regime vector c carries volume) -- it is relocated,
                # never deleted, which is the standing rule.
                m = self.mem_ln(m)
            elif self.FLAGS["memnorm"]:
                m = m + self.mem_alpha * (self.mem_ln(m) - m)
            p = self.FLAGS["memdrop"]
            if p > 0.0 and self.training:
                # One Bernoulli for the whole batch (see docstring), and the 1/(1-p)
                # rescale so E[m_read] is unchanged and no test-time correction is
                # needed.
                if torch.rand((), device=m.device) < p:
                    return torch.zeros_like(m)
                return m / (1.0 - p)
            return m

    class TGDetectDriftRobust(TGDetectDrift):
        """TGDetectDrift with the same two memory fixes. Same docstring applies.

        Kept as a separate class rather than a mixin because Drift overrides
        __init__ (it rebuilds feat/head/film), and parameter draw ORDER is what makes
        the superset proof numerical instead of rhetorical. Subclassing Drift and
        appending the two new parameters preserves that order exactly.
        """
        FLAGS = {"memnorm": False, "memdrop": 0.0, "memhard": False}

        def __init__(self):
            # Written out rather than aliased to TGDetectRobust.__init__: a zero-arg
            # super() compiles to a __class__ cell bound to the class the method was
            # DEFINED in, so borrowing the function raises
            #   TypeError: super(type, obj): obj must be an instance or subtype of type
            super().__init__()
            self.mem_ln = nn.LayerNorm(MEM)
            self.mem_alpha = nn.Parameter(torch.zeros(MEM))

        mem_read = TGDetectRobust.mem_read

    def rb(name, base, **flags):
        f = dict(TGDetectRobust.FLAGS, **flags)
        return type(name, (base,), {
            "FLAGS": f,
            "__doc__": base.__doc__ + "\n\nVariant: " + ", ".join(
                f"{k}={v}" for k, v in f.items() if v) or "anchor control",
        })

    def v2(name, **flags):
        f = dict(TGDetectV2.FLAGS, **flags)
        return type(name, (TGDetectV2,), {
            "FLAGS": f,
            "uses_ex": bool(f["pair"] or f["beacon"]),
            "__doc__": "TGDetectV2 ablation: " + ", ".join(
                f"{k}={v}" for k, v in f.items() if v),
        })

    M = {"PlainMLP": PlainMLP, "Pomsathit_GCN": PomsathitGCN,
         "TGDetect_TGN": TGDetectTGN, "TGDetect_Drift": TGDetectDrift}

    # ── memory-robustness grid (A7/A8) ───────────────────────────────────────
    # R_Off is the control: TGDetectRobust with both flags off, which must score
    # IDENTICALLY to TGDetect_TGN on the same seed. Same role V2_Off plays for the
    # V2 grid -- if it differs, the two extra parameters are leaking and no other
    # row in the grid means anything.
    M["R_Off"] = rb("R_Off", TGDetectRobust)
    M["R_MemNorm"] = rb("R_MemNorm", TGDetectRobust, memnorm=True)
    M["R_MemDrop"] = rb("R_MemDrop", TGDetectRobust, memdrop=0.15)
    M["R_Both"] = rb("R_Both", TGDetectRobust, memnorm=True, memdrop=0.15)
    # Same two fixes on top of the drift machinery. TGDetect_Drift already beat
    # TGDetect_TGN by +0.088 MACRO AUC-PR on the Neris holdout (1.4 SE, directional
    # not conclusive), so this is the combination most likely to clear Pomsathit.
    M["RD_Off"] = rb("RD_Off", TGDetectDriftRobust)
    M["RD_Both"] = rb("RD_Both", TGDetectDriftRobust, memnorm=True, memdrop=0.15)

    # ── A13: memory normalisation IMPOSED rather than learned ────────────────
    # The first real-data suite killed the learnable version: R_MemNorm scored
    # 0.4921 MACRO against the anchor's 0.5666 on the notime/Neris exam. The
    # explanation is in TGDetectRobust.mem_read -- a gate trained by ERM on the
    # distribution where the shortcut WORKS has no gradient reason to close it.
    # memhard removes the gate and hard-wires alpha = 1, which turns the
    # scale-invariance from something the optimiser may choose into a property of
    # the hypothesis class. This is the ONE variant whose invariance is a theorem
    # rather than a hope.
    M["R_MemHard"] = rb("R_MemHard", TGDetectRobust, memhard=True)
    M["R_HardDrop"] = rb("R_HardDrop", TGDetectRobust, memhard=True, memdrop=0.15)
    M["RD_MemHard"] = rb("RD_MemHard", TGDetectDriftRobust, memhard=True)

    # ── ablation grid ────────────────────────────────────────────────────────
    # V2_Off exists ONLY as the superset control: it must score identically to
    # TGDetect_TGN given the same seed. If it does not, the harness is lying.
    M["V2_Off"] = v2("V2_Off")
    # TGDetect_Drift expressed in the V2 flag system: one z-scale (Z_REF) + FiLM +
    # regime-aware head. Same three ideas, but entering through zero-initialised side
    # projections instead of a WIDENED feat/head.
    #
    # Why this model has to exist. TGDetect_Drift claims in its docstring to be
    # "EXACTLY TGDetectTGN" at initialisation, but it widens feat (FEAT_IN -> FEAT_IN*2)
    # and head (+CTX_BASE), so it cannot even load the anchor's state_dict:
    #     size mismatch for feat.0.weight: checkpoint [128, 22] vs model [128, 44]
    # The claim is true only for a hand-built weight surgery, never for the anchor
    # checkpoint. Every V2 variant loads the anchor and reproduces its logits to
    # 0.000e+00, so here the superset property is a TEST rather than a promise.
    #
    # It is also the missing cell in the grid: V2_N4_Z3 is zs=3, so the existing
    # registry has never contained Drift's actual hypothesis (zs=1) in comparable form.
    M["V2_Drift"] = v2("V2_Drift", zs=1, film=True)
    M["V2_N1_Beacon"] = v2("V2_N1_Beacon", beacon=True)
    M["V2_N2_Pair"] = v2("V2_N2_Pair", pair=True)
    M["V2_N4_Z3"] = v2("V2_N4_Z3", zs=3, film=True)
    M["V2_N5_Sync"] = v2("V2_N5_Sync", film=True, sync=True)
    # combinations: the pair-state pair (N1+N2) share the same streaming state, and
    # the context pair (N4+N5) share the same FiLM path, so those are the two
    # cheapest interactions to look for before the full stack.
    M["V2_N1N2"] = v2("V2_N1N2", beacon=True, pair=True)
    M["V2_N4N5"] = v2("V2_N4N5", zs=3, film=True, sync=True)
    M["V2_All"] = v2("V2_All", zs=3, film=True, sync=True, pair=True, beacon=True)
    return M
