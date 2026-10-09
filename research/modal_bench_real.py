#!/usr/bin/env python3
"""
TG-Detect vs Pomsathit (2025) — honest head-to-head on REAL data (CTU-13).

Protocols
─────────
P-temporal  : per scenario, strict chronological 60/15/25 train/val/test.
P-family    : leave-one-malware-family-out (train families != test family).

Models (all receive IDENTICAL 22-dim causal features and identical budgets)
──────────────────────────────────────────────────────────────────────────
  LogReg          tabular control — leakage gate
  HistGB          tabular control — strong non-graph baseline
  Pomsathit_GCN   3-layer GCN, hidden 128, attention aggregation, softmax
                  classification, 10% reservoir rehearsal buffer, operating on
                  time-window snapshot graphs (as in the paper).
  TGDetect_TGN    continuous-time: vectorised memory bank with risk-modulated
                  decay, multi-resolution time encoder, causal (dt>0) masked
                  temporal attention.

Eval hygiene
────────────
  * threshold chosen on VALIDATION only, never on test
  * AUC-PR is primary (prevalence is <1%); ROC-AUC, MCC, F1, P, R, FPR/day too
  * multiple seeds -> mean +- std; bootstrap CI on the TGN-minus-GCN delta
  * McNemar paired test on the test-set decisions
  * inference latency measured as ms/event with no_grad, separate from training

Run:
  modal run modal_bench_real.py::bench
  modal run modal_bench_real.py::bench --protocol family --seeds 3
"""

import modal

app = modal.App("tg-detect-bench-real")
volume = modal.Volume.from_name("tg-detect-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch==2.4.0", "numpy==2.1.3", "scikit-learn==1.5.2", "scipy==1.14.1"
)

DATA_DIR = "/data/real/ctu13"

FAMILY = {
    "s01": "Neris", "s02": "Neris", "s03": "Rbot", "s04": "Rbot", "s05": "Virut",
    "s06": "Menti", "s07": "Sogou", "s08": "Murlo", "s09": "Neris", "s10": "Rbot",
    "s11": "Rbot", "s12": "NSIS.ay", "s13": "Virut",
}


@app.function(image=image, gpu="L4", volumes={"/data": volume}, timeout=21600, memory=65536, cpu=8.0)
def bench(protocol: str = "temporal", seeds: int = 3, epochs: int = 3,
          max_events_per_scenario: int = 0, scenarios: str = "all"):
    import glob, json, math, os, random, time
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import stats as sps
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                                 precision_score, recall_score, matthews_corrcoef,
                                 precision_recall_curve, confusion_matrix)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FEAT_IN, HID, MEM, TDIM, HEADS = 22, 128, 128, 32, 4
    BATCH, LR = 2048, 1e-3
    GCN_LAYERS, TGN_LAYERS = 3, 3     # equal depth: TGN adds time+memory, not capacity
    REHEARSAL_FRAC = 0.10
    POS_W = 20.0               # identical positive class weight for both graph models
    NODE_BUCKETS = 1 << 17     # hashed node embedding: identical for both models

    # ────────────────────────── load ──────────────────────────
    keys = sorted(FAMILY) if scenarios == "all" else scenarios.split(",")
    streams, offset = [], 0
    for k in keys:
        p = f"{DATA_DIR}/{k}.npz"
        if not os.path.exists(p):
            print(f"  ! missing {p}, skipping"); continue
        z = np.load(p)
        X, ts, src, dst, y = z["X"], z["ts"], z["src"], z["dst"], z["y"]
        if max_events_per_scenario and len(y) > max_events_per_scenario:
            X, ts, src, dst, y = (a[:max_events_per_scenario] for a in (X, ts, src, dst, y))
        n_nodes = int(max(src.max(), dst.max())) + 1
        # time is stored as absolute unix seconds (~1.3e9); float32 has ~7 significant
        # digits there, i.e. ~64 s granularity, which would destroy every inter-arrival
        # signal. Rebase per scenario in float64 FIRST, then narrow to float32.
        ts = (ts.astype(np.float64) - float(ts[0])).astype(np.float32)
        # local ids drive the per-scenario memory bank; the embedding index is a
        # scenario-salted hash so no model can carry "this IP is the bot" across captures.
        salt = np.uint64(offset * 0x9E3779B1 + 1)
        def h(a):
            return ((a.astype(np.uint64) + salt) * np.uint64(0x9E3779B97F4A7C15)
                    % np.uint64(NODE_BUCKETS)).astype(np.int64)
        streams.append({"key": k, "family": FAMILY[k], "X": np.nan_to_num(X),
                        "ts": ts, "src": src.astype(np.int64), "dst": dst.astype(np.int64),
                        "hsrc": h(src), "hdst": h(dst),
                        "y": y.astype(np.float32), "n_nodes": n_nodes})
        offset += n_nodes
    TOTAL_NODES = offset
    assert streams, "no scenario npz found — run modal_data_real.py::prepare first"

    tot_ev = sum(len(s["y"]) for s in streams)
    tot_pos = sum(int(s["y"].sum()) for s in streams)
    print("=" * 100)
    print(f"  REAL-DATA BENCHMARK — CTU-13 | protocol={protocol} | seeds={seeds} | epochs={epochs}")
    print(f"  {len(streams)} scenarios | {tot_ev:,} events | {tot_pos:,} positive "
          f"({tot_pos/tot_ev*100:.4f}%) | {TOTAL_NODES:,} nodes | device={dev}")
    print("=" * 100)

    # feature standardisation fitted on TRAIN portions only (per protocol, below)
    def build_folds():
        """Segments are (stream, warm_start, eval_start, eval_end). Events in
        [warm_start, eval_start) are replayed with no_grad purely to warm the
        memory bank / last-seen clocks, exactly as a deployed detector would
        have processed them. They are never scored and never trained on."""
        folds = []
        if protocol == "temporal":
            tr, va, te = [], [], []
            for s in streams:
                n = len(s["y"]); a, b = int(n * 0.60), int(n * 0.75)
                tr.append((s, 0, 0, a)); va.append((s, 0, a, b)); te.append((s, 0, b, n))
            folds.append({"name": "temporal", "train": tr, "val": va, "test": te})
        else:
            fams = sorted({s["family"] for s in streams})
            for fam in fams:
                held = [s for s in streams if s["family"] == fam]
                rest = [s for s in streams if s["family"] != fam]
                if not rest or not held:
                    continue
                tr, va = [], []
                for s in rest:
                    n = len(s["y"]); c = int(n * 0.85)
                    tr.append((s, 0, 0, c)); va.append((s, 0, c, n))
                te = [(s, 0, 0, len(s["y"])) for s in held]
                folds.append({"name": f"holdout_{fam}", "train": tr, "val": va, "test": te})
        return folds

    def cat(segs, field):
        return np.concatenate([s[field][i:j] for s, _, i, j in segs]) if segs else np.empty(0)

    # ───────────────────────── models ─────────────────────────
    class TimeEnc(nn.Module):
        """multi-resolution log-time encoder"""
        def __init__(self, d):
            super().__init__()
            self.w = nn.Parameter(torch.logspace(-4, 1, d).float())
            self.b = nn.Parameter(torch.zeros(d))
        def forward(self, dt):
            x = torch.log1p(torch.clamp(dt, min=0.0)).unsqueeze(-1)
            return torch.cos(x * self.w + self.b)

    def seg_softmax(scores, index, n_seg):
        """numerically stable softmax over groups given by index"""
        mx = torch.full((n_seg,) + scores.shape[1:], -1e30, device=scores.device)
        mx = mx.index_reduce(0, index, scores, "amax", include_self=True)
        e = torch.exp(scores - mx[index])
        den = torch.zeros((n_seg,) + scores.shape[1:], device=scores.device)
        den = den.index_add(0, index, e)
        return e / (den[index] + 1e-12)

    class PlainMLP(nn.Module):
        """Ablation control: identical trunk/head capacity, NO graph, NO node identity,
        NO memory, NO time. Any gap between this and the graph models is what the
        structure actually buys; any gap between this and HistGB is neural-vs-tree
        on tabular features and has nothing to do with graphs."""
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(FEAT_IN, HID), nn.ReLU(), nn.Linear(HID, HID))
            self.head = nn.Sequential(nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 1))

        def forward(self, x, ms, md, hsrc, hdst, dt):
            return self.head(self.feat(x)).squeeze(-1), None

    class PomsathitGCN(nn.Module):
        """3-layer GCN + attention aggregation + softmax head, on window snapshots.

        No node-identity embedding: the scenario salt makes hashed IDs untransferable,
        so such a table can only memorise noise. Initial node state is the mean of
        incident edge features — the standard featureless-graph initialisation.
        """
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(FEAT_IN, HID), nn.ReLU(), nn.Linear(HID, HID))
            self.W = nn.ModuleList([nn.Linear(HID * 2, HID) for _ in range(GCN_LAYERS)])
            self.att = nn.ModuleList([nn.Linear(HID * 2, 1) for _ in range(GCN_LAYERS)])
            self.head = nn.Sequential(nn.Linear(HID * 3, HID), nn.ReLU(), nn.Linear(HID, 2))

        def forward(self, x, ms, md, hsrc, hdst, dt):
            uniq, inv = torch.unique(torch.cat([hsrc, hdst]), return_inverse=True)
            E = hsrc.numel()
            ls, ld = inv[:E], inv[E:]
            N = uniq.numel()
            ef = self.feat(x)
            deg = torch.zeros(N, 1, device=ef.device)
            deg = deg.index_add(0, ls, torch.ones(E, 1, device=ef.device))
            deg = deg.index_add(0, ld, torch.ones(E, 1, device=ef.device))
            h = torch.zeros(N, HID, device=ef.device)
            h = h.index_add(0, ls, ef).index_add(0, ld, ef) / deg.clamp(min=1.0)
            for W, A in zip(self.W, self.att):
                hs_i, hd_i = h[ls], h[ld]
                a_in = seg_softmax(A(torch.cat([hd_i, hs_i], -1)), ld, N)
                agg = torch.zeros_like(h).index_add(0, ld, a_in * torch.relu(W(torch.cat([hs_i, ef], -1))))
                a_out = seg_softmax(A(torch.cat([hs_i, hd_i], -1)), ls, N)
                agg = agg.index_add(0, ls, a_out * torch.relu(W(torch.cat([hd_i, ef], -1))))
                h = F.relu(h + agg)                                       # 3 hops, residual
            logits2 = self.head(torch.cat([h[ls], h[ld], ef], -1))
            return logits2[:, 1] - logits2[:, 0], None


    class TGDetectTGN(nn.Module):
        """TG-Detect: the Pomsathit window-graph attention PLUS two additions —
        (1) a multi-resolution time encoding on every message, and
        (2) a persistent per-node memory bank with risk-modulated decay.

        Deliberately a strict superset of the baseline: same depth, same
        aggregation scheme, same node initialisation, no node-identity embedding.
        Any difference in score is attributable to time + memory alone.
        """
        def __init__(self):
            super().__init__()
            self.feat = nn.Sequential(nn.Linear(FEAT_IN, HID), nn.ReLU(), nn.Linear(HID, HID))
            self.tenc = TimeEnc(TDIM)
            self.mproj = nn.Linear(MEM, HID)                  # memory -> node state
            self.W = nn.ModuleList([nn.Linear(HID * 2 + TDIM, HID) for _ in range(TGN_LAYERS)])
            self.att = nn.ModuleList([nn.Linear(HID * 2 + TDIM, 1) for _ in range(TGN_LAYERS)])
            self.ln = nn.ModuleList([nn.LayerNorm(HID) for _ in range(TGN_LAYERS)])
            self.cell = nn.GRUCell(HID + HID + TDIM, MEM)
            self.head = nn.Sequential(nn.Linear(HID * 3 + MEM, HID), nn.ReLU(), nn.Linear(HID, 1))

        def forward(self, x, ms, md, hsrc, hdst, dt):
            uniq, inv = torch.unique(torch.cat([hsrc, hdst]), return_inverse=True)
            E = hsrc.numel()
            ls, ld = inv[:E], inv[E:]
            N = uniq.numel()
            ef = self.feat(x)
            te = self.tenc(dt)
            causal = (dt > 0).float()                          # 0 => no prior contact
            one = torch.ones(E, 1, device=ef.device)
            deg = torch.zeros(N, 1, device=ef.device).index_add(0, ls, one).index_add(0, ld, one)
            # identical featureless init to the GCN ...
            h = torch.zeros(N, HID, device=ef.device)
            h = h.index_add(0, ls, ef).index_add(0, ld, ef) / deg.clamp(min=1.0)
            # ... plus the causal memory of what each node did before (the addition)
            mp = torch.zeros(N, HID, device=ef.device)
            mp = mp.index_add(0, ls, self.mproj(ms) * causal.unsqueeze(-1))
            mp = mp.index_add(0, ld, self.mproj(md))
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
            logits = self.head(torch.cat([hs, hd, ef, ms], -1)).squeeze(-1)
            new_mem = self.cell(torch.cat([hs, ef, te], -1), ms)
            return logits, new_mem

    # ───────────────────── streaming runner ─────────────────────
    class Reservoir:
        def __init__(self, cap):
            self.cap, self.buf, self.n = cap, [], 0
        def add(self, item):
            self.n += 1
            if len(self.buf) < self.cap:
                self.buf.append(item)
            else:
                j = random.randint(0, self.n - 1)
                if j < self.cap:
                    self.buf[j] = item
        def sample(self):
            return random.choice(self.buf) if self.buf else None

    def run_epoch(model, segs, mu, sd, train, opt=None, buf=None, is_tgn=True,
                  collect=False, timeit=False):
        """Process each scenario as an independent chronological stream.

        A segment is (stream, warm_start, eval_start, eval_end). Events in
        [warm_start, eval_start) are pushed through the model under no_grad so
        the memory bank and last-seen clocks reach the state a live detector
        would be in; they are never scored and never contribute to the loss.
        """
        scores, labels = [], []
        n_seen, fwd_time = 0, 0.0
        for s, w0, i0, i1 in segs:
            n_nodes = s["n_nodes"]
            mem = torch.zeros(n_nodes, MEM, device=dev)
            last = torch.full((n_nodes,), -1.0, device=dev)   # -1 => never seen
            X = torch.from_numpy(((s["X"][w0:i1] - mu) / sd).astype(np.float32)).to(dev)
            ts = torch.from_numpy(s["ts"][w0:i1]).float().to(dev)
            sr = torch.from_numpy(s["src"][w0:i1]).to(dev)
            ds = torch.from_numpy(s["dst"][w0:i1]).to(dev)
            hsr = torch.from_numpy(s["hsrc"][w0:i1]).to(dev)
            hds = torch.from_numpy(s["hdst"][w0:i1]).to(dev)
            yy = torch.from_numpy(s["y"][w0:i1]).to(dev)
            n = X.shape[0]
            warm = i0 - w0                       # rows before this are warm-up only
            bounds = []                          # batch edges, never straddling `warm`
            for a in range(0, warm, BATCH):
                bounds.append((a, min(a + BATCH, warm), True))
            for a in range(warm, n, BATCH):
                bounds.append((a, min(a + BATCH, n), False))
            for b0, b1, is_warm in bounds:
                x, sb, db = X[b0:b1], sr[b0:b1], ds[b0:b1]
                hsb, hdb, tb, yb = hsr[b0:b1], hds[b0:b1], ts[b0:b1], yy[b0:b1]
                ls = last[sb]
                # dt == 0 marks "no prior contact"; the TGN's causal gate keys off it,
                # so a never-before-seen source must not get a huge spurious dt.
                dt = torch.where(ls >= 0, torch.clamp(tb - ls, min=0.0), torch.zeros_like(tb))
                ms, md = mem[sb], mem[db]
                t_fwd = time.perf_counter()
                if train and not is_warm:
                    logits, nm = model(x, ms, md, hsb, hdb, dt)
                    w = torch.where(yb > 0, POS_W, 1.0)   # same class weight for both models
                    loss = F.binary_cross_entropy_with_logits(logits, yb, weight=w)
                    if buf is not None and buf.buf:       # rehearsal replay (both models)
                        r = buf.sample()
                        rl, _ = model(r[0], r[1], r[2], r[3], r[4], r[5])
                        rw = torch.where(r[6] > 0, POS_W, 1.0)
                        loss = loss + F.binary_cross_entropy_with_logits(rl, r[6], weight=rw)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    if buf is not None and (yb.sum() > 0 or random.random() < 0.05):
                        buf.add((x, ms.detach(), md.detach(), hsb, hdb, dt, yb))
                else:
                    with torch.no_grad():
                        logits, nm = model(x, ms, md, hsb, hdb, dt)
                if timeit and not is_warm:
                    if dev.type == "cuda":
                        torch.cuda.synchronize()
                    fwd_time += time.perf_counter() - t_fwd
                    n_seen += (b1 - b0)
                with torch.no_grad():
                    if is_tgn:
                        risk = torch.sigmoid(logits)
                        # risk-modulated decay: memory of risky endpoints persists longer
                        ld = last[db]
                        dtd = torch.where(ld >= 0, torch.clamp(tb - ld, min=0.0),
                                          torch.zeros_like(tb))
                        dec = torch.exp(-dtd / (3600.0 * (1.0 + 9.0 * risk)))
                        mem = mem.index_copy(0, db, md * dec.unsqueeze(-1))
                        mem = mem.index_copy(0, sb, nm.detach())
                    last = last.index_copy(0, sb, tb).index_copy(0, db, tb)
                if collect and not is_warm:
                    scores.append(torch.sigmoid(logits).detach().float().cpu().numpy())
                    labels.append(yb.detach().cpu().numpy())
        if collect:
            return np.concatenate(scores), np.concatenate(labels), fwd_time, n_seen
        return None, None, fwd_time, n_seen

    def metrics_at(y, p, thr, span_days):
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            "AUC_PR": float(average_precision_score(y, p)),
            "ROC_AUC": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
            "F1": float(f1_score(y, pred, zero_division=0)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "MCC": float(matthews_corrcoef(y, pred)) if len(np.unique(pred)) > 1 else 0.0,
            "FP_per_day": float(fp / max(span_days, 1e-6)),
            "threshold": float(thr), "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        }

    def pick_thr(yv, pv):
        pr, rc, th = precision_recall_curve(yv, pv)
        f = 2 * pr * rc / (pr + rc + 1e-12)
        i = int(np.argmax(f[:-1])) if len(th) else 0
        return float(th[i]) if len(th) else 0.5

    def span_days_of(segs):
        """Total wall-clock days covered by the evaluated slices. Timestamps are
        rebased per scenario, so a global max-min over a concatenation would be
        meaningless — sum the per-scenario spans instead."""
        tot = 0.0
        for s, _, i, j in segs:
            if j > i:
                tot += float(s["ts"][j - 1] - s["ts"][i]) / 86400.0
        return max(tot, 1e-6)

    # ───────────────────────── main loop ─────────────────────────
    all_results = {}
    for fold in build_folds():
        fname = fold["name"]
        Xtr_np = cat(fold["train"], "X")
        mu = Xtr_np.mean(0); sd = Xtr_np.std(0); sd[sd < 1e-6] = 1.0
        ytr = cat(fold["train"], "y"); yva = cat(fold["val"], "y"); yte = cat(fold["test"], "y")
        span_days = span_days_of(fold["test"])
        print(f"\n{'#'*100}\n# FOLD {fname} | train {len(ytr):,} ({int(ytr.sum())} pos) | "
              f"val {len(yva):,} ({int(yva.sum())} pos) | test {len(yte):,} ({int(yte.sum())} pos) | "
              f"test span {span_days:.2f} d\n{'#'*100}", flush=True)
        if yte.sum() == 0 or ytr.sum() == 0:
            print("  ! degenerate fold, skipped"); continue

        fold_res = {}

        # ---- tabular controls (leakage gate + strong non-graph baseline)
        Xtr = (Xtr_np - mu) / sd
        Xva = (cat(fold["val"], "X") - mu) / sd
        Xte = (cat(fold["test"], "X") - mu) / sd
        for tname, clf in (("LogReg", LogisticRegression(max_iter=300, class_weight="balanced", n_jobs=-1)),
                           ("HistGB", HistGradientBoostingClassifier(max_iter=200, class_weight="balanced",
                                                                     random_state=0))):
            t0 = time.time()
            clf.fit(Xtr, ytr)
            pv = clf.predict_proba(Xva)[:, 1]; pt = clf.predict_proba(Xte)[:, 1]
            thr = pick_thr(yva, pv)
            m = metrics_at(yte, pt, thr, span_days)
            m["fit_s"] = round(time.time() - t0, 1)
            fold_res[tname] = {"seeds": [m], "pred": (pt >= thr).astype(int)}
            print(f"  {tname:<14} AUC-PR {m['AUC_PR']:.4f} | F1 {m['F1']:.4f} | "
                  f"P {m['Precision']:.4f} | R {m['Recall']:.4f} | MCC {m['MCC']:.4f}", flush=True)

        # ---- graph models, multiple seeds
        for mname, ctor, is_tgn in (("MLP_nograph", PlainMLP, False),
                                    ("Pomsathit_GCN", PomsathitGCN, False),
                                    ("TGDetect_TGN", TGDetectTGN, True)):
            per_seed, last_pred = [], None
            for sd_i in range(seeds):
                torch.manual_seed(sd_i); np.random.seed(sd_i); random.seed(sd_i)
                os.environ["PYTHONHASHSEED"] = str(sd_i)
                model = ctor().to(dev)
                opt = torch.optim.Adam(model.parameters(), lr=LR)
                cap = max(8, int(len(ytr) * REHEARSAL_FRAC / BATCH))
                buf = Reservoir(cap)
                t0 = time.time()
                # early stopping / epoch selection on VALIDATION AUC-PR — applied
                # identically to both graph models so neither is under- or over-trained.
                best_ap, best_state, best_ep = -1.0, None, -1
                for ep in range(epochs):
                    model.train()
                    run_epoch(model, fold["train"], mu, sd, True, opt, buf, is_tgn)
                    model.eval()
                    pv_e, yv_e, _, _ = run_epoch(model, fold["val"], mu, sd, False,
                                                 is_tgn=is_tgn, collect=True)
                    ap_e = float(average_precision_score(yv_e, pv_e)) if yv_e.sum() > 0 else 0.0
                    if ap_e > best_ap:
                        best_ap, best_ep = ap_e, ep
                        best_state = {kk: vv.detach().clone() for kk, vv in model.state_dict().items()}
                if best_state is not None:
                    model.load_state_dict(best_state)
                train_s = time.time() - t0
                model.eval()
                pv, yv2, _, _ = run_epoch(model, fold["val"], mu, sd, False, is_tgn=is_tgn, collect=True)
                pt, yt2, fwd_s, n_ev = run_epoch(model, fold["test"], mu, sd, False,
                                                 is_tgn=is_tgn, collect=True, timeit=True)
                thr = pick_thr(yv2, pv)
                m = metrics_at(yt2, pt, thr, span_days)
                m["train_s"] = round(train_s, 1)
                m["best_epoch"] = best_ep
                m["val_AUC_PR"] = round(best_ap, 4)
                m["infer_ms_per_event"] = round(fwd_s / max(n_ev, 1) * 1000, 6)
                m["params"] = sum(p.numel() for p in model.parameters())
                per_seed.append(m)
                last_pred = (pt >= thr).astype(int)
                print(f"  {mname:<14} seed{sd_i} AUC-PR {m['AUC_PR']:.4f} | F1 {m['F1']:.4f} | "
                      f"P {m['Precision']:.4f} | R {m['Recall']:.4f} | MCC {m['MCC']:.4f} | "
                      f"ep{best_ep} (val AP {best_ap:.4f}) | "
                      f"{m['infer_ms_per_event']*1000:.2f} us/ev | train {train_s:.0f}s", flush=True)
                del model, opt, buf
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
            fold_res[mname] = {"seeds": per_seed, "pred": last_pred}

        # ---- aggregate + significance
        def agg(k, mkey):
            v = [s[mkey] for s in fold_res[k]["seeds"]]
            return float(np.mean(v)), float(np.std(v))

        print(f"\n  {'model':<15} {'AUC-PR':>16} {'F1':>16} {'ROC-AUC':>16} {'MCC':>16}")
        print("  " + "-" * 82)
        for k in fold_res:
            row = "  " + f"{k:<15}"
            for mk in ("AUC_PR", "F1", "ROC_AUC", "MCC"):
                m_, s_ = agg(k, mk)
                row += f"{m_:>10.4f}+-{s_:<5.4f}"
            print(row)

        if "TGDetect_TGN" in fold_res and "Pomsathit_GCN" in fold_res:
            a = np.array([s["AUC_PR"] for s in fold_res["TGDetect_TGN"]["seeds"]])
            b = np.array([s["AUC_PR"] for s in fold_res["Pomsathit_GCN"]["seeds"]])
            delta = float(a.mean() - b.mean())
            boot = [float(np.mean(np.random.choice(a, len(a))) - np.mean(np.random.choice(b, len(b))))
                    for _ in range(5000)]
            lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
            pa, pb = fold_res["TGDetect_TGN"]["pred"], fold_res["Pomsathit_GCN"]["pred"]
            ca = (pa == yte); cb = (pb == yte)
            n01 = int((~ca & cb).sum()); n10 = int((ca & ~cb).sum())
            mc = sps.binomtest(n10, n10 + n01, 0.5).pvalue if (n10 + n01) > 0 else 1.0
            hgb = fold_res["HistGB"]["seeds"][0]["AUC_PR"]
            print(f"\n  AUC-PR delta (TGN - Pomsathit) = {delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
                  f"  {'SIGNIFICANT' if lo > 0 else 'NOT significant'}")
            print(f"  McNemar exact p = {mc:.3e}  (TGN-only-correct {n10:,} vs GCN-only-correct {n01:,})")
            print(f"  vs non-graph HistGB AUC-PR {hgb:.4f}: TGN "
                  f"{'beats' if a.mean() > hgb else 'LOSES TO'} the tabular baseline "
                  f"({a.mean()-hgb:+.4f})")
            fold_res["_delta"] = {"auc_pr_delta": delta, "ci95": [lo, hi],
                                  "mcnemar_p": float(mc), "n_tgn_only": n10, "n_gcn_only": n01,
                                  "histgb_auc_pr": hgb}

        all_results[fname] = {k: (v["seeds"] if isinstance(v, dict) and "seeds" in v else v)
                              for k, v in fold_res.items()}
        volume.commit()

    out = f"{DATA_DIR}/bench_{protocol}.json"
    with open(out, "w") as f:
        json.dump({"protocol": protocol, "seeds": seeds, "epochs": epochs,
                   "n_events": tot_ev, "n_positive": tot_pos,
                   "prevalence": tot_pos / tot_ev, "results": all_results}, f, indent=2, default=str)
    volume.commit()
    print(f"\nwrote {out}")
    return all_results


@app.local_entrypoint()
def main(protocol: str = "temporal", seeds: int = 3, epochs: int = 3,
         max_events_per_scenario: int = 0, scenarios: str = "all"):
    bench.remote(protocol=protocol, seeds=seeds, epochs=epochs,
                 max_events_per_scenario=max_events_per_scenario, scenarios=scenarios)
