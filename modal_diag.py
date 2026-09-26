#!/usr/bin/env python3
"""
SUPERSEDED by modal_lab.py :: LabCPU.diag — do not run this file.

Kept only as the record of the original hypothesis statements. The live version
lives in the deployed lab, which additionally (a) reuses a warm container instead
of rebuilding, (b) adds ablations E/F for the keep-every-feature fix, and
(c) recalibrates before prior adaptation. Use:

    modal deploy modal_lab.py
    python trigger.py diag

TG-Detect — DIAGNOSTIC: why did the 13-scenario temporal fold invert?

CPU-only, no model training beyond sklearn controls. Purpose is to falsify or
confirm H1-H4 before any architecture work, because a ROC-AUC of 0.12 is a
distribution-shift signature, not a capacity signature.

  H1  threshold transfer failure   val prevalence 0.68% vs test 6.37%
  H2  cumulative-counter features (13,14,17,18,19) are non-stationary and
      carry an inverted sign at test time
  H3  pooling 13 scenarios with 0.055%-8.86% prevalence makes the pooled
      metric an artifact dominated by a few captures
  H4  the first-N cap biases train toward the pre-infection period

Run:
  modal run modal_diag.py::diag
"""

import modal

app = modal.App("tg-detect-diag")
volume = modal.Volume.from_name("tg-detect-data", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.1.3", "scikit-learn==1.5.2", "scipy==1.14.1"
)

DATA_DIR = "/data/real/ctu13"

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
# H2 suspects: cumulative lifetime counters + drifting running stats
CUMULATIVE = [13, 14, 17, 18, 19]
STATIONARY = [i for i in range(22) if i not in CUMULATIVE]


@app.function(image=image, volumes={"/data": volume}, timeout=7200,
              memory=65536, cpu=8.0)
def diag(max_events_per_scenario: int = 600000, fit_sample: int = 800000):
    import os, json, numpy as np
    from scipy import stats as sps
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (average_precision_score, roc_auc_score,
                                 f1_score, precision_recall_curve)

    rng = np.random.default_rng(0)
    keys = sorted(FAMILY)

    # ───────────────── load, replicating the bench exactly ─────────────────
    streams = []
    for k in keys:
        p = f"{DATA_DIR}/{k}.npz"
        if not os.path.exists(p):
            print(f"  ! missing {p}"); continue
        z = np.load(p)
        X, ts, y = z["X"], z["ts"], z["y"]
        n_full, pos_full = len(y), int(y.sum())
        if max_events_per_scenario and len(y) > max_events_per_scenario:
            X, ts, y = (a[:max_events_per_scenario] for a in (X, ts, y))
        ts = (ts.astype(np.float64) - float(ts[0])).astype(np.float32)
        streams.append({"key": k, "family": FAMILY[k],
                        "X": np.nan_to_num(X), "ts": ts, "y": y.astype(np.float32),
                        "n_full": n_full, "pos_full": pos_full})

    # ══════════════════════════ H4 ══════════════════════════
    print("\n" + "=" * 100)
    print("H4 — does the first-N cap bias train toward the pre-infection period?")
    print("=" * 100)
    print(f"  {'scen':5s} {'family':9s} {'n_full':>11s} {'prev_full':>10s} "
          f"{'n_cap':>10s} {'prev_cap':>10s} {'ratio':>7s}   prevalence by stream quintile (capped)")
    h4 = {}
    for s in streams:
        pf = s["pos_full"] / s["n_full"] * 100
        pc = float(s["y"].mean()) * 100
        q = [float(a.mean()) * 100 for a in np.array_split(s["y"], 5)]
        h4[s["key"]] = {"prev_full": pf, "prev_cap": pc, "quintiles": q}
        qs = " ".join(f"{v:6.3f}" for v in q)
        r = pc / pf if pf > 1e-9 else float("nan")
        print(f"  {s['key']:5s} {s['family']:9s} {s['n_full']:11,d} {pf:9.4f}% "
              f"{len(s['y']):10,d} {pc:9.4f}% {r:7.2f}   {qs}")

    # ───────────────── folds: 60/15/25 chronological per scenario ─────────────────
    tr, va, te, seg = [], [], [], {}
    for s in streams:
        n = len(s["y"]); a, b = int(n * 0.60), int(n * 0.75)
        seg[s["key"]] = (a, b, n)
        tr.append((s, 0, a)); va.append((s, a, b)); te.append((s, b, n))

    def cat(segs, field):
        return np.concatenate([s[field][i:j] for s, i, j in segs])

    Xtr, ytr = cat(tr, "X"), cat(tr, "y")
    Xva, yva = cat(va, "X"), cat(va, "y")
    Xte, yte = cat(te, "X"), cat(te, "y")
    for nm, y in (("train", ytr), ("val", yva), ("test", yte)):
        print(f"  {nm:5s} n={len(y):>10,d}  pos={int(y.sum()):>8,d}  prev={y.mean()*100:7.4f}%")
    base_te = float(yte.mean())
    print(f"\n  RANDOM-BASELINE AUC-PR on test = prevalence = {base_te:.4f}")
    print(f"  train->test prevalence ratio   = {yte.mean()/ytr.mean():.2f}x")

    # ══════════════════════════ H2a ══════════════════════════
    print("\n" + "=" * 100)
    print("H2a — per-feature train vs test shift (KS) and drift vs stream position")
    print("=" * 100)
    print(f"  {'idx':>3s} {'feature':20s} {'KS':>7s} {'mu_tr':>9s} {'mu_te':>9s} "
          f"{'d/sd_tr':>8s} {'rho_pos':>8s}  flag")
    # NOTE: sub_tr must stay SORTED — the drift statistic below correlates each
    # feature against stream position, which an unsorted random subsample would
    # destroy (rho would be ~0 for everything and the test would look negative).
    # Drift must be measured WITHIN each scenario. `train` is a concatenation of
    # 13 separate captures, each restarting its own ramp from zero, so a Spearman
    # against the global concatenated index sees a sawtooth and washes the drift
    # out (observed: rho 0.08 on a feature with KS 0.80). Average the per-capture
    # rank correlation instead.
    def drift_per_scenario(col):
        rs = []
        for s, lo, hi in tr:
            v = s["X"][lo:hi, col]
            if len(v) < 1000 or float(np.ptp(v)) < 1e-9:
                continue
            step = max(len(v) // 20000, 1)
            v = v[::step].astype(np.float64)
            r = sps.spearmanr(np.arange(len(v), dtype=np.float64), v).statistic
            if np.isfinite(r):
                rs.append(float(r))
        return float(np.mean(rs)) if rs else float("nan")

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

    # ══════════════════════════ H2c ══════════════════════════
    # The direct test of the inversion mechanism. Drift alone does NOT invert a
    # model — verified on a synthetic fixture where features 13/14/17 drifted hard
    # (rho 0.85, KS 0.80) yet ROC stayed at 1.00. Inversion requires the feature's
    # RELATIONSHIP TO THE LABEL to change sign between train and test. So measure
    # the point-biserial correlation with y separately in each window and flag the
    # sign flips. These, not merely the shifted features, are the culprits.
    print("\n" + "=" * 100)
    print("H2c — does any feature FLIP the sign of its label correlation?")
    print("       (this is what actually inverts a model, not drift per se)")
    print("=" * 100)
    print(f"  {'idx':>3s} {'feature':20s} {'r_train':>9s} {'r_val':>9s} {'r_test':>9s} "
          f"{'|dr|':>7s}  flag")
    def pbis(Xs, ys, col):
        v = Xs[:, col].astype(np.float64)
        sd = v.std()
        if sd < 1e-12 or ys.min() == ys.max():
            return float("nan")
        return float(np.corrcoef(v, ys.astype(np.float64))[0, 1])

    h2c, flips = {}, []
    for i, name in enumerate(FEAT):
        r_tr = pbis(Xtr, ytr, i); r_va = pbis(Xva, yva, i); r_te = pbis(Xte, yte, i)
        flag = ""
        if np.isfinite(r_tr) and np.isfinite(r_te):
            if r_tr * r_te < 0 and min(abs(r_tr), abs(r_te)) > 0.01:
                flag += " SIGN-FLIP"
                flips.append((name, r_tr, r_te))
            if abs(r_te - r_tr) > 0.10:
                flag += " WEAKENS" if abs(r_te) < abs(r_tr) else " STRENGTHENS"
        if i in CUMULATIVE: flag += " [suspect]"
        h2c[name] = {"r_train": r_tr, "r_val": r_va, "r_test": r_te}
        print(f"  {i:3d} {name:20s} {r_tr:9.4f} {r_va:9.4f} {r_te:9.4f} "
              f"{abs(r_te-r_tr):7.4f} {flag}")
    if flips:
        print(f"\n  {len(flips)} SIGN-FLIPPING feature(s) — these invert the model:")
        for nm, a_, b_ in sorted(flips, key=lambda t: -abs(t[1] - t[2])):
            print(f"    {nm:20s} train {a_:+.4f} -> test {b_:+.4f}")
    else:
        print("\n  no sign flips: H2c REJECTED, inversion must come from elsewhere")

    # ───────────────── shared eval helper ─────────────────
    def evaluate(tag, pva, pte):
        ap = float(average_precision_score(yte, pte))
        roc = float(roc_auc_score(yte, pte))
        # H1: threshold from val, vs oracle threshold on test
        pr, rc, th = precision_recall_curve(yva, pva)
        f = 2 * pr * rc / (pr + rc + 1e-12)
        thr_va = float(th[int(np.argmax(f[:-1]))]) if len(th) else 0.5
        f1_va = float(f1_score(yte, (pte >= thr_va).astype(int), zero_division=0))
        pr2, rc2, th2 = precision_recall_curve(yte, pte)
        f2 = 2 * pr2 * rc2 / (pr2 + rc2 + 1e-12)
        f1_or = float(np.max(f2[:-1])) if len(th2) else 0.0
        # quantile-matched threshold: fire on the top q of test, q = val-optimal rate
        rate = float((pva >= thr_va).mean())
        thr_q = float(np.quantile(pte, 1 - max(rate, 1e-9)))
        f1_q = float(f1_score(yte, (pte >= thr_q).astype(int), zero_division=0))
        lift = ap / base_te
        print(f"  {tag:34s} AUC-PR {ap:.4f} ({lift:5.2f}x rnd) | ROC {roc:.4f} "
              f"(flip {1-roc:.4f}) | F1 val-thr {f1_va:.4f} | F1 q-match {f1_q:.4f} "
              f"| F1 oracle {f1_or:.4f}")
        return {"tag": tag, "auc_pr": ap, "lift": lift, "roc": roc,
                "f1_val_thr": f1_va, "f1_qmatch": f1_q, "f1_oracle": f1_or}

    def fit_eval(tag, cols, transform=None):
        A, B, C = Xtr[:, cols], Xva[:, cols], Xte[:, cols]
        if transform is not None:
            A, B, C = transform(A, B, C)
        mu, sd = A.mean(0), A.std(0); sd[sd < 1e-6] = 1.0
        As, Bs, Cs = (A - mu) / sd, (B - mu) / sd, (C - mu) / sd
        idx = rng.choice(len(ytr), size=min(fit_sample, len(ytr)), replace=False)
        out = []
        lr = LogisticRegression(max_iter=300, class_weight="balanced")
        lr.fit(As[idx], ytr[idx])
        out.append(evaluate(f"{tag} / LogReg",
                            lr.predict_proba(Bs)[:, 1], lr.predict_proba(Cs)[:, 1]))
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                            class_weight="balanced", random_state=0)
        gb.fit(As[idx], ytr[idx])
        out.append(evaluate(f"{tag} / HistGB",
                            gb.predict_proba(Bs)[:, 1], gb.predict_proba(Cs)[:, 1]))
        return out

    # ══════════════════════════ H1 + H2b ══════════════════════════
    print("\n" + "=" * 100)
    print("H1 + H2b — feature-set ablation. If H2 holds, dropping the cumulative")
    print("           counters pushes ROC-AUC back above 0.5 even as val AP falls.")
    print("=" * 100)
    results = []
    results += fit_eval("A all-22", list(range(22)))
    results += fit_eval("B minus-cumulative(13,14,17,18,19)", STATIONARY)
    results += fit_eval("C cumulative-ONLY", CUMULATIVE)

    # D: batch rank-normalisation within each split slice. This is an UPPER BOUND,
    # not a deployable transform — ranking within the test slice peeks at the test
    # marginal (not at labels, so it is not label leakage, but it is not causal
    # either). Its only job is to answer "if the shift in these columns were
    # removed, how much performance comes back?". The deployable fix is to change
    # the FEATURES in modal_data_real.py to windowed rates, then re-measure.
    def rank_norm(A, B, C):
        def rk(M):
            out = np.empty_like(M)
            for j in range(M.shape[1]):
                o = np.argsort(M[:, j], kind="stable")
                r = np.empty(len(o), dtype=np.float32)
                r[o] = np.linspace(0.0, 1.0, len(o), dtype=np.float32)
                out[:, j] = r
            return out
        return rk(A), rk(B), rk(C)

    results += fit_eval("D all-22 rank-norm (UPPER BOUND)", list(range(22)), rank_norm)

    # ══════════════════════════ H3 ══════════════════════════
    print("\n" + "=" * 100)
    print("H3 — is the pooled test metric an artifact of a few captures?")
    print("=" * 100)
    cols = STATIONARY
    A = Xtr[:, cols]; mu, sd = A.mean(0), A.std(0); sd[sd < 1e-6] = 1.0
    idx = rng.choice(len(ytr), size=min(fit_sample, len(ytr)), replace=False)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                        class_weight="balanced", random_state=0)
    gb.fit((A[idx] - mu) / sd, ytr[idx])
    print(f"  {'scen':5s} {'family':9s} {'n_test':>10s} {'pos':>8s} {'prev':>9s} "
          f"{'share_pos':>10s} {'AUC-PR':>8s} {'lift':>7s} {'ROC':>8s}")
    h3 = {}
    tot_pos_te = int(yte.sum())
    for s, i, j in te:
        Xs = (s["X"][i:j][:, cols] - mu) / sd
        ys = s["y"][i:j]
        p = gb.predict_proba(Xs)[:, 1]
        npos = int(ys.sum()); prev = float(ys.mean())
        if npos == 0 or npos == len(ys):
            print(f"  {s['key']:5s} {s['family']:9s} {len(ys):10,d} {npos:8,d} "
                  f"{prev*100:8.4f}% {npos/max(tot_pos_te,1)*100:9.2f}%   (degenerate)")
            continue
        ap = float(average_precision_score(ys, p)); roc = float(roc_auc_score(ys, p))
        h3[s["key"]] = {"n": len(ys), "pos": npos, "prev": prev, "auc_pr": ap, "roc": roc}
        print(f"  {s['key']:5s} {s['family']:9s} {len(ys):10,d} {npos:8,d} "
              f"{prev*100:8.4f}% {npos/max(tot_pos_te,1)*100:9.2f}% "
              f"{ap:8.4f} {ap/max(prev,1e-9):7.2f} {roc:8.4f}")
    if h3:
        aps = np.array([v["auc_pr"] for v in h3.values()])
        lifts = np.array([v["auc_pr"] / max(v["prev"], 1e-9) for v in h3.values()])
        share = sorted((v["pos"] / max(tot_pos_te, 1) for v in h3.values()), reverse=True)
        print(f"\n  per-scenario AUC-PR  min {aps.min():.4f}  max {aps.max():.4f}  "
              f"spread {aps.max()/max(aps.min(),1e-9):.1f}x")
        print(f"  per-scenario lift    min {lifts.min():.2f}x max {lifts.max():.2f}x")
        print(f"  top-3 scenarios hold {sum(share[:3])*100:.1f}% of all test positives")

    out = {"h4": h4, "h2a": h2a, "h2c": h2c, "results": results, "h3": h3,
           "base_te": base_te, "prev_ratio": float(yte.mean() / ytr.mean())}
    with open(f"{DATA_DIR}/diag.json", "w") as fh:
        json.dump(out, fh, indent=1)
    volume.commit()
    print(f"\nwrote {DATA_DIR}/diag.json")
    return out


@app.local_entrypoint()
def main(max_events_per_scenario: int = 600000, fit_sample: int = 800000):
    diag.remote(max_events_per_scenario=max_events_per_scenario, fit_sample=fit_sample)
