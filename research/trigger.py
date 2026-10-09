#!/usr/bin/env python3
"""
Thin local client for the deployed TG-Detect lab.

Deploy once (only needed again when modal_lab.py / modal_common.py change):

    modal deploy modal_lab.py

Then every experiment is a trigger with no image build and no data reload:

    python trigger.py info
    python trigger.py smoke
    python trigger.py diag
    python trigger.py bench --models TGDetect_TGN,TGDetect_Drift --seeds 3 --epochs 6
    python trigger.py ablate --seeds 3 --epochs 4
    python trigger.py ablate --variants V2_All --seeds 5 --epochs 6 --logit-adj

The first trigger after a deploy pays a cold start (container boot + npz load +
causal-context build, and the context is then cached on the Volume). Subsequent
triggers land on the warm container and start computing immediately, which is the
whole point of deploy-and-trigger over `modal run`.

Results are printed here and also written to results/<cmd>-<timestamp>.json.

Long runs (`suite`, multi-seed `bench`/`ablate`) should be launched detached:

    python trigger.py suite --plan robust --detach
    python trigger.py poll <call-id>          # reattach any time, from anywhere

`.remote()` is client-attached: Modal cancels the input as soon as this local
process goes away (terminal closed, laptop asleep, network drop), which is why a
two-hour suite dies mid-step. `--detach` uses `.spawn()`, so the work is owned by
the server and survives the client.
"""

import argparse
import json
import os
import sys
import time

APP = "tg-detect-lab"


def _instance(cls_name, cap, halflife):
    try:
        import modal
    except ImportError:
        sys.exit("modal is not installed locally:  pip install modal")
    try:
        C = modal.Cls.from_name(APP, cls_name)
    except Exception as e:
        sys.exit(f"could not resolve {APP}/{cls_name}: {e}\n"
                 f"deploy it first:  modal deploy modal_lab.py")
    return C(cap=cap, halflife=halflife)


def _save(cmd, out):
    os.makedirs("results", exist_ok=True)
    p = f"results/{cmd}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n[saved] {p}")


def _report(cmd, out, t0):
    if isinstance(out, dict) and out.get("log"):
        print(out["log"], end="")
    print(f"\n[trigger] {cmd} finished in {time.time() - t0:.1f}s wall")
    if isinstance(out, dict):
        _save(cmd, out)


def _launch(cmd, method, detach, t0, **kwargs):
    """Blocking `.remote()` by default, server-owned `.spawn()` when detached."""
    if not detach:
        _report(cmd, method.remote(**kwargs), t0)
        return
    call = method.spawn(**kwargs)
    os.makedirs("results", exist_ok=True)
    with open("results/last_call.txt", "w") as fh:
        fh.write(call.object_id)
    print(f"[detached] {cmd} running server-side as {call.object_id}\n"
          f"           it keeps going if you close this terminal\n"
          f"           logs:   modal app logs {APP}\n"
          f"           result: python trigger.py poll {call.object_id}")


def _poll(call_id, wait):
    import modal
    call = modal.FunctionCall.from_id(call_id)
    t0 = time.time()
    if not wait:
        try:
            out = call.get(timeout=0)
        except Exception as e:
            # A still-running call raises TimeoutError with an EMPTY message, so
            # branch on the type, never on str(e).
            if type(e).__name__ == "TimeoutError":
                print(f"[poll] {call_id} still running "
                      f"(modal app logs {APP} to watch it)")
                return
            raise
    else:
        print(f"[poll] waiting on {call_id} ...")
        out = call.get()
    _report("poll", out, t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

    def add_detach(p):
        # SUPPRESS so the subparser copy never overwrites a --detach given
        # before the subcommand with its own default.
        p.add_argument("--detach", action="store_true",
                       default=argparse.SUPPRESS,
                       help="spawn server-side and return immediately. Without "
                            "this the run is tied to THIS process: closing the "
                            "terminal cancels the Modal input mid-step. Use it "
                            "for anything longer than a few minutes, then "
                            "`trigger.py poll <id>`.")

    ap.add_argument("--cap", type=int, default=600000,
                    help="max events per scenario (selects the warm container)")
    ap.add_argument("--halflife", type=int, default=20000,
                    help="EWMA half-life in events for the causal z-score / context")
    add_detach(ap)
    sub = ap.add_subparsers(dest="cmd", required=True)

    add_detach(sub.add_parser("info",
                              help="cheapest trigger; proves the container is warm"))
    add_detach(sub.add_parser("smoke",
                              help="GPU forward-pass + superset check for all models"))

    d = sub.add_parser("diag", help="H1-H4 distribution-shift diagnostic (CPU)")
    d.add_argument("--fit-sample", type=int, default=800000)
    add_detach(d)

    b = sub.add_parser("bench", help="train and compare graph models (GPU)")
    b.add_argument("--models", default="Pomsathit_GCN,TGDetect_TGN,TGDetect_Drift")
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--epochs", type=int, default=6)
    b.add_argument("--no-calibrate", action="store_true",
                   help="skip isotonic recalibration + SLD prior adaptation")
    b.add_argument("--logit-adj", action="store_true",
                   help="N3: train on logit + tau*log(pi1/pi0) instead of POS_W")
    b.add_argument("--features", default="all",
                   choices=["all", "notime", "noid", "minimal"],
                   help="all=every feature. notime=drop the 4 engineered timing "
                        "features, so architecture is the ONLY route to temporal "
                        "information (the fair architecture test). noid=drop the 3 "
                        "host-fingerprint features. minimal=drop both.")
    b.add_argument("--protocol", default="temporal",
                   choices=["temporal", "capture", "family"],
                   help="temporal=60/15/25 inside each capture (easy: the same bot "
                        "host is in train AND test). capture=hold out whole "
                        "captures. family=hold out a whole malware family (the "
                        "generalisation claim).")
    b.add_argument("--dro", action="store_true",
                   help="replace ERM with online Group DRO over captures. ERM "
                        "minimises the AVERAGE risk, so a shortcut that works on "
                        "most of the training mass and inverts on a minority "
                        "capture still lowers the loss. DRO minimises the WORST "
                        "group's risk, so a sign flip anywhere is paid for at full "
                        "weight. Prints the learned group weights.")
    b.add_argument("--holdout", default="",
                   help="which family (e.g. Neris) or comma-list of captures "
                        "(e.g. s10,s11) to hold out; default picks one")
    b.add_argument("--select", default="none", choices=["none", "macro", "worst"],
                   help="A10 best-epoch selection on the VALIDATION split instead "
                        "of using whatever the last epoch left behind. worst = the "
                        "MINIMUM per-capture val AUC-PR (prefer it: the held-out "
                        "family is the hardest group, so an epoch that has already "
                        "collapsed on one training capture is not the one to ship). "
                        "macro = the mean. Costs one no-grad val pass per epoch.")
    b.add_argument("--rankloss", type=float, default=0.0,
                   help="A12 weight on a within-batch pairwise ranking surrogate. "
                        "We GRADE on AUC-PR (a ranking functional) and TRAIN on "
                        "weighted cross-entropy (a calibration functional), and "
                        "softplus(-(s+ - s-)) is a genuine upper bound on 1-AUC. "
                        "Try 1.0.")
    b.add_argument("--decorr", type=float, default=0.0,
                   help="A14 weight on rho(score, host-activity)^2, computed per "
                        "batch. Targets the measured failure directly: the score "
                        "rides host activity, and on the 16.6%%-prevalence capture "
                        "the busiest hosts ARE the bots, so the score inverts. "
                        "Unlike a learnable gate, ERM cannot decline to pay this. "
                        "Try 1.0.")
    b.add_argument("--decorr-scope", default="batch", choices=["batch", "capture"],
                   help="SCOPE at which --decorr centres its correlation. 'batch' is "
                        "the original and is left byte-identical. 'capture' centres "
                        "on the RUNNING CAPTURE mean instead. Why it matters: by the "
                        "law of total covariance, Cov(s,a) = E_b[Cov(s,a|b)] + "
                        "Cov_b(E[s|b],E[a|b]), and a per-batch penalty sees only the "
                        "first term. The activity columns are CUMULATIVE counters, so "
                        "they ramp with time and batches are contiguous windows -- a "
                        "time-ordered offline rig measured the split at 59.1%% within "
                        "vs 40.9%% BETWEEN, i.e. batch scope is structurally blind to "
                        "two fifths of the association it is meant to remove. "
                        "See .labtest/research/e8_decorr_scope.py.")
    b.add_argument("--memprobe", action="store_true",
                   help="measure whether a host's EVENT COUNT is linearly decodable "
                        "from its node memory: R2(log k) from the raw memory, from "
                        "the LayerNormed memory, and after removing one rank-1 "
                        "direction fitted on held-out nodes. This ARBITRATES "
                        "memhard. LN(a*m+b)==LN(m) is a theorem, but it is "
                        "invariance to RESCALING; if the GRU's retention gates "
                        "differ across dimensions then saturation (1-z**k) rotates "
                        "the memory DIRECTION with k, and LayerNorm is blind to "
                        "rotations. Also prints the trained retention gate and the "
                        "||m||-vs-k curve: a FLAT curve with horizon ~1 means the "
                        "memory has no horizon and there is no leak to remove, which "
                        "is a different and worse finding. Costs one extra ridge fit "
                        "per test capture, no extra training. "
                        "See .labtest/research/e1b_real_gru.py and e9_probe_selftest.py.")
    b.add_argument("--ensemble", action="store_true",
                   help="A11 add one extra row per model that rank-averages the "
                        "seeds already trained. Free (no extra training) and it is "
                        "the cheapest variance reduction available.")
    b.add_argument("--probe-oracle", action="store_true",
                   help="also report the best-epoch-by-TEST MACRO. Reads test "
                        "labels, so it is a HEADROOM BOUND and never a result -- it "
                        "answers 'is epoch selection worth having', nothing else.")
    b.add_argument("--repeats", type=int, default=1,
                   help="run each (model, seed) N times under a byte-identical "
                        "configuration. The spread of those N IS the resolution "
                        "limit of every comparison in the run: CUDA's index_add "
                        "backward is order-nondeterministic and a 480k-event "
                        "recurrent chain amplifies it (measured: 0.7319 vs 0.5074 "
                        "MACRO for the same model at the same seed).")
    b.add_argument("--vrex", type=float, default=0.0,
                   help="V-REx objective weight beta (Krueger 2021): add "
                        "beta*Var_capture(risk) so the model EQUALISES per-capture "
                        "risk instead of just minimising the average. Complements "
                        "--dro rather than duplicating it: DRO minimises the WORST "
                        "capture's risk and its bound holds only inside the convex "
                        "hull of training captures; s09 is a 22x prevalence "
                        "EXTRAPOLATION outside that hull, where equal-risk (VREx) is "
                        "the property preserved under affine env shifts. Measured on "
                        "the sign-flip toy (15 seeds, VARIED shortcut): test ROC ERM "
                        "0.396 / DRO 0.707 / VREx 0.775. When shortcut strength is "
                        "constant across captures the variance signal vanishes and "
                        "VREx needs --memhard to lean on. Try 1.0. "
                        "See .labtest/research/e9_vrex_vs_dro.py.")
    b.add_argument("--plain-decay", action="store_true",
                   help="use dt-ONLY memory decay, removing the risk term from the "
                        "retention time-constant. The default couples decay to the "
                        "model's own sigmoid(logit): risky endpoints keep memory 10x "
                        "longer, memory norm grows with predicted risk, and that norm "
                        "is an activity-like signal the scorer reads back -- a "
                        "forward-dynamics positive feedback loop that AMPLIFIES the "
                        "very host-activity shortcut --decorr fights (measured "
                        "+0.1318+-0.0050 correlation lift, t=+26.1, 20/20 seeds, "
                        "e9 H-B). Under no_grad, so .detach() would not help; only "
                        "removing the risk term does. Genuine recency is untouched.")
    add_detach(b)

    q = sub.add_parser("suite",
                       help="run a whole experiment PLAN in one warm container "
                            "(one stream load, folds swapped server-side, one "
                            "cross-step summary table at the end)")
    q.add_argument("--plan", default="stability",
                   choices=["stability", "robust", "quick", "families"],
                   help="stability = THE CURRENT ONE (~1h40m): measure the noise "
                        "floor, then epoch-selection + seed ensembling, then the "
                        "ranking/decorrelation objectives, then imposed memory "
                        "normalisation. robust = the previous plan; it ran, and its "
                        "own control step showed run-to-run noise (0.22 MACRO) "
                        "exceeded every effect it was measuring. quick = 1 seed / "
                        "3 epochs pipeline check. families = replicate on Neris, "
                        "Rbot and Virut.")
    q.add_argument("--seeds", type=int, default=3,
                   help="seeds for steps that do not pin their own (the control "
                        "step is an identity check and stays at 1)")
    q.add_argument("--epochs", type=int, default=6)
    q.add_argument("--no-calibrate", action="store_true",
                   help="skip isotonic + SLD. Ranking metrics are unaffected, so "
                        "this only costs you the F1 columns and saves a little time.")
    add_detach(q)

    v = sub.add_parser("ablate", help="N1-N5 hypothesis grid, paired against TGDetect_TGN")
    v.add_argument("--variants", default="",
                   help="comma list; default is the full grid incl. combinations")
    v.add_argument("--seeds", type=int, default=3)
    v.add_argument("--epochs", type=int, default=4)
    v.add_argument("--logit-adj", action="store_true")
    v.add_argument("--calibrate", action="store_true",
                   help="also run isotonic + SLD (slower; ranking metrics unaffected)")
    add_detach(v)

    p = sub.add_parser("poll", help="fetch the result of a --detach'd run")
    p.add_argument("call_id", nargs="?", default="",
                   help="default: the id written to results/last_call.txt")
    p.add_argument("--wait", action="store_true",
                   help="block until it finishes instead of reporting status")

    a = ap.parse_args()
    detach = getattr(a, "detach", False)
    t0 = time.time()

    if a.cmd == "poll":
        cid = a.call_id
        if not cid:
            try:
                with open("results/last_call.txt") as fh:
                    cid = fh.read().strip()
            except OSError:
                sys.exit("no call id given and results/last_call.txt is missing")
        _poll(cid, a.wait)
        return

    if a.cmd in ("info", "diag"):
        lab = _instance("LabCPU", a.cap, a.halflife)
        if a.cmd == "info":
            if detach:
                _launch(a.cmd, lab.info, True, t0)
            else:
                out = lab.info.remote()
                print(json.dumps(out, indent=1))
                _report(a.cmd, out, t0)
        else:
            _launch(a.cmd, lab.diag, detach, t0, fit_sample=a.fit_sample)
    else:
        lab = _instance("LabGPU", a.cap, a.halflife)
        if a.cmd == "smoke":
            _launch(a.cmd, lab.smoke, detach, t0)
        elif a.cmd == "suite":
            _launch(a.cmd, lab.suite, detach, t0, plan=a.plan, seeds=a.seeds,
                    epochs=a.epochs, calibrate=not a.no_calibrate)
        elif a.cmd == "ablate":
            _launch(a.cmd, lab.ablate, detach, t0, variants=a.variants,
                    seeds=a.seeds, epochs=a.epochs, logit_adj=a.logit_adj,
                    calibrate=a.calibrate)
        else:
            _launch(a.cmd, lab.bench, detach, t0, models=a.models, seeds=a.seeds,
                    epochs=a.epochs, calibrate=not a.no_calibrate,
                    logit_adj=a.logit_adj, features=a.features,
                    protocol=a.protocol, holdout=a.holdout, dro=a.dro,
                    select=a.select, rankloss=a.rankloss, decorr=a.decorr,
                    decorr_scope=a.decorr_scope, memprobe=a.memprobe,
                    ensemble=a.ensemble, probe_oracle=a.probe_oracle,
                    repeats=a.repeats, vrex=a.vrex, plain_decay=a.plain_decay)


if __name__ == "__main__":
    main()
