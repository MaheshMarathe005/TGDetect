"""Two-GPU DEDALE host training: GNN+rehearsal vs the full TGNN hybrid, with a
threshold-honest evaluation that separates RANKING quality from fixed-threshold F1.

Each arm runs in its own container on its own L4 (spawned in parallel). It consumes a
prepared npz on the Volume (built by dedale_data.py) and reports, per seed:
  * f1 at the default 0.5 threshold            -> the "bad default threshold" view
  * f1 at a threshold chosen on VALIDATION and frozen before test -> honest calibration
  * f1 at the test-set-oracle threshold        -> optimistic upper bound, NOT a
                                                  generalization estimate (shown for the
                                                  user's "good ranking, bad threshold" test)
  * pr_ap (average precision) and roc_auc      -> threshold-free ranking quality
Full test probabilities + an F1-vs-threshold curve are saved for both models so the
crossover the user hypothesised can be drawn directly.

  python -m modal run modal_dedale_train.py            # both arms, parallel L4s
  python -m modal run modal_dedale_train.py --epochs 60 --seeds 7 --minutes 90
"""
import os
import modal

app = modal.App("tg-detect-dedale-train")
volume = modal.Volume.from_name("tg-detect-dedale-v1", create_if_missing=True)
SRC = ("windows_apt_models", "tgnn_components", "causal_tgn", "temporal_cnn",
       "pomsathit_baseline", "causal_data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", "numpy==1.26.4", "scikit-learn==1.5.2")
         .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
         .add_local_python_source(*SRC))

PREP = "/vol/prepared/dedale_host/prepared.npz"
RESULTS = "/vol/results/dedale_host"
ARMS = ("GNN_H", "TGNN_Q_A_C")


def _select_threshold(y, p):
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    precision, recall, thr = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thr[np.flatnonzero(f1 == f1.max())[-1]]) if len(thr) else 0.5


def _metrics(y, p, threshold):
    from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                                 average_precision_score, roc_auc_score)
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"threshold": float(threshold), "f1": float(f1_score(y, pred, zero_division=0)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "pr_ap": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "fpr": float(fp / max(tn + fp, 1)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _f1_curve(y, p, steps=200):
    import numpy as np
    ts = np.linspace(0.0, 1.0, steps + 1)
    f1s = []
    for t in ts:
        pred = p >= t
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return ts.tolist(), f1s

@app.function(image=image, gpu="L4", timeout=12600, volumes={"/vol": volume})
def train_arm(arm: str, epochs: int = 60, seeds: int = 7, minutes: float = 90.0):
    """Train one arm across seeds on its own L4; time-boxed. Returns a per-seed + mean summary."""
    import hashlib, json, subprocess, time
    from pathlib import Path
    import numpy as np
    import torch
    from windows_apt_models import make_model
    from temporal_cnn import rehearsal_schedule

    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    volume.reload()
    d = np.load(PREP)
    x = torch.from_numpy(d["x"]).cuda()
    windows = x[torch.from_numpy(d["ego"]).cuda()]
    valid = torch.from_numpy(d["valid"]).cuda(); ages = torch.from_numpy(d["ages"]).cuda()
    y_np = d["y"].astype(np.int64); y = torch.from_numpy(y_np).cuda()
    tr = int(d["train_end"]); va = int(d["val_end"]); n = len(y_np); input_dim = x.shape[1]
    out = Path(RESULTS); out.mkdir(parents=True, exist_ok=True)

    def forward(model, ix):
        return model.forward_events(windows[ix], valid[ix], ages[ix], None, None, None)[0]

    results = []; start = time.monotonic()
    for seed in range(seeds):
        stem = f"{arm}_seed{seed}"
        if (out / f"{stem}.json").exists():
            results.append(json.loads((out / f"{stem}.json").read_text())); continue
        if seed >= 3 and (time.monotonic() - start) / 60.0 > minutes:
            print(f"{arm}: stopping after {seed} seeds, time budget {minutes} min reached", flush=True); break
        torch.manual_seed(seed); np.random.seed(seed)
        rid, sizes = rehearsal_schedule(tr, epochs, 256, seed)
        model = make_model(arm, input_dim).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        torch.manual_seed(seed + 10000); t0 = time.monotonic(); step = 0; losses = []
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(epochs):
            model.train(); total = 0.0
            for lo in range(0, tr, 256):
                hi = min(lo + 256, tr); opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(forward(model, slice(lo, hi)), y[lo:hi])
                sz = int(sizes[step])
                if sz:
                    ix = torch.as_tensor(rid[step, :sz], device="cuda")
                    loss = loss + torch.nn.functional.cross_entropy(forward(model, ix), y[ix])
                assert torch.isfinite(loss); loss.backward(); opt.step(); total += loss.item(); step += 1
            losses.append(total)
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"{arm} seed{seed} epoch {epoch+1}/{epochs} loss {total:.3f}", flush=True)
        torch.cuda.synchronize(); seconds = time.monotonic() - t0
        model.eval(); prob = []
        with torch.no_grad():
            for lo in range(0, n, 256):
                logits = forward(model, slice(lo, min(lo + 256, n)))
                prob.append(logits.softmax(-1)[:, 1].cpu().numpy())
        p = np.concatenate(prob)
        t_val = _select_threshold(y_np[tr:va], p[tr:va])
        t_oracle = _select_threshold(y_np[va:], p[va:])
        ts, f1s = _f1_curve(y_np[va:], p[va:])
        torch.save({"state_dict": model.cpu().state_dict(), "arm": arm, "input_dim": input_dim,
                    "seed": seed, "epochs": epochs}, out / f"{stem}.pt")
        np.savez_compressed(out / f"{stem}_predictions.npz", probability=p, y=y_np,
                            train_end=tr, val_end=va, ids=d.get("ids"))
        result = {"arm": arm, "seed": seed, "epochs": epochs, "train_seconds": seconds,
                  "val_threshold": t_val, "oracle_threshold": t_oracle,
                  "test_default_0.5": _metrics(y_np[va:], p[va:], 0.5),
                  "test_val_threshold": _metrics(y_np[va:], p[va:], t_val),
                  "test_oracle_threshold": _metrics(y_np[va:], p[va:], t_oracle),
                  "validation_at_val_threshold": _metrics(y_np[tr:va], p[tr:va], t_val),
                  "f1_curve": {"thresholds": ts, "f1": f1s},
                  "epoch_losses": losses, "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                  "checkpoint_sha256": hashlib.sha256((out / f"{stem}.pt").read_bytes()).hexdigest()}
        (out / f"{stem}.json").write_text(json.dumps(result, indent=2)); results.append(result)
        volume.commit(); del model, opt; torch.cuda.empty_cache()

    def agg(key, sub):
        vals = [r[sub][key] for r in results]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [round(v, 6) for v in vals]}
    summary = {"arm": arm, "seeds_done": len(results), "input_dim": input_dim,
               "gpu": torch.cuda.get_device_name(), "partitions": {"train_end": tr, "val_end": va, "n": n},
               "positives": {"train": int(y_np[:tr].sum()), "val": int(y_np[tr:va].sum()), "test": int(y_np[va:].sum())},
               "metrics": {v: {k: agg(k, v) for k in ("f1", "precision", "recall", "fpr", "pr_ap", "roc_auc")}
                           for v in ("test_default_0.5", "test_val_threshold", "test_oracle_threshold")}}
    (Path(RESULTS) / f"summary_{arm}.json").write_text(json.dumps(summary, indent=2)); volume.commit()
    return summary

@app.local_entrypoint()
def main(epochs: int = 60, seeds: int = 7, minutes: float = 90.0):
    """Spawn both arms concurrently (two containers, two L4s), wait, and print a comparison."""
    import json
    from pathlib import Path
    calls = {arm: train_arm.spawn(arm, epochs, seeds, minutes) for arm in ARMS}
    summaries = {arm: call.get() for arm, call in calls.items()}
    out = Path("artifacts/dedale_v1"); out.mkdir(parents=True, exist_ok=True)
    (out / "train_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")

    print("\n== threshold-free ranking quality (does the temporal model RANK better?) ==")
    for arm in ARMS:
        m = summaries[arm]["metrics"]["test_val_threshold"]
        print(f"  {arm:12} seeds={summaries[arm]['seeds_done']}  "
              f"PR-AP {m['pr_ap']['mean']*100:6.2f}±{m['pr_ap']['sd']*100:.2f}  "
              f"ROC-AUC {m['roc_auc']['mean']*100:6.2f}±{m['roc_auc']['sd']*100:.2f}")
    for view, label in (("test_default_0.5", "F1 @ default 0.5 threshold (the 'bad threshold' view)"),
                        ("test_val_threshold", "F1 @ validation-frozen threshold (honest calibration)"),
                        ("test_oracle_threshold", "F1 @ test-oracle threshold (optimistic upper bound)")):
        print(f"\n== {label} ==")
        for arm in ARMS:
            m = summaries[arm]["metrics"][view]
            print(f"  {arm:12} F1 {m['f1']['mean']*100:6.2f}±{m['f1']['sd']*100:.2f}  "
                  f"P {m['precision']['mean']*100:6.2f}  R {m['recall']['mean']*100:6.2f}  "
                  f"FPR {m['fpr']['mean']*100:5.3f}")
    print("\nSaved per-seed metrics, checkpoints, predictions and F1-vs-threshold curves to",
          RESULTS, "on the Volume.")

