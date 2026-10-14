"""Final TGNN (TGNN_Q_A_C) trained on the DEDALE host 15:1 reservoir-subsampled
dataset in /prepared/dedale_host. This is NOT the full-unsubsampled build and should
not be described as a ~365:1 imbalance experiment.

VERIFIED RESULTS & CHECKPOINTS (COMPLETED 2026-09-30, FETCHED FROM tg-detect-dedale-v1):
  * Volume source: tg-detect-dedale-v1/results/dedale_host_15x
  * Local copy: artifacts/dedale_host_15x/
  * Dataset: 153,056 events (91,833 train / 30,611 val / 30,612 test), 15:1 reservoir ratio
  * Architecture: TGNN_Q_A_C (72 input dims, 3x128 GAT + Time + History-GRU + Attention + Causal CNN)
  * Target Hardware: NVIDIA L4 GPU, 80 epochs per seed, 3 seeds evaluated
  * Best Model: Seed 0 -> TGNN_Q_A_C_final.pt (SHA-256: 34ffc5e0a4d61c69d67f0dceee950305bd156ac2ad156903b9bfc9625b61ee33)
  * Test Metrics (Mean ± SD across Seeds 0-2):
      - Test ROC-AUC:             0.9954 ± 0.0013  [Seed 0: 0.9968, Seed 1: 0.9951, Seed 2: 0.9942]
      - Test PR-AP:               88.37 ± 0.70%    [Seed 0: 88.55%, Seed 1: 87.59%, Seed 2: 88.96%]
      - Test F1 (@ default 0.5):  86.22 ± 0.57%    [Seed 0: 85.88%, Seed 1: 86.88%, Seed 2: 85.90%]
      - Test Recall (@ 0.5):      94.37 ± 0.90%    [Seed 0: 94.89%, Seed 1: 94.89%, Seed 2: 93.33%]
      - Test Precision (@ 0.5):   79.37 ± 0.86%    [Seed 0: 78.43%, Seed 1: 80.12%, Seed 2: 79.56%]
      - Test FPR (@ 0.5):          0.58 ± 0.03%    [Seed 0: 0.62%,  Seed 1: 0.56%,  Seed 2: 0.57%]
      - Test F1 (@ Val Thr):      85.72 ± 0.12%    [Val Thresholds: 0.707, 0.767, 0.697]
      - Test F1 (@ Oracle Thr):   86.53 ± 0.45%    [Oracle Thresholds: 0.536, 0.537, 0.540]

DEPLOYMENT MODE: deploy the App once, then use `python this_file.py --spawn ...` to
submit a server-side FunctionCall that survives closing the local terminal.

  * class-weighted cross-entropy (balanced from TRAIN counts only) handles the imbalance;
  * periodic model+optimizer+replay+RNG checkpoints to the Volume, so a Modal preemption
    RESUMES a multi-hour GPU run instead of restarting it from zero;
  * the best-validation-PR-AP snapshot among evaluated checkpoints is the model we keep
    (honest val selection; the
    test set is never used to pick a model or a threshold for the headline number);
  * threshold-honest test report: PR-AP / ROC-AUC (ranking, threshold-free) + F1 at the
    0.5 / validation-frozen / test-oracle thresholds, plus the full F1-vs-threshold curve
    and the raw test probabilities for later plotting.

The replay policy is byte-identical to temporal_cnn.rehearsal_schedule (Algorithm R,
capacity train_end//10, seed+937, sample<=26 before admitting the current batch).

  modal deploy modal_dedale_final_8cpu_16gb.py --name tg-detect-dedale-final
  python modal_dedale_final_8cpu_16gb.py --spawn --seeds 3 --epochs 80 --minutes 390 --train-batch 512 --infer-batch 1024 --ckpt-every 5
"""
import modal

app = modal.App("tg-detect-dedale-final")
volume = modal.Volume.from_name("tg-detect-dedale-v1", create_if_missing=True)
SRC = ("windows_apt_models", "tgnn_components", "causal_tgn", "temporal_cnn",
       "pomsathit_baseline", "causal_data")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", "numpy==1.26.4", "scikit-learn==1.5.2")
         .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
         .add_local_python_source(*SRC))

PREP = "/vol/prepared/dedale_host/prepared.npz"
RESULTS = "/vol/results/dedale_host_15x"
ARM = "TGNN_Q_A_C"


def _select_threshold(y, p):
    """Threshold that maximises F1 on the given (y, p); ties -> largest threshold."""
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
    ts = np.linspace(0.0, 1.0, steps + 1); f1s = []
    for t in ts:
        pred = p >= t
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return ts.tolist(), f1s


class ReplayBuffer:
    """Online reservoir replay, policy-identical to temporal_cnn.rehearsal_schedule:
    sample <=26 stored TRAIN event ids BEFORE admitting the current batch (Algorithm R,
    capacity train_end//10, RNG seeded seed+937). Fully picklable for preemption resume."""
    def __init__(self, train_end, seed):
        import random
        self.cap = train_end // 10
        self.rng = random.Random(seed + 937)
        self.items = []
        self.seen = 0

    def sample_then_admit(self, lo, hi):
        ids = self.rng.sample(self.items, min(26, len(self.items)))
        for event in range(lo, hi):
            self.seen += 1
            if len(self.items) < self.cap:
                self.items.append(event)
            elif self.cap:
                j = self.rng.randrange(self.seen)
                if j < self.cap:
                    self.items[j] = event
        return ids

    def state(self):
        return {"rng": self.rng.getstate(), "items": self.items, "seen": self.seen}

    def load(self, st):
        self.rng.setstate(st["rng"]); self.items = list(st["items"]); self.seen = st["seen"]


GPU = "L4"
CPU = 8
MEMORY_MB = 16384
TRAIN_BATCH = 512
INFER_BATCH = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


@app.function(image=image, gpu=GPU, timeout=25200, cpu=CPU, memory=MEMORY_MB, volumes={"/vol": volume})
def train_final(
    seeds: int = 3,
    epochs: int = 80,
    minutes: float = 390.0,
    ckpt_every: int = 5,
    train_batch: int = TRAIN_BATCH,
    infer_batch: int = INFER_BATCH,
):
    """Train TGNN_Q_A_C on the DEDALE 15:1 reservoir-subsampled dataset across `seeds`,
    each up to `epochs`, sharing a global `minutes` wall-clock budget. Periodic checkpoints
    make it preemption-safe."""
    import hashlib, json, time
    from pathlib import Path
    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score
    from windows_apt_models import make_model

    if train_batch <= 0 or infer_batch <= 0:
        raise ValueError("train_batch and infer_batch must be positive")
    if ckpt_every <= 0:
        raise ValueError("ckpt_every must be positive")
    if seeds <= 0 or epochs <= 0 or minutes <= 0:
        raise ValueError("seeds, epochs, and minutes must be positive")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    volume.reload()
    prep_path = Path(PREP)
    audit_path = prep_path.with_name("audit.json")
    if not prep_path.is_file():
        raise FileNotFoundError(
            f"Prepared dataset not found at {PREP}. "
            f"Expected Volume path tg-detect-dedale-v1/prepared/dedale_host/prepared.npz"
        )
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing audit file next to prepared dataset: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if audit.get("subsampling") != "reservoir_15x" or audit.get("neg_per_pos_requested") != 15:
        raise RuntimeError(
            "This TGNN script expects the 15:1 DEDALE build; "
            f"audit reports subsampling={audit.get('subsampling')!r}, "
            f"neg_per_pos_requested={audit.get('neg_per_pos_requested')!r}"
        )
    d = np.load(PREP)
    xg = torch.from_numpy(d["x"]).cuda()
    windows = xg[torch.from_numpy(d["ego"]).cuda()]; del xg
    valid = torch.from_numpy(d["valid"]).cuda(); ages = torch.from_numpy(d["ages"]).cuda()
    y_np = d["y"].astype(np.int64); y = torch.from_numpy(y_np).cuda()
    tr = int(d["train_end"]); va = int(d["val_end"]); n = len(y_np); input_dim = int(windows.shape[-1])
    ids_all = d["ids"] if "ids" in d.files else None
    out = Path(RESULTS); out.mkdir(parents=True, exist_ok=True)

    pos_tr = int(y_np[:tr].sum()); neg_tr = tr - pos_tr
    w = torch.tensor([tr / (2.0 * max(neg_tr, 1)), tr / (2.0 * max(pos_tr, 1))],
                     dtype=torch.float32, device="cuda")
    sig = (f"{ARM}|n={n}|tr={tr}|va={va}|epochs={epochs}|train_batch={train_batch}|"
           f"infer_batch={infer_batch}|lr={LEARNING_RATE}|wd={WEIGHT_DECAY}|replay=R_v1")
    print(f"DEDALE 15:1 final train: n={n} train={tr} (pos={pos_tr}, {neg_tr/max(pos_tr,1):.1f}:1) "
          f"val={va-tr} test={n-va} input_dim={input_dim} class_weight={w.tolist()} "
          f"seeds={seeds} epochs={epochs} batch={train_batch} infer_batch={infer_batch} "
          f"budget={minutes}m cpu={CPU} mem={MEMORY_MB//1024}GiB gpu={torch.cuda.get_device_name()}", flush=True)

    def forward(model, ix):
        return model.forward_events(windows[ix], valid[ix], ages[ix], None, None, None)[0]

    def probs(model, lo, hi):
        model.eval(); out_p = []
        with torch.no_grad():
            for b in range(lo, hi, infer_batch):
                out_p.append(forward(model, slice(b, min(b + infer_batch, hi))).softmax(-1)[:, 1].float().cpu().numpy())
        return np.concatenate(out_p) if out_p else np.zeros(0, dtype=np.float32)

    start = time.monotonic(); results = []
    for seed in range(seeds):
        stem = f"{ARM}_seed{seed}"
        result_path = out / f"{stem}.json"
        if result_path.exists():
            try:
                old = json.loads(result_path.read_text())
                if old.get("run_signature") == sig and old.get("status") == "complete":
                    results.append(old)
                    print(f"{stem}: matching completed result found; skipping", flush=True)
                    continue
            except Exception:
                pass
        ckpt_path = out / f"{stem}.ckpt.pt"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        import random
        random.seed(seed)
        model = make_model(ARM, input_dim).cuda()
        opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        replay = ReplayBuffer(tr, seed)
        start_epoch = 0; best_ap = -1.0; best_state = None; losses = []; seed_seconds = 0.0
        if ckpt_path.exists():
            try:
                ck = torch.load(ckpt_path, map_location="cuda")
                if ck.get("sig") == sig and ck.get("seed") == seed:
                    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                    replay.load(ck["replay"]); start_epoch = ck["epoch"] + 1
                    best_ap = ck["best_ap"]; best_state = ck["best_state"]; losses = ck["losses"]
                    seed_seconds = ck.get("seed_seconds", 0.0)
                    torch.set_rng_state(ck["torch_rng"].cpu())
                    torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
                    np.random.set_state(ck["np_rng"])
                    print(f"{stem}: resume at epoch {start_epoch}/{epochs} best_val_pr_ap={best_ap:.4f}", flush=True)
                else:
                    print(f"{stem}: ckpt sig mismatch; fresh start", flush=True)
            except Exception as ex:
                print(f"{stem}: ckpt load failed ({ex}); fresh start", flush=True)

        def save_ckpt(epoch):
            tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save({"sig": sig, "seed": seed, "epoch": epoch, "best_ap": best_ap,
                        "best_state": best_state, "losses": losses, "seed_seconds": seed_seconds,
                        "model": model.state_dict(), "opt": opt.state_dict(),
                        "replay": replay.state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(), "np_rng": np.random.get_state()}, tmp)
            tmp.replace(ckpt_path); volume.commit()

        torch.cuda.reset_peak_memory_stats()
        stopped = False
        for epoch in range(start_epoch, epochs):
            if (time.monotonic() - start) / 60.0 > minutes:
                print(f"{stem}: global time budget {minutes}m reached before epoch {epoch}", flush=True)
                stopped = True; break
            t0 = time.monotonic(); model.train(); running = torch.zeros((), device="cuda")
            for lo in range(0, tr, train_batch):
                hi = min(lo + train_batch, tr); opt.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(forward(model, slice(lo, hi)), y[lo:hi], weight=w)
                rep = replay.sample_then_admit(lo, hi)
                if rep:
                    ix = torch.as_tensor(rep, device="cuda")
                    loss = loss + torch.nn.functional.cross_entropy(forward(model, ix), y[ix], weight=w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step(); running = running + loss.detach()
            total = float(running.item()); losses.append(total); seed_seconds += time.monotonic() - t0
            if not np.isfinite(total):
                raise RuntimeError(f"{stem}: non-finite epoch loss at epoch {epoch}")
            if (epoch + 1) % ckpt_every == 0 or epoch == epochs - 1:
                ap = float(average_precision_score(y_np[tr:va], probs(model, tr, va)))
                if ap > best_ap:
                    best_ap = ap
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                print(f"{stem} epoch {epoch+1}/{epochs} loss {total:.3f} val_pr_ap {ap:.4f} "
                      f"best {best_ap:.4f} [{(time.monotonic()-start)/60.0:.1f}m]", flush=True)
                save_ckpt(epoch)

        if best_state is None:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_ap = float(average_precision_score(y_np[tr:va], probs(model, tr, va)))
        model.load_state_dict({k: v.cuda() for k, v in best_state.items()})
        p = probs(model, 0, n)
        t_val = _select_threshold(y_np[tr:va], p[tr:va]); t_oracle = _select_threshold(y_np[va:], p[va:])
        ts, f1s = _f1_curve(y_np[va:], p[va:])
        torch.save({"state_dict": best_state, "arm": ARM, "input_dim": input_dim, "seed": seed,
                    "epochs_trained": len(losses), "best_val_pr_ap": best_ap}, out / f"{stem}.pt")
        np.savez_compressed(out / f"{stem}_predictions.npz", probability=p, y=y_np,
                            train_end=tr, val_end=va, ids=ids_all)
        result = {"run_signature": sig, "status": "complete", "arm": ARM, "seed": seed,
                  "epochs_trained": len(losses), "epochs_target": epochs, "train_batch": train_batch,
                  "infer_batch": infer_batch, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                  "time_boxed": stopped, "train_seconds": seed_seconds, "best_val_pr_ap": best_ap,
                  "val_threshold": t_val, "oracle_threshold": t_oracle,
                  "test_default_0.5": _metrics(y_np[va:], p[va:], 0.5),
                  "test_val_threshold": _metrics(y_np[va:], p[va:], t_val),
                  "test_oracle_threshold": _metrics(y_np[va:], p[va:], t_oracle),
                  "validation_at_val_threshold": _metrics(y_np[tr:va], p[tr:va], t_val),
                  "f1_curve": {"thresholds": ts, "f1": f1s}, "epoch_losses": losses,
                  "class_weight": w.tolist(), "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                  "checkpoint_sha256": hashlib.sha256((out / f"{stem}.pt").read_bytes()).hexdigest()}
        (out / f"{stem}.json").write_text(json.dumps(result, indent=2)); results.append(result)
        if ckpt_path.exists():
            ckpt_path.unlink()
        volume.commit(); del model, opt; torch.cuda.empty_cache()
        if stopped:
            print(f"stopping after seed {seed}: global budget reached", flush=True); break

    def agg(key, sub):
        vals = [r[sub][key] for r in results]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [round(v, 6) for v in vals]}
    best_seed = max(results, key=lambda r: r["best_val_pr_ap"]) if results else None
    if best_seed is not None:
        import shutil
        shutil.copy(out / f"{ARM}_seed{best_seed['seed']}.pt", out / f"{ARM}_final.pt")
    summary = {"run_signature": sig, "arm": ARM, "dataset": "dedale_host (reservoir_15x)",
               "seeds_done": len(results), "input_dim": input_dim, "cpu": CPU, "memory_gib": MEMORY_MB // 1024,
               "train_batch": train_batch, "infer_batch": infer_batch, "learning_rate": LEARNING_RATE,
               "weight_decay": WEIGHT_DECAY,
               "gpu": torch.cuda.get_device_name(),
               "subsampling": audit.get("subsampling"),
               "neg_per_pos_requested": audit.get("neg_per_pos_requested"),
               "positives_total": audit.get("positives_matched"),
               "negatives_kept": audit.get("negatives_kept"),
               "partitions": {"train_end": tr, "val_end": va, "n": n},
               "positives": {"train": int(y_np[:tr].sum()), "val": int(y_np[tr:va].sum()), "test": int(y_np[va:].sum())},
               "imbalance_train": round(neg_tr / max(pos_tr, 1), 2), "class_weight": w.tolist(),
               "final_model_seed": (best_seed["seed"] if best_seed else None),
               "final_model_file": f"{ARM}_final.pt", "model_selection": "max validation PR-AP",
               "metrics": {v: {k: agg(k, v) for k in ("f1", "precision", "recall", "fpr", "pr_ap", "roc_auc")}
                           for v in ("test_default_0.5", "test_val_threshold", "test_oracle_threshold")}}
    (out / "summary_final.json").write_text(json.dumps(summary, indent=2)); volume.commit()
    return summary


def spawn_deployed(seeds: int = 3, epochs: int = 80, minutes: float = 390.0,
                   train_batch: int = TRAIN_BATCH, infer_batch: int = INFER_BATCH,
                   ckpt_every: int = 5):
    """Submit training to the already-deployed App and exit immediately.

    The deployed FunctionCall is independent of this local process, so the terminal
    can be closed without cancelling the training input.
    """
    fn = modal.Function.from_name("tg-detect-dedale-final", "train_final")
    call = fn.spawn(
        seeds=seeds, epochs=epochs, minutes=minutes, ckpt_every=ckpt_every,
        train_batch=train_batch, infer_batch=infer_batch,
    )
    print(f"function_call_id={call.object_id}")
    print("Logs: modal function logs tg-detect-dedale-final/train_final -f")
    print("Calls: modal function calls tg-detect-dedale-final/train_final --show-function-call-id --tail 10")
    return call.object_id


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Spawn the deployed TGNN training job.")
    ap.add_argument("--spawn", action="store_true", help="spawn the already-deployed train_final function and exit")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--minutes", type=float, default=390.0)
    ap.add_argument("--train-batch", type=int, default=TRAIN_BATCH)
    ap.add_argument("--infer-batch", type=int, default=INFER_BATCH)
    ap.add_argument("--ckpt-every", type=int, default=5)
    args = ap.parse_args()
    if args.spawn:
        spawn_deployed(args.seeds, args.epochs, args.minutes, args.train_batch, args.infer_batch, args.ckpt_every)
