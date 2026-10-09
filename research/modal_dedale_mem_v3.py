"""V3 memory fine-tuning on the existing DEDALE host dataset.

Each seed transfers its trained QAC encoder/head, adds separate endpoint reads,
64-dimensional fast/slow states, exact pre-event prefix scans and corrected soft
admission penalties. The initial zero-read-gate model is a validation-selectable
fallback. Results live in a new directory and include warm-start provenance.
This is additional fine-tuning, not a matched-budget baseline comparison.

Deploy: python3 -m modal deploy modal_dedale_mem_v3.py
Spawn: python3 modal_dedale_mem_v3.py --spawn --seeds 3 --mem-epochs 80
"""
import modal

app = modal.App("tg-detect-dedale-mem-v3")
volume = modal.Volume.from_name("tg-detect-dedale-v1", create_if_missing=True)
SRC = ("windows_apt_models", "tgnn_components", "causal_tgn", "temporal_cnn",
       "pomsathit_baseline", "causal_data", "node_memory_models", "node_memory_v3", "test_node_memory_v3")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.4.1", "numpy==1.26.4", "scikit-learn==1.5.2", "pytest")
         .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
         .add_local_python_source(*SRC))

PREP = "/vol/prepared/dedale_host_mem/prepared.npz"
RESULTS = "/vol/results/dedale_host_mem_v3"
ARMS_DEFAULT = ("TGNN_Node_Mem_V3",)
SEL_ARMS = ("TGNN_Node_Mem_V3", "TGNN_Node_Mem_V3_Fixed")
MEM_ARMS = ("TGNN_Node_Mem", "TGNN_Node_Mem_Fixed")   # V1 arms (not in the param-matched sweep)
GPU = "L4"; CPU = 8; MEMORY_MB = 16384
TRAIN_BATCH = 512; INFER_BATCH = 512
LEARNING_RATE = 1e-4; WEIGHT_DECAY = 1e-5
CTRL_LR = 1e-4
R_TARGET = 0.25          # V1 budget anchor (kept for back-compat)
R_SEL_TARGET = 0.20      # V2 target kept-FRACTION of candidates (centre of [0.10, 0.30])
LAMBDA_BUDGET = 0.05
LAMBDA_REDUNDANCY = 0.01
MEM_CHUNK = 128          # vectorised batch-granular memory chunk (v1 efficiency)
GAT_LAYERS = 2           # BOTH TGNNs share this depth so they match parameter-for-parameter
TBPTT_K = 3              # Prefix reads also supply within-batch task gradients.
ARCH_VERSION = "v3-prefix-dualscale-endpoint64-20261002"


def _select_threshold(y, p):
    """Threshold that maximises F1 on (y, p); ties -> largest threshold."""
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
    """Global reservoir replay, policy-identical to the incumbent run (Algorithm R, capacity
    train_end//10, RNG seed+937, sample<=26 before admitting the current batch). This is the
    third, slowest time scale and is deliberately independent of per-entity node memory."""
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


@app.function(image=image, gpu=GPU, timeout=25200, cpu=CPU, memory=MEMORY_MB, volumes={"/vol": volume})
def train_all(seeds: int = 3, epochs: int = 80, mem_epochs: int = 80, minutes: float = 600.0,
              ckpt_every: int = 5, train_batch: int = TRAIN_BATCH, infer_batch: int = INFER_BATCH,
              arms: str = ",".join(ARMS_DEFAULT), lat_events: int = 3000):
    """Train every arm across `seeds`, select on val PR-AP, report threshold-honest test metrics,
    and finally measure a latency matrix. Shares a global `minutes` wall-clock budget; periodic
    checkpoints make each (arm, seed) preemption-safe."""
    import hashlib, json, time, random
    from pathlib import Path
    import numpy as np
    import torch
    from torch.nn import functional as F
    from sklearn.metrics import average_precision_score
    from windows_apt_models import make_model
    from node_memory_v3 import make_mem_model, admission_penalties

    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    volume.reload()
    import pytest, test_node_memory_v3
    import os
    os.environ['TGNN_TEST_DEVICE'] = 'cuda'
    if pytest.main([test_node_memory_v3.__file__, '-q']) != 0:
        raise RuntimeError('V3 GPU acceptance tests failed; training not started')
    prep = Path(PREP)
    if not prep.is_file():
        raise FileNotFoundError(f"Prepared dataset not found at {PREP}; build dedale_host_mem first "
                                f"(python dedale_data.py --spawn --neg-per-pos 15 --out-name dedale_host_mem)")
    audit = json.loads(prep.with_name("audit.json").read_text())
    d = np.load(PREP)
    for k in ("src", "dst", "times"):
        if k not in d.files:
            raise RuntimeError(f"prepared.npz lacks '{k}'; rebuild dedale_host_mem with the entity-id save")
    xg = torch.from_numpy(d["x"]).cuda()
    windows = xg[torch.from_numpy(d["ego"]).cuda()]; del xg
    valid = torch.from_numpy(d["valid"]).cuda(); ages = torch.from_numpy(d["ages"]).cuda()
    y_np = d["y"].astype(np.int64); y = torch.from_numpy(y_np).cuda()
    src_np = d["src"].astype(np.int64); dst_np = d["dst"].astype(np.int64); times_np = d["times"].astype(np.float64)
    tr = int(d["train_end"]); va = int(d["val_end"]); n = len(y_np); input_dim = int(windows.shape[-1])
    ids_all = d["ids"] if "ids" in d.files else None
    n_nodes = int(d["n_nodes"]) if "n_nodes" in d.files else (int(max(src_np.max(), dst_np.max())) + 1)
    out = Path(RESULTS); out.mkdir(parents=True, exist_ok=True)
    arm_list = [a for a in arms.split(",") if a]
    pos_tr = int(y_np[:tr].sum()); neg_tr = tr - pos_tr
    w = torch.tensor([tr / (2.0 * max(neg_tr, 1)), tr / (2.0 * max(pos_tr, 1))], dtype=torch.float32, device="cuda")
    print(f"DEDALE node-mem retest: n={n} train={tr}(pos={pos_tr},{neg_tr/max(pos_tr,1):.1f}:1) val={va-tr} "
          f"test={n-va} input_dim={input_dim} nodes={int(d['n_nodes']) if 'n_nodes' in d.files else '?'} "
          f"arms={arm_list} seeds={seeds} epochs={epochs} mem_epochs={mem_epochs} "
          f"gpu={torch.cuda.get_device_name()}", flush=True)

    def make(arm):
        """Construct the V3 model or an optional memory-free control."""
        if arm == "GNN_H":
            return make_model(arm, input_dim)
        return make_mem_model(arm, input_dim, n_nodes, mem_chunk=MEM_CHUNK, gat_layers=GAT_LAYERS)

    def probs_window(model, lo, hi):
        """Memory-free batched probabilities (GNN_H / TGNN_Q_A_C)."""
        model.eval(); outp = []
        with torch.no_grad():
            for b in range(lo, hi, infer_batch):
                e = min(b + infer_batch, hi)
                lg = model.forward_events(windows[b:e], valid[b:e], ages[b:e], None, None, None)[0]
                outp.append(lg.softmax(-1)[:, 1].float().cpu().numpy())
        return np.concatenate(outp) if outp else np.zeros(0, dtype=np.float32)

    def probs_stream(model, hi_eval):
        """Streaming probabilities over the chronological prefix 0..hi_eval WITH node memory
        (causally warm); detach between chunks so evaluation uses O(1) memory."""
        model.eval(); model.reset_state(); outp = []
        with torch.no_grad():
            for b in range(0, hi_eval, infer_batch):
                e = min(b + infer_batch, hi_eval)
                lg = model.forward_events(windows[b:e], valid[b:e], ages[b:e],
                                          src_np[b:e], dst_np[b:e], times_np[b:e])[0]
                outp.append(lg.softmax(-1)[:, 1].float().cpu().numpy())
                model.detach_state()
        return np.concatenate(outp)

    def val_ap(model, arm):
        p = probs_stream(model, va)[tr:va] if arm in SEL_ARMS else probs_window(model, tr, va)
        return float(average_precision_score(y_np[tr:va], p))

    def train_nonmem_epoch(model, opt, replay):
        model.train(); running = torch.zeros((), device="cuda")
        for lo in range(0, tr, train_batch):
            hi = min(lo + train_batch, tr); opt.zero_grad(set_to_none=True)
            lg = model.forward_events(windows[lo:hi], valid[lo:hi], ages[lo:hi], None, None, None)[0]
            loss = F.cross_entropy(lg, y[lo:hi], weight=w)
            rep = replay.sample_then_admit(lo, hi)
            if rep:
                ix = torch.as_tensor(rep, device="cuda")
                rl = model.forward_events(windows[ix], valid[ix], ages[ix], None, None, None)[0]
                loss = loss + F.cross_entropy(rl, y[ix], weight=w)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); running = running + loss.detach()
        return float(running.item())

    def train_mem_epoch(model, opt, opt_ctrl, replay, adaptive):
        """Streaming chronological training with per-entity memory; truncated BPTT via
        detach_state between batches. Two parameter groups: main (task + replay) and, when
        adaptive, controller (shaped by the budget + redundancy regularizers and the
        future-prediction signal that flows task -> gate -> later reads)."""
        model.train(); model.reset_state(); running = torch.zeros((), device="cuda")
        for lo in range(0, tr, train_batch):
            hi = min(lo + train_batch, tr)
            opt.zero_grad(set_to_none=True)
            if opt_ctrl is not None:
                opt_ctrl.zero_grad(set_to_none=True)
            lg, aux = model.forward_events(windows[lo:hi], valid[lo:hi], ages[lo:hi],
                                           src_np[lo:hi], dst_np[lo:hi], times_np[lo:hi])
            task = F.cross_entropy(lg, y[lo:hi], weight=w)
            loss = task
            rep = replay.sample_then_admit(lo, hi)
            if rep:
                ix = torch.as_tensor(rep, device="cuda")
                loss = loss + F.cross_entropy(model.replay_logits(windows[ix], valid[ix], ages[ix]),
                                              y[ix], weight=w)
            if adaptive:
                g = aux["gate"]; nov = aux["novelty"]
                budget = (g.mean() - R_TARGET) ** 2
                redundancy = (g * (1 - nov).clamp(min=0)).mean()
                loss = loss + LAMBDA_BUDGET * budget + LAMBDA_REDUNDANCY * redundancy
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if opt_ctrl is not None:
                opt_ctrl.step()
            model.detach_state(); running = running + task.detach()
        return float(running.item())

    def train_sel_epoch(model, opt, opt_ctrl, replay):
        """Streaming training for the SELECTIVE-admission arm (V2).  Truncated BPTT over TBPTT_K
        batches: the memory graph is kept un-detached across the window so a batch's gated writes
        feed later-batch reads, giving the controller (phi) a genuine FUTURE-utility gradient
        (L_future), on top of the per-batch budget (admitted fraction -> r_t) and redundancy
        (do not spend writes on low-novelty events).  theta learns on task + replay over the same
        window; the two optimisers step together at the end of each K-batch window."""
        model.train(); model.reset_state(); running = torch.zeros((), device="cuda")
        opt.zero_grad(set_to_none=True); opt_ctrl.zero_grad(set_to_none=True)
        win = torch.zeros((), device="cuda"); k = 0
        starts = list(range(0, tr, train_batch))
        for bi, lo in enumerate(starts):
            hi = min(lo + train_batch, tr)
            lg, aux = model.forward_events(windows[lo:hi], valid[lo:hi], ages[lo:hi],
                                           src_np[lo:hi], dst_np[lo:hi], times_np[lo:hi])
            task = F.cross_entropy(lg, y[lo:hi], weight=w)
            loss = task
            rep = replay.sample_then_admit(lo, hi)
            if rep:
                ix = torch.as_tensor(rep, device="cuda")
                loss = loss + F.cross_entropy(model.replay_logits(windows[ix], valid[ix], ages[ix]),
                                              y[ix], weight=w)
            if model.adaptive:
                budget, redundancy = admission_penalties(aux, R_SEL_TARGET)
                loss = loss + LAMBDA_BUDGET * budget + LAMBDA_REDUNDANCY * redundancy
            win = win + loss; k += 1; running = running + task.detach()
            if k == TBPTT_K or bi == len(starts) - 1:
                (win / k).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                diagnostics = {name: float(p.grad.norm().item()) for name, p in model.named_parameters()
                               if p.grad is not None and name.startswith(('write_src', 'write_dst', 'read_gate', 'controller'))}
                model.training_diagnostics = {"gradient_norms": diagnostics,
                    "mean_admission": float(aux['admit'].detach().mean().item()),
                    "mean_read_gate_abs": float(aux['read_gate'].detach().abs().mean().item()),
                    "mean_fast_weight": float(aux['effective_fast_weight'].detach().mean().item()),
                    "mean_slow_weight": float(aux['effective_slow_weight'].detach().mean().item())}
                opt.step(); opt_ctrl.step()
                opt.zero_grad(set_to_none=True); opt_ctrl.zero_grad(set_to_none=True)
                model.detach_state(); win = torch.zeros((), device="cuda"); k = 0
        return float(running.item())

    def measure_latency(model, is_mem):
        """Latency matrix row: online (batch=1 streaming) p50/p95/p99 ms, batched throughput
        (events/s), params, peak inference GPU MB. Memory arms are warmed on 0..va first."""
        import numpy as _np
        model.eval(); torch.cuda.synchronize()
        lo = va; hi = min(va + lat_events, n)
        if is_mem:
            with torch.no_grad():
                model.reset_state()
                for b in range(0, va, infer_batch):
                    e = min(b + infer_batch, va)
                    model.forward_events(windows[b:e], valid[b:e], ages[b:e],
                                         src_np[b:e], dst_np[b:e], times_np[b:e])
                    model.detach_state()
        per_event = []
        with torch.no_grad():
            for i in range(lo, hi):
                sl = slice(i, i + 1); torch.cuda.synchronize(); t0 = time.perf_counter()
                if is_mem:
                    model.forward_events(windows[sl], valid[sl], ages[sl], src_np[sl], dst_np[sl], times_np[sl])
                    model.detach_state()
                else:
                    model.forward_events(windows[sl], valid[sl], ages[sl], None, None, None)
                torch.cuda.synchronize(); per_event.append((time.perf_counter() - t0) * 1e3)
        per_event = _np.array(per_event[200:] if len(per_event) > 400 else per_event)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
        if is_mem:
            probs_stream(model, n)
        else:
            probs_window(model, 0, n)
        torch.cuda.synchronize(); batched_s = time.perf_counter() - t0
        return {"params": int(sum(p.numel() for p in model.parameters())),
                "online_batch1_ms_mean": float(per_event.mean()), "online_batch1_ms_p50": float(_np.percentile(per_event, 50)),
                "online_batch1_ms_p95": float(_np.percentile(per_event, 95)), "online_batch1_ms_p99": float(_np.percentile(per_event, 99)),
                "batched_events_per_s": float(n / batched_s), "peak_infer_gpu_mb": float(torch.cuda.max_memory_allocated() / 1e6),
                "lat_events_online": int(hi - lo)}

    start = time.monotonic(); all_arms = {}; latency = {}; stopped_global = False
    for arm in arm_list:
        is_mem = arm in SEL_ARMS; adaptive = arm == 'TGNN_Node_Mem_V3'
        ep = mem_epochs if is_mem else epochs
        sig = (f"{ARCH_VERSION}|{arm}|n={n}|tr={tr}|va={va}|ep={ep}|tb={train_batch}|ib={infer_batch}"
               f"|lr={LEARNING_RATE}|wd={WEIGHT_DECAY}|replay=R_v1")
        if is_mem:
            sig += (f"|ctrl=clr{CTRL_LR},rt{R_SEL_TARGET},lb{LAMBDA_BUDGET},lrr{LAMBDA_REDUNDANCY},k{TBPTT_K}"
                    f"|candidate=fixed,nov0.5,r[0.10,0.30],gat{GAT_LAYERS}")
        elif arm == "TGNN_Q_A_C":   # param-matched 2-GAT baseline (memory arch, memory bypassed) -> re-trains
            sig += f"|arch=memsel_nomem,gat{GAT_LAYERS}"
        arm_results = []
        for seed in range(seeds):
            stem = f"{arm}_seed{seed}"; result_path = out / f"{stem}.json"
            if result_path.exists():
                try:
                    old = json.loads(result_path.read_text())
                    if old.get("run_signature") == sig and old.get("status") == "complete":
                        arm_results.append(old); print(f"{stem}: cached complete; skip", flush=True); continue
                except Exception:
                    pass
            ckpt_path = out / f"{stem}.ckpt.pt"
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); np.random.seed(seed); random.seed(seed)
            model = make(arm).cuda()
            warm_path = Path('/vol/results/dedale_host_mem') / f'TGNN_Q_A_C_seed{seed}.pt'
            if not warm_path.is_file():
                raise FileNotFoundError(f'Missing same-seed QAC warm start: {warm_path}')
            warm = torch.load(warm_path, map_location='cpu')
            current = model.state_dict()
            transferred = {key: value for key, value in warm['state_dict'].items()
                           if key in current and current[key].shape == value.shape
                           and not key.startswith(('controller.', 'attn_score.'))}
            required = [key for key in current if key.startswith(('layers.', 'cnn.', 'head.', 'mem_norm.',
                        'time_encoder.', 'history_gru.', 'query.', 'key.', 'value.'))]
            if any(key not in transferred for key in required):
                raise RuntimeError('Incomplete encoder/head warm start')
            model.load_state_dict(transferred, strict=False)
            warm_sha = hashlib.sha256(warm_path.read_bytes()).hexdigest()
            if is_mem:
                opt = torch.optim.Adam(model.main_parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
                opt_ctrl = torch.optim.Adam(model.controller_parameters(), lr=CTRL_LR, weight_decay=WEIGHT_DECAY)
            else:
                opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY); opt_ctrl = None
            replay = ReplayBuffer(tr, seed)
            start_epoch = 0; losses = []; seed_seconds = 0.0
            best_ap = val_ap(model, arm)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(f'{stem}: QAC warm start val_ap={best_ap:.6f}; transferred={len(transferred)}', flush=True)
            if ckpt_path.exists():
                try:
                    ck = torch.load(ckpt_path, map_location="cuda")
                    if ck.get("sig") == sig and ck.get("seed") == seed:
                        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
                        if opt_ctrl is not None and ck.get("opt_ctrl") is not None:
                            opt_ctrl.load_state_dict(ck["opt_ctrl"])
                        replay.load(ck["replay"]); start_epoch = ck["epoch"] + 1
                        best_ap = ck["best_ap"]; best_state = ck["best_state"]; losses = ck["losses"]
                        seed_seconds = ck.get("seed_seconds", 0.0)
                        torch.set_rng_state(ck["torch_rng"].cpu())
                        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]]); np.random.set_state(ck["np_rng"])
                        print(f"{stem}: resume epoch {start_epoch}/{ep} best_val_pr_ap={best_ap:.4f}", flush=True)
                    else:
                        print(f"{stem}: ckpt sig mismatch; fresh start", flush=True)
                except Exception as ex:
                    print(f"{stem}: ckpt load failed ({ex}); fresh start", flush=True)

            def save_ckpt(epoch):
                tmp = ckpt_path.with_suffix(".pt.tmp")
                torch.save({"sig": sig, "seed": seed, "epoch": epoch, "best_ap": best_ap, "best_state": best_state,
                            "losses": losses, "seed_seconds": seed_seconds, "model": model.state_dict(),
                            "opt": opt.state_dict(), "opt_ctrl": (opt_ctrl.state_dict() if opt_ctrl is not None else None),
                            "replay": replay.state(), "torch_rng": torch.get_rng_state(),
                            "cuda_rng": torch.cuda.get_rng_state_all(), "np_rng": np.random.get_state()}, tmp)
                tmp.replace(ckpt_path); volume.commit()

            torch.cuda.reset_peak_memory_stats(); stopped = False
            for epoch in range(start_epoch, ep):
                if (time.monotonic() - start) / 60.0 > minutes:
                    print(f"{stem}: time budget {minutes}m reached before epoch {epoch}", flush=True)
                    stopped = True; stopped_global = True; break
                t0 = time.monotonic()
                total = (train_sel_epoch(model, opt, opt_ctrl, replay) if is_mem
                         else train_nonmem_epoch(model, opt, replay))
                losses.append(total); seed_seconds += time.monotonic() - t0
                progress = {'architecture': ARCH_VERSION, 'arm': arm, 'seed': seed,
                            'epoch': epoch + 1, 'epochs_target': ep, 'loss': total,
                            'training_diagnostics': getattr(model, 'training_diagnostics', {}),
                            'train_seconds': seed_seconds, 'warm_start_sha256': warm_sha}
                (out / f'{stem}_progress.json').write_text(json.dumps(progress, indent=2))
                volume.commit()
                print(f'{stem} epoch {epoch+1}/{ep} loss={total:.4f} seconds={seed_seconds:.1f}', flush=True)
                if not np.isfinite(total):
                    raise RuntimeError(f"{stem}: non-finite epoch loss at epoch {epoch}")
                if (epoch + 1) % ckpt_every == 0 or epoch == ep - 1:
                    ap = val_ap(model, arm)
                    if ap > best_ap:
                        best_ap = ap; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    print(f"{stem} ep {epoch+1}/{ep} loss {total:.3f} val_pr_ap {ap:.4f} best {best_ap:.4f} "
                          f"[{(time.monotonic()-start)/60.0:.1f}m]", flush=True)
                    save_ckpt(epoch)

            if best_state is None:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_ap = val_ap(model, arm)
            model.load_state_dict({k: v.cuda() for k, v in best_state.items()})
            p = probs_stream(model, n) if is_mem else probs_window(model, 0, n)
            t_val = _select_threshold(y_np[tr:va], p[tr:va]); t_oracle = _select_threshold(y_np[va:], p[va:])
            ts, f1s = _f1_curve(y_np[va:], p[va:])
            torch.save({"state_dict": best_state, "arm": arm, "input_dim": input_dim, "seed": seed,
                        "epochs_trained": len(losses), "best_val_pr_ap": best_ap}, out / f"{stem}.pt")
            np.savez_compressed(out / f"{stem}_predictions.npz", probability=p, y=y_np, train_end=tr, val_end=va, ids=ids_all)
            result = {"run_signature": sig, "status": "complete", "arm": arm, "seed": seed, "is_mem": is_mem,
                      "architecture": ARCH_VERSION, "warm_start_path": str(warm_path),
                      "warm_start_sha256": warm_sha, "warm_start_epochs": warm.get('epochs_trained'),
                      "training_protocol": "same-seed QAC warm start plus additional memory fine-tuning",
                      "active_parameters": sum(p.numel() for p in model.parameters()),
                      "optimizer_steps": len(losses) * ((int(np.ceil(tr / train_batch)) + TBPTT_K - 1) // TBPTT_K),
                      "training_diagnostics": getattr(model, 'training_diagnostics', {}),
                      "adaptive": adaptive, "epochs_trained": len(losses), "epochs_target": ep, "train_batch": train_batch,
                      "infer_batch": infer_batch, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                      "time_boxed": stopped, "train_seconds": seed_seconds, "best_val_pr_ap": best_ap,
                      "val_threshold": t_val, "oracle_threshold": t_oracle,
                      "test_default_0.5": _metrics(y_np[va:], p[va:], 0.5), "test_val_threshold": _metrics(y_np[va:], p[va:], t_val),
                      "test_oracle_threshold": _metrics(y_np[va:], p[va:], t_oracle),
                      "validation_at_val_threshold": _metrics(y_np[tr:va], p[tr:va], t_val),
                      "f1_curve": {"thresholds": ts, "f1": f1s}, "epoch_losses": losses, "class_weight": w.tolist(),
                      "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
                      "checkpoint_sha256": hashlib.sha256((out / f"{stem}.pt").read_bytes()).hexdigest()}
            (out / f"{stem}.json").write_text(json.dumps(result, indent=2)); arm_results.append(result)
            if ckpt_path.exists():
                ckpt_path.unlink()
            volume.commit(); del model, opt; opt_ctrl = None; torch.cuda.empty_cache()
            if stopped:
                print(f"{stem}: stopping (global budget reached)", flush=True); break
        all_arms[arm] = arm_results
        if arm_results and not stopped_global:
            best = max(arm_results, key=lambda r: r["best_val_pr_ap"])
            lm = make(arm).cuda(); ckf = torch.load(out / f"{arm}_seed{best['seed']}.pt", map_location="cuda")
            lm.load_state_dict({k: v.cuda() for k, v in ckf["state_dict"].items()})
            latency[arm] = measure_latency(lm, is_mem); del lm; torch.cuda.empty_cache()
            print(f"{arm} latency {latency[arm]}", flush=True)
            import shutil; shutil.copy(out / f"{arm}_seed{best['seed']}.pt", out / f"{arm}_final.pt")
        if stopped_global:
            print("global budget reached; stopping arm sweep", flush=True); break

    def agg(results, key, sub):
        vals = [r[sub][key] for r in results]
        return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "values": [round(v, 6) for v in vals]}
    thresholds = ("test_default_0.5", "test_val_threshold", "test_oracle_threshold")
    arms_summary = {}
    for arm, res in all_arms.items():
        if not res:
            continue
        vap = [r["best_val_pr_ap"] for r in res]
        arms_summary[arm] = {
            "seeds_done": len(res), "is_mem": res[0]["is_mem"], "adaptive": res[0]["adaptive"],
            "epochs_target": res[0]["epochs_target"], "mean_train_seconds": float(np.mean([r["train_seconds"] for r in res])),
            "best_val_pr_ap": {"mean": float(np.mean(vap)), "sd": float(np.std(vap, ddof=1)) if len(vap) > 1 else 0.0,
                               "values": [round(v, 6) for v in vap]},
            "metrics": {v: {k: agg(res, k, v) for k in ("f1", "precision", "recall", "fpr", "pr_ap", "roc_auc")}
                        for v in thresholds},
            "final_model_seed": max(res, key=lambda r: r["best_val_pr_ap"])["seed"],
            "final_model_file": f"{arm}_final.pt" if arm in latency else None}
    summary = {"dataset": "dedale_host (reservoir_15x, +entity ids src/dst)",
               "subsampling": audit.get("subsampling"), "neg_per_pos_requested": audit.get("neg_per_pos_requested"),
               "partitions": {"train_end": tr, "val_end": va, "n": n},
               "positives": {"train": int(y_np[:tr].sum()), "val": int(y_np[tr:va].sum()), "test": int(y_np[va:].sum())},
               "imbalance_train": round(neg_tr / max(pos_tr, 1), 2), "class_weight": w.tolist(),
               "cpu": CPU, "memory_gib": MEMORY_MB // 1024, "gpu": torch.cuda.get_device_name(),
               "seeds_requested": seeds, "train_batch": train_batch, "infer_batch": infer_batch,
               "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "model_selection": "max validation PR-AP",
               "architecture": ARCH_VERSION,
               "training_protocol": "QAC warm start plus fine-tuning; not a matched-budget comparison",
               "memory": {"endpoint_dim": 64, "read": "exact pre-event prefix",
                          "candidate_threshold": "learned fixed threshold", "scales": [.5, .005]},
               "controller": {"stages": "1=fixed learned threshold UNION novel; 2=retention; 3=dual-scale prefix EMA",
                              "nov_tau": 0.5, "r_min": 0.10, "r_max": 0.30,
                              "r_sel_target": R_SEL_TARGET, "tbptt_k": TBPTT_K, "ctrl_lr": CTRL_LR,
                              "lambda_budget": LAMBDA_BUDGET, "lambda_redundancy": LAMBDA_REDUNDANCY},
               "arms": arms_summary, "latency_matrix": latency, "stopped_global": stopped_global}
    (out / "summary_mem.json").write_text(json.dumps(summary, indent=2))
    (out / "latency_matrix.json").write_text(json.dumps(latency, indent=2))
    volume.commit()
    return summary


@app.function(image=image, cpu=1, memory=2048, timeout=25200, volumes={"/vol": volume})
def wait_and_train(seeds: int = 3, epochs: int = 80, mem_epochs: int = 80, minutes: float = 600.0,
                   ckpt_every: int = 5, train_batch: int = TRAIN_BATCH, infer_batch: int = INFER_BATCH,
                   arms: str = ",".join(ARMS_DEFAULT), lat_events: int = 3000, wait_minutes: float = 180.0):
    """Cheap CPU coordinator: poll the Volume until dedale_host_mem/prepared.npz (with entity ids)
    is ready, then invoke the GPU train_all. One server-side spawn launches the whole pipeline
    without burning GPU time while the data rebuild is still running."""
    import time
    from pathlib import Path
    import numpy as np
    deadline = time.monotonic() + wait_minutes * 60
    while True:
        volume.reload()
        p = Path(PREP)
        if p.is_file():
            try:
                if "src" in np.load(PREP).files and p.with_name("audit.json").is_file():
                    print(f"data ready at {PREP}; launching train_all", flush=True); break
            except Exception as ex:
                print(f"data present but not readable yet ({ex}); waiting", flush=True)
        if time.monotonic() > deadline:
            raise TimeoutError(f"dedale_host_mem not ready after {wait_minutes} min")
        time.sleep(60)
    call = train_all.spawn(seeds=seeds, epochs=epochs, mem_epochs=mem_epochs, minutes=minutes,
                           ckpt_every=ckpt_every, train_batch=train_batch, infer_batch=infer_batch,
                           arms=arms, lat_events=lat_events)
    print(f"spawned train_all function_call_id={call.object_id}", flush=True)
    return call.object_id


def spawn_deployed(seeds: int = 3, epochs: int = 80, mem_epochs: int = 80, minutes: float = 600.0,
                   ckpt_every: int = 5, train_batch: int = TRAIN_BATCH, infer_batch: int = INFER_BATCH,
                   arms: str = ",".join(ARMS_DEFAULT), lat_events: int = 3000, wait: bool = False):
    """Submit training to the already-deployed App and exit; the call survives terminal close.
    wait=True spawns the CPU coordinator that waits for the data rebuild, then trains on GPU."""
    name = "wait_and_train" if wait else "train_all"
    fn = modal.Function.from_name("tg-detect-dedale-mem-v3", name)
    kw = dict(seeds=seeds, epochs=epochs, mem_epochs=mem_epochs, minutes=minutes, ckpt_every=ckpt_every,
              train_batch=train_batch, infer_batch=infer_batch, arms=arms, lat_events=lat_events)
    call = fn.spawn(**kw)
    print(f"function_call_id={call.object_id} ({name})")
    print("Logs: modal app logs tg-detect-dedale-mem-v3")
    return call.object_id


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Spawn the deployed node-memory comparison job.")
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--mem-epochs", type=int, default=80)
    ap.add_argument("--minutes", type=float, default=600.0)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--train-batch", type=int, default=TRAIN_BATCH)
    ap.add_argument("--infer-batch", type=int, default=INFER_BATCH)
    ap.add_argument("--arms", default=",".join(ARMS_DEFAULT))
    ap.add_argument("--lat-events", type=int, default=3000)
    ap.add_argument("--wait", action="store_true",
                    help="spawn the CPU coordinator that waits for the data rebuild, then trains")
    args = ap.parse_args()
    if args.spawn:
        spawn_deployed(args.seeds, args.epochs, args.mem_epochs, args.minutes, args.ckpt_every,
                       args.train_batch, args.infer_batch, args.arms, args.lat_events, args.wait)
