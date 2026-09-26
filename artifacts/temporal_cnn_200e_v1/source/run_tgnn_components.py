"""Frozen 200-epoch component experiment; invoked by the Modal L4 wrapper."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch.nn import functional as F

from run_causal_pilot import metrics, select_threshold
from tgnn_components import ARMS, ComponentTGNN, EfficientReservoir

DATA_HASH = "cf4c0d3c9eba674db8297923f97b664d3de5c5dea412b7ebca444410949a80c6"
SOURCES = ["pomsathit_baseline.py", "causal_tgn.py", "causal_data.py", "run_causal_pilot.py", "report_tgnn_components.py",
           "tgnn_components.py", "test_causal_pilot.py", "test_tgnn_components.py",
           "run_tgnn_components.py", "modal_tgnn_components.py", "TGNN_COMPONENT_PROTOCOL.md", "REFERENCE_RESULTS.json"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def scored(y, p, threshold):
    result = metrics(y, p, threshold)
    result["accuracy"] = (result["tp"] + result["tn"]) / len(y)
    result["balanced_accuracy"] = .5 * (result["recall"] + 1 - result["fpr"])
    return result


class Data:
    def __init__(self, path, device):
        self.path = Path(path)
        if digest(path) != DATA_HASH:
            raise ValueError("Frozen dataset hash mismatch")
        a = np.load(path, allow_pickle=False)
        for name in ("x", "y", "ego", "valid", "ages"):
            setattr(self, name, torch.from_numpy(a[name]).to(device))
        for name in ("src", "dst", "times", "row_ids"):
            setattr(self, name, a[name])
        self.train_end, self.val_end, self.n = int(a["train_end"]), int(a["val_end"]), len(a["y"])
        self.labels = a["y"]
        assert (self.train_end, self.val_end, self.n) == (9830, 13107, 16384)
        assert [int(self.labels[s].sum()) for s in (slice(0,9830), slice(9830,13107), slice(13107,None))] == [91,25,25]

    def batch(self, ids):
        idx = ids if isinstance(ids, slice) else torch.as_tensor(ids, device=self.x.device)
        return (self.x[self.ego[idx]], self.valid[idx], self.ages[idx], self.src[ids], self.dst[ids], self.times[ids])


def verify_result(output, row, data):
    name = row["run_id"]
    saved = np.load(output / f"{name}_predictions.npz", allow_pickle=False)
    np.testing.assert_array_equal(saved["y"], data.labels)
    np.testing.assert_array_equal(saved["raw_row_id"], data.row_ids)
    val, test = slice(data.train_end, data.val_end), slice(data.val_end, data.n)
    threshold = select_threshold(saved["y"][val], saved["probability"][val])
    assert threshold == row["threshold"], name
    assert scored(saved["y"][val], saved["probability"][val], threshold) == row["validation"], name
    assert scored(saved["y"][test], saved["probability"][test], threshold) == row["test"], name
    assert digest(output / f"{name}.pt") == row["checkpoint_sha256"], name
    assert digest(output / f"{name}_predictions.npz") == row["predictions_sha256"], name
    assert row["epochs"] == 200 and len(row["epoch_losses"]) == 200


def summarize(results):
    fields = ("f1", "pr_ap", "precision", "recall", "fpr", "roc_auc", "accuracy", "balanced_accuracy")
    summary = {"completed_runs": len(results), "expected_runs": 60, "arms": {}}
    indexed = {(r["arm"], r["seed"]): r for r in results}
    for arm in ARMS:
        rows = [r for r in results if r["arm"] == arm]
        if not rows:
            continue
        group = {"seeds": len(rows), "parameters": rows[0]["parameters"]}
        for split in ("validation", "test"):
            group[split] = {f: {"mean": float(np.mean([r[split][f] for r in rows])),
                                  "sd": float(np.std([r[split][f] for r in rows], ddof=1)) if len(rows)>1 else None}
                            for f in fields}
        for field in ("train_seconds", "inference_seconds", "peak_gpu_memory_bytes", "live_state_bytes"):
            group[field] = float(np.mean([r[field] for r in rows]))
        group["paired_test_differences"] = {}
        for anchor in ("GNN_H", "TGNN"):
            matched = [r for r in rows if (anchor, r["seed"]) in indexed]
            if not matched:
                continue
            group["paired_test_differences"][anchor] = {}
            for field in fields:
                diffs = [r["test"][field] - indexed[(anchor, r["seed"])]["test"][field] for r in matched]
                group["paired_test_differences"][anchor][field] = {
                    "seeds": len(diffs), "values": diffs, "mean": float(np.mean(diffs)),
                    "sd": float(np.std(diffs, ddof=1)) if len(diffs)>1 else None}
        summary["arms"][arm] = group
    complete = [arm for arm, g in summary["arms"].items() if g["seeds"] == 5]
    if complete:
        summary["validation_preferred"] = max(complete, key=lambda a: summary["arms"][a]["validation"]["f1"]["mean"])
        summary["highest_observed_test_f1_exploratory"] = max(complete, key=lambda a: summary["arms"][a]["test"]["f1"]["mean"])
    summary["claim_limit"] = "Previously inspected test, only 25 positives; exploratory component comparison requires new-capture confirmation."
    return summary


def run_study(data_path, output_path, commit=lambda: None):
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available() or "L4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("This registered experiment requires an NVIDIA L4")
    device = torch.device("cuda:0")
    root = Path(__file__).parent
    sources = {name: digest(root / name) for name in SOURCES}
    env = dict(os.environ, TGNN_TEST_DEVICE="cuda:0")
    preflight = subprocess.run([sys.executable, "-m", "pytest", "-q", "test_tgnn_components.py"],
                               cwd=root, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(preflight.stdout, flush=True)
    (output / "gpu_preflight.txt").write_text(preflight.stdout)
    if preflight.returncode:
        commit()
        raise RuntimeError("GPU correctness tests failed; training prohibited")
    data = Data(data_path, device)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_hashes"] != sources or manifest["data_sha256"] != DATA_HASH:
            raise ValueError("Cannot resume: frozen sources or data differ")
    else:
        manifest = {"created_utc": utc(), "source_hashes": sources, "data_sha256": DATA_HASH,
                    "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
                    "python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
                    "epochs": 200, "batch_size": 256, "seeds": list(range(5)), "arms": ARMS,
                    "gpu_preflight": "PASS", "profile": "prathampansare25", "training_only_positive_weight": float(np.sqrt((9830-91)/91)),
                    "nvidia_smi": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"], text=True).strip()}
        write_json(manifest_path, manifest)
        (output / "source").mkdir(exist_ok=True)
        for name in SOURCES:
            shutil.copy2(root / name, output / "source" / name)
    commit()
    results = []
    ordered = ["GNN_H", "TGNN"] + [arm for arm in ARMS if arm not in {"GNN_H", "TGNN"}]
    for arm in ordered:
        factors = ARMS[arm]
        for seed in range(5):
            name = f"{arm}_seed{seed}"
            result_path = output / f"{name}.json"
            if result_path.exists():
                row = json.loads(result_path.read_text())
                assert row["source_hashes"] == sources and row["data_sha256"] == DATA_HASH
                verify_result(output, row, data)
                results.append(row)
                print(json.dumps({"phase": "resume_verified", "run_id": name}), flush=True)
                continue
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = ComponentTGNN(10, factors).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)
            torch.manual_seed(seed + 10000)
            reservoir = EfficientReservoir(983, data.train_end, seed + 937)
            weight = torch.tensor([1., np.sqrt((data.train_end-91)/91)], dtype=torch.float32, device=device) if "W" in factors else None
            losses, replay_visits = [], 0
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for epoch in range(200):
                model.train()
                model.reset_state()
                total = torch.zeros((), device=device)
                for lo in range(0, data.train_end, 256):
                    hi = min(lo+256, data.train_end)
                    optimizer.zero_grad(set_to_none=True)
                    logits, context = model.forward_events(*data.batch(slice(lo,hi)))
                    live = F.cross_entropy(logits, data.y[lo:hi], weight=weight)
                    loss = live
                    items = reservoir.sample(26)
                    if items:
                        ids = np.array([event for event, _ in items])
                        replay_context = torch.stack([ctx for _, ctx in items])
                        replay_logits = model.replay(*data.batch(ids)[:3], replay_context)
                        loss = loss + F.cross_entropy(replay_logits, data.y[torch.as_tensor(ids, device=device)], weight=weight)
                        replay_visits += len(items)
                    loss.backward()
                    optimizer.step()
                    model.detach_state()
                    total += live.detach() * (hi-lo)
                    # unbind makes only views; rejected reservoir items aren't cloned.
                    for event, ctx in zip(range(lo,hi), context.detach().unbind()):
                        reservoir.admit(event, ctx)
                losses.append(float(total / data.train_end))
                if not np.isfinite(losses[-1]):
                    raise RuntimeError(f"Nonfinite training loss: {name}")
                if (epoch+1) % 25 == 0:
                    progress = {"updated_utc": utc(), "phase": "TRAINING", "run_id": name, "epoch": epoch+1,
                                "epochs": 200, "completed_runs": len(results), "expected_runs": 60,
                                "live_loss": losses[-1], "run_elapsed_seconds": time.perf_counter()-started}
                    print(json.dumps(progress), flush=True)
                    write_json(output / "progress.json", progress)
                    commit()
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-started
            peak = torch.cuda.max_memory_allocated()
            checkpoint = output / f"{name}.pt"
            torch.save({"state_dict": {k: v.detach().cpu() for k,v in model.state_dict().items()},
                        "factors": factors, "seed": seed, "epochs": 200}, checkpoint)
            model.eval()
            model.reset_state()
            chunks = []
            torch.cuda.synchronize()
            infer_start = time.perf_counter()
            with torch.no_grad():
                for lo in range(0, data.n, 256):
                    logits, _ = model.forward_events(*data.batch(slice(lo, min(lo+256,data.n))))
                    chunks.append(logits.softmax(-1)[:,1].cpu().numpy())
            torch.cuda.synchronize()
            infer_elapsed = time.perf_counter()-infer_start
            probability = np.concatenate(chunks)
            val, test = slice(data.train_end, data.val_end), slice(data.val_end,data.n)
            threshold = select_threshold(data.labels[val], probability[val])
            prediction_path = output / f"{name}_predictions.npz"
            np.savez_compressed(prediction_path, probability=probability, y=data.labels, raw_row_id=data.row_ids)
            row = {"run_id": name, "arm": arm, "seed": seed, "factors": factors, "status": "COMPLETE", "completed_utc": utc(),
                   "epochs": 200, "batch_size": 256, "threshold": threshold, "epoch_losses": losses,
                   "validation": scored(data.labels[val], probability[val], threshold),
                   "test": scored(data.labels[test], probability[test], threshold),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "parameters_by_tensor": {n: p.numel() for n,p in model.named_parameters()},
                   "train_seconds": elapsed, "inference_seconds": infer_elapsed, "peak_gpu_memory_bytes": peak,
                   "live_state_bytes": sum(v.numel()*v.element_size() for v in model.memory.values()) + sum(v.numel()*v.element_size() for _,_,v in model.pending),
                   "plan_cache_bytes": sum(read.numel()*read.element_size()+update.numel()*update.element_size() for _,read,update,_ in model._plans.values()),
                   "training_live_visits": data.train_end*200, "training_replay_visits": replay_visits,
                   "reservoir_capacity": 983, "data_sha256": DATA_HASH, "source_hashes": sources,
                   "checkpoint_sha256": digest(checkpoint), "predictions_sha256": digest(prediction_path),
                   "gpu": torch.cuda.get_device_name(0)}
            verify_result(output, row, data)
            row["artifact_verification"] = "PASS"
            write_json(result_path, row)
            results.append(row)
            write_json(output / "summary.json", summarize(results))
            write_json(output / "progress.json", {"phase": "RUN_COMPLETE", "updated_utc": utc(), "run_id": name,
                       "completed_runs": len(results), "expected_runs": 60})
            commit()
            print(json.dumps({"phase": "result", "run_id": name, "validation": row["validation"], "test": row["test"], "train_seconds": elapsed}), flush=True)
            del model, optimizer, reservoir, logits, context, loss
            torch.cuda.empty_cache()
    for row in results:
        verify_result(output, row, data)
    summary = summarize(results)
    write_json(output / "summary.json", summary)
    write_json(output / "verification.json", {"status": "PASS", "runs": len(results), "gpu_tests": "PASS",
               "source_hashes": sources, "data_sha256": DATA_HASH, "saved_prediction_metrics": "PASS", "checkpoint_hashes": "PASS", "verified_utc": utc()})
    write_json(output / "progress.json", {"phase": "COMPLETE", "completed_runs": len(results), "expected_runs": 60, "updated_utc": utc()})
    from report_tgnn_components import render
    render(output)
    commit()
    print(json.dumps({"phase": "COMPLETE", "summary": summary}), flush=True)
    return summary
