"""One Modal L4 invocation for all matched 200-epoch B+H versus B+T+H runs."""
from __future__ import annotations

import modal

APP_NAME = "tg-detect-matched-time-rehearsal"
REMOTE_DATA = "/root/scenario06_pilot16384.npz"
REMOTE_META = "/root/scenario06_pilot16384.json"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .uv_pip_install(
        "torch==2.7.1",
        "numpy==2.2.6",
        "scikit-learn==1.7.2",
    )
    .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_python_source(
        "pomsathit_baseline", "causal_data", "causal_tgn", "run_causal_pilot"
    )
    .add_local_file("data/ctu13_verified/scenario06_pilot16384.npz", REMOTE_DATA)
    .add_local_file("data/ctu13_verified/scenario06_pilot16384.json", REMOTE_META)
    .add_local_file("ADAPTED_PROTOCOL.md", "/root/ADAPTED_PROTOCOL.md")
    .add_local_file("MATCHED_TIME_REHEARSAL_PROTOCOL.md", "/root/MATCHED_TIME_REHEARSAL_PROTOCOL.md")
    .add_local_file("run_matched_time_rehearsal.py", "/root/run_matched_time_rehearsal.py")
    .add_local_file("modal_matched_time_rehearsal.py", "/root/modal_matched_time_rehearsal.py")
)

app = modal.App(APP_NAME, image=image)


@app.function(gpu="L4", cpu=4.0, memory=16384, timeout=14400)
def run_all():
    import hashlib
    import json
    import os
    import platform
    import resource
    import shutil
    import subprocess
    import sys
    import tarfile
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np
    import sklearn
    import torch
    from torch.nn import functional as F

    from causal_tgn import CausalTGNN, Reservoir
    from run_causal_pilot import metrics, select_threshold

    epochs = 200
    seeds = list(range(5))
    arms = {"B_H": ("H",), "B_T_H": ("T", "H")}
    output = Path("/tmp/modal_matched_time_rehearsal_200e")
    output.mkdir(parents=True, exist_ok=False)

    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    def write_json(path, value):
        temporary = Path(str(path) + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    class Data:
        def __init__(self, path, meta_path, device):
            self.path = Path(path)
            self.meta = json.loads(Path(meta_path).read_text())
            if file_hash(path) != self.meta["processed_sha256"]:
                raise ValueError("Processed dataset checksum mismatch")
            arrays = np.load(path, allow_pickle=False)
            self.x = torch.from_numpy(arrays["x"]).to(device)
            self.y = torch.from_numpy(arrays["y"]).to(device)
            self.ego = torch.from_numpy(arrays["ego"]).to(device)
            self.valid = torch.from_numpy(arrays["valid"]).to(device)
            self.ages = torch.from_numpy(arrays["ages"]).to(device)
            self.src = arrays["src"]
            self.dst = arrays["dst"]
            self.times = arrays["times"]
            self.row_ids = arrays["row_ids"]
            self.train_end = int(arrays["train_end"])
            self.val_end = int(arrays["val_end"])
            self.n = len(self.src)
            for name, lo, hi, minimum in [
                ("train", 0, self.train_end, 20),
                ("validation", self.train_end, self.val_end, 10),
                ("test", self.val_end, self.n, 10),
            ]:
                positives = int(self.y[lo:hi].sum().item())
                if min(positives, hi - lo - positives) < minimum:
                    raise ValueError(f"{name} fails class support: {positives}/{hi-lo}")

        def batch(self, ids):
            if isinstance(ids, slice):
                tensor_ids = ids
                numpy_ids = ids
            else:
                numpy_ids = np.asarray(ids, dtype=np.int64)
                tensor_ids = torch.as_tensor(numpy_ids, device=self.x.device)
            return (
                self.x[self.ego[tensor_ids]],
                self.valid[tensor_ids],
                self.ages[tensor_ids],
                self.src[numpy_ids], self.dst[numpy_ids], self.times[numpy_ids],
            )

    if not torch.cuda.is_available():
        raise RuntimeError("L4 requested but CUDA is unavailable")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    properties = torch.cuda.get_device_properties(device)
    print(json.dumps({"phase": "environment", "nvidia_smi": nvidia_smi,
                      "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                      "device": str(properties)}), flush=True)

    torch.manual_seed(20260923)
    first = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
    torch.manual_seed(20260923)
    second = torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    data = Data(REMOTE_DATA, REMOTE_META, device)

    # Confirm paired initialization: every shared baseline tensor is bit-identical.
    torch.manual_seed(0)
    smoke_b = CausalTGNN(10, ("H",)).to(device)
    torch.manual_seed(0)
    smoke_t = CausalTGNN(10, ("T", "H")).to(device)
    temporal_names = {"time_encoder.weight", "time_encoder.bias"}
    shared_t = {name: value for name, value in smoke_t.state_dict().items()
                if name not in temporal_names}
    for name, value in smoke_b.state_dict().items():
        torch.testing.assert_close(value, shared_t[name], rtol=0, atol=0)
    ids = slice(0, 8)
    smoke_t.train()
    logits, _ = smoke_t.forward_events(*data.batch(ids))
    F.cross_entropy(logits, data.y[ids]).backward()
    if smoke_t.time_encoder.weight.grad is None or not torch.count_nonzero(smoke_t.time_encoder.weight.grad):
        raise RuntimeError("Time encoder failed CUDA gradient smoke test")
    del smoke_b, smoke_t, logits
    torch.cuda.empty_cache()

    source_paths = {
        "pomsathit_baseline.py": "/root/pomsathit_baseline.py",
        "causal_data.py": "/root/causal_data.py",
        "causal_tgn.py": "/root/causal_tgn.py",
        "run_causal_pilot.py": "/root/run_causal_pilot.py",
        "run_matched_time_rehearsal.py": "/root/run_matched_time_rehearsal.py",
        "modal_matched_time_rehearsal.py": "/root/modal_matched_time_rehearsal.py",
        "ADAPTED_PROTOCOL.md": "/root/ADAPTED_PROTOCOL.md",
        "MATCHED_TIME_REHEARSAL_PROTOCOL.md": "/root/MATCHED_TIME_REHEARSAL_PROTOCOL.md",
    }
    source_hashes = {name: file_hash(path) for name, path in source_paths.items()}
    source_output = output / "source"
    source_output.mkdir()
    for name, path in source_paths.items():
        shutil.copy2(path, source_output / name)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "app": APP_NAME, "modal_client_requested_gpu": "L4",
        "nvidia_smi": nvidia_smi, "gpu_name": properties.name,
        "gpu_total_memory": properties.total_memory,
        "python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "numpy": np.__version__, "sklearn": sklearn.__version__, "platform": platform.platform(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "epochs": epochs, "batch_size": 256, "seeds": seeds,
        "arms": {name: list(factors) for name, factors in arms.items()},
        "data_sha256": file_hash(REMOTE_DATA), "data_meta": data.meta,
        "source_hashes": source_hashes, "cuda_smoke": "PASS",
        "paired_shared_initialization": "PASS", "tiny_backward": "PASS",
    }
    write_json(output / "run_manifest.json", manifest)

    def predict(model):
        model.eval()
        model.reset_state()
        chunks = []
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for lo in range(0, data.n, 256):
                hi = min(lo + 256, data.n)
                logits, _ = model.forward_events(*data.batch(slice(lo, hi)))
                chunks.append(logits.softmax(-1)[:, 1].cpu().numpy())
        torch.cuda.synchronize()
        return np.concatenate(chunks), time.perf_counter() - started

    def train_one(arm, factors, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = CausalTGNN(10, factors).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)
        torch.manual_seed(seed + 10000)
        reservoir = Reservoir(data.train_end // 10, data.train_end, seed + 937)
        losses = []
        replay_visits = 0
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            model.reset_state()
            total = 0.0
            for lo in range(0, data.train_end, 256):
                hi = min(lo + 256, data.train_end)
                optimizer.zero_grad(set_to_none=True)
                logits, context = model.forward_events(*data.batch(slice(lo, hi)))
                live_loss = F.cross_entropy(logits, data.y[lo:hi])
                replay_items = reservoir.sample(26)
                loss = live_loss
                if replay_items:
                    replay_ids = np.array([event_id for event_id, _ in replay_items])
                    stored_context = torch.stack([saved for _, saved in replay_items])
                    replay_logits = model.replay(*data.batch(replay_ids)[:3], stored_context)
                    loss = loss + F.cross_entropy(replay_logits, data.y[
                        torch.as_tensor(replay_ids, device=device)])
                    replay_visits += len(replay_items)
                loss.backward()
                optimizer.step()
                model.detach_state()
                total += float(live_loss.detach()) * (hi - lo)
                for event_id in range(lo, hi):
                    reservoir.admit(event_id, context[event_id - lo])
            losses.append(total / data.train_end)
            if (epoch + 1) % 25 == 0:
                print(json.dumps({"phase": "train", "arm": arm, "seed": seed,
                                  "epoch": epoch + 1, "live_loss": losses[-1]}), flush=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        peak_gpu_bytes = torch.cuda.max_memory_allocated(device)
        run_id = f"{arm}_seed{seed}"
        checkpoint = output / f"{run_id}.pt"
        state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        torch.save({"state_dict": state, "factors": list(factors), "seed": seed,
                    "epochs": epochs}, checkpoint)
        probabilities, inference_seconds = predict(model)
        y = data.y.detach().cpu().numpy()
        validation = slice(data.train_end, data.val_end)
        test = slice(data.val_end, data.n)
        threshold = select_threshold(y[validation], probabilities[validation])
        result = {
            "run_id": run_id, "arm": arm, "factors": list(factors), "seed": seed,
            "status": "COMPLETE", "epochs": epochs, "batch_size": 256,
            "threshold": threshold, "epoch_losses": losses,
            "validation": metrics(y[validation], probabilities[validation], threshold),
            "test": metrics(y[test], probabilities[test], threshold),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "train_seconds": train_seconds, "inference_seconds": inference_seconds,
            "inference_events_per_second": data.n / inference_seconds,
            "training_live_visits": data.train_end * epochs,
            "training_replay_visits": replay_visits,
            "reservoir_capacity": reservoir.capacity,
            "replay_context_bytes": sum(saved.numel() * saved.element_size() + 8
                                        for _, saved in reservoir.items),
            "peak_gpu_memory_bytes": peak_gpu_bytes,
            "peak_process_rss_kib_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": file_hash(checkpoint),
            "data_sha256": file_hash(REMOTE_DATA), "source_hashes": source_hashes,
            "gpu": properties.name, "scope": "matched paper-specified adapted architecture",
        }
        np.savez_compressed(output / f"{run_id}_predictions.npz", probability=probabilities,
                            y=y, raw_row_id=data.row_ids)
        write_json(output / f"{run_id}.json", result)
        print(json.dumps({"phase": "result", "run_id": run_id,
                          "test": result["test"], "train_seconds": train_seconds}), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
        return result

    results = []
    for arm, factors in arms.items():
        for seed in seeds:
            results.append(train_one(arm, factors, seed))

    fields = ["f1", "pr_ap", "roc_auc", "precision", "recall", "fpr"]
    aggregate = {}
    for arm in arms:
        rows = [row for row in results if row["arm"] == arm]
        aggregate[arm] = {
            "seeds": len(rows), "parameters": rows[0]["parameters"],
            **{field: {"mean": float(np.mean([row["test"][field] for row in rows])),
                       "sd": float(np.std([row["test"][field] for row in rows], ddof=1))}
               for field in fields},
            "mean_train_seconds": float(np.mean([row["train_seconds"] for row in rows])),
            "mean_peak_gpu_memory_bytes": float(np.mean([row["peak_gpu_memory_bytes"] for row in rows])),
        }
    indexed = {(row["arm"], row["seed"]): row for row in results}
    paired = {}
    for field in fields:
        differences = [indexed[("B_T_H", seed)]["test"][field] -
                       indexed[("B_H", seed)]["test"][field] for seed in seeds]
        paired[field] = {"seed_differences": differences,
                         "mean": float(np.mean(differences)),
                         "sd": float(np.std(differences, ddof=1))}
    summary = {
        "comparison": "B_T_H minus B_H", "arms": aggregate,
        "paired_test_difference": paired,
        "claim_limit": "Same previously inspected CTU test; missing-arm completion, not independent confirmation.",
    }
    write_json(output / "summary.json", summary)

    # Recompute every saved threshold and test metric before exporting.
    validation = slice(data.train_end, data.val_end)
    test = slice(data.val_end, data.n)
    for result in results:
        run_id = result["run_id"]
        saved = np.load(output / f"{run_id}_predictions.npz", allow_pickle=False)
        threshold = select_threshold(saved["y"][validation], saved["probability"][validation])
        if threshold != result["threshold"]:
            raise RuntimeError(f"Threshold verification failed: {run_id}")
        if metrics(saved["y"][test], saved["probability"][test], threshold) != result["test"]:
            raise RuntimeError(f"Metric verification failed: {run_id}")
        if file_hash(output / f"{run_id}.pt") != result["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint verification failed: {run_id}")
    verification = {
        "verified_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        "runs": len(results), "gpu": properties.name, "cuda_smoke": "PASS",
        "paired_shared_initialization": "PASS", "tiny_backward": "PASS",
        "prediction_metric_recomputation": "PASS", "checkpoint_hashes": "PASS",
        "data_hash": "PASS", "source_hashes": "PASS",
    }
    write_json(output / "verification.json", verification)
    archive = Path("/tmp/modal_matched_time_rehearsal_200e.tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    print(json.dumps({"phase": "complete", "summary": summary,
                      "verification": verification, "archive_bytes": archive.stat().st_size}), flush=True)
    return {"summary": summary, "verification": verification,
            "archive_sha256": file_hash(archive), "archive": archive.read_bytes()}


@app.local_entrypoint()
def main():
    import hashlib
    import json
    import tarfile
    from pathlib import Path

    returned = run_all.remote()
    archive = Path("artifacts/modal_matched_time_rehearsal_200e.tar.gz")
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(returned["archive"])
    if hashlib.sha256(archive.read_bytes()).hexdigest() != returned["archive_sha256"]:
        raise RuntimeError("Downloaded Modal archive hash mismatch")
    destination = Path("artifacts")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            candidate = (destination / member.name).resolve()
            if destination.resolve() not in candidate.parents and candidate != destination.resolve():
                raise RuntimeError("Unsafe archive path")
        tar.extractall(destination)
    print(json.dumps({"downloaded": str(archive), "archive_sha256": returned["archive_sha256"],
                      "summary": returned["summary"], "verification": returned["verification"]}, indent=2))
