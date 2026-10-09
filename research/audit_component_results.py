"""Post-run completeness, metric, checkpoint and frozen-anchor verification."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_tgnn_components import ARMS, DATA_HASH, Data, digest, summarize, verify_result, write_json
from tgnn_components import ComponentTGNN


def main():
    root = Path(__file__).parent
    output = root / "artifacts/components_200e_v1"
    manifest = json.loads((output / "run_manifest.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    cloud_check = json.loads((output / "verification.json").read_text())
    assert progress["phase"] == "COMPLETE" and progress["completed_runs"] == 60
    assert cloud_check["status"] == "PASS" and cloud_check["runs"] == 60
    assert manifest["gpu"] == "NVIDIA L4" and manifest["epochs"] == 200
    assert manifest["data_sha256"] == DATA_HASH
    assert manifest["arms"] == {a: list(f) for a, f in ARMS.items()}
    for name, expected in manifest["source_hashes"].items():
        assert digest(output / "source" / name) == expected, name
        assert digest(root / name) == expected, f"Current source differs: {name}"
    data = Data(root / "data/ctu13_verified/scenario06_pilot16384.npz", torch.device("cpu"))
    expected_ids = {f"{arm}_seed{seed}" for arm in ARMS for seed in range(5)}
    assert {p.stem for p in output.glob("*_seed*.json")} == expected_ids
    results = []
    for arm, factors in ARMS.items():
        for seed in range(5):
            name = f"{arm}_seed{seed}"
            row = json.loads((output / f"{name}.json").read_text())
            assert row["run_id"] == name and row["arm"] == arm and row["seed"] == seed
            assert row["factors"] == list(factors) and row["status"] == "COMPLETE"
            assert row["source_hashes"] == manifest["source_hashes"] and row["data_sha256"] == DATA_HASH
            assert row["batch_size"] == 256 and row["training_live_visits"] == 9830 * 200
            assert row["reservoir_capacity"] == 983 and row["gpu"] == "NVIDIA L4"
            assert np.isfinite(row["epoch_losses"]).all()
            verify_result(output, row, data)
            checkpoint = torch.load(output / f"{name}.pt", map_location="cpu", weights_only=True)
            assert checkpoint["epochs"] == 200 and checkpoint["seed"] == seed
            assert tuple(checkpoint["factors"]) == factors
            model = ComponentTGNN(10, factors)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert sum(p.numel() for p in model.parameters()) == row["parameters"]
            probability = np.load(output / f"{name}_predictions.npz")["probability"]
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            results.append(row)
    saved_summary = json.loads((output / "summary.json").read_text())
    # Match the runner's original accumulation order for exact floating arithmetic.
    ordered = ["GNN_H", "TGNN"] + [a for a in ARMS if a not in {"GNN_H", "TGNN"}]
    results.sort(key=lambda r: (ordered.index(r["arm"]), r["seed"]))
    assert summarize(results) == saved_summary, "Aggregate or paired summary mismatch"
    anchors = []
    original = root / "artifacts/modal_matched_time_rehearsal_200e"
    for current_arm, old_arm in (("GNN_H", "B_H"), ("TGNN", "B_T_H")):
        for seed in range(5):
            current = f"{current_arm}_seed{seed}"
            old = f"{old_arm}_seed{seed}"
            p = np.load(output / f"{current}_predictions.npz")["probability"]
            q = np.load(original / f"{old}_predictions.npz")["probability"]
            new_weights = torch.load(output / f"{current}.pt", weights_only=True, map_location="cpu")["state_dict"]
            old_weights = torch.load(original / f"{old}.pt", weights_only=True, map_location="cpu")["state_dict"]
            anchors.append({"run_id": current, "frozen_run_id": old,
                            "predictions_bit_identical": bool(np.array_equal(p, q)),
                            "maximum_probability_difference": float(np.max(np.abs(p-q))),
                            "weights_bit_identical": all(torch.equal(new_weights[k], old_weights[k]) for k in old_weights)})
    groups = saved_summary["arms"]
    max_val = max(g["validation"]["f1"]["mean"] for g in groups.values())
    ties = [a for a, g in groups.items() if g["validation"]["f1"]["mean"] == max_val]
    temporal_best = max((a for a in ARMS if a.startswith("TGNN")), key=lambda a: groups[a]["test"]["f1"]["mean"])
    checks = {"status": "PASS", "verified_utc": datetime.now(timezone.utc).isoformat(), "runs": len(results),
              "complete_registered_grid": "PASS", "all_checkpoints_load_strictly": "PASS", "all_weights_finite": "PASS",
              "saved_predictions_metrics_thresholds": "PASS", "aggregate_and_paired_summary": "PASS", "source_and_data_hashes": "PASS",
              "anchor_reproduction": anchors, "validation_f1_maximizers": ties, "best_observed_tgnn_test_f1": temporal_best,
              "summary_sha256": digest(output / "summary.json"), "manifest_sha256": digest(output / "run_manifest.json")}
    write_json(output / "local_verification.json", checks)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
