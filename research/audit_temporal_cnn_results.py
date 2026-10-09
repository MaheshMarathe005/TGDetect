"""Independent audit of the completed temporal-CNN study."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from run_tgnn_components import DATA_HASH, Data, digest, verify_result, write_json
from temporal_cnn import CNN_ARMS, make_model


def main():
    root = Path(__file__).parent
    output = root / "artifacts/temporal_cnn_200e_v2"
    manifest = json.loads((output / "run_manifest.json").read_text())
    verification = json.loads((output / "verification.json").read_text())
    progress = json.loads((output / "progress.json").read_text())
    assert progress["phase"] == "COMPLETE" and progress["completed_runs"] == 40
    assert verification["status"] == "PASS" and verification["runs"] == 40
    assert manifest["data_sha256"] == DATA_HASH and manifest["epochs"] == 200
    assert manifest["arms"] == {a: list(f) for a, f in CNN_ARMS.items()}
    for name, expected in manifest["source_hashes"].items():
        assert digest(output / "source" / name) == expected, name
    data = Data(root / "data/ctu13_verified/scenario06_pilot16384.npz", torch.device("cpu"))
    rows = []
    for arm, factors in CNN_ARMS.items():
        for seed in range(5):
            run_id = f"{arm}_seed{seed}"
            row = json.loads((output / f"{run_id}.json").read_text())
            assert row["status"] == "COMPLETE" and row["factors"] == list(factors)
            assert row["source_hashes"] == manifest["source_hashes"]
            assert row["data_sha256"] == DATA_HASH and row["epochs"] == 200
            verify_result(output, row, data)
            checkpoint = torch.load(output / f"{run_id}.pt", weights_only=True, map_location="cpu")
            model = make_model(arm)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            assert checkpoint["seed"] == seed and checkpoint["epochs"] == 200
            assert sum(p.numel() for p in model.parameters()) == row["parameters"]
            assert all(torch.isfinite(p).all() for p in model.parameters())
            pred = np.load(output / f"{run_id}_predictions.npz", allow_pickle=False)["probability"]
            assert np.isfinite(pred).all() and np.all((pred >= 0) & (pred <= 1))
            rows.append(row)
    assert len(rows) == 40
    summary = json.loads((output / "summary.json").read_text())
    for arm in CNN_ARMS:
        group = [r for r in rows if r["arm"] == arm]
        for split in ("validation", "test"):
            for field in ("f1", "pr_ap", "precision", "recall", "fpr", "roc_auc"):
                mean = float(np.mean([r[split][field] for r in group]))
                sd = float(np.std([r[split][field] for r in group], ddof=1))
                assert abs(mean-summary["arms"][arm][split][field]["mean"]) < 1e-12
                assert abs(sd-summary["arms"][arm][split][field]["sd"]) < 1e-12
    anchor = []
    previous = root / "artifacts/components_200e_v1"
    for arm in ("GNN_H", "TGNN"):
        for seed in range(5):
            current = np.load(output/f"{arm}_seed{seed}_predictions.npz",allow_pickle=False)["probability"]
            old = np.load(previous/f"{arm}_seed{seed}_predictions.npz",allow_pickle=False)["probability"]
            anchor.append({"run_id": f"{arm}_seed{seed}", "predictions_bit_identical": bool(np.array_equal(current, old)),
                           "max_probability_difference": float(np.max(np.abs(current-old)))})
    assert all(item["predictions_bit_identical"] for item in anchor)
    checks = {"status":"PASS","verified_utc":datetime.now(timezone.utc).isoformat(),"runs":40,
              "source_data_hashes":"PASS","checkpoint_architectures_and_finite_weights":"PASS",
              "saved_predictions_thresholds_metrics":"PASS","aggregate_statistics":"PASS",
              "anchor_reproduction":anchor,"cnn_prediction_files":40,
              "full_run_summary_sha256":digest(output/"summary.json"),
              "full_run_manifest_sha256":digest(output/"run_manifest.json")}
    write_json(output/"local_verification.json",checks)
    print(json.dumps(checks,indent=2))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
