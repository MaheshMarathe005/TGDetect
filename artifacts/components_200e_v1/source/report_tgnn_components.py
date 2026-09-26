"""Verify and preserve the reference; render every registered component result."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_causal_pilot import metrics, select_threshold


def freeze_reference(root):
    from run_tgnn_components import DATA_HASH, digest, write_json
    directory = root / "artifacts/modal_matched_time_rehearsal_200e"
    manifest = json.loads((directory / "run_manifest.json").read_text())
    assert manifest["data_sha256"] == DATA_HASH and manifest["epochs"] == 200
    for name, expected in manifest["source_hashes"].items():
        assert digest(directory / "source" / name) == expected, name
    rows = []
    for arm in ("B_H", "B_T_H"):
        for seed in range(5):
            run = f"{arm}_seed{seed}"
            row = json.loads((directory / f"{run}.json").read_text())
            saved = np.load(directory / f"{run}_predictions.npz", allow_pickle=False)
            threshold = select_threshold(saved["y"][9830:13107], saved["probability"][9830:13107])
            assert threshold == row["threshold"]
            assert metrics(saved["y"][13107:], saved["probability"][13107:], threshold) == row["test"]
            assert digest(directory / f"{run}.pt") == row["checkpoint_sha256"]
            rows.append({k: row[k] for k in ("run_id", "arm", "seed", "threshold", "validation", "test", "parameters", "train_seconds", "checkpoint_sha256")})
    summary = json.loads((directory / "summary.json").read_text())
    for arm in ("B_H", "B_T_H"):
        r = [v for v in rows if v["arm"] == arm]
        for field in ("f1", "precision", "recall", "pr_ap", "roc_auc", "fpr"):
            assert abs(summary["arms"][arm][field]["mean"] - np.mean([v["test"][field] for v in r])) < 1e-12
            assert abs(summary["arms"][arm][field]["sd"] - np.std([v["test"][field] for v in r], ddof=1)) < 1e-12
    record = {"status": "VERIFIED_FROZEN_REFERENCE", "app_id": "ap-2ukQHuXPf3dCLIi5g6MtYx", "gpu": "NVIDIA L4",
              "epochs": 200, "seeds": list(range(5)), "batch_size": 256,
              "data_sha256": DATA_HASH, "partitions": {"train": [9830,91], "validation": [3277,25], "test": [3277,25]},
              "summary": summary, "runs": rows, "manifest_sha256": digest(directory / "run_manifest.json"),
              "archive_sha256": digest(root / "artifacts/modal_matched_time_rehearsal_200e.tar.gz"),
              "source_hashes": manifest["source_hashes"], "sd_definition": "sample SD across five seeds, ddof=1",
              "threshold": "per-run validation-max-F1, largest threshold on ties"}
    assert record["archive_sha256"] == "2d75e1168e8e0c45a2ba883abe22fd5bce62c3dc6b3b7f4c293fe92661d812c7"
    write_json(root / "REFERENCE_RESULTS.json", record)
    print("Reference verified: all 10 runs, metrics, thresholds, checkpoints, source snapshots and archive.")


def render(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    def fmt(g, split, field):
        stat = g[split][field]
        return f"{100*stat['mean']:.2f} ± {100*stat['sd']:.2f}" if stat["sd"] is not None else f"{100*stat['mean']:.2f} (one seed)"
    lines = ["# TGNN component results", "", f"Completed: {summary['completed_runs']}/60 runs. Values are percentages, mean ± sample SD.", "",
             "Frozen prior reference: GNN + rehearsal F1 **86.61 ± 4.84%**; TGNN F1 **84.52 ± 2.63%**.",
             "Full frozen details: REFERENCE_RESULTS.json and MATCHED_TIME_REHEARSAL_REPORT.md.", "",
             "TGNN = GNN + time + rehearsal. Q = neighbor-history GRU, M = persistent EMA memory, A = additional attention, N = head normalization, W = weighted loss.", "",
             "| Variant | Seeds | Validation F1 | Test F1 | Test PR-AP | Precision | Recall | FPR | Parameters |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm,g in summary["arms"].items():
        vals = [fmt(g,"validation","f1")] + [fmt(g,"test",f) for f in ("f1","pr_ap","precision","recall","fpr")]
        lines.append(f"| {arm} | {g['seeds']} | " + " | ".join(vals) + f" | {g['parameters']:,} |")
    lines.extend(["", f"Preferred by mean validation F1: **{summary.get('validation_preferred', 'pending')}**.",
                  f"Highest observed mean test F1 (exploratory): **{summary.get('highest_observed_test_f1_exploratory','pending')}**.", "",
                  summary["claim_limit"], "", "## Per-seed audit", "",
                  "| Run | Threshold | Val F1 | Test F1 | PR-AP | TP | FP | FN | TN | Train seconds |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for arm in summary["arms"]:
        for seed in range(5):
            path = directory / f"{arm}_seed{seed}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            t = r["test"]
            lines.append(f"| {r['run_id']} | {r['threshold']:.9g} | {100*r['validation']['f1']:.2f} | {100*t['f1']:.2f} | {100*t['pr_ap']:.2f} | {t['tp']} | {t['fp']} | {t['fn']} | {t['tn']} | {r['train_seconds']:.2f} |")
    lines.extend(["", "All per-seed metrics, paired differences, timings, state/cache memory, losses, predictions, checkpoints, source hashes and verification are saved alongside this report.", ""])
    (directory / "TGNN_COMPONENT_REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-reference", action="store_true")
    parser.add_argument("--directory", default="artifacts/components_200e_v1")
    args = parser.parse_args()
    if args.freeze_reference:
        freeze_reference(Path(__file__).parent)
    else:
        render(args.directory)
