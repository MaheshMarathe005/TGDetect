"""Read-only data/model diagnostics for the long-history hypothesis; no training."""
import hashlib
import json
import csv
from pathlib import Path

import numpy as np


def main():
    root = Path(__file__).parent
    path = root / "data/ctu13_verified/scenario06_pilot16384.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["processed_sha256"]
    a = np.load(path, allow_pickle=False)
    t, src, dst = a["times"], a["src"], a["dst"]
    n, train_end, val_end = len(t), int(a["train_end"]), int(a["val_end"])
    assert np.all(np.diff(t) >= 0)
    prior = np.zeros((n, 2), dtype=np.int64)
    counts, pending, last = {}, [], None
    for i, (s, d, time) in enumerate(zip(src, dst, t)):
        if time != last:
            for ps, pd in pending:
                for node in {ps, pd}:
                    counts[node] = counts.get(node, 0) + 1
            pending = []
        prior[i] = counts.get(int(s), 0), counts.get(int(d), 0)
        pending.append((int(s), int(d)))
        last = time
    partitions = {}
    for name, lo, hi in [("train", 0, train_end), ("validation", train_end, val_end), ("test", val_end, n)]:
        mask = a["valid"][lo:hi, 1:]
        age_seconds = np.expm1(a["ages"][lo:hi, 1:][mask].astype(np.float64))
        history = mask.sum(1)
        partitions[name] = {
            "events": hi-lo, "positive_events": int(a["y"][lo:hi].sum()),
            "span_minutes": float((t[hi-1]-t[lo])/60),
            "events_with_prior_endpoint_history": int((prior[lo:hi].max(1)>0).sum()),
            "events_with_four_graph_neighbors": int((history == 4).sum()),
            "mean_graph_neighbor_count": float(history.mean()),
            "prior_endpoint_event_count_median": np.median(prior[lo:hi], axis=0).tolist(),
            "neighbor_age_seconds_quantiles_50_90_99": np.quantile(age_seconds, [.5,.9,.99]).tolist(),
        }
    summary = json.loads((root / "artifacts/components_200e_v1/summary.json").read_text())
    groups = summary["arms"]
    effects = []
    for base, memory in [("TGNN", "TGNN_M"), ("TGNN_A", "TGNN_M_A"),
                         ("TGNN_Q", "TGNN_Q_M"), ("TGNN_Q_A", "TGNN_Q_M_A")]:
        effects.append({"without_memory": base, "with_memory": memory,
                        "f1_difference_percentage_points": 100*(groups[memory]["test"]["f1"]["mean"]-groups[base]["test"]["f1"]["mean"]),
                        "pr_ap_difference_percentage_points": 100*(groups[memory]["test"]["pr_ap"]["mean"]-groups[base]["test"]["pr_ap"]["mean"])})
    result = {
        "analysis_date": "2026-09-26", "data_sha256": meta["processed_sha256"],
        "sample_events": n, "raw_events": meta["raw_rows"],
        "fraction_of_raw_events_retained": n/meta["raw_rows"],
        "raw_to_sample_ratio": meta["raw_rows"]/n,
        "sample_span_seconds": float(t[-1]-t[0]), "sample_span_hours": float((t[-1]-t[0])/3600),
        "sample_spans_full_capture_completion_range": "Yes: frozen preprocessing selects linspace indices including the first and last sorted raw events",
        "partitions": partitions, "matched_memory_effects": effects,
        "ema_direct_coefficient_mass_in_last_updates": {str(k): 1-2**(-k) for k in (1,4,5,7,10)},
        "old_state_retention_after_updates": {str(k): 2**(-k) for k in (1,4,5,7,10)},
        "caveat": "EMA coefficients concern direct stored embeddings, not a strict bound on all historical information inside embeddings. Decay follows endpoint updates, not elapsed wall time. No training performed."
    }
    out = root / "research/findings"
    out.mkdir(parents=True, exist_ok=True)
    (out / "memory_horizon_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    # Keep a byte-identical aggregate copy and a complete, human-readable seed export.
    artifacts = root / "artifacts/components_200e_v1"
    raw_summary = (artifacts / "summary.json").read_bytes()
    assert hashlib.sha256(raw_summary).hexdigest() == "1be3420aae31cd555189f82d92456563e1bf8a04bed7e63202462381badc70db"
    (out / "component_summary.json").write_bytes(raw_summary)
    metric_names = ("f1", "pr_ap", "precision", "recall", "fpr", "roc_auc", "accuracy", "balanced_accuracy", "tp", "fp", "fn", "tn")
    rows = []
    for arm in groups:
        for seed in range(5):
            r = json.loads((artifacts / f"{arm}_seed{seed}.json").read_text())
            assert r["epochs"] == 200 and r["status"] == "COMPLETE"
            row = {k: r[k] for k in ("run_id", "arm", "seed", "epochs", "threshold", "parameters", "train_seconds", "inference_seconds", "peak_gpu_memory_bytes", "checkpoint_sha256", "predictions_sha256", "data_sha256")}
            row["factors"] = "+".join(r["factors"])
            for split in ("validation", "test"):
                row.update({f"{split}_{key}": r[split][key] for key in metric_names})
            rows.append(row)
    assert len(rows) == 60
    with (out / "component_runs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
