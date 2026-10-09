"""Inspect/download the CNN study and independently verify all completed runs."""
import argparse
import json
from pathlib import Path

import modal


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--download",action="store_true")
    parser.add_argument("--telemetry",action="store_true")
    args=parser.parse_args()
    volume=modal.Volume.from_name("tg-detect-temporal-cnn-v1")
    remote="temporal_cnn_200e_v2"
    for name in ("progress.json","concurrency_calibration.json","gpu_utilization_summary.json"):
        try:
            value=json.loads(b"".join(volume.read_file(f"{remote}/{name}")))
        except FileNotFoundError:
            continue
        if name=="concurrency_calibration.json":
            value={"selected_workers":value["selected_workers"],"measurements":[{k:v for k,v in m.items() if k!="worker_measurements"} for m in value["measurements"]]}
        print(json.dumps({name:value},indent=2),flush=True)
    if args.telemetry:
        try:
            raw=b"".join(volume.read_file(f"{remote}/gpu_telemetry.jsonl")).decode()
            samples=[]
            for line in raw.splitlines():
                try:
                    row=json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("phase")=="production" and "gpu_utilization_percent" in row:
                    samples.append(row)
            recent=samples[-60:]
            if recent:
                print(json.dumps({"recent_production_telemetry":{"samples":len(recent),"latest_utc":recent[-1]["utc"],
                    **{key:{"mean":sum(r[key] for r in recent)/len(recent),"max":max(r[key] for r in recent)}
                       for key in ("gpu_utilization_percent","memory_used_mib","power_watts")}}},indent=2),flush=True)
        except FileNotFoundError:
            pass
    if not args.download:
        return
    output=Path("artifacts")/remote
    output.mkdir(parents=True,exist_ok=True)
    for entry in volume.iterdir(remote,recursive=True):
        if entry.type!=modal.volume.FileEntryType.FILE:
            continue
        destination=output/Path(entry.path).relative_to(remote)
        if output.resolve() not in destination.resolve().parents:
            raise ValueError("Unsafe remote path")
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(destination.suffix+".download")
        with temporary.open("wb") as handle:
            for block in volume.read_file(entry.path):
                handle.write(block)
        temporary.replace(destination)
    import numpy as np
    import torch
    from run_tgnn_components import Data,digest,verify_result,write_json
    from temporal_cnn import CNN_ARMS,make_model
    data=Data("data/ctu13_verified/scenario06_pilot16384.npz",torch.device("cpu"))
    manifest=json.loads((output/"run_manifest.json").read_text())
    for name,expected in manifest["source_hashes"].items():
        assert digest(output/"source"/name)==expected,name
    count=0
    reproduction=[]
    for arm in CNN_ARMS:
        for seed in range(5):
            name=f"{arm}_seed{seed}"
            path=output/f"{name}.json"
            if not path.exists():
                continue
            row=json.loads(path.read_text())
            assert row["source_hashes"]==manifest["source_hashes"]
            verify_result(output,row,data)
            checkpoint=torch.load(output/f"{name}.pt",map_location="cpu",weights_only=True)
            model=make_model(arm)
            model.load_state_dict(checkpoint["state_dict"],strict=True)
            assert all(torch.isfinite(p).all() for p in model.parameters())
            assert sum(p.numel() for p in model.parameters())==row["parameters"]
            count+=1
            previous=Path("artifacts/components_200e_v1")/f"{name}_predictions.npz"
            if previous.exists():
                a=np.load(previous)["probability"]
                b=np.load(output/f"{name}_predictions.npz")["probability"]
                reproduction.append({"run_id":name,"predictions_bit_identical":bool(np.array_equal(a,b)),
                                     "max_probability_difference":float(np.max(np.abs(a-b)))})
    write_json(output/"local_verification.json",{"status":"PASS","verified_runs":count,"expected_runs":40,
               "source_snapshots":"PASS","checkpoints_and_metrics":"PASS","prior_control_predictions":reproduction})
    print(f"Downloaded and verified {count}/40 runs",flush=True)


if __name__=="__main__":
    main()
