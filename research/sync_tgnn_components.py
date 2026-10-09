"""Read progress or download/verify the running study without starting GPU jobs."""
import argparse
import json
from pathlib import Path

import modal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    volume = modal.Volume.from_name("tg-detect-components-v1")
    remote = "components_200e_v1"
    for filename in ("progress.json", "summary.json"):
        try:
            value = json.loads(b"".join(volume.read_file(f"{remote}/{filename}")))
        except FileNotFoundError:
            continue
        if filename == "summary.json":
            value = {"completed_runs": value["completed_runs"], "arms": {
                arm: {"seeds": g["seeds"], "test_f1": g["test"]["f1"], "test_pr_ap": g["test"]["pr_ap"]}
                for arm,g in value["arms"].items()}}
        print(json.dumps({filename: value}, indent=2))
    if not args.download:
        return
    output = Path("artifacts") / remote
    output.mkdir(parents=True, exist_ok=True)
    for entry in volume.iterdir(remote, recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        relative = Path(entry.path).relative_to(remote)
        destination = output / relative
        if output.resolve() not in destination.resolve().parents:
            raise ValueError("Unexpected remote path")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        with temporary.open("wb") as handle:
            for block in volume.read_file(entry.path):
                handle.write(block)
        temporary.replace(destination)
    from run_tgnn_components import Data, digest, verify_result
    import torch
    data = Data("data/ctu13_verified/scenario06_pilot16384.npz", torch.device("cpu"))
    manifest = json.loads((output / "run_manifest.json").read_text())
    for filename, expected in manifest["source_hashes"].items():
        assert digest(output / "source" / filename) == expected, filename
    count = 0
    for path in output.glob("*_seed*.json"):
        row = json.loads(path.read_text())
        assert row["source_hashes"] == manifest["source_hashes"]
        verify_result(output, row, data)
        count += 1
    if (output / "summary.json").exists():
        from report_tgnn_components import render
        render(output)
    print(f"Downloaded and independently verified {count} completed runs in {output}")


if __name__ == "__main__":
    main()
