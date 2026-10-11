# TGDetect — V3 and ORTHRUS package

This directory is an independent snapshot of the V3 research, recorded V3/ORTHRUS comparison results, and the complete dashboard application. Its local CADETS E3 data and trained V3 checkpoint are supplied as a checksum-verified GitHub release asset; large data files are excluded from Git history. No file links point back to the parent project.

- `research/sliding_node_mem_v3/`: complete V3 implementation, preparation, pretraining, training, inference, profiling, ablations and comparison.
- `research/architectures/`: V3 and ORTHRUS diagrams and component descriptions.
- `research/orthrus_official/`: authors' ORTHRUS source/configuration snapshot; `orthrus_comparison/` contains the comparison adapter and protocol.
- `research/references/ORTHRUS-paper.pdf`: ORTHRUS reference paper.
- `results/v3_vs_orthrus/`: unchanged recorded accuracy, latency and provenance JSON files.
- `results/sliding_v3/`: V3 profiling and development measurements.
- `results/dashboard_validation/`: recorded browser evidence and local data verification.
- `dashboard/`: full backend, frontend, runtime, upload graphs and attack-chain investigation code.
- `dashboard/data/cadets_v3/`: local checkpoint, tokens, event arrays, saved scores and investigation database.
- `artifacts/`: compatibility copy of result directories used by existing experiment scripts.

V3 imports code from `sliding_node_mem_v2`, `sliding_node_mem_upgrade`, `sliding_node_mem`, and several shared research modules. These are required compatibility dependencies, not additional selected research versions. Frozen files are unchanged so checkpoint source-hash validation still works. V4 experiments/results and unrelated research drafts are excluded.

## Start the dashboard

From **this directory**, use Python 3.10+ with the dependencies in `requirements.txt`. `environment-record.json` records the versions used to verify this copy. Choose the appropriate Torch CPU/CUDA wheel for your machine; the dashboard runs on CPU. Modal is needed only for existing cloud scripts and downloading artifacts, not for normal dashboard operation.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/download_local_data.py
python -m uvicorn dashboard.backend:app --host 127.0.0.1 --port 8000
```

In another terminal:

```sh
cd dashboard/frontend
npm ci
npm run build
npm run start
```

Open http://localhost:3000. Use **Detection graph** to select local CADETS, upload endpoint-event logs, select IPs/hosts/processes/files, and follow connected malicious or benign predictions. See `dashboard/CADETS.md` for upload schemas and investigation details. If ports 8000/3000 are already occupied, stop those services or change the ports and `TGDETECT_API_ORIGIN`.

The complete legacy Sysmon dashboard source is retained, but its independent DEDALE model artifacts are outside this V3-only research package. The default CADETS workspace is fully supplied.

## Research and results

`research/architectures/README.md` identifies the tested configurations. `results/README.md` explains the result files and protocol limitations. Original manifests retain historical `/study/...` Modal paths as provenance; these are not links to local parent files. V3 local training accepts the copied prepared corpus:

```sh
PYTHONPATH=research python -m sliding_node_mem_v3.run --help
```

Cloud scripts are included for reproducibility but have not been launched by this packaging task. Their Modal volume names and account access must exist before use. ORTHRUS source and measured comparison results are included; an ORTHRUS trained checkpoint is not present locally in this snapshot.

## Verify

```sh
python scripts/verify_package.py
python -m pytest research/test_sliding_v3.py research/test_orthrus_universal.py research/test_cadets_dashboard.py research/test_graph_uploads.py -q
```

`PACKAGE_MANIFEST.json` lists the packaged files and SHA-256 checksums. Python caches, Git history and frontend build/dependency caches are excluded from the manifest. `VALIDATION.json` records checks performed on this independent copy.

The release asset URL, size and SHA-256 are recorded in `DATA_BUNDLE.json`. Run `python scripts/download_local_data.py` after cloning. It downloads the approximately 262 MiB bundle, checks its checksum, extracts only the expected data directory, and verifies each extracted file against the package manifest. No Modal account or GPU is required.
