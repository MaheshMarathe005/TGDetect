# Research workspace

Run local experiment commands from this directory so `artifacts/` and the
`data/` link resolve consistently. The active app is in `../web_app/`; datasets
are in `../datasets/`; PDFs and manuscript assets are in `../research paper/`.
Historical complete project versions are preserved in `archives/` and excluded
from test collection. One-off browser checks and mock UI backend scripts were
removed; regression tests and the supporting fixture generator remain.

See [codebase notes](CODEBASE_NOTES.md), [script inventory](SCRIPT_INVENTORY.md),
and [cleanup/move record](RESTRUCTURE_MANIFEST.json).

# TG-Detect

Fresh reproducibility study of temporal graph modeling versus a GNN with rehearsal. Work follows [the Astra prompt](ASTRA_FROM_SCRATCH_PROMPT.md); [the research ledger](ASTRA_RESEARCH_LEDGER.md) is the authoritative status and research memory.

The project now lives directly in this directory. Redundant reset wrapper folders and stale research outputs were removed after a verified backup. Original source, data and both paper copies are preserved. [Reset inventory](RESET_INVENTORY.json) records the backup path, every original file hash and cleanup decisions.

## Current state

- [TGNN Final](tgnn_final/README.md): new version assembled one component at a
  time. Currently contains only the selected original V3 key-value encoder;
  causal provenance GAT is the next planned component study.

- [ORTHRUS / V3 / V4-KV study](orthrus_comparison/README.md): pinned released ORTHRUS architecture and dependency tracing, matched benign reconstruction on full E3 CADETS, and a shared key-value event encoder for heterogeneous telemetry. [Collected results and status](artifacts/orthrus_e3_study/RESULTS.md) distinguish published paper values from measured outputs. Collect committed progress with `~/main_env/bin/python sync_orthrus_study.py` from this directory.

- [Temporal CNN study](TEMPORAL_CNN_PROTOCOL.md): 40 registered runs comparing standalone causal Conv1d, TGNN+CNN, TGNN+GRU+CNN and TGNN+GRU+attention+CNN against four matched controls. The [launcher](modal_temporal_cnn.py) benchmarks independent-worker concurrency on one L4 and saves GPU utilization/memory/power telemetry. Run status and verified downloads: `~/main_env/bin/python sync_temporal_cnn.py --download`.
- [Temporal CNN findings](TEMPORAL_CNN_FINDINGS.md): 40/40 runs verified. CNN+rehearsal reached **97.11 ± 1.16% F1**; TGNN+GRU+attention+CNN reached **87.16 ± 6.41%**. The standalone result is a five-position local-window finding, not proof of persistent long-term-memory superiority.

- [Research findings collection](findings/README.md): all 60 runs exported with detailed metrics and hashes. [Data scale and long-term memory review](findings/memory_and_data_scale.md) explains the measured 2.93% event retention, full 2.15-hour span, local-GRU versus persistent-memory distinction, and proposed controlled tests. No scaling experiment has been run yet.

- **Component study complete: 60/60 runs verified.** [Full findings](TGNN_COMPONENT_FINDINGS.md): best observed TGNN F1 is **GRU + additional attention, 86.39 ± 3.27%**, versus GNN+rehearsal **86.61 ± 4.84%** and plain TGNN **84.52 ± 2.63%**. All ten reference reruns reproduce original weights and predictions exactly. Nine variants tie at perfect validation F1; there is no unique validation winner. [Local audit](artifacts/components_200e_v1/local_verification.json).

- Frozen reference: **GNN + rehearsal F1 86.61 ± 4.84%; TGNN F1 84.52 ± 2.63%**, five seeds and 200 epochs on L4. [Detailed verified record](REFERENCE_RESULTS.json) preserves all ten runs, thresholds, confusion counts, metrics and artifact hashes.
- [Component study protocol](TGNN_COMPONENT_PROTOCOL.md): 60 registered L4 runs testing GRU, persistent memory, additional attention and their combinations, plus normalization and weighted loss. [New model](tgnn_components.py), [runner](run_tgnn_components.py) and [Modal launcher](modal_tgnn_components.py) preserve the original source/results. Local acceptance: 45 passed before launch. Results are committed to Modal Volume `tg-detect-components-v1`, directory `components_200e_v1`, and downloaded to `artifacts/components_200e_v1` after completion.

- [Paper specification](BASELINE_PAPER_SPEC.md): independently extracted, with explicit unknowns.
- [Architecture audit](ARCHITECTURE_AUDIT.md): three incompatible baseline paths, tensor flow, data provenance and leakage findings.
- [Canonical baseline](pomsathit_baseline.py): executable, explicitly documented `PomsathitAdapted`; the exact paper-spec class remains blocked. [Adapted protocol](ADAPTED_PROTOCOL.md) and [architecture manifest](ADAPTED_BASELINE_ARCHITECTURE.json) record the choices.
- `../datasets/fixtures/real/ctu13/s01.npz` through `s03.npz` are **synthetic fixtures**, verified by exact array comparison. Folder names do not establish real-data provenance. Retained `ctx_*.npz` files are unverified derived caches.
- [TGNN pilot report](TGNN_PILOT_REPORT.md): 40 completed neural runs, 10 feature-control fits, and 50 memory/time interventions on freshly downloaded CTU-13 scenario 6. Five seeds, 3 epochs, 16,384 sampled real flows. The pilot TGNN has no convincing advantage; the feature-only tree scored highest on this small test.
- [Matched 200-epoch L4 report](MATCHED_TIME_REHEARSAL_REPORT.md): direct
  `GNN+rehearsal` versus `GNN+time+rehearsal` comparison. Time improves PR-AP and
  precision but lowers selected-threshold F1 and recall on this capture.
- [Research continuation](RESEARCH_UPDATE.md): additional primary sources and their implications.
- Legacy `modal_*.py` and `trigger.py` remain audit targets with known defects. The new tested path is `causal_tgn.py` / `run_causal_pilot.py`.

## Local work

Use the existing environment requested by the project owner:

```bash
source ~/main_env/bin/activate
python audit_environment.py
python pomsathit_baseline.py
python audit_models.py
python -m pytest -q -p no:cacheprovider test_research_foundation.py
```

Current acceptance suite:

```bash
~/main_env/bin/python -m pytest -q -p no:cacheprovider test_causal_pilot.py test_research_foundation.py
```

Result: **21 passed, 4 xfailed**. All 16 new-path checks pass. The four expected failures document defects in preserved legacy code. Synthetic fixtures use temporary directories; original inputs remain untouched.

Reproduce the pilot using the existing environment (downloads are reused only after checksum verification):

```bash
~/main_env/bin/python fetch_pilot_data.py --scenario 06
~/main_env/bin/python causal_data.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/main_env/bin/python run_causal_pilot.py --output artifacts/causal_pilot_reproduction
~/main_env/bin/python report_causal_pilot.py
```

The report verifier targets the preserved original `artifacts/causal_pilot_v1` run.
New runs should use a new output directory. The runner rejects changed source/data
when resuming. `ABLATION_MATRIX.json` represents the most recently invoked run;
the original matrix snapshot, source snapshot, results and verification remain under
their artifact directory.
`temporal_toy.py` is a separate synthetic memory-learning acceptance experiment.

The environment has CPU PyTorch. GPU-heavy experiments belong on Modal after its
runtime is verified. The complete factor study and unseen-capture evaluation remain
outstanding. Pilot metrics cannot establish an APT or paper-reproduction claim.

## Registered component study

```bash
~/main_env/bin/python -m pytest -q test_causal_pilot.py test_tgnn_components.py
~/main_env/bin/python report_tgnn_components.py --freeze-reference
~/main_env/bin/python -m modal run --detach modal_tgnn_components.py
```

Do not launch a duplicate while the study is active. The launcher can resume completed
runs after interruption only with identical frozen source and dataset hashes.
Each completed run saves a checkpoint, predictions, per-epoch losses, validation-only
threshold, metrics, resource measurements and verification. At completion the remote
job writes `TGNN_COMPONENT_REPORT.md`, `summary.json` and `verification.json`.

Read progress or download and verify finished runs without launching training:

```bash
~/main_env/bin/python sync_tgnn_components.py
~/main_env/bin/python sync_tgnn_components.py --download
```
# TGDetect

## KAIROS comparison and node memory V4

The [shared DEDALE experiment](kairos_comparison/README.md) implements the
KAIROS encoder/decoder, window correlation and investigation stages alongside
V3 and experimental V4 node memory. V4 adds elapsed-time decay and separate
endpoint admission. All arms use the same provenance feature schema and data
splits; their original training objectives differ.

Run `~/main_env/bin/python sync_kairos_comparison.py` from this directory to
read the durable Modal job status. Add `--download` after completion to fetch
and audit the comparison artifacts.

The benign-only follow-up is stored under `artifacts/kairos_dedale_shared41_benign/`. `TGNN_Node_Mem_V3_Benign` and `TGNN_Node_Mem_V4_Benign` use only wholly benign training windows with one-class binary targets; their attack-probability scores are compared with KAIROS queue scores.

## Consolidated results

See [RESULTS_LEDGER.md](RESULTS_LEDGER.md) for the collected metrics from the shared KAIROS/DEDALE runs, node-memory studies, component ablations, temporal CNN, causal/matched-time, DARPA E3, and Windows-APT transfer diagnostics.
