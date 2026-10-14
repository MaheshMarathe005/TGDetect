# Codebase review and cleanup

The pre-move scan read **1,088 source files / 190,088 lines** across the active
project, three packaged versions and bundled references. Python files parsed
successfully. Dependencies, build caches and Git internals were excluded. The
inventory records module docstrings, functions/classes with line numbers and
docstrings, imports, source hashes, and literal data/path references. This is a
structural code review; it does not certify every historical experiment.

## Active application

- `web_app/frontend/src/app/page.tsx` opens the CADETS workspace. `legacy/page.tsx`
  retains the separate Sysmon UI. Components under `components/cadets/` implement
  the console, investigation and detection graph; `components/tgdetect/` and
  `lib/tgdetect/` provide the earlier dashboard pages, graph helpers and API adapters.
- `web_app/backend/backend.py` creates FastAPI, loads prepared legacy telemetry,
  builds checkpoint registries, verifies checkpoint hashes and metadata, runs
  chronological streaming inference, and registers the CADETS/upload routers.
- `cadets.py` reads the prepared CADETS corpus and SQLite investigation index,
  provides node/event evidence, directed chronological chains, reconstruction
  explanations and exact checkpoint replay. It imports the V3 research packages
  from the root `research/` directory when needed.
- `graph_datasets.py` parses uploaded JSON/JSONL/CSV, checks event times, normalizes
  entities, scores with the verified V3 checkpoint, stores isolated dataset graphs,
  and serves entity search and connected neighborhoods.
- `teammate_api.py` adapts legacy artifact-backed models, events, datasets,
  analytics and exports for the earlier frontend. Its experiment results are
  separate from CADETS and are currently absent from this checkout.
- `web_app/backend/scripts/download_cadets.py` retrieves/verifies the original
  CADETS checkpoint and preparation inputs. `prepare_cadets.py` builds the local
  investigation index and recorded-score views. Invoke them as
  `python -m web_app.backend.scripts.<name>` from the repository root.
- `web_app/model/` contains inference definitions: causal GAT/TGN, component
  combinations, temporal CNN, selective node memory, V3 memory and the adapted
  baseline; `preprocessing.py` implements the 72-feature Sysmon extractor. Their
  internal relative imports remain intact. Weights remain in `datasets/` and
  experiment outputs rather than being embedded in source code.

## Research scripts and packages

- `sliding_node_mem/`, `sliding_node_mem_upgrade/`, `sliding_node_mem_v2/`,
  `sliding_node_mem_v3/`, and `sliding_node_mem_v4/` implement successive temporal
  graph pipelines: event encoding, windows, graph context, training, node memory,
  benchmarking, ablation, and evaluation. V3 is the active CADETS checkpoint path;
  V4 and the earlier packages remain research implementations.
- `tgnn_final/` assembles the selected V3 encoder and compatibility definitions.
  It records further planned components; it is not a complete replacement for
  the running V3 pipeline.
- `encoder_colab/` and `gat_study/` create standalone notebooks and perform
  encoder rarity/stability and graph-attention experiments, diagnostics and
  chronological data-slice audits. Their tests and recorded reports are retained.
- `orthrus_comparison/` prepares matched provenance benchmarks, constructs
  reference/shared-encoder models, runs evaluation and dependency tracing, and
  reports results. `orthrus_official/` is the bundled upstream reference source;
  functions named `testing` there are evaluation code, not temporary scripts.
- `kairos_comparison/` implements the shared KAIROS encoder/decoder, anomaly
  scoring, window correlation, audit and comparison stages.
- `publication/` contains reproducible study models, preprocessing, run logic,
  analysis, latency measurement and table/figure generation. Protocol JSONs and
  reference code remain here. Manuscript assets and generated paper outputs moved
  to `research paper/publication/`.
- `causal_data.py`, `dedale_data.py`, `darpa_pilot_data.py`, `windows_apt_data.py`,
  `universal_events.py` and data download/preparation scripts acquire telemetry,
  encode events and construct chronological splits. Raw dataset helpers remain
  beside their datasets in `datasets/raw/`.
- `run_*.py` train or evaluate a named study. `modal_*.py` package its source and
  data for cloud workers, execute experiments, and sometimes collect outputs.
  They were retained even when old because they record reproducibility logic.
  Importing or launching these commands may require Modal configuration.
- `sync_*.py` read durable remote status, retrieve outputs and verify results.
  `audit_*.py`, `check_*.py`, `report_*.py`, budget/profiling scripts and threshold
  calibration scripts inspect provenance, hashes, controls, metrics and latency.
  A name containing `check` or `profile` does not by itself make a file disposable.
- `test_*.py` contain maintained model/API/data regression checks and documented
  expected failures. `research/support/harness.py` is the retained synthetic
  fixture generator used by the foundation regression and environment audit.
  Its persistent fixture/caches moved to `datasets/fixtures/`.

See `SCRIPT_INVENTORY.md` for each individual script and
`CODEBASE_INVENTORY.json` for its definitions, dependencies and source hashes.

## Removed temporary scripts

Four unique `browser_check*.py` scripts used fixed browser selectors, a local
Chromium executable and one-off screenshot/export checks for previous UI work.
Three appeared in all four project versions: all **13 copies** were removed.
Their existing screenshots, exports and verification evidence were preserved.
`new/mock_backend.py` served embedded fake entities, illustrative scores and mock
API endpoints for UI testing; it was removed as well. Each unique script was
reviewed in full, and duplicate copies were verified by SHA-256 before deletion.
The exact paths, checksums, definitions and reasons are in
`RESTRUCTURE_MANIFEST.json`.

## Layout and preservation

The active application moved from `dashboard/` to `web_app/`, with model code in
`web_app/model/`. Imports now use `web_app.backend` and `web_app.model`. The
backend finds the repository root from its new depth and reads
`datasets/application/` and `research/artifacts/`.

Original `data/` became `datasets/raw/`; `research/data` is a relative symlink
for research commands run from `research/`. Saved experiment outputs moved to
`research/artifacts/`; affected local result collectors were updated. Papers,
extracted paper text and presentation assets moved to `research paper/`.

`old/`, `new/` and `tgdetectv3/` were preserved under `research/archives/`, including
their Git metadata and their distinct implementations. Their datasets physically
live in `datasets/archives/`; relative links preserve archived application paths.
Archived package manifests remain historical originals, including checksums for
the deleted temporary scripts. Archives and source snapshots are excluded from
active pytest collection. Generated Python/pytest/Next caches were removed.

All original retained files were checked at their relocated paths. Every original
dataset file retains the same inode and byte size after filesystem renaming. The
manifest includes before/after hashes for retained source files and a separate
list of intentional source edits. No model weights or dataset bytes were edited.
Existing user modifications and staged changes were carried with their files;
Git was not reset, staged, committed or pushed.

## Validation and existing limitations

The active CADETS/API/upload/model/foundation suite completed with **47 passed
and 4 documented expected failures**. It includes real checkpoint replay,
chronological chain checks and uploaded-event scores against saved losses.
The frontend production build passed, including TypeScript and static routes.
Frontend lint reports **13 pre-existing errors**; all affected frontend source
files have unchanged before/after hashes.

The broader initial selection also found **10 failures** in the optional legacy
Sysmon tests because the separate DEDALE result directories were absent before
cleanup. The model checkpoint files alone do not supply the missing registries,
calibration results and saved predictions. These tests remain in place, and no
results or metrics were fabricated to make them pass. Other historical research
paths may require separately downloaded experiment artifacts or cloud access.

Wider test collection now finds **261 tests** without errors or duplicate archive
collection. The notebook bundle regression was fixed to restore imported package
state after loading its temporary standalone source bundle. Its previously
persistent incomplete packages hid the live model modules during collection. A
second suite covering notebook encoder behavior, V3, graph attention, slice audits
and upgraded pipelines completed with **73 passed**. The two verification suites
overlap on V3 tests; their counts are separate run totals. The environment audit
also parsed 252 live Python files, verified current source receipts, inspected 15
fixture datasets, and reported no unexpected source changes.
