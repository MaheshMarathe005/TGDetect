# TG-Detect dashboard

The active frontend is the teammate Next.js interface copied from
`/home/mahesh/Desktop/tgdetect_frountend_temp_pratham/TGDetect-Temporal-Graph/frontend`.
Its MIT license is retained in `frontend/LICENSE`. The previous Vite frontend is
preserved locally in `../research/frontend_legacy/`.

Start the backend from the repository root:

```sh
/home/mahesh/main_env/bin/python -m uvicorn dashboard.backend:app --host 127.0.0.1 --port 8000
```

Start the frontend in another terminal:

```sh
cd dashboard/frontend
npm ci
npm run build
npm run start
```

Open http://localhost:3000. Next.js proxies `/api` to the backend. Set
`TGDETECT_API_ORIGIN` when building/running if the backend uses another address.
The frontend has no embedded benchmark metrics or demo-model fallback.

The eight teammate sections use `/api/ui` adapters in `teammate_api.py`; the
original backend API remains available. Model selection lists the nine matched
Modal checkpoints. The default is selected by validation average precision.
The model page shows actual architecture, saved training history, evaluation,
matched comparison at both thresholds, and live checkpoint inference. Sessions
isolate memory and verify checkpoint hashes and metadata. No frontend operation
starts training.

The benchmark contains 153,056 prepared DEDALE events, with benign sampling.
Recorded process-to-host observations are graph edges. Ground truth and model
predictions are separate. Entity-time investigation groups labeled malicious
events from one source process, splitting gaps above 300 seconds. They are not
causal attack-path proof; MITRE annotations and explicit causal-parent links are
unavailable. Graph rendering samples up to 300 nodes / 600 edges from the loaded
window; the canvas reports its actual displayed counts. Analytics blocks are
visualization aggregates, not model training snapshots.

Datasets accepts up to 20 MB of chronological Sysmon/Winlogbeat JSON arrays or
JSONL. It uses the exact recovered training scaler (verified against all prepared
features with zero absolute error), 72-feature extraction, and the selected saved
checkpoint. Uploaded labels remain unknown. An upload's scores record the model
that produced them; choosing another model does not silently replace those scores.
Choose a new dataset ID to process the raw upload with another checkpoint. Only
Sysmon telemetry is compatible with these models; NetFlow/synthetic parser options
are excluded. Uploaded data and scores are stored locally under `dashboard/data`.
Artifacts exports stream complete JSON derived from the selected real dataset.

Validation:

```sh
/home/mahesh/main_env/bin/python -m pytest -q research/test_dashboard.py research/test_teammate_dashboard.py research/test_node_memory_v3.py
/home/mahesh/main_env/bin/python research/browser_check.py
```

Browser screenshots and verification: `research/artifacts/teammate_frontend_review/`.
