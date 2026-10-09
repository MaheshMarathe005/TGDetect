# TG-Detect

The running application lives in `dashboard/`. Training, testing, model versions,
datasets, experiment artifacts, and research reports live in [research/](research/README.md).

## Run the project

From the repository root, start the backend:

```sh
/home/mahesh/main_env/bin/python -m uvicorn dashboard.backend:app --host 127.0.0.1 --port 8000
```

In another terminal, start the frontend:

```sh
cd dashboard/frontend
npm ci
npm run build
npm run start
```

Open http://localhost:3000. See [dashboard instructions](dashboard/README.md) for configuration.

`dashboard/runtime/` contains the model definitions and preprocessing required for
inference. Runtime telemetry and uploaded datasets remain in `dashboard/data/`.
The backend reads saved experiment results and checkpoints from `research/artifacts/`.

## Research and tests

Run research scripts from their new directory so existing relative paths work:

```sh
cd research
/home/mahesh/main_env/bin/python -m pytest -q
```

To check the dashboard from the repository root:

```sh
/home/mahesh/main_env/bin/python -m pytest -q research/test_dashboard.py research/test_teammate_dashboard.py research/test_node_memory_v3.py
```

Research script contents and experiment artifacts are preserved. The move inventory
and original script hashes are in [research/RELOCATION_MANIFEST.json](research/RELOCATION_MANIFEST.json).
