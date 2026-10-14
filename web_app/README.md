# Web application

`frontend/` contains the Next.js UI. `backend/` contains the FastAPI APIs,
checkpoint loading, dataset upload and preparation commands. `model/` contains
inference model definitions and preprocessing; experiment implementations remain
in the local, ignored `../research/` folder. CADETS replay and upload inference
require those research packages; they are temporarily excluded from GitHub.

Run from the repository root:

```sh
python -m uvicorn web_app.backend.backend:app --host 127.0.0.1 --port 8000
npm --prefix web_app/frontend run dev
```

Data and uploads are stored in `../datasets/application/`. Saved experiment
results are read from `../research/artifacts/`. The default workspace uses CADETS
V3; the optional `/legacy` Sysmon workspace requires its separate result artifacts.
See [CADETS instructions](backend/CADETS.md).
