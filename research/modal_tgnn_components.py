"""One resumable L4 job for the registered 60-run study."""
from pathlib import Path

import modal

APP_NAME = "tg-detect-component-study"
VOLUME_NAME = "tg-detect-components-v1"
RUN_DIR = "components_200e_v1"
MODULES = ["pomsathit_baseline", "causal_tgn", "causal_data", "run_causal_pilot", "tgnn_components", "run_tgnn_components", "report_tgnn_components"]
image = (modal.Image.debian_slim(python_version="3.10")
         .uv_pip_install("torch==2.7.1", "numpy==2.2.6", "scikit-learn==1.7.2", "pytest==9.0.2")
         .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
         .add_local_python_source(*MODULES)
         .add_local_file("data/ctu13_verified/scenario06_pilot16384.npz", "/root/scenario06_pilot16384.npz"))
for name in ["test_causal_pilot.py", "test_tgnn_components.py", "modal_tgnn_components.py", "TGNN_COMPONENT_PROTOCOL.md", "REFERENCE_RESULTS.json"]:
    image = image.add_local_file(name, f"/root/{name}")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
app = modal.App(APP_NAME, image=image)


@app.function(gpu="L4", cpu=4, memory=16384, timeout=86400, volumes={"/results": volume}, max_containers=1)
def run_all():
    from run_tgnn_components import run_study
    return run_study("/root/scenario06_pilot16384.npz", f"/results/{RUN_DIR}", volume.commit)


@app.local_entrypoint()
def main():
    import json
    summary = run_all.remote()
    output = Path("artifacts") / RUN_DIR
    output.mkdir(parents=True, exist_ok=True)
    # Each file is already durable; download via Volume even after a detached job.
    for entry in volume.iterdir(RUN_DIR, recursive=True):
        if entry.type == modal.volume.FileEntryType.FILE:
            relative = Path(entry.path).relative_to(RUN_DIR)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as handle:
                for block in volume.read_file(entry.path):
                    handle.write(block)
    print(json.dumps(summary, indent=2))
