"""One L4 allocation; benchmarked concurrency among independent training runs."""
import modal

VOLUME_NAME="tg-detect-temporal-cnn-v1"
RUN_DIR="temporal_cnn_200e_v2"
image=(modal.Image.debian_slim(python_version="3.10")
       .uv_pip_install("torch==2.7.1","numpy==2.2.6","scikit-learn==1.7.2","pytest==9.0.2")
       .env({"CUBLAS_WORKSPACE_CONFIG":":4096:8","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
       .add_local_python_source("pomsathit_baseline","causal_tgn","causal_data","tgnn_components","run_causal_pilot",
                                "run_tgnn_components","temporal_cnn","run_temporal_cnn")
       .add_local_file("data/ctu13_verified/scenario06_pilot16384.npz","/root/scenario06_pilot16384.npz"))
for name in ("test_temporal_cnn.py","modal_temporal_cnn.py","TEMPORAL_CNN_PROTOCOL.md"):
    image=image.add_local_file(name,f"/root/{name}")
volume=modal.Volume.from_name(VOLUME_NAME,create_if_missing=True)
app=modal.App("tg-detect-temporal-cnn",image=image)


@app.function(gpu="L4",cpu=16,memory=49152,timeout=14400,volumes={"/results":volume},max_containers=1)
def run_all():
    from run_temporal_cnn import run_study
    return run_study("/root/scenario06_pilot16384.npz",f"/results/{RUN_DIR}",volume.commit)


@app.local_entrypoint()
def main():
    import json
    from pathlib import Path
    summary=run_all.remote()
    output=Path("artifacts")/RUN_DIR
    for entry in volume.iterdir(RUN_DIR,recursive=True):
        if entry.type==modal.volume.FileEntryType.FILE:
            target=output/Path(entry.path).relative_to(RUN_DIR)
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("wb") as handle:
                for block in volume.read_file(entry.path):
                    handle.write(block)
    print(json.dumps(summary,indent=2))
