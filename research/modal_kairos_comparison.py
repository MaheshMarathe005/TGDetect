"""Prepare shared provenance data, then run each matched seed on one Modal L4.

From repository root: PYTHONPATH=research python -m modal run --detach
research/modal_kairos_comparison.py --epochs 50 --seeds 3
"""
from pathlib import Path
import json
import modal

app = modal.App('tg-detect-kairos-comparison-v1')
volume = modal.Volume.from_name('tg-detect-dedale-v1')
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.4.1', 'torch-geometric==2.6.1', 'numpy==1.26.4',
                 'scikit-learn==1.5.2', 'networkx==3.4.2', 'pytest==8.3.5')
    .env({'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
    .add_local_python_source('kairos_comparison', 'node_memory_v4', 'node_memory_v3',
        'node_memory_models', 'matched_memory_training', 'tgnn_components', 'temporal_cnn',
        'pomsathit_baseline', 'causal_tgn', 'causal_data', 'test_kairos_comparison'))

PREPARED = '/vol/prepared/dedale_kairos_shared41_v1'
RESULTS = '/vol/results/dedale_kairos_shared41_v4_benign2'


@app.function(image=image, cpu=2, memory=4096, timeout=600, volumes={'/vol': volume})
def preflight():
    import zipfile
    import torch
    volume.reload()
    with zipfile.ZipFile('/vol/raw/system_logs_winlogbeat.zip') as z:
        members = z.infolist()
        result = dict(raw_bytes=Path('/vol/raw/system_logs_winlogbeat.zip').stat().st_size,
            member_count=len(members), uncompressed_bytes=sum(i.file_size for i in members),
            examples=[i.filename for i in members[:12]])
    result['existing_prepared'] = [str(p) for p in Path('/vol/prepared').glob('*')]
    result['torch'] = torch.__version__
    return result


@app.function(image=image, cpu=16, memory=65536, timeout=86400, volumes={'/vol': volume})
def prepare_data():
    import json
    from pathlib import Path
    import zipfile
    from kairos_comparison.data import prepare, read_member, timestamp, LABEL_MEMBER
    volume.reload()
    # The prepared store is immutable once its manifest and array are present;
    # avoid rescanning the 27 GB archive on resumed comparison runs.
    if all(Path(PREPARED, name).exists() for name in ('manifest.json', 'prepared.npz', 'nodes.json', 'events.sqlite')):
        return json.loads(Path(PREPARED, 'manifest.json').read_text())
    with zipfile.ZipFile('/vol/raw/system_logs_labels.zip') as z:
        times = [timestamp(e) for e in read_member(z, LABEL_MEMBER)]
    try:
        # Same established active-attack interval as the earlier DEDALE study.
        # All supported benign events are retained, not reservoir sampled.
        return prepare('/vol/raw/system_logs_winlogbeat.zip', '/vol/raw/system_logs_labels.zip', PREPARED,
                       start=min(times), end=max(times)+.000001, workers=8, on_checkpoint=volume.commit)
    finally:
        volume.commit()


@app.function(image=image, gpu='L4', cpu=8, memory=65536, timeout=86400,
              volumes={'/vol': volume}, max_containers=3)
def train_seed(seed: int, epochs: int = 50, smoke_only: bool = False):
    import os
    import pytest
    import test_kairos_comparison
    from kairos_comparison.run import train_one, report
    volume.reload()
    os.environ['TGNN_COMPARISON_TEST_DEVICE'] = 'cuda'
    # Full pipeline smoke trains all benign-only arms on the actual GPU first.
    if pytest.main([test_kairos_comparison.__file__, '-q']) != 0:
        raise RuntimeError('Acceptance checks failed')
    if smoke_only:
        import torch
        return dict(status='GPU acceptance passed', gpu=torch.cuda.get_device_name(0))
    out = Path(RESULTS) / f'seed{seed}'
    results = []
    arms = ['KAIROS', 'TGNN_Node_Mem_V3_Benign', 'TGNN_Node_Mem_V4_Benign']
    arms = arms[seed % 3:] + arms[:seed % 3]
    for arm in arms:
        try:
            results.append(train_one(Path(PREPARED)/'prepared.npz', out, arm, seed,
                                     epochs=epochs, batch=1024, device='cuda', on_checkpoint=volume.commit))
            report(out)
        finally:
            volume.commit()
    return results


@app.function(image=image, cpu=1, memory=8192, timeout=86400, volumes={'/vol': volume})
def pipeline(epochs: int = 50, seeds: int = 3):
    """Deployed coordinator survives the lifetime of the launching local process."""
    import traceback
    from kairos_comparison.audit import audit
    if not 1 <= seeds <= 3 or epochs < 1:
        raise ValueError('Positive epochs and 1..3 seeds required')
    state_path = Path(RESULTS) / 'pipeline_status.json'
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = dict(status='preparing', epochs=epochs, seeds=seeds)
    state_path.write_text(json.dumps(state, indent=2)); volume.commit()
    try:
        volume.reload()
        preparation = json.loads((Path(PREPARED) / 'manifest.json').read_text())
        state.update(status='training', preparation=preparation)
        state_path.write_text(json.dumps(state, indent=2)); volume.commit()
        results = list(train_seed.starmap([(seed, epochs) for seed in range(seeds)]))
        volume.reload()
        checked = audit(Path(PREPARED)/'prepared.npz', RESULTS, expected_seeds=seeds, expected_arms=['KAIROS', 'TGNN_Node_Mem_V3_Benign', 'TGNN_Node_Mem_V4_Benign'])
        state.update(status='complete', audit=checked)
        state_path.write_text(json.dumps(state, indent=2)); volume.commit()
        return state
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        state_path.write_text(json.dumps(state, indent=2)); volume.commit()
        raise


@app.local_entrypoint()
def main(epochs: int = 50, seeds: int = 3, prepare_only: bool = False, check_only: bool = False, skip_prepare: bool = False, smoke_only: bool = False):
    if epochs < 1 or seeds < 1 or seeds > 3:
        raise ValueError('Positive epochs and 1..3 seeds required')
    if check_only:
        print(json.dumps(preflight.remote(), indent=2))
        return
    if smoke_only:
        print(json.dumps(train_seed.remote(seed=0, smoke_only=True), indent=2))
        return
    if not skip_prepare:
        print(json.dumps(prepare_data.remote(), indent=2))
    if not prepare_only:
        for result in train_seed.starmap([(seed, epochs) for seed in range(seeds)]):
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--spawn', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    if not args.spawn:
        parser.error('Deploy with modal deploy, then invoke this script with --spawn')
    function = modal.Function.from_name('tg-detect-kairos-comparison-v1', 'pipeline')
    call = function.spawn(epochs=args.epochs, seeds=args.seeds)
    launch = dict(app='tg-detect-kairos-comparison-v1', function_call_id=call.object_id,
                  epochs=args.epochs, seeds=args.seeds, remote_results=RESULTS,
                  remote_prepared=PREPARED)
    path = Path('artifacts/kairos_dedale_shared41/launch.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(launch, indent=2))
    print(json.dumps(launch, indent=2))
