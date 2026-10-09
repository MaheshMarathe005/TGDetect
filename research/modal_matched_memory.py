"""Deploy then spawn three independent matched-seed GPU jobs."""
import json
from pathlib import Path

import modal

app = modal.App('tg-detect-memory-matched')
volume = modal.Volume.from_name('tg-detect-dedale-v1')
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.4.1', 'numpy==1.26.4', 'scikit-learn==1.5.2', 'pytest')
         .env({'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})
         .add_local_python_source('matched_memory_training', 'node_memory_v3', 'node_memory_models',
                                  'test_node_memory_v3', 'tgnn_components', 'temporal_cnn',
                                  'pomsathit_baseline', 'causal_tgn', 'causal_data'))


@app.function(image=image, gpu='L4', cpu=8, memory=16384, timeout=14400,
              volumes={'/vol': volume}, max_containers=3)
def train(seed: int, epochs: int = 80):
    import os
    import pytest
    import test_node_memory_v3
    import torch
    from matched_memory_training import train_seed
    os.environ['TGNN_TEST_DEVICE'] = 'cuda'
    torch.use_deterministic_algorithms(True)
    if pytest.main([test_node_memory_v3.__file__, '-q']) != 0:
        raise RuntimeError('GPU acceptance gates failed')
    return train_seed(seed, volume, epochs)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--spawn', action='store_true')
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    if args.spawn:
        function = modal.Function.from_name('tg-detect-memory-matched', 'train')
        calls = [{'seed': seed, 'function_call_id': function.spawn(seed=seed, epochs=args.epochs).object_id}
                 for seed in range(3)]
        manifest = dict(app='tg-detect-memory-matched', protocol='matched-memory-v1-20261002',
                        epochs=args.epochs, arms=['TGNN_Q_A_C', 'TGNN_Node_Mem_V3_Fixed', 'TGNN_Node_Mem_V3'],
                        remote_results='/results/dedale_host_mem_matched', calls=calls)
        path = Path('artifacts/dedale_host_mem_matched/launch.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
