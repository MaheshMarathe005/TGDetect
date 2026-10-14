"""Matched B+H versus B+T+H at the paper-stated 200 epochs."""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch.nn import functional as F

from causal_data import sha256
from causal_tgn import CausalTGNN, Reservoir
from run_causal_pilot import PilotData, metrics, predict, save_json, select_threshold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data/ctu13_verified/scenario06_pilot16384.npz'
OUT = ROOT / 'artifacts/matched_time_rehearsal_200e'
EPOCHS = 200
SEEDS = list(range(5))
ARMS = {'B_H': ('H',), 'B_T_H': ('T', 'H')}
SOURCES = [
    'pomsathit_baseline.py', 'causal_data.py', 'causal_tgn.py',
    'run_causal_pilot.py', 'run_matched_time_rehearsal.py',
    'ADAPTED_PROTOCOL.md', 'MATCHED_TIME_REHEARSAL_PROTOCOL.md',
]


def train(data, arm, factors, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = CausalTGNN(data.x.shape[1], factors)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)
    torch.manual_seed(seed + 10000)
    reservoir = Reservoir(data.train_end // 10, data.train_end, seed + 937)
    replay_visits = 0
    losses = []
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        model.reset_state()
        total = 0.0
        for lo in range(0, data.train_end, 256):
            hi = min(lo + 256, data.train_end)
            optimizer.zero_grad(set_to_none=True)
            logits, context = model.forward_events(*data.batch(slice(lo, hi)))
            live_loss = F.cross_entropy(logits, data.y[lo:hi])
            replay_items = reservoir.sample(26)
            loss = live_loss
            if replay_items:
                ids = np.array([event_id for event_id, _ in replay_items])
                stored_context = torch.stack([context for _, context in replay_items])
                replay_logits = model.replay(*data.batch(ids)[:3], stored_context)
                loss = loss + F.cross_entropy(replay_logits, data.y[ids])
                replay_visits += len(replay_items)
            loss.backward()
            optimizer.step()
            model.detach_state()
            total += float(live_loss.detach()) * (hi - lo)
            for event_id in range(lo, hi):
                reservoir.admit(event_id, context[event_id - lo])
        losses.append(total / data.train_end)
        if (epoch + 1) % 25 == 0:
            print(json.dumps({'arm': arm, 'seed': seed, 'epoch': epoch + 1,
                              'live_loss': losses[-1]}), flush=True)

    train_seconds = time.perf_counter() - started
    run_id = f'{arm}_seed{seed}'
    checkpoint = OUT / f'{run_id}.pt'
    torch.save({'state_dict': model.state_dict(), 'factors': list(factors),
                'seed': seed, 'epochs': EPOCHS}, checkpoint)
    probabilities, inference_seconds = predict(model, data, seed=seed)
    validation = slice(data.train_end, data.val_end)
    test = slice(data.val_end, data.n)
    threshold = select_threshold(data.y[validation].numpy(), probabilities[validation])
    source_hashes = {name: sha256(ROOT / name) for name in SOURCES}
    result = {
        'run_id': run_id, 'arm': arm, 'factors': list(factors), 'seed': seed,
        'status': 'COMPLETE', 'epochs': EPOCHS, 'batch_size': 256,
        'threshold': threshold, 'epoch_losses': losses,
        'validation': metrics(data.y[validation].numpy(), probabilities[validation], threshold),
        'test': metrics(data.y[test].numpy(), probabilities[test], threshold),
        'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'train_seconds': train_seconds, 'inference_seconds': inference_seconds,
        'inference_events_per_second': data.n / inference_seconds,
        'training_live_visits': data.train_end * EPOCHS,
        'training_replay_visits': replay_visits,
        'reservoir_capacity': reservoir.capacity,
        'replay_context_bytes': sum(context.numel() * context.element_size() + 8
                                    for _, context in reservoir.items),
        'peak_process_rss_kib_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'checkpoint_bytes': checkpoint.stat().st_size,
        'checkpoint_sha256': sha256(checkpoint),
        'data_sha256': sha256(data.path), 'source_hashes': source_hashes,
        'scope': 'matched adapted architecture comparison; original Wazuh data unavailable',
    }
    np.savez_compressed(OUT / f'{run_id}_predictions.npz', probability=probabilities,
                        y=data.y.numpy(), raw_row_id=data.row_ids)
    save_json(OUT / f'{run_id}.json', result)
    print(json.dumps({'run_id': run_id, 'test': result['test'],
                      'train_seconds': train_seconds}), flush=True)
    return result


def summarize(results):
    fields = ['f1', 'pr_ap', 'roc_auc', 'precision', 'recall', 'fpr']
    arms = {}
    for arm in ARMS:
        rows = [row for row in results if row['arm'] == arm]
        arms[arm] = {
            'seeds': len(rows), 'parameters': rows[0]['parameters'],
            **{field: {'mean': float(np.mean([row['test'][field] for row in rows])),
                       'sd': float(np.std([row['test'][field] for row in rows], ddof=1))}
               for field in fields},
            'mean_train_seconds': float(np.mean([row['train_seconds'] for row in rows])),
        }
    paired = {}
    indexed = {(row['arm'], row['seed']): row for row in results}
    for field in fields:
        values = [indexed[('B_T_H', seed)]['test'][field] -
                  indexed[('B_H', seed)]['test'][field] for seed in SEEDS]
        paired[field] = {'seed_differences': values, 'mean': float(np.mean(values)),
                         'sd': float(np.std(values, ddof=1))}
    return {'comparison': 'B_T_H minus B_H', 'arms': arms,
            'paired_test_difference': paired,
            'claim_limit': 'Same inspected CTU test; missing factorial arm completion, not independent confirmation.'}


def verify(data, results):
    validation = slice(data.train_end, data.val_end)
    test = slice(data.val_end, data.n)
    for result in results:
        run_id = result['run_id']
        assert sha256(OUT / f'{run_id}.pt') == result['checkpoint_sha256']
        saved = np.load(OUT / f'{run_id}_predictions.npz', allow_pickle=False)
        threshold = select_threshold(saved['y'][validation], saved['probability'][validation])
        assert threshold == result['threshold']
        assert metrics(saved['y'][test], saved['probability'][test], threshold) == result['test']
        assert result['epochs'] == EPOCHS and result['reservoir_capacity'] == data.train_end // 10
        assert result['source_hashes'] == {name: sha256(ROOT / name) for name in SOURCES}
    return {'status': 'PASS', 'runs': len(results), 'checkpoints': 'PASS',
            'prediction_metric_recomputation': 'PASS', 'source_hashes': 'PASS',
            'data_sha256': sha256(data.path),
            'verified_utc': datetime.now(timezone.utc).isoformat()}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True, exist_ok=True)
    data = PilotData(DATA)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
        'executable': sys.executable, 'torch': torch.__version__, 'numpy': np.__version__,
        'sklearn': sklearn.__version__, 'platform': platform.platform(), 'device': 'cpu',
        'threads': 1, 'epochs': EPOCHS, 'batch_size': 256, 'seeds': SEEDS,
        'arms': {name: list(factors) for name, factors in ARMS.items()},
        'data_sha256': sha256(DATA),
        'source_hashes': {name: sha256(ROOT / name) for name in SOURCES},
    }
    manifest_path = OUT / 'run_manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ['epochs', 'batch_size', 'seeds', 'arms', 'data_sha256', 'source_hashes']:
            if previous[key] != manifest[key]:
                raise ValueError(f'Resume manifest mismatch: {key}')
    else:
        save_json(manifest_path, manifest)

    results = []
    for arm, factors in ARMS.items():
        for seed in SEEDS:
            path = OUT / f'{arm}_seed{seed}.json'
            result = json.loads(path.read_text()) if path.exists() else train(data, arm, factors, seed)
            results.append(result)
            if len(results) == len(ARMS) * len(SEEDS):
                save_json(OUT / 'summary.json', summarize(results))
    verification = verify(data, results)
    save_json(OUT / 'verification.json', verification)
    print(json.dumps({'summary': summarize(results), 'verification': verification}, indent=2), flush=True)


if __name__ == '__main__':
    main()
