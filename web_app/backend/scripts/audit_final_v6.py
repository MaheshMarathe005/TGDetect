"""Replay V6 separately; compare archived channels and raw/offline metrics.

Run from repository root: python -m web_app.backend.scripts.audit_final_v6
"""
import importlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from web_app.model.final_v6_candidate import ROOT, RUN, digest, load_model, modules


def main():
    torch.set_num_threads(2)
    package = modules()
    study = importlib.import_module(package + '.study_v3')
    output = ROOT / 'datasets/application/final_v6_candidate/audit'
    output.mkdir(parents=True, exist_ok=True)
    model = load_model()
    report = {'checkpoint_sha256': digest(RUN / 'model.pt'),
              'strict_load': True, 'device': 'cpu', 'torch': str(torch.__version__),
              'parameters': sum(p.numel() for p in model.parameters()),
              'model_config': model.config, 'datasets': {}}
    for name, saved_dir in [('cadets', RUN), ('theia', ROOT / 'final_v6_results/v6_s0_transfer_theia')]:
        data = study.DataBundle(ROOT / 'datasets/application/final_v6_candidate' / name)
        dest = output / name
        dest.mkdir(exist_ok=True)
        started = time.monotonic()
        rows = {}
        differences = {}
        for split in ('val', 'test'):
            print('Replaying', name, split, flush=True)
            rows[split] = study.export_channels(data, model, split, dest, 1024)
            archived = np.memmap(saved_dir / (split + '_channels.bin'), dtype=study.CHANNEL_DTYPE, mode='r')
            differences[split] = {
                'events': len(rows[split]),
                'metadata_equal': all(np.array_equal(rows[split][k], archived[k]) for k in study.CHANNEL_DTYPE.names if k not in ('ce', 'pair')),
                'ce_max_absolute_error': float(np.max(np.abs(rows[split]['ce'] - archived['ce']))),
                'pair_max_absolute_error': float(np.max(np.abs(rows[split]['pair'] - archived['pair'])))}
        test = rows['test']
        ids = np.unique(np.concatenate([test['src'], test['dst']]))
        scores = np.zeros(data.n_nodes)
        np.maximum.at(scores, test['src'], test['ce'])
        np.maximum.at(scores, test['dst'], test['ce'])
        threshold = float(rows['val']['ce'].max())
        raw = study.metrics(ids, scores[ids], data.truth, set(ids[scores[ids] > threshold]))
        evaluation = study.evaluate_channels(data, rows['val'], test, dest)
        recorded = json.loads((saved_dir / 'evaluation.json').read_text())
        result = {'data': data.signature, 'replay_seconds': time.monotonic() - started,
                  'channels': differences, 'raw_neural_paper_threshold': raw,
                  'raw_threshold': threshold,
                  'offline_pipeline': {k: v['metrics'] for k, v in evaluation.items()},
                  'archived_metrics_match': all(
                      evaluation[k]['metrics'][m] == recorded[k]['metrics'][m]
                      for k in recorded for m in ('tp', 'fp', 'fn', 'tn'))}
        report['datasets'][name] = result
        (output / 'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        print(name, json.dumps(result), flush=True)
    print('Audit saved:', output / 'audit.json', flush=True)


if __name__ == '__main__':
    main()
