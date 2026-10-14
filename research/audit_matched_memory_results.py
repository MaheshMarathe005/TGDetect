"""Verify matched budgets, saved model hashes and independently computed metrics."""
import hashlib
import json
from pathlib import Path

import numpy as np
from matched_memory_training import ARMS, metrics, threshold_for

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / 'artifacts/dedale_host_mem_matched/results'


def main():
    reference = None
    groups = {arm: [] for arm in ARMS}
    for seed in range(3):
        rows = []
        for arm in ARMS:
            stem = f'{arm}_seed{seed}'
            r = json.loads((FOLDER / f'{stem}.json').read_text())
            a = np.load(FOLDER / f'{stem}_predictions.npz', allow_pickle=False)
            y, p = a['y'], a['probability']
            tr, va = int(a['train_end']), int(a['val_end'])
            identity = (y, a['ids'], tr, va)
            if reference is None:
                reference = identity
            else:
                assert np.array_equal(reference[0], y) and np.array_equal(reference[1], a['ids'])
                assert reference[2:] == (tr, va)
            assert r['status'] == 'complete' and r['epochs_trained'] == 80
            assert hashlib.sha256((FOLDER / f'{stem}.pt').read_bytes()).hexdigest() == r['checkpoint_sha256']
            threshold = threshold_for(y[tr:va], p[tr:va])
            assert abs(threshold - r['val_threshold']) < 1e-12
            for name, t in [('test_val_threshold', threshold), ('test_default_0_5', .5)]:
                actual = metrics(y[va:], p[va:], t)
                for k, v in actual.items():
                    assert np.isclose(v, r[name][k], atol=1e-10), (arm, seed, k)
            rows.append(r)
            groups[arm].append(r)
        for field in ['warm_start_sha256', 'dataset_sha256', 'optimizer_steps', 'training_live_visits',
                      'training_replay_visits', 'replay_schedule_sha256', 'train_batch', 'infer_batch',
                      'tbptt_k', 'learning_rate', 'epochs_trained']:
            assert len({r[field] for r in rows}) == 1, (seed, field)
    summary = {'status': 'verified', 'protocol': 'Matched additional training from same-seed QAC warm starts',
               'checks': ['same labels/event IDs/splits', 'same warm-start hashes per seed',
                          'same steps/live visits/replay schedule', 'checkpoint hashes', 'thresholds', 'metrics'],
               'arms': {}}
    for arm, rows in groups.items():
        item = {'seeds': [{'seed': r['seed'], 'threshold': r['val_threshold'],
                          'best_epoch': r['best_epoch'], 'validation_pr_ap': r['best_val_pr_ap'],
                          'metrics': r['test_val_threshold']} for r in rows]}
        for name in ('test_val_threshold', 'test_default_0_5'):
            item[name] = {k: {'mean': float(np.mean([r[name][k] for r in rows])),
                              'sd': float(np.std([r[name][k] for r in rows], ddof=1))}
                          for k in rows[0][name] if k != 'threshold'}
        item['latency'] = {k: float(np.mean([r['latency'][k] for r in rows]))
                           for k in ('p50_ms', 'p95_ms', 'batched_events_per_s')}
        summary['arms'][arm] = item
    dest = FOLDER.parent
    (dest / 'verified_comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# Matched memory comparison', '', 'All nine runs finished 80 additional epochs. All budget and artifact checks passed.', '',
             'Same per-seed QAC warm start, learning rate 1e-4, 512-event batches, three-batch accumulation, 4,800 optimizer steps, and identical replay schedules. Only the memory treatment differs.', '',
             '## Test metrics at validation-selected thresholds', '',
             '| Arm | Accuracy | Precision | Recall | Specificity | F1 | PR-AP | ROC-AUC | FP | FN |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = summary['arms'][arm]['test_val_threshold']
        cells = [f'{m[k]["mean"]:.5f}' for k in ('accuracy','precision','recall','specificity','f1','pr_ap','roc_auc')]
        cells += [f'{m[k]["mean"]:.1f}' for k in ('fp','fn')]
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Fixed 0.5 operating threshold', '', '| Arm | Precision | Recall | F1 | FP | FN |',
              '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = summary['arms'][arm]['test_default_0_5']
        lines.append('| ' + arm + ' | ' + ' | '.join(f'{m[k]["mean"]:.5f}' for k in ('precision','recall','f1')) +
                     ' | ' + ' | '.join(f'{m[k]["mean"]:.1f}' for k in ('fp','fn')) + ' |')
    lines += ['', 'Latency settings match across arms (batch 512, 1,000 online events, first 200 omitted). Separate seed jobs ran concurrently on distinct L4s; an isolated repeat is still needed for a precise hardware comparison.', '',
              'This evaluates additional training from trained QAC checkpoints, not equal-budget training from scratch. Multiple changes form the V3 treatment, so architectural attribution requires further ablations. Three seeds on one chronological split do not establish cross-dataset generalization.']
    (dest / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({arm: {name: {k: round(v['mean'], 6) for k,v in item[name].items()}
                           for name in ('test_val_threshold','test_default_0_5')}
                      for arm,item in summary['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
