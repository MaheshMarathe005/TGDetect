"""Independently verify downloaded V3/previous predictions and write a report."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

ROOT = Path(__file__).resolve().parent / 'artifacts/dedale_host_mem_v3'
ARMS = ('GNN_H', 'TGNN_Q_A_C', 'TGNN_Node_Mem_Sel', 'TGNN_Node_Mem_V3')


def metrics(y, p, threshold):
    tn, fp, fn, tp = confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel()
    return dict(tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
                accuracy=float((tn + tp) / len(y)),
                precision=float(tp / max(tp + fp, 1)), recall=float(tp / max(tp + fn, 1)),
                specificity=float(tn / max(tn + fp, 1)), f1=float(2 * tp / max(2 * tp + fp + fn, 1)),
                pr_ap=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)))


def main():
    records, reference = {}, None
    for arm in ARMS:
        folder = ROOT / ('results_v3' if arm.endswith('V3') else 'results_previous')
        rows = []
        for seed in range(3):
            record = json.loads((folder / f'{arm}_seed{seed}.json').read_text())
            a = np.load(folder / f'{arm}_seed{seed}_predictions.npz', allow_pickle=True)
            y, p = a['y'], a['probability']
            tr, va = int(a['train_end']), int(a['val_end'])
            assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
            identity = (y, a['ids'], tr, va)
            if reference is None:
                reference = identity
            else:
                assert np.array_equal(y, reference[0]), 'Labels differ'
                assert np.array_equal(a['ids'], reference[1]), 'Event IDs differ'
                assert (tr, va) == reference[2:], 'Splits differ'
            assert record['status'] == 'complete' and not record['time_boxed']
            prec, rec, thresholds = precision_recall_curve(y[tr:va], p[tr:va])
            vf1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-15)
            selected = float(thresholds[np.flatnonzero(vf1 == vf1.max())[-1]])
            assert np.isclose(selected, record['val_threshold'], atol=1e-12, rtol=0)
            row = {'seed': seed, 'threshold': selected, 'epochs': record['epochs_trained'],
                   'train_seconds': record['train_seconds'], 'validation_pr_ap': record['best_val_pr_ap']}
            for key, threshold in [('test_val_threshold', selected), ('test_default_0.5', .5)]:
                row[key] = metrics(y[va:], p[va:], threshold)
                for metric, value in record[key].items():
                    if metric in row[key]:
                        assert np.isclose(row[key][metric], value, atol=1e-10), (arm, seed, metric)
            assert np.isclose(average_precision_score(y[tr:va], p[tr:va]), record['best_val_pr_ap'])
            rows.append(row)
        records[arm] = {'seeds': rows}
        for key in ('test_val_threshold', 'test_default_0.5'):
            records[arm][key] = {
                metric: {'mean': float(np.mean([row[key][metric] for row in rows])),
                         'sd': float(np.std([row[key][metric] for row in rows], ddof=1))}
                for metric in rows[0][key]}
    report = {'verification': 'All 12 prediction files match labels, event IDs and splits; metrics and validation thresholds verified',
              'test_events': len(reference[0]) - reference[3],
              'arms': records,
              'limitation': 'V3 uses 80 additional fine-tuning epochs after the QAC warm start; memory-specific gains need a matched additional-training control.'}
    (ROOT / 'verified_comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Verified node-memory V3 results', '', report['verification'] + '.', '',
             'Means of three per-seed test metrics at validation-selected thresholds:', '',
             '| Arm | TN | FP | FN | TP | Accuracy | Precision | Recall | Specificity | F1 | PR-AP | ROC-AUC |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        m = records[arm]['test_val_threshold']
        keys = ['tn', 'fp', 'fn', 'tp', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'pr_ap', 'roc_auc']
        cells = [f'{m[k]["mean"]:.1f}' if k in keys[:4] else f'{m[k]["mean"]:.5f}' for k in keys]
        lines.append('| ' + arm + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## V3 per seed', '', '| Seed | Threshold | TN | FP | FN | TP | Precision | Recall | F1 | AP |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in records[ARMS[-1]]['seeds']:
        m = row['test_val_threshold']
        lines.append(f'| {row["seed"]} | {row["threshold"]:.6f} | {m["tn"]} | {m["fp"]} | {m["fn"]} | {m["tp"]} | {m["precision"]:.5f} | {m["recall"]:.5f} | {m["f1"]:.5f} | {m["pr_ap"]:.5f} |')
    fixed = records[ARMS[-1]]['test_default_0.5']
    lines += ['', f'V3 fixed-0.5 mean F1: {fixed["f1"]["mean"]:.5f}; precision: {fixed["precision"]["mean"]:.5f}; recall: {fixed["recall"]["mean"]:.5f}.', '',
              report['limitation'], '',
              'Seed 0 selected threshold 0.002703 on validation; it gives 171 test false positives versus 23 at 0.5. Validation/test distributions transfer poorly. Fixed 0.5 was pre-specified; a new threshold chosen after seeing test results requires fresh evaluation.', '',
              'All three V3 seeds completed 80 fine-tuning epochs without time boxing. Fine-tuning training time totals approximately 31.6 minutes, excluding validation, startup and latency measurement.', '',
              'Recorded latency: V3 online p50 8.342 ms versus old memory 10.697 ms and QAC 5.995 ms. V3 batched throughput 45,277 events/s versus old memory 72,613 and QAC 154,153. These are separate runs with different inference batches (V3 512, previous 1024) and latency sample sizes (1000 versus 3000); they do not establish a matched-setting efficiency improvement.']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({arm: {key: {m: round(v['mean'], 6) for m, v in records[arm][key].items()}
                          for key in ('test_val_threshold', 'test_default_0.5')} for arm in ARMS}, indent=2))
    print(report['verification'])


if __name__ == '__main__':
    main()
