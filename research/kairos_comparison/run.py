"""Train/evaluate KAIROS, original V3 control, and improved V4 on one archive.

Real performance runs require a preparation manifest and complete class support.
KAIROS sees only wholly benign training windows. TGNN uses train attack labels.
No checkpoint, threshold, normalization, or rareness choice uses test labels.
"""
import argparse
import copy
import hashlib
import json
import importlib
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix, precision_recall_curve

from node_memory_v3 import TGNNNodeMemV3, admission_penalties
from node_memory_v4 import TGNNNodeMemV4
from matched_memory_training import Replay
from .data import SCHEMA, sha256
from .model import Kairos
from .detection import fit_rareness, detect, summaries

PROTOCOL = 'dedale-kairos-shared41-v1'


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


def finite_scores(scores):
    return np.nan_to_num(np.asarray(scores, dtype=float), neginf=-1e30, posinf=1e30)


def metrics(y, scores, threshold, strict=False):
    scores = finite_scores(scores)
    pred = scores > threshold if strict else scores >= threshold
    tn, fp, fn, tp = map(int, confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return dict(threshold=float(threshold), threshold_comparison='>' if strict else '>=',
        tn=tn, fp=fp, fn=fn, tp=tp, precision=tp/max(tp+fp, 1), recall=tp/max(tp+fn, 1),
        f1=2*tp/max(2*tp+fp+fn, 1), false_positive_rate=fp/(fp+tn) if fp+tn else None,
        pr_ap=float(average_precision_score(y, scores)) if np.any(y) else None,
        roc_auc=float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None)


def f1_threshold(y, p):
    precision, recall, threshold = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(threshold[np.flatnonzero(f1 == f1.max())[-1]])


def window_values(windows, labels, score):
    ids, starts = np.unique(windows, return_index=True)
    return ids, np.maximum.reduceat(labels, starts), np.maximum.reduceat(score, starts)


def benign_windows(windows, labels):
    _, starts, inverse = np.unique(windows, return_index=True, return_inverse=True)
    return (np.maximum.reduceat(labels, starts) == 0)[inverse]


def train_one(prepared, output, arm, seed=0, epochs=50, batch=1024, device='cpu', horizon=3, on_checkpoint=None):
    prepared, output = Path(prepared), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(prepared.with_name('manifest.json').read_text())
    digest = sha256(prepared)
    if manifest.get('schema') != SCHEMA or manifest.get('dataset_sha256') != digest or not manifest.get('no_subsampling'):
        raise ValueError('Verified shared, unsubsampled preparation required')
    source_hashes = {}
    for module_name in ['kairos_comparison.data', 'kairos_comparison.model', 'kairos_comparison.detection',
                        'kairos_comparison.run', 'node_memory_v4', 'node_memory_v3', 'node_memory_models',
                        'temporal_cnn', 'tgnn_components', 'causal_tgn', 'pomsathit_baseline',
                        'matched_memory_training', 'causal_data']:
        source = Path(importlib.import_module(module_name).__file__)
        source_hashes[module_name] = sha256(source)
        dest = output / 'source' / (module_name.replace('.', '/') + '.py')
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and sha256(dest) != source_hashes[module_name]:
            raise ValueError('Source changed in existing output directory; use a new run directory')
        dest.write_bytes(source.read_bytes())
    signature = dict(protocol=PROTOCOL, dataset_sha256=digest, source_hashes=source_hashes,
                     epochs=epochs, batch=batch, horizon=horizon, seed=seed, arm=arm)
    existing_path = output / f'{arm}_seed{seed}.json'
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if existing.get('run_signature') != signature:
            raise ValueError('Existing result has a different run signature; choose a new directory')
        if existing.get('status') == 'complete':
            return existing
    with np.load(prepared, allow_pickle=False) as archive:
        # NpzFile re-decompresses a field on every access. Materialize once;
        # minibatch slicing must never re-read the whole event stream.
        a = {k: archive[k] for k in archive.files}
    tr, va, n = int(a['train_end']), int(a['val_end']), len(a['y'])
    labels, windows = a['y'], a['window']
    for name, lo, hi in [('train', 0, tr), ('validation', tr, va), ('test', va, n)]:
        if len(np.unique(labels[lo:hi])) != 2:
            raise ValueError(f'{name} lacks both classes. Do not claim a supervised comparison; register new date boundaries.')
    if not benign_windows(windows[tr:va], labels[tr:va]).any():
        raise ValueError('No wholly benign validation window for KAIROS calibration')
    torch.set_num_threads(4)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    torch.backends.cudnn.benchmark = False
    # Native scatter kernels can be nondeterministic on CUDA; record rather than
    # silently promising bitwise deterministic reproduction of upstream PyG.
    x = torch.as_tensor(a['x'], device=device)
    src = torch.as_tensor(a['src'], device=device)
    dst = torch.as_tensor(a['dst'], device=device)
    # KAIROS upstream uses nanoseconds. Offset preserves units without float epoch loss.
    ts = torch.as_tensor(np.rint((a['times'] - a['times'][0]) * 1e9).astype('int64'), device=device)
    y = torch.as_tensor(labels, device=device)
    rel = torch.as_tensor(a['relation'], device=device)
    is_kairos = arm == 'KAIROS'
    benign_tgnn = arm.endswith('_Benign')
    if is_kairos:
        model = Kairos(int(a['n_nodes'])).to(device)
        mask = benign_windows(windows[:tr], labels[:tr])
        train_idx = np.flatnonzero(mask)
        if len(np.unique(windows[train_idx])) < 2:
            raise ValueError('KAIROS requires at least two wholly benign training windows')
        train_ids = torch.as_tensor(train_idx, device=device)
        train_s, train_d, train_t, train_x = src[train_ids], dst[train_ids], ts[train_ids], x[train_ids]
        train_rel = rel[train_ids]
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, eps=1e-8, weight_decay=.01)
    else:
        base_arm = arm[:-len('_Benign')] if benign_tgnn else arm
        cls = TGNNNodeMemV4 if base_arm == 'TGNN_Node_Mem_V4' else TGNNNodeMemV3
        model = cls(input_dim=41, n_nodes=int(a['n_nodes'])).to(device)
        ego = torch.as_tensor(a['ego'], device=device)
        valid = torch.as_tensor(a['valid'], device=device)
        ages = torch.as_tensor(a['ages'], device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        replay = Replay(tr, seed)
        count = np.bincount(labels[:tr], minlength=2)
        class_weight = torch.as_tensor(tr/(2*count), dtype=x.dtype, device=device)
        train_idx = np.arange(tr)
        if benign_tgnn:
            benign_mask = benign_windows(windows[:tr], labels[:tr])
            train_idx = np.flatnonzero(benign_mask)
            if len(np.unique(windows[train_idx])) < 2:
                raise ValueError('Benign TGNN requires at least two wholly benign training windows')

    def tg_forward(lo, hi, indices=None):
        ids = np.arange(lo, hi) if indices is None else indices
        return model.forward_events(x[ego[ids]], valid[ids], ages[ids],
                                    a['src'][ids], a['dst'][ids], a['times'][ids])

    def evaluate(end):
        model.eval(); model.reset_state()
        scores = []
        with torch.no_grad():
            for lo in range(0, end, batch):
                hi = min(lo + batch, end)
                if is_kairos:
                    logits = model.score(src[lo:hi], dst[lo:hi], ts, x)
                    value = F.cross_entropy(logits, rel[lo:hi], reduction='none')
                    model.observe(src[lo:hi], dst[lo:hi], ts[lo:hi], x[lo:hi])
                else:
                    logits, _ = tg_forward(lo, hi)
                    value = logits.softmax(-1)[:, 1]
                model.detach_state()
                scores.append(value.cpu().numpy())
        return np.concatenate(scores)

    history, best, best_ap, best_epoch, steps = [], None, -1., 0, 0
    resume_path = output / f'{arm}_seed{seed}_resume.pt'
    completed_epochs, prior_seconds = 0, 0.
    if resume_path.exists():
        resume = torch.load(resume_path, map_location=device, weights_only=False)
        if resume['run_signature'] != signature:
            raise ValueError('Resume checkpoint signature mismatch')
        model.load_state_dict(resume['model'], strict=not is_kairos)
        optimizer.load_state_dict(resume['optimizer'])
        history, best = resume['history'], resume['best']
        best_ap, best_epoch, steps = resume['best_ap'], resume['best_epoch'], resume['steps']
        completed_epochs, prior_seconds = resume['completed_epochs'], resume['elapsed_seconds']
        torch.set_rng_state(resume['torch_rng'].cpu())
        np.random.set_state(resume['numpy_rng']); random.setstate(resume['python_rng'])
        if device.startswith('cuda'):
            torch.cuda.set_rng_state_all([v.cpu() for v in resume['cuda_rng']])
        if not is_kairos:
            replay = resume['replay']
    start = time.monotonic()
    for epoch in range(completed_epochs, epochs):
        model.train(); model.reset_state(); optimizer.zero_grad(set_to_none=True)
        losses, accumulated, count_batches = [], 0., 0
        length = len(train_idx)
        for lo in range(0, length, batch):
            hi = min(lo + batch, length)
            if is_kairos:
                optimizer.zero_grad(set_to_none=True)
                logits = model.score(train_s[lo:hi], train_d[lo:hi], train_t, train_x)
                loss = F.cross_entropy(logits, train_rel[lo:hi])
                model.observe(train_s[lo:hi], train_d[lo:hi], train_t[lo:hi], train_x[lo:hi])
                loss.backward(); optimizer.step(); model.detach_state(); steps += 1
            else:
                ids = train_idx[lo:hi]
                logits, aux = tg_forward(lo, hi, ids)
                target = torch.zeros(len(ids), dtype=torch.long, device=device) if benign_tgnn else y[ids]
                loss = F.cross_entropy(logits, target, weight=None if benign_tgnn else class_weight)
                picked = [] if benign_tgnn else replay.sample_then_admit(int(ids[0]), int(ids[-1])+1)
                if picked:
                    ix = torch.as_tensor(picked, device=device)
                    loss = loss + F.cross_entropy(model.replay_logits(x[ego[ix]], valid[ix], ages[ix]),
                                                  y[ix], weight=class_weight)
                budget, redundancy = admission_penalties(aux)
                loss = loss + .05*budget + .01*redundancy
                accumulated = accumulated + loss
                count_batches += 1
                if count_batches == horizon or hi == length:
                    (accumulated / count_batches).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    model.detach_state(); accumulated, count_batches = 0., 0
                    steps += 1
            losses.append(float(loss.detach()))
        row = dict(epoch=epoch+1, loss=float(np.mean(losses)), optimizer_steps=steps)
        if not np.isfinite(row['loss']):
            raise RuntimeError('Nonfinite loss')
        if not is_kairos and ((epoch+1) % 5 == 0 or epoch+1 == epochs):
            scores = evaluate(va)
            ap = float(average_precision_score(labels[tr:va], scores[tr:va]))
            row['validation_ap'] = ap
            if ap > best_ap:
                best_ap, best_epoch = ap, epoch+1
                best = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row)
        elapsed = prior_seconds + time.monotonic()-start
        json_write(output / f'{arm}_seed{seed}_progress.json', dict(history=history, elapsed_seconds=elapsed))
        if (epoch+1) % 5 == 0 or epoch+1 == epochs:
            state = model.learned_state() if is_kairos else {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            resume = dict(run_signature=signature, model=state, optimizer=optimizer.state_dict(),
                history=history, best=best, best_ap=best_ap, best_epoch=best_epoch, steps=steps,
                completed_epochs=epoch+1, elapsed_seconds=elapsed, torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else [],
                replay=None if is_kairos else replay)
            temporary = resume_path.with_suffix('.tmp')
            torch.save(resume, temporary); temporary.replace(resume_path)
            if on_checkpoint:
                on_checkpoint()
        print(f'{arm} seed={seed} epoch={epoch+1}/{epochs} loss={row["loss"]:.5f}', flush=True)
    if is_kairos:
        best, best_epoch = model.learned_state(), epochs
    else:
        model.load_state_dict(best)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    eval_start = time.monotonic()
    scores = evaluate(n)
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    eval_seconds = time.monotonic() - eval_start
    result = dict(protocol=PROTOCOL, run_signature=signature, arm=arm, seed=seed, epochs=epochs, best_epoch=best_epoch,
        dataset_sha256=digest, feature_sha256=hashlib.sha256(a['x'].tobytes()).hexdigest(),
        event_ids_sha256=hashlib.sha256(a['ids'].tobytes()).hexdigest(),
        train_end=tr, val_end=va, batch=batch, tbptt=horizon if not is_kairos else 1,
        train_events_per_epoch=len(train_idx), optimizer_steps=steps,
        training_objective=('benign edge-type reconstruction' if is_kairos else ('benign-only binary TGNN objective' if benign_tgnn else 'supervised attack classification + replay + admission penalties')),
        training_selection=('wholly benign train windows' if (is_kairos or benign_tgnn) else 'all train events'),
        checkpoint_selection='final epoch (released protocol)' if is_kairos else 'validation event PR-AP',
        parameter_count=sum(p.numel() for p in model.parameters()),
        device=device, torch_version=torch.__version__, pyg_version=__import__('torch_geometric').__version__,
        cuda_scatter_may_be_nondeterministic=device.startswith('cuda'),
        elapsed_seconds=prior_seconds+time.monotonic()-start, history=history,
        evaluation_seconds=eval_seconds, batched_events_per_second=n/eval_seconds,
        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/1024**2 if device.startswith('cuda') else None)
    vwin, vy, vp = window_values(windows[tr:va], labels[tr:va], scores[tr:va])
    twin, ty, tp = window_values(windows[va:], labels[va:], scores[va:])
    if is_kairos:
        rarity_src, rarity_dst = a['rarity_ids'][a['src']], a['rarity_ids'][a['dst']]
        rare = fit_rareness(rarity_src[train_idx], rarity_dst[train_idx], windows[train_idx])
        benign_val = benign_windows(windows[tr:va], labels[tr:va])
        if not benign_val.any():
            raise ValueError('No wholly benign validation window for paper threshold calibration')
        val_idx = np.arange(tr, va)[benign_val]
        vr = detect(rarity_src[val_idx], rarity_dst[val_idx], windows[val_idx], scores[val_idx], rare)
        log_beta = max([r['retrospective_score'] for r in vr if np.isfinite(r['retrospective_score'])], default=0.)
        test_rows = detect(rarity_src[va:], rarity_dst[va:], windows[va:], scores[va:], rare)
        online = finite_scores([r['online_score'] for r in test_rows])
        retrospective = finite_scores([r['retrospective_score'] for r in test_rows])
        result['window_online'] = metrics(ty, online, log_beta, strict=True)
        result['window_retrospective'] = metrics(ty, retrospective, log_beta, strict=True)
        result['threshold_rule'] = 'maximum benign-validation queue log score; strict exceedance'
        edge_threshold = float(scores[val_idx].max())
        result['event_diagnostic'] = metrics(labels[va:], scores[va:], edge_threshold, strict=True)
        result['event_diagnostic_note'] = 'Extra edge-level diagnostic; paper detects windows/queues.'
        json_write(output / f'{arm}_seed{seed}_summary_graphs.json',
            summaries(a['src'][va:], a['dst'][va:], a['relation'][va:], scores[va:], test_rows, log_beta, seed))
        for row in test_rows:
            for k in ('log_window_score', 'online_score', 'retrospective_score'):
                row[k] = float(finite_scores([row[k]])[0])
        json_write(output / f'{arm}_seed{seed}_queues.json', test_rows)
        json_write(output / f'{arm}_seed{seed}_rareness.json', rare)
        checkpoint_threshold = log_beta
    else:
        threshold = f1_threshold(labels[tr:va], scores[tr:va])
        result['event'] = metrics(labels[va:], scores[va:], threshold)
        # Separate validation-selected window threshold; never compare event F1
        # to KAIROS window F1 as if they were the same task.
        window_threshold = f1_threshold(vy, vp)
        result['window_online'] = metrics(ty, tp, window_threshold)
        result['threshold_rule'] = 'validation F1, separately for events and max-score 15-minute windows'
        checkpoint_threshold = threshold
    stem = output / f'{arm}_seed{seed}'
    torch.save(dict(state_dict=best, arm=arm, n_nodes=int(a['n_nodes']), input_dim=41,
                    protocol=PROTOCOL, schema=SCHEMA, dataset_sha256=digest,
                    threshold=checkpoint_threshold, best_epoch=best_epoch), str(stem)+'.pt')
    np.savez_compressed(str(stem)+'_predictions.npz', score=scores, y=labels, ids=a['ids'],
                        window=windows, train_end=tr, val_end=va)
    result['status'] = 'complete'
    json_write(str(stem)+'.json', result)
    return result


def report(output):
    output = Path(output)
    rows = []
    for path in sorted(output.rglob('*_seed*.json')):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and data.get('status') == 'complete' and data.get('protocol') == PROTOCOL:
            rows.append(data)
    if not rows:
        return
    for key in ('dataset_sha256', 'feature_sha256', 'event_ids_sha256', 'train_end', 'val_end'):
        if len({r[key] for r in rows}) != 1:
            raise ValueError(f'Unmatched result archives: {key}')
    all_benign = all(r.get('training_selection') == 'wholly benign train windows' for r in rows)
    regime = ('All arms use only wholly benign training windows; TGNN scores are one-class attack-probability diagnostics.' if all_benign else 'KAIROS and benign TGNN arms use only benign training windows; supervised TGNN arms use attack labels.')
    lines = ['# Shared DEDALE comparison', '',
        regime,
        'All arms receive the same 41-dimensional provenance messages and chronological held-out events.',
        'Results are for the supported Sysmon subset, not a complete KAIROS reproduction on DARPA.', '',
        '| Model | Seeds | Online window F1 | Window PR-AP | Window recall | Window FPR |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in sorted({r['arm'] for r in rows}):
        selected = [r for r in rows if r['arm'] == arm]
        values = []
        for metric in ('f1', 'pr_ap', 'recall', 'false_positive_rate'):
            vals = [r['window_online'][metric] for r in selected if r['window_online'][metric] is not None]
            values.append(f'{np.mean(vals):.4f} ± {np.std(vals, ddof=1):.4f}' if len(vals)>1 else f'{vals[0]:.4f}' if vals else 'undefined')
        lines.append(f'| {arm} | {len(selected)} | ' + ' | '.join(values) + ' |')
    lines.extend(['', '## Event diagnostics', '',
        ('KAIROS uses reconstruction error with a benign-validation threshold; benign TGNN arms use attack-probability scores with validation F1 thresholds.' if all_benign else 'KAIROS uses reconstruction error with a benign-validation threshold; supervised TGNN arms use validation F1 thresholds.'), '',
        '| Model | Event F1 | Event PR-AP | Event ROC-AUC | Precision | Recall |',
        '|---|---:|---:|---:|---:|---:|'])
    for arm in sorted({r['arm'] for r in rows}):
        selected = [r.get('event', r.get('event_diagnostic')) for r in rows if r['arm']==arm]
        vals = [f'{np.mean([r[k] for r in selected]):.4f}' for k in ('f1','pr_ap','roc_auc','precision','recall')]
        lines.append(f'| {arm} | ' + ' | '.join(vals) + ' |')
    lines.extend(['', '## Resources', '',
        'Batched throughput includes chronological state reconstruction and CPU output transfer; no claim of event-at-a-time latency.', '',
        '| Model | Parameters | Mean elapsed seconds | Mean events/s |', '|---|---:|---:|---:|'])
    for arm in sorted({r['arm'] for r in rows}):
        selected = [r for r in rows if r['arm']==arm]
        lines.append(f'| {arm} | {selected[0]["parameter_count"]} | '
                     f'{np.mean([r["elapsed_seconds"] for r in selected]):.1f} | '
                     f'{np.mean([r["batched_events_per_second"] for r in selected]):.1f} |')
    (output / 'RESULTS.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arms', nargs='+', choices=['KAIROS', 'TGNN_Node_Mem_V3', 'TGNN_Node_Mem_V4'],
                   default=['KAIROS', 'TGNN_Node_Mem_V3', 'TGNN_Node_Mem_V4'])
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.epochs < 1 or args.batch < 1:
        p.error('epochs and batch must be positive')
    for seed in args.seeds:
        for arm in args.arms:
            train_one(args.prepared, args.output, arm, seed, args.epochs, args.batch, args.device)
            report(args.output)
