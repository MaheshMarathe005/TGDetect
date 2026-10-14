"""Equal additional-training comparison from shared per-seed QAC checkpoints."""
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from torch.nn import functional as F

from node_memory_v3 import make_mem_model, admission_penalties

PROTOCOL = 'matched-memory-v1-20261002'
ARMS = ('TGNN_Q_A_C', 'TGNN_Node_Mem_V3_Fixed', 'TGNN_Node_Mem_V3')
RESULTS = '/vol/results/dedale_host_mem_matched'


def threshold_for(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]]) if len(thresholds) else .5


def metrics(y, p, threshold):
    tn, fp, fn, tp = map(int, confusion_matrix(y, p >= threshold, labels=[0, 1]).ravel())
    return dict(threshold=threshold, tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=(tn + tp) / len(y), precision=tp / max(tp + fp, 1),
                recall=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                f1=2 * tp / max(2 * tp + fp + fn, 1),
                pr_ap=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)))


class Replay:
    def __init__(self, n, seed):
        self.cap, self.seen, self.items = n // 10, 0, []
        self.rng = random.Random(seed + 937)

    def sample_then_admit(self, lo, hi):
        picked = self.rng.sample(self.items, min(26, len(self.items)))
        for event in range(lo, hi):
            self.seen += 1
            if len(self.items) < self.cap:
                self.items.append(event)
            elif self.cap:
                j = self.rng.randrange(self.seen)
                if j < self.cap:
                    self.items[j] = event
        return picked


def train_seed(seed, volume, epochs=80, batch=512, horizon=3):
    """Run all arms sequentially on one GPU, in rotated order across seeds."""
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    volume.reload()
    prep = Path('/vol/prepared/dedale_host_mem/prepared.npz')
    dataset_sha = hashlib.sha256(prep.read_bytes()).hexdigest()
    a = np.load(prep)
    x = torch.from_numpy(a['x']).cuda()
    windows = x[torch.from_numpy(a['ego']).cuda()]
    del x
    valid, ages = torch.from_numpy(a['valid']).cuda(), torch.from_numpy(a['ages']).cuda()
    labels = a['y'].astype(np.int64)
    y = torch.from_numpy(labels).cuda()
    src, dst, times = a['src'], a['dst'], a['times']
    tr, va, n = int(a['train_end']), int(a['val_end']), len(labels)
    nodes, dim = int(a['n_nodes']), int(windows.shape[-1])
    positive = int(labels[:tr].sum())
    weight = torch.tensor([tr / (2 * (tr - positive)), tr / (2 * positive)], device='cuda')
    root = Path(RESULTS)
    root.mkdir(parents=True, exist_ok=True)
    warm_path = Path('/vol/results/dedale_host_mem') / f'TGNN_Q_A_C_seed{seed}.pt'
    warm_sha = hashlib.sha256(warm_path.read_bytes()).hexdigest()
    warm = torch.load(warm_path, map_location='cpu', weights_only=False)
    outcomes = []
    order = ARMS[seed % len(ARMS):] + ARMS[:seed % len(ARMS)]
    for arm in order:
        stem = f'{arm}_seed{seed}'
        signature = f'{PROTOCOL}|{dataset_sha}|{warm_sha}|{arm}|{seed}|{epochs}|{batch}|{horizon}'
        result_path = root / f'{stem}.json'
        if result_path.exists():
            old = json.loads(result_path.read_text())
            if old.get('run_signature') == signature and old.get('status') == 'complete':
                outcomes.append(old)
                continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        model = make_mem_model(arm, dim, nodes).cuda()
        state = model.state_dict()
        transfer = {k: v for k, v in warm['state_dict'].items()
                    if k in state and v.shape == state[k].shape
                    and not k.startswith(('controller.', 'attn_score.'))}
        required = [k for k in state if k.startswith(('layers.', 'cnn.', 'head.', 'mem_norm.',
                    'time_encoder.', 'history_gru.', 'query.', 'key.', 'value.'))]
        if any(k not in transfer for k in required):
            raise RuntimeError('Incomplete shared warm start')
        model.load_state_dict(transfer, strict=False)
        # Reset dropout RNG identically after architecture construction and loading.
        torch.manual_seed(seed + 10000)
        torch.cuda.manual_seed_all(seed + 10000)
        opt = torch.optim.Adam(model.main_parameters(), lr=1e-4, weight_decay=1e-5)
        ctrl = torch.optim.Adam(model.controller_parameters(), lr=1e-4, weight_decay=1e-5)
        replay = Replay(tr, seed)
        replay_hash = hashlib.sha256()
        history, steps, replay_visits = [], 0, 0
        best_epoch = 0

        def probabilities(end):
            model.eval()
            model.reset_state()
            parts = []
            with torch.no_grad():
                for lo in range(0, end, batch):
                    hi = min(end, lo + batch)
                    logits, _ = model.forward_events(windows[lo:hi], valid[lo:hi], ages[lo:hi],
                                                      src[lo:hi], dst[lo:hi], times[lo:hi])
                    parts.append(logits.softmax(-1)[:, 1].cpu().numpy())
                    model.detach_state()
            return np.concatenate(parts)

        initial = probabilities(va)
        best_ap = float(average_precision_score(labels[tr:va], initial[tr:va]))
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        start = time.monotonic()
        print(f'{stem}: initial validation AP={best_ap:.6f}', flush=True)
        for epoch in range(epochs):
            model.train()
            model.reset_state()
            opt.zero_grad(set_to_none=True)
            ctrl.zero_grad(set_to_none=True)
            losses, window, count = [], torch.zeros((), device='cuda'), 0
            for lo in range(0, tr, batch):
                hi = min(tr, lo + batch)
                logits, aux = model.forward_events(windows[lo:hi], valid[lo:hi], ages[lo:hi],
                                                  src[lo:hi], dst[lo:hi], times[lo:hi])
                task = F.cross_entropy(logits, y[lo:hi], weight=weight)
                loss = task
                picked = replay.sample_then_admit(lo, hi)
                replay_hash.update(np.asarray([epoch, lo, *picked], dtype=np.int64).tobytes())
                replay_visits += len(picked)
                if picked:
                    ix = torch.as_tensor(picked, device='cuda')
                    loss = loss + F.cross_entropy(model.replay_logits(windows[ix], valid[ix], ages[ix]),
                                                  y[ix], weight=weight)
                if model.use_memory and model.adaptive:
                    budget, redundancy = admission_penalties(aux)
                    loss = loss + .05 * budget + .01 * redundancy
                losses.append(task.detach())
                window = window + loss
                count += 1
                if count == horizon or hi == tr:
                    (window / count).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    opt.step()
                    ctrl.step()
                    steps += 1
                    opt.zero_grad(set_to_none=True)
                    ctrl.zero_grad(set_to_none=True)
                    model.detach_state()
                    window, count = torch.zeros((), device='cuda'), 0
            loss_value = float(torch.stack(losses).mean().item())
            if not np.isfinite(loss_value):
                raise RuntimeError(f'{stem}: nonfinite loss')
            row = dict(epoch=epoch + 1, loss=loss_value, optimizer_steps=steps)
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                p = probabilities(va)
                ap = float(average_precision_score(labels[tr:va], p[tr:va]))
                row['validation_pr_ap'] = ap
                if ap > best_ap:
                    best_ap, best_epoch = ap, epoch + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append(row)
            progress = dict(protocol=PROTOCOL, arm=arm, seed=seed, epoch=epoch + 1,
                            epochs_target=epochs, best_val_pr_ap=best_ap,
                            elapsed_seconds=time.monotonic() - start, history=history)
            (root / f'{stem}_progress.json').write_text(json.dumps(progress))
            if (epoch + 1) % 5 == 0:
                torch.save(dict(state_dict=best_state, arm=arm, seed=seed, input_dim=dim,
                                n_nodes=nodes, best_epoch=best_epoch, protocol=PROTOCOL,
                                dataset_sha256=dataset_sha), root / f'{stem}_best.pt')
            volume.commit()
            print(f'{stem} epoch={epoch+1}/{epochs} loss={loss_value:.5f} val_best={best_ap:.5f}', flush=True)
        model.load_state_dict(best_state)
        p = probabilities(n)
        threshold = threshold_for(labels[tr:va], p[tr:va])
        checkpoint = dict(state_dict=best_state, arm=arm, seed=seed, input_dim=dim,
                          n_nodes=nodes, epochs_trained=epochs, best_epoch=best_epoch,
                          protocol=PROTOCOL, dataset_sha256=dataset_sha,
                          val_threshold=threshold, feature_schema='dedale-sysmon72-v1',
                          warm_start_sha256=warm_sha)
        torch.save(checkpoint, root / f'{stem}.pt')
        np.savez_compressed(root / f'{stem}_predictions.npz', probability=p, y=labels, ids=a['ids'],
                            train_end=tr, val_end=va)
        result = dict(status='complete', run_signature=signature, protocol=PROTOCOL, arm=arm, seed=seed,
                      epochs_trained=epochs, best_epoch=best_epoch, best_val_pr_ap=best_ap,
                      val_threshold=threshold, warm_start_sha256=warm_sha, dataset_sha256=dataset_sha,
                      optimizer_steps=steps, training_live_visits=tr * epochs,
                      training_replay_visits=replay_visits, replay_schedule_sha256=replay_hash.hexdigest(),
                      train_batch=batch, infer_batch=batch, tbptt_k=horizon, learning_rate=1e-4,
                      parameter_count=sum(v.numel() for v in model.parameters()),
                      elapsed_seconds=time.monotonic() - start, history=history,
                      test_val_threshold=metrics(labels[va:], p[va:], threshold),
                      test_default_0_5=metrics(labels[va:], p[va:], .5),
                      checkpoint_sha256=hashlib.sha256((root / f'{stem}.pt').read_bytes()).hexdigest())
        # Identical latency settings and chronological warm prefix for every arm.
        model.eval()
        probabilities(va)
        durations = []
        with torch.no_grad():
            for i in range(va, min(va + 1000, n)):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model.forward_events(windows[i:i+1], valid[i:i+1], ages[i:i+1],
                                     src[i:i+1], dst[i:i+1], times[i:i+1])
                torch.cuda.synchronize()
                durations.append((time.perf_counter() - t0) * 1000)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            probabilities(n)
            torch.cuda.synchronize()
            throughput = n / (time.perf_counter() - t0)
        result['latency'] = dict(batch=batch, online_events=len(durations),
                                p50_ms=float(np.median(durations[200:])),
                                p95_ms=float(np.quantile(durations[200:], .95)),
                                batched_events_per_s=throughput,
                                concurrent_gpu_jobs=True)
        result_path.write_text(json.dumps(result, indent=2))
        volume.commit()
        outcomes.append(result)
        del model, opt, ctrl
        torch.cuda.empty_cache()
    for key in ('warm_start_sha256', 'dataset_sha256', 'optimizer_steps', 'training_live_visits',
                'training_replay_visits', 'replay_schedule_sha256'):
        assert len({row[key] for row in outcomes}) == 1, f'Matched gate failed: {key}'
    (root / f'seed{seed}_matched_verification.json').write_text(json.dumps(
        dict(status='passed', seed=seed, protocol=PROTOCOL, arms=list(ARMS),
             matched_fields=['warm_start', 'dataset', 'steps', 'live_visits', 'replay_visits', 'replay_schedule']), indent=2))
    volume.commit()
    return outcomes
