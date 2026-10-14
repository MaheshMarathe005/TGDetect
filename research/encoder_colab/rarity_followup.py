"""Controlled rarity follow-up. No masked-pretraining recipe in this matrix."""
import copy
import hashlib
import json
import math
import platform
from pathlib import Path
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from experiment import (Config, CandidateEncoder, PilotModel, PackedEncoder, TokenBank,
    FieldCMS, prepare_data, load_data, pretrain, batches, seed_all, score, benchmark,
    metrics, bootstrap_delta, save_json, digest, sync, acceptance_check)


@dataclass
class FollowupConfig(Config):
    root: str = '/content/tgdetect_rarity_followup'
    epochs: int = 8
    seeds: tuple = (0, 1, 2, 3, 4)
    split_seeds: tuple = (20261013, 20261014)
    reference_rows: int = 1000
    bootstrap: int = 300
    latency_repeats: int = 40


MATRIX = {
    'old': ('zero', False),
    'prefix_log_film': ('prefix_log', False),
    'frozen_log_film': ('frozen_log', False),
    'frozen_surprise_film': ('frozen_surprise', False),
    'shuffled_surprise_film': ('shuffled_surprise', False),
    'frozen_surprise_interaction': ('frozen_surprise', True),
}


def read_fields(sketch, rows):
    return np.array([[sketch.read(*json.loads(pair)) for pair in record] for record in rows], dtype=np.float32)


def insert_fields(sketch, rows):
    for record in rows:
        for pair in record:
            sketch.add(*json.loads(pair))


def count_variants(data, cfg):
    """Disjoint reference records never enter optimizer or validation."""
    rng = np.random.default_rng(cfg.split_seed)
    n = len(data['train']['ids'])
    if not 1 <= cfg.reference_rows < n:
        raise ValueError('reference_rows must be positive and smaller than train')
    picked = rng.permutation(n)[:cfg.reference_rows]
    task_ix = np.setdiff1d(np.arange(n), picked)  # retained source order
    reference = data['train']['field_identity'][picked]
    task_data = {split: {k: (v[task_ix] if split == 'train' else v) for k, v in values.items()}
                 for split, values in data.items()}
    frozen = FieldCMS(cfg.cms_width); insert_fields(frozen, reference)
    started = time.perf_counter()
    reference_counts = {split: read_fields(frozen, values['field_identity']) for split, values in task_data.items()}
    read_seconds = time.perf_counter()-started
    prefix = FieldCMS(cfg.cms_width); prefix.table[:] = frozen.table
    prefix_train = []
    for record in task_data['train']['field_identity']:
        prefix_train.append(read_fields(prefix, [record])[0])
        insert_fields(prefix, [record])
    prefix_counts = {'train': np.asarray(prefix_train)}
    prefix_counts.update({split: read_fields(prefix, values['field_identity'])
                          for split, values in task_data.items() if split != 'train'})
    # Bounded normalized novelty, not an exact probabilistic likelihood.
    scale = math.log1p(cfg.reference_rows)
    surprises = {split: np.clip((scale-np.log1p(count))/scale, 0, 1).astype(np.float32)
                 for split, count in reference_counts.items()}
    variants = {}
    for name in ('zero', 'prefix_log', 'frozen_log', 'frozen_surprise', 'shuffled_surprise'):
        variants[name] = {}
        for split, values in task_data.items():
            if name == 'zero':
                rarity = np.zeros_like(values['rarity'])
            elif name == 'prefix_log':
                rarity = np.log1p(prefix_counts[split])
            elif name == 'frozen_log':
                rarity = np.log1p(reference_counts[split])
            else:
                rarity = surprises[split].copy()
                if name == 'shuffled_surprise':
                    # Each field shuffled independently; same marginals, broken value association.
                    generator = np.random.default_rng(cfg.split_seed + len(split))
                    for field in range(1, rarity.shape[1]):
                        generator.shuffle(rarity[:, field])
            rarity = rarity.astype(np.float32)
            rarity[:, 0] = 0  # Prevent count of synthetic event-type field exposing stream age.
            variants[name][split] = {**values, 'rarity': rarity}
    audit = {'reference_rows': len(picked), 'task_rows': len(task_ix),
             'reference_identity_sha256': hashlib.sha256(data['train']['ids'][picked].tobytes()).hexdigest(),
             'task_identity_sha256': hashlib.sha256(task_data['train']['ids'].tobytes()).hexdigest(),
             'reference_disjoint_from_optimizer': True, 'synthetic_type_rarity_zero': True,
             'reference_count_read_seconds': read_seconds,
             'cpu_read_ms_per_record': 1000*read_seconds/sum(len(v['ids']) for v in task_data.values()),
             'latency_scope': 'CPU count reads measured separately; no claim of integrated streaming latency',
             'novelty_definition': 'clip((log1p(reference_size)-log1p(count))/log1p(reference_size),0,1)',
             'prefix_vs_frozen_means': {split: {'prefix_log': float(variants['prefix_log'][split]['rarity'][:, 1:].mean()),
                                               'frozen_log': float(variants['frozen_log'][split]['rarity'][:, 1:].mean())}
                                       for split in task_data},
             'unseen_field_fraction': {s: float((c[:, 1:] == 0).mean()) for s,c in reference_counts.items()}}
    return variants, surprises, reference, audit


def tail_scale(reference, values):
    """Benign median/IQR normalization: preserve scores beyond observed tail."""
    median = np.median(reference)
    q25, q75 = np.quantile(reference, [.25, .75])
    scale = max(float(q75-q25), float(np.std(reference))*.1, 1e-6)
    return (values-median)/scale


def family_metrics(y, family, values, calibration):
    result = metrics(y, values, calibration)
    threshold = result['fpr_target_0.01']['threshold']
    by_family = {}
    for name in sorted(set(family[y == 1])):
        use = family == name
        by_family[name] = {'n': int(use.sum()), 'recall': float((values[use] > threshold).mean())}
    non_neptune = (y == 1) & (family != 'neptune.')
    result['non_neptune_n'] = int(non_neptune.sum())
    result['non_neptune_recall'] = float((values[non_neptune] > threshold).mean()) if non_neptune.any() else None
    result['macro_family_recall'] = float(np.mean([v['recall'] for v in by_family.values()])) if by_family else None
    result['family_recall'] = by_family
    return result


def initialize(bank, classes, arm, seed, tokens, device):
    seed_all(seed)
    baseline = PilotModel(bank, classes, [], {})
    seed_all(seed+10000)
    prototype = PilotModel(bank, classes, [], {'rarity': True, 'interaction': True})
    mode, interaction = MATRIX[arm]
    options = {} if arm == 'old' else {'rarity': True, 'interaction': interaction}
    model = PilotModel(bank, classes, [], options).to(device)
    state = {k:v for k,v in prototype.state_dict().items() if k in model.state_dict()}
    state.update(baseline.state_dict()); model.load_state_dict(state)
    with torch.no_grad():
        model.encoder.tokens.weight.copy_(tokens.to(device))
    return model


def train_one(cfg, bank, data, surprise, classes, tokens, arm, seed):
    out = Path(cfg.root)/'results'/f'{arm}_s{seed}'; out.mkdir(parents=True, exist_ok=False)
    model = initialize(bank, classes, arm, seed, tokens, cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []; best = math.inf; steps = 0
    if cfg.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    sync(cfg); start = time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train(); total = 0.; seen = 0
        order = np.random.default_rng(seed*1000+epoch).permutation(len(data['train']['ids']))
        for batch in batches(data['train'], cfg, order):
            loss = F.cross_entropy(model(batch['ids'], batch['rarity']), batch['service'])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); steps += 1
            total += float(loss.detach())*len(batch['ids']); seen += len(batch['ids'])
        val = float(score(model, data['val'], cfg, [], False)[:, 0].mean())
        history.append({'epoch': epoch+1, 'train_ce': total/seen, 'benign_val_ce': val})
        if val < best:
            best = val; torch.save({'model': model.state_dict(), 'epoch': epoch+1}, out/'best.pt')
    sync(cfg); seconds = time.perf_counter()-start
    state = torch.load(out/'best.pt', map_location=cfg.device, weights_only=False)
    model.load_state_dict(state['model']); model.eval()
    cal = score(model, data['cal'], cfg, [], False)[:,0]
    test = score(model, data['test'], cfg, [], False)[:,0]
    half = len(cal)//2; y = data['test']['y']; family = data['test']['family']
    primary = family_metrics(y, family, test, cal[half:])
    # Same frozen reference rarity for ALL models, including old.
    rarity_cal = surprise['cal'][:,1:].mean(1); rarity_test = surprise['test'][:,1:].mean(1)
    fusion = {}
    for weight in (.25, .5, 1.):
        a = tail_scale(cal[:half], cal[half:])+weight*tail_scale(rarity_cal[:half], rarity_cal[half:])
        b = tail_scale(cal[:half], test)+weight*tail_scale(rarity_cal[:half], rarity_test)
        fusion[str(weight)] = family_metrics(y, family, b, a)
    # Counterfactual does not retrain: quantify reliance on learned rarity input.
    disabled = {k: {**v, 'rarity': np.zeros_like(v['rarity'])} for k,v in data.items()}
    zero_test = score(model, disabled['test'], cfg, [], False)[:,0]
    zero_cal = score(model, disabled['cal'], cfg, [], False)[:,0]
    reliance = family_metrics(y, family, zero_test, zero_cal[half:])
    timing = benchmark(model, data['test'], cfg, [])
    result = {'arm': arm, 'seed': seed, 'split_seed': cfg.split_seed, 'primary': primary,
              'predeclared_late_fusion': fusion, 'zero_rarity_counterfactual': reliance,
              'epochs': cfg.epochs, 'updates': steps, 'checkpoint_epoch': state['epoch'],
              'history': history, 'training_seconds': seconds, 'latency': timing,
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if cfg.device.startswith('cuda') else None,
              'parameters': sum(p.numel() for p in model.parameters()), 'checkpoint_sha256': digest(out/'best.pt')}
    np.savez_compressed(out/'scores.npz', y=y, family=family, service=test, calibration=cal,
                        rarity_test=rarity_test, rarity_calibration=rarity_cal, zero_rarity=zero_test,
                        zero_rarity_calibration=zero_cal)
    save_json(out/'result.json', result)
    print(arm, seed, 'AP', round(primary['ap'],4), 'F1', round(primary['fpr_target_0.01']['f1'],4), flush=True)
    del model, optimizer
    return result


def run_followup(cfg=FollowupConfig(), frame=None):
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Choose GPU in Colab, or explicitly set device="cpu" for smoke tests.')
    root = Path(cfg.root); root.mkdir(parents=True, exist_ok=True)
    if (root/'runs.csv').exists():
        raise FileExistsError('Completed study exists. Choose a fresh root.')
    torch.set_num_threads(2)
    if cfg.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,cfg.memory_budget_gib*2**30/total))
    save_json(root/'config.json', asdict(cfg))
    save_json(root/'environment.json', {'gpu': torch.cuda.get_device_name() if cfg.device.startswith('cuda') else 'CPU smoke',
              'torch': torch.__version__, 'cuda': torch.version.cuda, 'python': platform.python_version(),
              'source_sha256': digest(__file__)})
    frame = load_data(cfg) if frame is None else frame
    rows = []; results = []; paired = []
    for split_seed in cfg.split_seeds:
        split_cfg = copy.copy(cfg); split_cfg.split_seed = split_seed
        split_cfg.root = str(root/f'split_{split_seed}')
        original, keys, specs, classes, audit = prepare_data(split_cfg, frame)
        variants, surprise, reference, count_audit = count_variants(original, split_cfg)
        save_json(Path(split_cfg.root)/'count_protocol.json', count_audit)
        # Rebuild SGNS corpus from task records only (no validation/test).
        from orthrus_comparison.universal import value_parts
        words = []
        for record in variants['zero']['train']['field_identity']:
            sequence = []
            for raw in record:
                _, value = json.loads(raw); kind, parts, _ = value_parts(value)
                if kind not in ('number','bool','missing'):
                    sequence.extend(parts)
            if len(sequence)>1:
                words.append(sequence)
        corpus = Path(split_cfg.root)/'benign_paths.jsonl'
        corpus.write_text(''.join(json.dumps(w)+'\n' for w in words))
        manifest = Path(split_cfg.root)/'manifest.json'
        value = json.loads(manifest.read_text()); value['checksums']['benign_paths.jsonl'] = digest(corpus)
        save_json(manifest, value)
        bank = TokenBank(Path(split_cfg.root)/'tokens')
        if cfg.device.startswith('cuda'):
            bank.to_device(cfg.device)
        acceptance_check(bank, variants['zero'], split_cfg)
        init_path = Path(split_cfg.root)/'initial_embeddings.pt'
        init = pretrain(split_cfg.root, init_path, epochs=1, batch=2048, projection_steps=30,
                        seed=split_seed, device=cfg.device)
        save_json(Path(split_cfg.root)/'shared_sgns.json', init)
        tokens = torch.load(init_path, map_location='cpu', weights_only=False)['tokens']
        half = len(surprise['cal'])//2
        raw = family_metrics(original['test']['y'], original['test']['family'], surprise['test'][:,1:].mean(1),
                             surprise['cal'][half:,1:].mean(1))
        save_json(Path(split_cfg.root)/'results/raw_rarity.json', raw)
        inverse_test = np.exp(-variants['frozen_log']['test']['rarity'][:,1:]).mean(1)
        inverse_cal = np.exp(-variants['frozen_log']['cal']['rarity'][:,1:]).mean(1)
        inverse = family_metrics(original['test']['y'], original['test']['family'], inverse_test, inverse_cal[half:])
        save_json(Path(split_cfg.root)/'results/raw_inverse_count.json', inverse)
        for seed in cfg.seeds:
            order = list(MATRIX); shift = seed % len(order); order = order[shift:]+order[:shift]
            for arm in order:
                try:
                    result = train_one(split_cfg, bank, variants[MATRIX[arm][0]], surprise, classes, tokens, arm, seed)
                except Exception as exc:
                    save_json(root/'failure.json', {'split': split_seed, 'seed': seed, 'arm': arm, 'error': repr(exc)})
                    raise
                results.append(result); m=result['primary']; op=m['fpr_target_0.01']
                rows.append({'arm':arm,'seed':seed,'split_seed':split_seed,'ap':m['ap'],'f1':op['f1'],
                             'recall':op['recall'],'fpr':op['test_fpr'],'non_neptune_recall':m['non_neptune_recall'],
                             'macro_family_recall':m['macro_family_recall'],'checkpoint_epoch':result['checkpoint_epoch'],
                             'encoder_ms':result['latency'][str(min(128,cfg.test))]['encoder']['p50_ms']})
            for new, control in [('prefix_log_film','old'),('frozen_log_film','prefix_log_film'),
                                 ('frozen_surprise_film','frozen_log_film'),('frozen_surprise_film','shuffled_surprise_film'),
                                 ('frozen_surprise_interaction','frozen_surprise_film')]:
                a=np.load(Path(split_cfg.root)/f'results/{control}_s{seed}/scores.npz')
                b=np.load(Path(split_cfg.root)/f'results/{new}_s{seed}/scores.npz')
                paired.append({'split_seed':split_seed,'seed':seed,'candidate':new,'control':control,
                               **bootstrap_delta(a['y'],a['service'],b['service'],cfg.bootstrap)})
        del bank, tokens
        if cfg.device.startswith('cuda'):
            torch.cuda.empty_cache()
    table = pd.DataFrame(rows);table.to_csv(root/'runs.csv',index=False)
    summary=table.groupby('arm')[['ap','f1','recall','fpr','non_neptune_recall','macro_family_recall','encoder_ms']].agg(['mean','std'])
    summary.to_csv(root/'summary.csv');save_json(root/'paired_comparisons.json',paired)
    print(summary.to_string(),flush=True)
    return table, summary
