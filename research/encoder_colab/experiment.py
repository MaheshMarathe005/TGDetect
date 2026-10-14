"""Self-contained encoder pilot; bundled with unmodified V3 sources in Colab."""
from __future__ import annotations
import hashlib
import json
import math
import platform
from pathlib import Path
import random
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import sklearn
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

from sliding_node_mem_v3.tokens import TokenWriter, TokenBank, PackedEncoder
from orthrus_comparison.universal import flatten, value_parts
from encoder_sgns import pretrain

FEATURES = '''duration protocol_type service flag src_bytes dst_bytes land wrong_fragment urgent
hot num_failed_logins logged_in num_compromised root_shell su_attempted num_root
num_file_creations num_shells num_access_files num_outbound_cmds is_host_login
is_guest_login count srv_count serror_rate srv_serror_rate rerror_rate srv_rerror_rate
same_srv_rate diff_srv_rate srv_diff_host_rate dst_host_count dst_host_srv_count
dst_host_same_srv_rate dst_host_diff_srv_rate dst_host_same_src_port_rate
dst_host_srv_diff_host_rate dst_host_serror_rate dst_host_srv_serror_rate
dst_host_rerror_rate dst_host_srv_rerror_rate'''.split()
# Pinned mirror/checksum used by scikit-learn's official dataset loader.
SOURCE_URL = 'https://ndownloader.figshare.com/files/5976042'
DATA_SHA256 = '8045aca0d84e70e622d1148d7df782496f6333bf6eb979a1b0837c42a9fd9561'


@dataclass
class Config:
    root: str = '/content/tgdetect_encoder_run'
    train: int = 5000
    val: int = 1000
    calibration: int = 2000
    test: int = 4000
    attack_fraction: float = .20
    batch: int = 128
    epochs: int = 3
    pretrain_epochs: int = 1
    seeds: tuple = (0, 1, 2)
    split_seed: int = 20261012
    bootstrap: int = 300
    latency_repeats: int = 60
    device: str = 'cuda'
    score_fields: int = 4
    cms_width: int = 65536
    memory_budget_gib: float = 15.0


ARMS = {
    'old': {},
    'old_extra_updates': {'extra': True},
    'rarity': {'rarity': True},
    'interaction': {'interaction': True},
    'interaction_extra_updates': {'interaction': True, 'extra': True},
    'interaction_masked': {'interaction': True, 'masked': True},
    'rarity_interaction': {'rarity': True, 'interaction': True},
    'rarity_interaction_extra_updates': {'rarity': True, 'interaction': True, 'extra': True},
    'full': {'rarity': True, 'interaction': True, 'masked': True},
    'full_contrastive': {'rarity': True, 'interaction': True, 'masked': True, 'contrastive': True},
}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FieldCMS:
    """New exact-string field sketch, separate from endpoint-signature counts."""
    def __init__(self, width=65536, depth=4):
        self.width, self.depth = width, depth
        self.table = np.zeros((depth, width), dtype=np.int64)

    def indices(self, key, value):
        text = json.dumps([key, value], sort_keys=True, separators=(',', ':'))
        raw = hashlib.blake2b(text.encode(), digest_size=8*self.depth).digest()
        return [int.from_bytes(raw[8*i:8*i+8], 'little') % self.width for i in range(self.depth)]

    def read(self, key, value):
        return int(min(self.table[i, h] for i, h in enumerate(self.indices(key, value))))

    def add(self, key, value):
        for i, h in enumerate(self.indices(key, value)):
            self.table[i, h] += 1


def load_data(cfg):
    root = Path(cfg.root); root.mkdir(parents=True, exist_ok=True)
    path = root / 'kddcup.data_10_percent.gz'
    if not path.exists():
        temp = path.with_suffix('.download')
        urls = [SOURCE_URL, 'https://figshare.com/ndownloader/files/5976042']
        errors = []
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=90) as src, temp.open('wb') as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                if digest(temp) != DATA_SHA256:
                    raise ValueError('Dataset SHA256 does not match the pinned scikit-learn archive')
                pd.read_csv(temp, compression='gzip', header=None, nrows=2)
                temp.replace(path); break
            except Exception as exc:
                errors.append(f'{url}: {exc}')
        else:
            raise RuntimeError('Dataset download failed; no synthetic fallback. ' + '; '.join(errors))
    if digest(path) != DATA_SHA256:
        raise ValueError('Cached dataset checksum mismatch; remove the corrupt download and retry')
    frame = pd.read_csv(path, header=None, names=FEATURES + ['attack'], compression='gzip')
    save_json(root / 'download.json', {'url': SOURCE_URL, 'sha256': digest(path), 'rows': len(frame)})
    return frame


def prepare_data(cfg, frame=None):
    """Split before fitting statistics. No exact encoder input crosses splits."""
    root = Path(cfg.root); root.mkdir(parents=True, exist_ok=True)
    frame = load_data(cfg) if frame is None else frame.copy()
    inputs = [c for c in FEATURES if c != 'service']
    frame['malicious'] = frame.attack.ne('normal.').astype(int)
    # Group by ALL encoder fields, excluding the reconstruction target.
    group = frame.groupby(inputs, dropna=False, sort=False)
    inconsistent = group[['service', 'malicious']].transform('nunique').max(axis=1).gt(1)
    clean = frame.loc[~inconsistent].drop_duplicates(inputs).copy()
    rng = np.random.default_rng(cfg.split_seed)
    benign = clean.index[clean.malicious.eq(0)].to_numpy(); rng.shuffle(benign)
    attacks = clean.index[clean.malicious.eq(1)].to_numpy(); rng.shuffle(attacks)
    nt_attack = round(cfg.test * cfg.attack_fraction); nt_benign = cfg.test - nt_attack
    needed = cfg.train + cfg.val + cfg.calibration + nt_benign
    if len(benign) < needed or len(attacks) < nt_attack:
        raise ValueError(f'Not enough unique records: benign={len(benign)}, attack={len(attacks)}')
    t = cfg.train; v = t + cfg.val; c = v + cfg.calibration
    split_indices = {'train': benign[:t], 'val': benign[t:v], 'cal': benign[v:c],
                     'test': np.r_[benign[c:c+nt_benign], attacks[:nt_attack]]}
    rng.shuffle(split_indices['test'])
    # Preserve source-row order only for the training-prefix count construction.
    split_indices['train'].sort()
    splits = {k: clean.loc[ix].copy() for k, ix in split_indices.items()}
    sets = [set(pd.util.hash_pandas_object(s[inputs], index=False).tolist()) for s in splits.values()]
    for i, a in enumerate(sets):
        for b in sets[i+1:]:
            assert a.isdisjoint(b), 'Encoder input overlap across splits'
    train = splits['train']
    service_values = ['<UNK>'] + sorted(train.service.unique().tolist())
    service_lookup = {s: i for i, s in enumerate(service_values)}
    # Predict each attribute using targets fitted ONLY on benign training.
    target_specs = {}
    for col in inputs:
        if pd.api.types.is_numeric_dtype(train[col]):
            cuts = np.unique(np.quantile(train[col].astype(float), np.linspace(0, 1, 17)[1:-1]))
            target_specs[col] = {'cuts': cuts.tolist(), 'classes': len(cuts) + 1}
        else:
            values = train[col].value_counts().index[:63].tolist()
            target_specs[col] = {'values': values, 'classes': len(values) + 1}
    bank_path = root / 'tokens'
    if bank_path.exists():
        raise FileExistsError('Use a fresh run directory; token bank already exists: ' + str(bank_path))
    writer = TokenWriter(bank_path)
    sketch = FieldCMS(cfg.cms_width); exact = Counter(); result = {}
    corpus = []
    train_keys = None
    for split, data in splits.items():
        ids, rarity, targets, target_unknown, labels, services, field_identity = [], [], [], [], [], [], []
        for _, row in data.iterrows():
            attrs = {key: row[key].item() if isinstance(row[key], np.generic) else row[key] for key in inputs}
            fields = [('@event_type', None)] + list(flatten(attrs))
            keys = [key for key, _ in fields]
            if train_keys is None:
                train_keys = keys
            assert keys == train_keys
            ids.append(writer.add({'attributes': attrs}))
            field_identity.append([json.dumps([key, value], sort_keys=True, separators=(',', ':')) for key, value in fields])
            rarity.append([math.log1p(sketch.read(key, value)) for key, value in fields])
            tg, unknown = [0], [False]
            for key, value in list(flatten(attrs)):
                # Flattened scalar key: remove its length prefix.
                col = key.split(':', 1)[1]; spec = target_specs[col]
                if 'cuts' in spec:
                    tg.append(int(np.searchsorted(spec['cuts'], float(value), side='right'))); unknown.append(False)
                else:
                    lookup = {v: i+1 for i, v in enumerate(spec['values'])}
                    target = lookup.get(value, 0); tg.append(target); unknown.append(target == 0)
            targets.append(tg); target_unknown.append(unknown)
            services.append(service_lookup.get(row.service, 0)); labels.append(int(row.malicious))
            if split == 'train':
                # Read ALL fields before committing the current record.
                for key, value in fields:
                    sketch.add(key, value)
                    exact[json.dumps([key, value], sort_keys=True)] += 1
                # Adapt original path-token SGNS to benign categorical telemetry.
                words = []
                for value in attrs.values():
                    kind, parts, _ = value_parts(value)
                    if kind not in ('number', 'bool', 'missing'):
                        words.extend(parts)
                if len(words) >= 2:
                    corpus.append(words)
        result[split] = {'ids': np.array(ids), 'rarity': np.array(rarity, np.float32),
                         'targets': np.array(targets), 'unknown': np.array(target_unknown),
                         'service': np.array(services), 'y': np.array(labels),
                         'field_identity': np.array(field_identity),
                         'family': data.attack.to_numpy().astype(str)}
    writer.close()
    corpus_path = root / 'benign_paths.jsonl'
    corpus_path.write_text(''.join(json.dumps(words) + '\n' for words in corpus))
    save_json(root / 'manifest.json', {'checksums': {'benign_paths.jsonl': digest(corpus_path),
              'tokens/tokens.json': digest(bank_path / 'tokens.json')}})
    errors = []
    for raw, count in exact.items():
        key, value = json.loads(raw); errors.append(sketch.read(key, value) - count)
    stats = {'raw_rows': len(frame), 'conflicting_rows_removed': int(inconsistent.sum()),
             'unique_inputs': len(clean), 'split_sizes': {k: len(v['ids']) for k, v in result.items()},
             'test_attack_fraction': float(result['test']['y'].mean()),
             'split_seed': cfg.split_seed, 'split_hashes': {k: hashlib.sha256(ix.tobytes()).hexdigest() for k, ix in split_indices.items()},
             'cms': {'width': cfg.cms_width, 'depth': 4, 'bytes': sketch.table.nbytes,
                     'mean_overcount': float(np.mean(errors)), 'max_overcount': int(max(errors)),
                     'fraction_exact': float(np.mean(np.array(errors) == 0))},
             'unseen_service_fraction': {k: float(np.mean(v['service'] == 0)) for k, v in result.items()},
             'field_target_unknown_fraction': {k: float(v['unknown'][:, 1:].mean()) for k, v in result.items()},
             'test_families': dict(Counter(result['test']['family'])),
             'temporal_claim': False, 'pretraining_corpus': 'benign training categorical values only'}
    stats['rarity_latency_scope'] = 'counts are prepared on CPU; encoder timing includes FiLM but excludes online counting'
    save_json(root / 'data_audit.json', stats)
    return result, train_keys, target_specs, len(service_values), stats


class CandidateEncoder(PackedEncoder):
    def __init__(self, bank, rarity=False, interaction=False, **unused):
        super().__init__(bank)
        self.use_rarity, self.use_interaction = rarity, interaction
        if rarity:
            self.film = nn.Linear(64, 128)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        if interaction:
            self.interact = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128,
                dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.mask_value = nn.Parameter(torch.zeros(64))
        self.field_table = None
        self.field_signature = None
        self.compact = False
        self.int8_tokens = None

    def train(self, mode=True):
        if mode:
            self.field_table = None
            self.field_signature = None
            self.int8_tokens = None
        return super().train(mode)

    def _apply(self, fn, recurse=True):
        self.field_table = None
        self.field_signature = None
        self.int8_tokens = None
        return super()._apply(fn, recurse=recurse)

    def embed(self, ids):
        if self.int8_tokens is None:
            return self.tokens(ids)
        quant, scale = self.int8_tokens
        return (quant[ids].float() * scale[ids]).to(self.tokens.weight.dtype)

    def fields(self, identities):
        if self.field_table is not None and self.field_signature != self._weight_signature():
            self.field_table = None
            self.field_signature = None
        if self.field_table is not None and not self.training and not torch.is_grad_enabled():
            p, k, refs, pads = self.field_table
            return p[refs[identities]], k[refs[identities]], pads[identities]
        data, pad = self.bank.fetch(identities, self.tokens.weight.device)
        words = data['word']; valid = words[..., 0] != 0
        weights = valid.to(self.tokens.weight.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
        flat = words.reshape(-1, words.shape[-2] * words.shape[-1])
        w = weights[..., None].expand_as(words).reshape_as(flat)
        if self.compact or self.int8_tokens is not None:
            # Compact IDs still require dynamic packing here; timed explicitly.
            valid_flat = flat != 0
            sizes = valid_flat.sum(-1)
            offsets = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
            chosen, chosen_w = flat[valid_flat], w[valid_flat]
            if self.int8_tokens is None:
                value = F.embedding_bag(chosen, self.tokens.weight, offsets, mode='sum',
                    per_sample_weights=chosen_w, include_last_offset=True, padding_idx=0)
            else:
                bags = torch.repeat_interleave(torch.arange(len(flat), device=flat.device), sizes)
                value = self.tokens.weight.new_zeros(len(flat), 64)
                value.index_add_(0, bags, self.embed(chosen) * chosen_w[:, None])
        else:
            offsets = torch.arange(0, flat.numel()+1, flat.shape[1], device=flat.device)
            value = F.embedding_bag(flat.flatten(), self.tokens.weight, offsets, mode='sum',
                per_sample_weights=w.flatten(), include_last_offset=True, padding_idx=0)
        value = value.reshape(len(identities), words.shape[1], 64)
        key = self.embed(data['key']).sum(-2)
        typed = self.kinds(data['kind']) + self.numeric(data['number'][..., None])
        return self.pair(torch.cat((key, value, typed), -1)), key, pad

    def forward(self, identities, rarity=None, hide=None, drop=None, return_fields=False):
        # Unique STATIC fields only: rarity/hide/drop belong to request occurrences.
        unique, inverse = torch.unique(identities, sorted=True, return_inverse=True)
        p, key, pad = self.fields(unique)
        p, key, pad = p[inverse], key[inverse], pad[inverse]
        if self.use_rarity:
            gamma, beta = self.film(key).chunk(2, -1)
            r = rarity[..., None].to(p.dtype)
            p = p * (1 + .1 * gamma.tanh() * r) + .1 * beta.tanh() * r
        if hide is not None:
            # Replace ENTIRE field representation, including kind/number/rarity.
            p = torch.where(hide[..., None], key + self.mask_value, p)
        if drop is not None:
            pad = pad | drop
            pad[:, 0] = False  # Always retain the event-type field.
            p = p.masked_fill(pad[..., None], 0)
        if self.use_interaction:
            p = self.interact(p, src_key_padding_mask=pad)
        pooled, _ = self.attention(self.query.expand(len(identities), -1, -1), p, p,
                                   key_padding_mask=pad, need_weights=False)
        out = self.output(pooled[:, 0])
        return (out, p) if return_fields else out


class PilotModel(nn.Module):
    def __init__(self, bank, classes, field_classes, options):
        super().__init__()
        self.options = options
        # Keep the actual original baseline class, not a replacement architecture.
        self.encoder = CandidateEncoder(bank, **options) if any(options.get(k) for k in ('rarity', 'interaction')) else PackedEncoder(bank)
        self.decoder = nn.Linear(64, classes)
        if options.get('masked'):
            self.mask_heads = nn.ModuleList(nn.Linear(64, n) for n in field_classes)

    def encode(self, ids, rarity, **kwargs):
        if isinstance(self.encoder, CandidateEncoder):
            return self.encoder(ids, rarity, **kwargs)
        return self.encoder(ids)

    def forward(self, ids, rarity):
        return self.decoder(self.encode(ids, rarity))

    def masked_loss(self, ids, rarity, targets, positions):
        hide = torch.zeros_like(targets, dtype=torch.bool)
        hide.scatter_(1, positions[:, None], True)
        _, fields = self.encode(ids, rarity, hide=hide, return_fields=True)
        losses = fields.new_zeros(len(ids))
        for pos in positions.unique().tolist():
            rows = positions == pos
            losses[rows] = F.cross_entropy(self.mask_heads[pos](fields[rows, pos]), targets[rows, pos], reduction='none')
        return losses


def batches(data, cfg, order=None):
    order = np.arange(len(data['ids'])) if order is None else order
    for start in range(0, len(order), cfg.batch):
        ix = order[start:start+cfg.batch]
        yield {k: torch.as_tensor(data[k][ix], device=cfg.device) for k in ('ids', 'rarity', 'targets', 'service')}


def sync(cfg):
    if cfg.device.startswith('cuda'):
        torch.cuda.synchronize()


def score(model, data, cfg, positions, include_masked=True):
    model.eval(); channels = []
    with torch.inference_mode():
        for batch in batches(data, cfg):
            ce = F.cross_entropy(model(batch['ids'], batch['rarity']), batch['service'], reduction='none')
            cols = [ce]
            if include_masked and model.options.get('masked'):
                losses = []
                for pos in positions:
                    where = torch.full_like(batch['ids'], pos)
                    losses.append(model.masked_loss(batch['ids'], batch['rarity'], batch['targets'], where))
                cols.append(torch.stack(losses).mean(0))
            channels.append(torch.stack(cols, -1).float().cpu().numpy())
    return np.concatenate(channels)


def cdf_fusion(reference, values):
    # Fixed equal-weight rank fusion; reference uses only benign calibration A.
    columns = []
    for j in range(reference.shape[1]):
        ref = np.sort(reference[:, j])
        columns.append(np.searchsorted(ref, values[:, j], side='right') / (len(ref)+1))
    return np.stack(columns, -1).mean(-1)


def calibrated_threshold(scores, fpr):
    # Strict > comparison; conservative finite-sample order statistic.
    n = len(scores); index = min(n-1, max(0, math.ceil((n+1)*(1-fpr))-1))
    return float(np.sort(scores)[index])


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return [None, None]
    p = successes/total; denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def metrics(y, values, calibration):
    result = {'ap': float(average_precision_score(y, values)), 'roc_auc': float(roc_auc_score(y, values)),
              'test_prevalence': float(np.mean(y))}
    for fpr in (.01, .05):
        threshold = calibrated_threshold(calibration, fpr)
        predicted = values > threshold
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        precision = tp/max(tp+fp, 1); recall = tp/max(tp+fn, 1)
        result[f'fpr_target_{fpr}'] = {'threshold': threshold, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'precision': precision, 'recall': recall, 'f1': 2*precision*recall/max(precision+recall, 1e-12),
            'test_fpr': fp/max(tn+fp, 1), 'false_alerts_per_1000_benign': 1000*fp/max(tn+fp, 1),
            'test_fpr_wilson95': wilson_interval(fp, tn+fp),
            'recall_wilson95': wilson_interval(tp, tp+fn),
            'precision_wilson95': wilson_interval(tp, tp+fp)}
    ranked = np.argsort(-values, kind='stable')
    result['precision_at'] = {str(k): float(y[ranked[:min(k, len(y))]].mean()) for k in (10, 30, 100)}
    return result


def train_arm(cfg, data, bank, n_services, field_classes, initialization, arm, seed, positions):
    out = Path(cfg.root)/'results'/f'{arm}_s{seed}'
    out.mkdir(parents=True, exist_ok=False)
    seed_all(seed); options = ARMS[arm]
    # Construct common modules under a fixed RNG stream, then copy to every arm.
    baseline = PilotModel(bank, n_services, field_classes, {})
    seed_all(seed + 10000)
    # Common added blocks get identical initial weights even if another arm
    # inserts FiLM before the interaction layer and consumes different RNG draws.
    prototype = PilotModel(bank, n_services, field_classes, ARMS['full'])
    model = PilotModel(bank, n_services, field_classes, options).to(cfg.device)
    state = {k: v for k, v in prototype.state_dict().items() if k in model.state_dict()}
    state.update(baseline.state_dict())
    model.load_state_dict(state, strict=True)
    del baseline, prototype
    with torch.no_grad():
        model.encoder.tokens.weight.copy_(initialization.to(cfg.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if cfg.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    history = []; steps = 0; best = math.inf
    sync(cfg); started = time.perf_counter()
    phases = [('pretrain', cfg.pretrain_epochs)] if options.get('masked') or options.get('extra') else []
    phases.append(('task', cfg.epochs))
    for phase, epochs in phases:
        for epoch in range(epochs):
            model.train()
            order = np.random.default_rng(seed + epoch + (1000 if phase == 'pretrain' else 0)).permutation(len(data['train']['ids']))
            loss_total = 0.; examples = 0
            for step, batch in enumerate(batches(data['train'], cfg, order)):
                if phase == 'pretrain' and options.get('masked'):
                    generator = torch.Generator(device=cfg.device).manual_seed(seed*100000 + epoch*1000 + step)
                    where = torch.randint(1, len(field_classes), (len(batch['ids']),), generator=generator, device=cfg.device)
                    loss = model.masked_loss(batch['ids'], batch['rarity'], batch['targets'], where).mean()
                    if options.get('contrastive'):
                        drops = torch.rand(batch['targets'].shape, generator=generator, device=cfg.device) < .15
                        a = F.normalize(model.encode(batch['ids'], batch['rarity']), dim=-1)
                        b = F.normalize(model.encode(batch['ids'], batch['rarity'], drop=drops), dim=-1)
                        logits = a @ b.T / .1
                        target = torch.arange(len(a), device=a.device)
                        loss = loss + .1*(F.cross_entropy(logits, target)+F.cross_entropy(logits.T, target))/2
                else:
                    loss = F.cross_entropy(model(batch['ids'], batch['rarity']), batch['service'])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step(); steps += 1
                loss_total += float(loss.detach())*len(batch['ids']); examples += len(batch['ids'])
            val = float(score(model, data['val'], cfg, positions, include_masked=False)[:, 0].mean())
            row = {'phase': phase, 'epoch': epoch+1, 'train_loss': loss_total/examples, 'benign_val_service_ce': val}
            history.append(row); print(arm, seed, row, flush=True)
            if phase == 'task' and val < best:
                best = val
                torch.save({'model': model.state_dict(), 'options': options, 'epoch': epoch+1, 'seed': seed}, out/'best.pt')
    sync(cfg); train_seconds = time.perf_counter()-started
    state = torch.load(out/'best.pt', map_location=cfg.device, weights_only=False)
    model.load_state_dict(state['model']); model.eval()
    cal_channels = score(model, data['cal'], cfg, positions)
    test_channels = score(model, data['test'], cfg, positions)
    half = len(cal_channels)//2
    # Service-only retains raw CE for every arm (including old); fused is secondary.
    primary = metrics(data['test']['y'], test_channels[:, 0], cal_channels[half:, 0])
    cutoff = primary['fpr_target_0.01']['threshold']
    family_recall = {}
    for family in sorted(set(data['test']['family'])):
        subset = data['test']['family'] == family
        if not data['test']['y'][subset].any():
            continue
        n = int(subset.sum()); caught = int((test_channels[subset, 0] > cutoff).sum())
        family_recall[family] = {'n': n, 'detected': caught, 'recall': caught/n,
                                 'wilson95': wilson_interval(caught, n)}
    fused_test = cdf_fusion(cal_channels[:half], test_channels)
    fused_cal = cdf_fusion(cal_channels[:half], cal_channels[half:])
    fused = metrics(data['test']['y'], fused_test, fused_cal)
    np.savez_compressed(out/'scores.npz', y=data['test']['y'], service=test_channels[:, 0], fused=fused_test,
                        channels=test_channels, calibration=cal_channels, family=data['test']['family'])
    timing = benchmark(model, data['test'], cfg, positions)
    result = {'arm': arm, 'seed': seed, 'primary_service': primary, 'secondary_fused': fused,
              'attack_family_recall_at_1pct_target': family_recall,
              'history': history, 'checkpoint_epoch': state['epoch'], 'optimizer_updates': steps,
              'training_seconds_including_validation': train_seconds,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'peak_cuda_bytes': torch.cuda.max_memory_allocated() if cfg.device.startswith('cuda') else None,
              'latency': timing, 'checkpoint_sha256': digest(out/'best.pt')}
    save_json(out/'result.json', result)
    return result, model


def timed(fn, cfg, repeats=None):
    repeats = cfg.latency_repeats if repeats is None else repeats
    with torch.inference_mode():
        for _ in range(5):
            fn()
        sync(cfg); samples = []
        for _ in range(repeats):
            sync(cfg); start = time.perf_counter(); fn(); sync(cfg)
            samples.append((time.perf_counter()-start)*1000)
    return {'p50_ms': float(np.median(samples)), 'p95_ms': float(np.quantile(samples, .95)),
            'p99_ms': float(np.quantile(samples, .99)), 'samples_ms': samples,
            'scope': 'synchronized wall time; prepared inputs resident; excludes preparation and batch wait'}


def benchmark(model, data, cfg, positions):
    model.eval(); results = {}
    for size in (1, 32, 128):
        size = min(size, len(data['ids']))
        batch = {k: torch.as_tensor(data[k][:size], device=cfg.device) for k in ('ids', 'rarity', 'targets', 'service')}
        ids, r = batch['ids'], batch['rarity']
        encoder = timed(lambda: model.encode(ids, r), cfg)
        def scoring():
            ce = F.cross_entropy(model(ids, r), batch['service'], reduction='none')
            if model.options.get('masked'):
                for pos in positions:
                    ce = ce + model.masked_loss(ids, r, batch['targets'], torch.full_like(ids, pos)) / len(positions)
            return ce
        full = timed(scoring, cfg)
        results[str(size)] = {'encoder': encoder, 'service_and_masked_channels': full,
                              'masked_forward_passes': len(positions) if model.options.get('masked') else 0}
    return results


def bootstrap_delta(y, old, new, repeats=300, seed=123):
    rng = np.random.default_rng(seed); groups = [np.flatnonzero(y == label) for label in (0, 1)]
    diffs = []
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        diffs.append(average_precision_score(y[ix], new[ix])-average_precision_score(y[ix], old[ix]))
    return {'delta_ap': float(average_precision_score(y, new)-average_precision_score(y, old)),
            'ci95_low': float(np.quantile(diffs, .025)), 'ci95_high': float(np.quantile(diffs, .975)),
            'scope': 'paired stratified record bootstrap; conditional on this test sample; not capture-level uncertainty'}


def acceptance_check(bank, data, cfg):
    """Runs on the selected Colab GPU before any experiment training."""
    seed_all(2026)
    old = PackedEncoder(bank).to(cfg.device).eval()
    candidate = CandidateEncoder(bank).to(cfg.device).eval()
    candidate.load_state_dict(old.state_dict(), strict=False)
    ids = torch.as_tensor(data['train']['ids'][:8], device=cfg.device)
    ids = torch.cat((ids, ids[:3]))
    with torch.inference_mode():
        reference = old(ids)
        error = float((candidate(ids)-reference).abs().max())
        candidate.compact = True
        compact_error = float((candidate(ids)-reference).abs().max())
    if max(error, compact_error) > 1e-4:
        raise AssertionError(f'Original-encoder parity failed on {cfg.device}: {error}, {compact_error}')
    result = {'device': cfg.device, 'wrapper_max_abs_error': error,
              'compact_max_abs_error': compact_error, 'tolerance': 1e-4}
    save_json(Path(cfg.root)/'results/encoder_acceptance.json', result)
    return result


def summarize(cfg, results):
    rows = []
    for result in results:
        metric = result['primary_service']; op = metric['fpr_target_0.01']
        rows.append({'arm': result['arm'], 'seed': result['seed'], 'ap': metric['ap'], 'roc_auc': metric['roc_auc'],
                     'f1': op['f1'], 'precision': op['precision'], 'recall': op['recall'], 'test_fpr': op['test_fpr'],
                     'fp': op['fp'], 'tp': op['tp'], 'fused_ap': result['secondary_fused']['ap'],
                     'train_seconds': result['training_seconds_including_validation'],
                     'parameters': result['parameter_count'], 'encoder_p50_b128': result['latency'][str(min(128, cfg.test))]['encoder']['p50_ms']})
    table = pd.DataFrame(rows)
    out = Path(cfg.root)/'results'; table.to_csv(out/'runs.csv', index=False)
    summary = table.groupby('arm').agg({c: ['mean', 'std'] for c in ('ap', 'f1', 'recall', 'test_fpr', 'fused_ap', 'encoder_p50_b128')})
    summary.to_csv(out/'seed_summary.csv')
    intervals = []
    for seed in cfg.seeds:
        base = np.load(out/f'old_s{seed}'/'scores.npz')
        for arm in sorted(set(table.arm)-{'old'}):
            candidate = np.load(out/f'{arm}_s{seed}'/'scores.npz')
            assert np.array_equal(base['y'], candidate['y'])
            intervals.append({'arm': arm, 'seed': seed, **bootstrap_delta(base['y'], base['service'], candidate['service'], cfg.bootstrap)})
    save_json(out/'paired_ap_intervals.json', intervals)
    return table, summary


def run(cfg=Config(), frame=None, arms=None):
    root = Path(cfg.root)
    if (root/'results'/'runs.csv').exists():
        raise FileExistsError('Completed results exist; choose a new Config.root to rerun.')
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab. CPU smoke runs require device="cpu" explicitly.')
    torch.set_num_threads(2)
    if cfg.device.startswith('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        total_memory = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1., cfg.memory_budget_gib*2**30/total_memory))
    root.mkdir(parents=True, exist_ok=True)
    save_json(root/'config.json', asdict(cfg))
    save_json(root/'environment.json', {'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__,
              'sklearn': sklearn.__version__, 'python': platform.python_version(), 'cuda': torch.version.cuda,
              'gpu': torch.cuda.get_device_name() if cfg.device.startswith('cuda') else 'CPU smoke only',
              'vram_bytes': torch.cuda.get_device_properties(0).total_memory if cfg.device.startswith('cuda') else None,
              'source_sha256': digest(__file__)})
    preparation_start = time.perf_counter()
    data, keys, specs, classes, audit = prepare_data(cfg, frame)
    audit['preparation_seconds_including_download'] = time.perf_counter()-preparation_start
    save_json(root/'data_audit.json', audit)
    # A cheap control detects whether learning adds anything beyond frequency.
    cal_rare = np.exp(-data['cal']['rarity'][:, 1:]).mean(1)
    test_rare = np.exp(-data['test']['rarity'][:, 1:]).mean(1)
    save_json(root/'results/rarity_only_control.json', {
        'definition': 'mean 1/(1+count(key,value)); excludes synthetic event-type field; no neural training',
        'metrics': metrics(data['test']['y'], test_rare, cal_rare[len(cal_rare)//2:])})
    bank = TokenBank(root/'tokens')
    if cfg.device.startswith('cuda'):
        bank.to_device(cfg.device)
    print('Encoder acceptance:', acceptance_check(bank, data, cfg), flush=True)
    field_classes = [1] + [specs[k.split(':', 1)[1]]['classes'] for k in keys[1:]]
    # Fixed field selection: training-only entropy; no test labels choose fields.
    entropies = []
    for i in range(1, len(keys)):
        counts = np.bincount(data['train']['targets'][:, i], minlength=field_classes[i]); p = counts[counts>0]/counts.sum()
        entropies.append((-float(-(p*np.log(p)).sum()), i))
    positions = [i for _, i in sorted(entropies)[:cfg.score_fields]]
    save_json(root/'masked_score_fields.json', {'positions': positions, 'keys': [keys[i] for i in positions],
              'selection': 'highest benign-training target entropy; fixed across all arms'})
    init_path = root/'initial_embeddings.pt'
    init_started = time.perf_counter()
    init_info = pretrain(root, init_path, epochs=1, batch=2048, projection_steps=30, seed=cfg.split_seed, device=cfg.device)
    sync(cfg)
    save_json(root/'shared_sgns.json', {**init_info, 'seconds': time.perf_counter()-init_started,
                                     'sha256': digest(init_path), 'shared_across_all_arms': True})
    init = torch.load(init_path, map_location='cpu', weights_only=False)['tokens']
    results = []; last_models = {}
    arms = list(ARMS) if arms is None else arms
    for seed in cfg.seeds:
        # Rotate arm order to reduce systematic thermal/startup bias.
        order = arms[seed % len(arms):] + arms[:seed % len(arms)]
        for arm in order:
            try:
                result, model = train_arm(cfg, data, bank, classes, field_classes, init, arm, seed, positions)
            except Exception as exc:
                save_json(root/'results'/f'{arm}_s{seed}_failure.json', {'error': repr(exc)})
                raise
            results.append(result)
            if seed == cfg.seeds[-1] and arm in ('old', 'full'):
                last_models[arm] = model.cpu()
            else:
                del model
            if cfg.device.startswith('cuda'):
                torch.cuda.empty_cache()
    table, summary = summarize(cfg, results)
    print(summary.to_string(), flush=True)
    return {'config': cfg, 'data': data, 'bank': bank, 'models': last_models, 'positions': positions,
            'table': table, 'summary': summary, 'audit': audit}
