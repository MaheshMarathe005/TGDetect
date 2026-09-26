"""Verified flow-completion pilot data; no legacy feature/cache imports."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FEATURES = ['log_dur', 'log_pkts', 'log_bytes', 'log_src_bytes', 'src_byte_ratio',
            'tcp', 'udp', 'icmp', 'dport_fraction', 'dport_missing']


def transport_port(proto, token):
    if proto not in {'tcp', 'udp'}:
        return -1
    try:
        p = int(token, 16) if token.lower().startswith('0x') else int(token)
    except ValueError:
        return -1
    if not 0 <= p <= 65535:
        raise ValueError('TCP/UDP port outside range')
    return p


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_scaler(x, train_end):
    mean = x[:train_end].mean(axis=0)
    std = x[:train_end].std(axis=0)
    return mean, np.where(std < 1e-6, 1., std)


def causal_egos(src, dst, times, neighbors=4):
    """Self + most recent strictly earlier incident events, independent of labels."""
    if np.any(np.diff(times) < 0):
        raise ValueError('Events must be sorted by observation time')
    n = len(times)
    indices = np.zeros((n, neighbors + 1), dtype=np.int64)
    valid = np.zeros_like(indices, dtype=bool)
    ages = np.zeros_like(indices, dtype=np.float32)
    history = defaultdict(lambda: deque(maxlen=neighbors))
    pending = []
    prior_time = None
    for i, (s, d, t) in enumerate(zip(src, dst, times)):
        if prior_time is None or t != prior_time:
            for old in pending:
                for node in {int(src[old]), int(dst[old])}:
                    history[node].append(old)
            pending = []
        recent = sorted(set(history[int(s)]) | set(history[int(d)]), reverse=True)[:neighbors]
        selected = [i] + recent
        indices[i, :len(selected)] = selected
        valid[i, :len(selected)] = True
        ages[i, :len(selected)] = np.log1p(t - times[selected])
        pending.append(i)
        prior_time = t
    return indices, valid, ages


def prepare(raw='data/ctu13_verified/scenario06.binetflow', samples=16384):
    raw = Path(raw)
    source = json.loads(raw.with_suffix('.source.json').read_text())
    if sha256(raw) != source['sha256']:
        raise ValueError('Raw download checksum mismatch')
    rows, nodes = [], {}
    with raw.open() as f:
        for row_id, r in enumerate(csv.DictReader(f)):
            dur, pkts, size, source_size = [float(r[k] or 0) for k in ('Dur', 'TotPkts', 'TotBytes', 'SrcBytes')]
            if not np.isfinite([dur, pkts, size, source_size]).all() or min(dur, pkts, size, source_size) < 0:
                raise ValueError(f'Invalid numeric record {row_id}')
            start = datetime.strptime(r['StartTime'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=timezone.utc).timestamp()
            label = r['Label']
            if 'Botnet' in label:
                y = 1
            elif 'Normal' in label or 'Background' in label:
                y = 0
            else:
                raise ValueError(f'Unknown label at {row_id}: {label}')
            proto = r['Proto'].lower()
            p = transport_port(proto, r['Dport'].strip())
            s = nodes.setdefault(r['SrcAddr'], len(nodes))
            d = nodes.setdefault(r['DstAddr'], len(nodes))
            x = [*np.log1p([dur, pkts, size, source_size]), source_size / max(size, 1),
                 float(proto == 'tcp'), float(proto == 'udp'), float(proto == 'icmp'),
                 max(p, 0) / 65535, float(p < 0)]
            rows.append((start + dur, row_id, s, d, y, x))
    rows.sort(key=lambda r: (r[0], r[1]))
    full_count = len(rows)
    if full_count < samples:
        raise ValueError('Requested sample exceeds capture')
    chosen = np.linspace(0, full_count - 1, samples, dtype=np.int64)
    rows = [rows[i] for i in chosen]
    times = np.array([r[0] for r in rows], dtype=np.float64)
    ids = np.array([r[1] for r in rows], dtype=np.int64)
    src = np.array([r[2] for r in rows], dtype=np.int64)
    dst = np.array([r[3] for r in rows], dtype=np.int64)
    y = np.array([r[4] for r in rows], dtype=np.int64)
    x = np.array([r[5] for r in rows], dtype=np.float32)
    ends = [int(samples * .6), int(samples * .8)]
    for k in range(2):
        while ends[k] < samples and times[ends[k]] == times[ends[k] - 1]:
            ends[k] += 1
    train_end, val_end = ends
    if not 0 < train_end < val_end < samples:
        raise ValueError('Invalid chronological boundaries')
    mean, std = fit_scaler(x, train_end)
    scaled = (x - mean) / std
    ego, valid, ages = causal_egos(src, dst, times)
    partitions = {}
    for name, lo, hi in [('train', 0, train_end), ('validation', train_end, val_end), ('test', val_end, samples)]:
        yy = y[lo:hi]
        if len(np.unique(yy)) != 2:
            raise ValueError(f'{name} lacks both classes; no performance experiment allowed')
        partitions[name] = {'start': lo, 'end': hi, 'count': hi-lo, 'positives': int(yy.sum()),
                            'start_time': float(times[lo]), 'end_time': float(times[hi-1])}
    meta = {'raw_source': source, 'raw_rows': full_count, 'selected_rows': samples,
            'selection': 'linspace positions in completion-time order; no label sampling',
            'split_id': 'completion_chronological_60_20_20_v1', 'partitions': partitions,
            'features': FEATURES, 'scaler_mean': mean.tolist(), 'scaler_std': std.tolist(),
            'observation_time': 'StartTime+Dur; exporter arrival unavailable',
            'label_mapping': 'Botnet=1; Normal and Background=0 (Background uncertain)',
            'selected_row_ids_sha256': hashlib.sha256(ids.tobytes()).hexdigest(),
            'neighbors': 4, 'sampled_nodes': len(np.unique(np.r_[src, dst]))}
    out = raw.parent / f'{raw.stem}_pilot{samples}.npz'
    np.savez_compressed(out, x=scaled, y=y, src=src, dst=dst, times=times, row_ids=ids,
                        ego=ego, valid=valid, ages=ages, train_end=train_end, val_end=val_end)
    meta['processed_sha256'] = sha256(out)
    out.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps({k: meta[k] for k in ('raw_rows', 'partitions', 'sampled_nodes', 'processed_sha256')}, indent=2))


if __name__ == '__main__':
    prepare()
