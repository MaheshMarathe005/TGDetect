"""DEDALE ECS/Sysmon -> shared provenance messages, with explicit coverage accounting.

No existing 72-feature archive can recover the lost file/socket graph. Raw logs
are required. Unsupported event semantics are excluded, never silently relabeled.
"""
import argparse
import bz2
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import zipfile

import numpy as np
from sklearn.feature_extraction import FeatureHasher

from causal_data import causal_egos

RELATIONS = ('START', 'CLOSE', 'CLONE', 'READ', 'WRITE', 'OPEN', 'EXEC', 'SEND', 'RECEIVE')
SCHEMA = 'dedale-kairos-hierarchical16-relation9-v1'
LABEL_MEMBER = 'system_labels/clients_1_and_2/malicious_events_class_1.jsonl'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def host(e):
    return str((e.get('host') or {}).get('name') or (e.get('agent') or {}).get('name') or
               (e.get('winlog') or {}).get('computer_name') or '').split('.')[0].upper()


def timestamp(e):
    dt = datetime.fromisoformat(e['@timestamp'].replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Timestamp timezone missing')
    return dt.timestamp()


def event_key(e):
    w = e.get('winlog') or {}
    # Timestamp disambiguates reused record IDs after a channel reset.
    return json.dumps([host(e), w.get('channel'), str(w.get('record_id')), e.get('@timestamp')], separators=(',', ':'))


def fields(e):
    # ECS moves common fields out of event_data; message fallback restores only
    # explicit Sysmon fields, never rule/MITRE/attack annotations.
    values = {}
    for line in str(e.get('message') or '').splitlines():
        key, sep, value = line.partition(': ')
        if sep:
            values[key.lower()] = value.strip()
    values.update({k.lower(): v for k, v in (e.get('winlog', {}).get('event_data') or {}).items()})
    return values


def process(e, f, prefix=''):
    p = e.get('process') or {}
    if prefix == 'parent':
        p = p.get('parent') or {}
    guid = p.get('entity_id') or f.get(prefix + 'processguid') or f.get(prefix + 'processuuid')
    path = p.get('executable') or f.get(prefix + 'image')
    if not path:
        return None
    if guid and str(guid).strip('{}0-'):
        key = f'{host(e)}|process|guid|{str(guid).lower()}'
    else:
        pid = p.get('pid') or f.get(prefix + 'processid')
        start = p.get('start') or f.get(prefix + 'processstarttime')
        if pid is not None and start:
            key = f'{host(e)}|process|pidstart|{pid}|{start}'
        else:
            # No unsupported assumption that PID identifies one lifetime.
            key = f'{host(e)}|process|isolated|{prefix}|{event_key(e)}'
    return key, 'subject', str(path).replace('\\', '/').lower()


def convert(e):
    """One supported raw event -> one directed edge; returns (edge, reason)."""
    ev, w = e.get('event') or {}, e.get('winlog') or {}
    provider = str(ev.get('provider') or w.get('provider_name') or '').lower()
    if 'sysmon' not in provider and ev.get('module') != 'sysmon':
        return None, 'not_sysmon'
    code = str(ev.get('code') or w.get('event_id'))
    if code not in {'1', '3', '5', '7', '11'}:
        return None, 'unsupported_code_' + code
    f = fields(e)
    proc = process(e, f)
    if proc is None:
        return None, 'missing_process_path'
    source, dest = proc, None
    if code == '1':
        source, dest, relation = process(e, f, 'parent'), proc, 'START'
    elif code == '5':
        dest, relation = proc, 'CLOSE'  # Termination has no second distinct entity.
    elif code in {'7', '11'}:
        path = (e.get('file') or {}).get('path') or f.get('imageloaded' if code == '7' else 'targetfilename')
        if path:
            path = str(path).replace('\\', '/').lower()
            dest = (f'{host(e)}|file|{path}', 'file', path)
        relation = 'READ' if code == '7' else 'WRITE'
    else:
        # Sysmon connection establishment is a proxy for SEND/RECEIVE, not a
        # complete system-call trace. This unavoidable adaptation is reported.
        initiated = str(f.get('initiated', '')).lower()
        if initiated not in {'true', 'false'}:
            return None, 'missing_network_direction'
        src_net, dst_net = e.get('source') or {}, e.get('destination') or {}
        sip, dip = src_net.get('ip') or f.get('sourceip'), dst_net.get('ip') or f.get('destinationip')
        sport, dport = src_net.get('port') or f.get('sourceport'), dst_net.get('port') or f.get('destinationport')
        if not sip or not dip or sport is None or dport is None:
            return None, 'missing_socket_attributes'
        path = f'{sip}:{sport}->{dip}:{dport}'
        dest = (f'{host(e)}|socket|{path}', 'netflow', path)
        relation = 'SEND' if initiated == 'true' else 'RECEIVE'
        if initiated == 'false':
            source, dest = dest, source
    if source is None or dest is None:
        return None, 'missing_endpoint'
    return (source, dest, RELATIONS.index(relation)), 'included'


@lru_cache(maxsize=100000)
def node_feature(kind, attribute):
    """Paper §4.1: sum signed character hashes over hierarchical prefixes.

    The released code includes a node-type token. list(text) explicitly preserves
    its old FeatureHasher character semantics with current scikit-learn versions.
    """
    if kind == 'netflow':
        chunks = []
        for endpoint in attribute.split('->'):
            ip, port = endpoint.rsplit(':', 1)
            pieces = ip.split('.' if '.' in ip else ':')
            sep = '.' if '.' in ip else ':'
            chunks.extend(sep.join(pieces[:i]) for i in range(1, len(pieces) + 1))
            chunks.append(':' + port)
    else:
        parts = attribute.split('/')
        chunks = ['/'.join(parts[:i]) for i in range(1, len(parts) + 1)]
    text = kind + ''.join(chunks)
    return FeatureHasher(n_features=16, input_type='string', alternate_sign=True).transform([list(text)]).toarray()[0].astype('float32')


def message(edge):
    src, dst, relation = edge
    onehot = np.eye(len(RELATIONS), dtype='float32')[relation]
    return np.concatenate((node_feature(src[1], src[2]), onehot, node_feature(dst[1], dst[2])))


def read_member(archive, member):
    with archive.open(member) as raw:
        stream = bz2.BZ2File(raw) if member.endswith('.bz2') else gzip.GzipFile(fileobj=raw) if member.endswith('.gz') else raw
        for line in io.TextIOWrapper(stream, encoding='utf-8'):
            if line.strip():
                yield json.loads(line)


def scan_member(args):
    raw, member, hosts, start, end, positive, ambiguous = args
    local, rows = Counter(), []
    with zipfile.ZipFile(raw) as archive:
        for e in read_member(archive, member):
            if host(e) not in hosts:
                continue
            t = timestamp(e)
            if (start is not None and t < start) or (end is not None and t >= end):
                continue
            key = event_key(e)
            local['raw_client_records'] += 1
            label = int(key in positive)
            local['raw_class1_records'] += label
            if key in ambiguous:
                local['excluded_class2_only'] += 1
                continue
            edge, reason = convert(e)
            local[reason] += 1
            if edge is None:
                local['excluded_class1'] += label
                continue
            s, d, rel = edge
            rows.append((key, t, s[0], d[0], rel, label, message(edge).tobytes(), json.dumps(s), json.dumps(d)))
    return member, local, rows


def prepare(raw, labels, output, hosts=('CLIENT1', 'CLIENT2'), start=None, end=None, workers=1, on_checkpoint=None):
    raw, labels, output = Path(raw), Path(labels), Path(output)
    if not zipfile.is_zipfile(raw):
        raise ValueError(f'Complete raw ZIP required: {raw}')
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(raw_sha256=sha256(raw), labels_sha256=sha256(labels), hosts=list(hosts),
                     start=start, end=end, schema=SCHEMA)
    signature['adapter_sha256'] = sha256(__file__)
    sigfile = output / 'preparation_signature.json'
    if sigfile.exists() and json.loads(sigfile.read_text()) != signature:
        raise ValueError('Output contains different preparation; choose a new directory')
    sigfile.write_text(json.dumps(signature, indent=2))
    positive, ambiguous = set(), set()
    with zipfile.ZipFile(labels) as archive:
        positive = {event_key(e) for e in read_member(archive, LABEL_MEMBER)}
        union = LABEL_MEMBER.replace('class_1.jsonl', 'class_1_and_2.jsonl')
        ambiguous = {event_key(e) for e in read_member(archive, union)} - positive
    db = sqlite3.connect(output / 'events.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, t REAL, src TEXT, dst TEXT, relation INTEGER, y INTEGER, x BLOB, sm TEXT, dm TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS completed (member TEXT PRIMARY KEY, counts TEXT)')
    completed = dict(db.execute('SELECT member, counts FROM completed'))
    counts = Counter()
    for saved in completed.values():
        counts.update(json.loads(saved))
    with zipfile.ZipFile(raw) as archive:
        members = []
        for info in archive.infolist():
            member = info.filename
            if not info.file_size or not member.endswith(('.jsonl', '.json', '.gz', '.bz2')) or member in completed:
                continue
            match = re.search(r'(\d{4}-\d{2}-\d{2})', member)
            if match and (start is not None or end is not None):
                date = datetime.fromisoformat(match[1]).replace(tzinfo=timezone.utc).timestamp()
                # One-day margin for filename timezone; event timestamps govern inclusion.
                if (start is not None and date + 2*86400 < start) or (end is not None and date - 86400 > end):
                    continue
            members.append(member)
    tasks = [(str(raw), m, hosts, start, end, positive, ambiguous) for m in members]
    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        results = pool.map(scan_member, tasks) if pool else map(scan_member, tasks)
        for member, local, rows in results:
            with db:
                for row in rows:
                    cur = db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?,?,?)',
                                     row)
                    local['unique_events'] += cur.rowcount
                    local['unique_class1'] += row[5] * cur.rowcount
                    local['isolated_identity_events'] += int('|isolated|' in row[2] or '|isolated|' in row[3]) * cur.rowcount
                db.execute('INSERT INTO completed VALUES (?,?)', (member, json.dumps(local)))
            counts.update(local)
            print(json.dumps(dict(member=member, counts=dict(counts))), flush=True)
            if on_checkpoint:
                on_checkpoint()
    finally:
        if pool:
            pool.shutdown()
    n = db.execute('SELECT count(*) FROM events').fetchone()[0]
    if n < 30:
        raise ValueError('Insufficient supported events')
    # Preallocate; read sorted SQL rows in a stream rather than duplicating them.
    arrays = {k: np.empty(n, dtype=dtype) for k, dtype in
              [('src', 'int64'), ('dst', 'int64'), ('times', 'float64'), ('y', 'int64'), ('relation', 'int64')]}
    x = np.empty((n, 41), dtype='float32')
    ids = np.empty(n, dtype='U64')
    nodes, metadata = {}, []
    for i, (key, t, s, d, rel, label, features, sm, dm) in enumerate(db.execute('SELECT * FROM events ORDER BY t, id')):
        for node, meta in ((s, sm), (d, dm)):
            if node not in nodes:
                nodes[node] = len(nodes)
                metadata.append(json.loads(meta))
        arrays['src'][i], arrays['dst'][i], arrays['times'][i] = nodes[s], nodes[d], t
        arrays['y'][i], arrays['relation'][i] = label, rel
        x[i] = np.frombuffer(features, dtype='float32')
        ids[i] = hashlib.sha256(key.encode()).hexdigest()
    db.close()
    # Label-independent boundaries moved to the next wall-clock window.
    windows = np.floor(arrays['times'] / 900).astype('int64')
    tr, va = [int(np.searchsorted(windows, windows[int(n * frac)], side='right')) for frac in (.6, .8)]
    if not 0 < tr < va < n:
        raise ValueError('Too few time windows for chronological 60/20/20 split')
    ego, valid, ages = causal_egos(arrays['src'], arrays['dst'], arrays['times'])
    # Released KAIROS IDF groups by entity description, not process lifetime ID.
    attribute_ids = {}
    rarity_ids = np.array([attribute_ids.setdefault((m[1], m[2]), len(attribute_ids))
                           for m in metadata], dtype='int64')
    np.savez_compressed(output / 'prepared.npz', **arrays, x=x, ids=ids, ego=ego, valid=valid,
                        ages=ages, train_end=tr, val_end=va, n_nodes=len(nodes), window=windows,
                        rarity_ids=rarity_ids)
    partitions = {name: dict(count=hi-lo, positives=int(arrays['y'][lo:hi].sum()),
                            windows=len(np.unique(windows[lo:hi])))
                  for name, lo, hi in [('train', 0, tr), ('validation', tr, va), ('test', va, n)]}
    manifest = dict(**signature, counts=dict(counts), partitions=partitions,
        feature_dim=41, relations=RELATIONS, no_subsampling=True, scaler=None,
        split='chronological 60/20/20 event quantiles rounded forward to 15-minute boundary',
        label='class1 external join; class2-only excluded; unmatched treated as background',
        dataset_sha256=sha256(output / 'prepared.npz'),
        adaptation='Sysmon subset; ImageLoad=READ, FileCreate=WRITE, process creation=START, termination=CLOSE self-edge, network connection=SEND/RECEIVE proxy. Unsupported operations excluded.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    (output / 'nodes.json').write_text(json.dumps(metadata))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True)
    p.add_argument('--labels', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--start', type=float)
    p.add_argument('--end', type=float)
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))
