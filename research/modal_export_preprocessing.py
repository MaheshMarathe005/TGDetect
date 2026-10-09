"""Recover the exact missing training scaler and metadata from source telemetry.

Matches already-selected event keys; does not resample or modify the dataset.
Export is accepted only if recovered standardized features match the prepared
matrix within floating-point tolerance.
"""
import modal

app = modal.App('tg-detect-preprocessing-export')
volume = modal.Volume.from_name('tg-detect-dedale-v1')
image = (modal.Image.debian_slim(python_version='3.11').pip_install('numpy==1.26.4')
         .add_local_python_source('dedale_data', 'causal_data'))


@app.function(image=image, cpu=16, memory=32768, timeout=10800, volumes={'/vol': volume})
def recover():
    import hashlib
    import json
    import re
    import zipfile
    from collections import defaultdict
    from datetime import datetime, timezone, timedelta
    from pathlib import Path
    import numpy as np
    from causal_data import fit_scaler
    from dedale_data import features, event_key, node_ids, _iter_events, _host, _epoch

    volume.reload()
    root = Path('/vol/prepared/dedale_host_mem')
    a = np.load(root / 'prepared.npz')
    audit = json.loads((root / 'audit.json').read_text())
    dest = root / 'preprocessing.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        if old.get('prepared_sha256') == audit['prepared_sha256']:
            return old['verification']
    ids = a['ids'].astype(str)
    # Winlogbeat exports can repeat a record key. Match timestamp too, and
    # retain all selected occurrences rather than dropping them in a dict.
    lookup = defaultdict(list)
    for i, key in enumerate(ids):
        lookup[key].append(i)
    raw = np.zeros_like(a['x'])
    found = np.zeros(len(ids), dtype=bool)
    entities, metadata = {}, [None] * len(ids)
    lo, hi = audit['modeling_window_epoch']
    hosts = set(audit['hosts'].split(','))
    date_lo = datetime.fromtimestamp(lo, timezone.utc).date() - timedelta(days=1)
    date_hi = datetime.fromtimestamp(hi, timezone.utc).date() + timedelta(days=1)
    with zipfile.ZipFile('/vol/raw/system_logs_winlogbeat.zip') as z:
        members = []
        for name in z.namelist():
            match = re.search(r'(\d{4}-\d{2}-\d{2})', name)
            if not name.endswith('/') and (not match or date_lo <= datetime.strptime(match[1], '%Y-%m-%d').date() <= date_hi):
                members.append(name)
        for mi, member in enumerate(members):
            for event in _iter_events(z, member, tuple(hosts)):
                if _host(event) not in hosts:
                    continue
                key = '|'.join(map(str, event_key(event)))
                candidates = lookup.get(key)
                if not candidates:
                    continue
                t = _epoch(event.get('@timestamp'))
                matching = [i for i in candidates if not found[i] and t is not None
                            and abs(t - float(a['times'][i])) <= 1e-6]
                if not matching:
                    continue
                s, d = node_ids(event)
                ed = (event.get('winlog') or {}).get('event_data') or {}
                proc = event.get('process') or {}
                event_info = dict(image=ed.get('Image') or proc.get('executable') or '',
                                   parent_image=ed.get('ParentImage') or '',
                                   event_code=str((event.get('event') or {}).get('code') or (event.get('winlog') or {}).get('event_id') or ''),
                                   host=_host(event), user=ed.get('User') or '',
                                   identity_kind='guid' if ed.get('ProcessGuid') else 'pid' if ed.get('ProcessId') or proc.get('pid') else 'provider_fallback')
                for i in matching:
                    raw[i] = features(event)
                    for node, token in [(int(a['src'][i]), s), (int(a['dst'][i]), d)]:
                        if node in entities and entities[node] != token:
                            raise RuntimeError('Inconsistent recovered entity map')
                        entities[node] = token
                    metadata[i] = event_info
                    found[i] = True
            print(f'metadata member {mi+1}/{len(members)} recovered={int(found.sum())}/{len(ids)}', flush=True)
            if found.all():
                break
    if not found.all():
        raise RuntimeError(f'Missing {int((~found).sum())} original events; no scaler exported')
    mean, std = fit_scaler(raw, int(a['train_end']))
    recovered = ((raw - mean) / std).astype(np.float32)
    np.testing.assert_allclose(recovered, a['x'], rtol=2e-5, atol=2e-5)
    export = dict(schema='dedale-sysmon72-v1', prepared_sha256=audit['prepared_sha256'],
                  mean=mean.tolist(), std=std.tolist(), n_nodes=int(a['n_nodes']),
                  entities={str(k): v for k, v in entities.items()},
                  verification=dict(status='passed', events=len(ids),
                                    max_abs_feature_error=float(np.max(np.abs(recovered - a['x'])))),
                  identity_note='Original Guid/PID/provider fallback preserved for checkpoint compatibility')
    (root / 'event_metadata.json').write_text(json.dumps(metadata))
    dest.write_text(json.dumps(export))
    volume.commit()
    return export['verification']


if __name__ == '__main__':
    function = modal.Function.from_name('tg-detect-preprocessing-export', 'recover')
    print(function.spawn().object_id)
