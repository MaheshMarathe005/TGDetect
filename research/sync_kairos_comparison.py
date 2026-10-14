"""Read durable KAIROS comparison status; optionally download and audit results."""
import argparse
import json
from pathlib import Path
import modal

REMOTE = '/results/dedale_kairos_shared41_v4_benign2'
PREPARED = '/prepared/dedale_kairos_shared41_v1'
LOCAL = Path(__file__).resolve().parent / 'artifacts/kairos_dedale_shared41'


def read_json(volume, path):
    return json.loads(b''.join(volume.read_file(path)))


def download(volume, remote, local, size=None):
    if local.exists() and size is not None and local.stat().st_size == size:
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    temp = local.with_name(local.name+'.download')
    with temp.open('wb') as out:
        for block in volume.read_file(remote):
            out.write(block)
    temp.replace(local)


def sync(fetch=False):
    volume = modal.Volume.from_name('tg-detect-dedale-v1')
    state = read_json(volume, REMOTE+'/pipeline_status.json')
    local_state = LOCAL/'pipeline_status.json'
    local_state.parent.mkdir(parents=True, exist_ok=True)
    local_state.write_text(json.dumps(state, indent=2))
    print(json.dumps({k:v for k,v in state.items() if k!='preparation'},indent=2))
    files = list(volume.iterdir(REMOTE, recursive=True))
    for entry in files:
        if entry.path.endswith('_progress.json'):
            row = read_json(volume, '/'+entry.path.lstrip('/'))
            latest = row['history'][-1] if row.get('history') else {}
            print(json.dumps(dict(file=entry.path,latest=latest,elapsed_seconds=row.get('elapsed_seconds'))))
    if not fetch:
        return state
    if state['status']!='complete':
        raise RuntimeError('Experiment has not completed; use status-only mode')
    for entry in files:
        # Preserve epoch-resume files remotely; final models/results suffice locally.
        if entry.type.name!='FILE' or entry.path.endswith('_resume.pt') or '/source/' in entry.path:
            continue
        relative=Path(entry.path).relative_to(REMOTE.lstrip('/'))
        download(volume,'/'+entry.path,LOCAL/'results'/relative,entry.size)
    for name in ('prepared.npz','manifest.json','nodes.json'):
        download(volume,PREPARED+'/'+name,LOCAL/'prepared'/name)
    from kairos_comparison.audit import audit
    checked = audit(LOCAL/'prepared/prepared.npz',LOCAL/'results',expected_seeds=state['seeds'])
    print(json.dumps(checked,indent=2))
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    sync(p.parse_args().download)
