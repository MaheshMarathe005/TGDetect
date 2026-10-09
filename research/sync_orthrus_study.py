"""Collect committed full CADETS study outputs and rebuild the result table."""
import argparse
import json
import time
import traceback
from pathlib import Path
import modal


def sync(output='artifacts/orthrus_e3_study',include_edges=False,include_checkpoints=True):
    from orthrus_comparison.report import report
    volume=modal.Volume.from_name('tg-detect-orthrus-e3-v1')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    paths=['/status.json','/acquisition.json','/restore.log','/environment.json','/prepared/manifest.json',
           '/prepared/ground_truth_nodes.json','/prepared/attack_nodes.json','/prepared/model_config.yml',
           '/raw/dataset_source.json']
    try:
        for item in volume.iterdir('/results',recursive=True):
            name='/'+item.path.lstrip('/')
            if name.endswith('.json') or (include_checkpoints and name.endswith('.pt')) or (include_edges and name.endswith('.jsonl')):
                paths.append(name)
    except (FileNotFoundError,modal.exception.NotFoundError):pass
    for path in paths:
        dest=root/path.lstrip('/')
        # Prepared inputs and final outputs are immutable. Resume checkpoints,
        # status, and epoch progress can change and must be refreshed.
        if dest.exists() and path!='/status.json' and not path.endswith(('/progress.json','/resume.pt')):
            continue
        try:
            chunks=volume.read_file(path)
            dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_suffix(dest.suffix+'.tmp')
            with temporary.open('wb') as f:
                for chunk in chunks:f.write(chunk)
            temporary.replace(dest)
        except (FileNotFoundError,modal.exception.NotFoundError):
            if dest.with_suffix(dest.suffix+'.tmp').exists():dest.with_suffix(dest.suffix+'.tmp').unlink()
    report(root)
    progress=[]
    for path in sorted((root/'results').glob('*/progress.json')):
        epochs=json.loads(path.read_text())
        if epochs:
            last=epochs[-1]
            progress.append(f"| {path.parent.name} | {last['epoch']} | {last['loss']:.6f} | {last['seconds']:.1f} |")
    if progress:
        with (root/'RESULTS.md').open('a') as handle:
            handle.write('\n## Training progress\n\nTraining losses are diagnostics, not detection metrics.\n\n'
                         '| Model | Completed epoch | Mean relation loss | Epoch seconds |\n'
                         '|---|---:|---:|---:|\n'+'\n'.join(progress)+'\n')
    state=root/'status.json'
    if state.exists():
        value=json.loads(state.read_text())
        print(json.dumps(value,indent=2)[:4000],flush=True)
        return value
    print('Acquisition status collected; pipeline status not available yet.',flush=True)
    return {}


def watch(output,interval):
    """Collect lightweight progress repeatedly, then archive all final outputs."""
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    previous=None
    while True:
        try:
            state=sync(output,include_checkpoints=False)
            epochs={p.parent.name:json.loads(p.read_text())[-1]
                    for p in (root/'results').glob('*/progress.json') if json.loads(p.read_text())}
            observation=dict(status=state.get('status'),epochs=epochs)
            if observation!=previous:
                with (root/'run_events.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(observed_at=time.time(),**observation))+'\n')
                print(json.dumps(observation),flush=True);previous=observation
            if state.get('status')=='complete':
                sync(output,include_edges=True,include_checkpoints=True)
                summary=(root/'RESULTS.md').read_text()
                ledger=Path(__file__).parent/'RESULTS_LEDGER.md'
                marker='\n## Completed full ORTHRUS E3 comparison\n'
                existing=ledger.read_text() if ledger.exists() else '# Results ledger\n'
                ledger.write_text(existing.split(marker)[0]+marker+'\n'+summary)
                return
            if state.get('status')=='failed':return
        except Exception:
            print(traceback.format_exc(),flush=True)
        time.sleep(interval)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='artifacts/orthrus_e3_study');p.add_argument('--include-edges',action='store_true')
    p.add_argument('--skip-checkpoints',action='store_true')
    p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=float,default=120)
    a=p.parse_args()
    if a.interval<10:p.error('--interval must be at least 10 seconds')
    if a.watch:watch(a.output,a.interval)
    else:sync(a.output,a.include_edges,not a.skip_checkpoints)
