"""Collect and report the interrupted 20/18/13 epoch checkpoint comparison."""
import argparse
import json
from pathlib import Path
import shutil
import time
import modal


def sync():
    from orthrus_comparison.report import report
    base=Path('artifacts/orthrus_e3_study');root=base/'checkpoint_comparison'
    root.mkdir(parents=True,exist_ok=True)
    volume=modal.Volume.from_name('tg-detect-orthrus-e3-v1')
    paths=['checkpoint_comparison/status.json']
    try:
        paths += [p.path for p in volume.iterdir('/checkpoint_comparison/results',recursive=True)
                  if p.path.endswith('.json')]
    except (FileNotFoundError,modal.exception.NotFoundError):pass
    for path in paths:
        relative=Path(path).relative_to('checkpoint_comparison');dest=root/relative
        if dest.exists() and dest.name not in ['status.json','evaluation_progress.json']:continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        try:
            with dest.with_suffix('.tmp').open('wb') as f:
                for chunk in volume.read_file('/'+path):f.write(chunk)
            dest.with_suffix('.tmp').replace(dest)
        except (FileNotFoundError,modal.exception.NotFoundError):pass
    for progress in (base/'results').glob('*/progress.json'):
        dest=root/'results'/progress.parent.name/'progress.json';dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(progress,dest)
    studies=report(root)
    p=root/'RESULTS.md';text=p.read_text().replace(
        'the final 20-epoch checkpoint','the latest saved checkpoints (ORTHRUS: 20 epochs; V3: 18; V4: 13)')
    text=text.replace('this study selects the final epoch without test labels',
                      'this comparison uses the latest saved checkpoint without selecting by test labels')
    text += '\n## Checkpoint and compute limits\n\nGPU credits ended during training. No further optimization is performed. All three saved models are evaluated on CPU using the same full validation/test graphs, cold split state, reconstruction scoring, threshold rules, and dependency tracing. Training budgets differ; this comparison cannot establish performance at equal epoch counts.\n\n'
    text += '| Model | Epoch | Last training loss | Training minutes |\n|---|---:|---:|---:|\n'
    for progress in sorted((root/'results').glob('*/progress.json')):
        history=json.loads(progress.read_text());last=history[-1]
        text+=f"| {progress.parent.name} | {last['epoch']} | {last['loss']:.6f} | {sum(e['seconds'] for e in history)/60:.1f} |\n"
    p.write_text(text)
    state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
    summary=dict(status=state.get('status','pending'),arms=[dict(arm=s['arm'],status=s['status'],
                    tracing=s.get('tracing',{}).get('status','pending')) for s in studies])
    print(json.dumps(summary),flush=True)
    if state.get('status')=='complete':
        ledger=Path('research/RESULTS_LEDGER.md');marker='\n## Saved ORTHRUS / V3 / V4 checkpoint comparison\n'
        ledger.write_text(ledger.read_text().split(marker)[0]+marker+'\n'+text)
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        state=sync()
        if not args.watch or state.get('status') in ['complete','failed']:break
        time.sleep(60)
