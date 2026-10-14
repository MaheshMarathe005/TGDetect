"""Compare only completed V4 runs with saved, seed-matched AttnShort results."""
import argparse
import json
from pathlib import Path
import statistics


def report(root):
    root=Path(root)
    results=root/'artifacts/sliding_v4'
    rows=[]
    for path in sorted(results.glob('V4_s*_e5.json')):
        new=json.loads(path.read_text())
        if new.get('status')!='complete':continue
        seed=new['signature']['seed']
        oldpath=root/f'artifacts/publication/V3_short_attention_s{seed}_e5.json'
        old=json.loads(oldpath.read_text())
        if old['signature']['prepared_sha256']!=new['signature']['prepared_sha256']:
            raise ValueError('Cannot compare different prepared datasets')
        keys=('epochs','batch','unroll_batches','seed','embedding_lr','pair_weight','pair_score_weight','false_alert_rate','frozen_eval')
        if any(old['signature'][k]!=new['signature'][k] for k in keys):
            raise ValueError('Training/scoring protocol differs from AttnShort')
        for name,r,p in [('V3-AttnShort',old,oldpath),('V4',new,path)]:
            m=r['metrics']
            rows.append(dict(model=name,seed=seed,epochs=5,parameters=r['parameter_count'],
                training_seconds=sum(e['seconds'] for e in r['history']),
                total_seconds=r.get('total_seconds'),peak_cuda_bytes=r.get('peak_cuda_bytes'),
                checkpoint_epoch=r['checkpoint_epoch'],source=str(p.relative_to(root)),**m))
    if not rows:raise ValueError('No completed V4 results; refusing to manufacture a comparison')
    lines=['# V4 versus saved V3-AttnShort','',
           'Same prepared CADETS split and five-epoch protocol; previous models were not retrained.',
           'Only completed, seed-matched runs appear. Test results are retrospective development evidence.','',
           '| Model | Seed | PR-AP | F1 | Precision | Recall | False positives | Train seconds |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['model']} | {r['seed']} | {r['pr_ap']:.6f} | {r['f1']:.6f} | {r['precision']:.6f} | {r['recall']:.6f} | {r['fp']} | {r['training_seconds']:.1f} |")
    lines+=['','Metrics are fractions, not percentages. Higher AP/F1 alone does not establish',
            'deployment readiness; review false positives, recall and runtime together.','',
            'V4 changes encoder width, temporal processing, episodic width and node updater',
            'together. Without ablations, improvements cannot be attributed to one component.']
    (results/'comparison.json').write_text(json.dumps(rows,indent=2))
    (results/'COMPARISON.md').write_text('\n'.join(lines)+'\n')
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]))
    print(json.dumps(report(parser.parse_args().root),indent=2))
