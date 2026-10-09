"""Evaluate and trace all saved checkpoints sequentially on local CPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import importlib.metadata
import platform


def run():
    base=Path('artifacts/orthrus_e3_study').resolve()
    root=base/'local_checkpoint_comparison';root.mkdir(exist_ok=True)
    data=base/'local_evaluation_data'
    env=dict(os.environ,PYTHONPATH=str(Path('research').resolve()))
    arms=['ORTHRUS_Reference','TGNN_Node_Mem_V3_Reconstruction','TGNN_Node_Mem_V4_KV']
    status=root/'status.json'
    def record(**values):
        temp=status.with_suffix('.tmp');temp.write_text(json.dumps(dict(observed_at=time.time(),**values),indent=2));temp.replace(status)
    try:
        record(status='waiting_for_verified_download',device='local CPU')
        manifest=json.loads((base/'prepared/manifest.json').read_text())
        prefix=str(Path(manifest['node_attributes']).parent.parent)+'/'
        needed=manifest['files']['val']+manifest['files']['test']
        needed+=[manifest[k] for k in ['node_attributes','ground_truth_nodes','attack_nodes']]
        # Downloads are renamed only after their checksum passes. The evaluator
        # checks them again before use; tracing graphs can finish in parallel.
        while not all((data/p.removeprefix(prefix)).is_file() for p in needed):time.sleep(10)
        (root/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),
            packages={name:importlib.metadata.version(name) for name in
                      ['torch','torch-geometric','numpy','scipy','scikit-learn','networkx']}),indent=2))
        for arm in arms:
            out=root/'results'/arm;out.mkdir(parents=True,exist_ok=True)
            shutil.copy2(base/'results'/arm/'progress.json',out/'progress.json')
            record(status='evaluating',arm=arm,device='local CPU')
            with (out/'evaluation.log').open('w') as log:
                subprocess.run([sys.executable,'-u','research/evaluate_orthrus_checkpoints.py',
                                '--manifest',str(base/'prepared/manifest.json'),
                                '--checkpoint',str(base/'results'/arm/'resume.pt'),
                                '--output',str(root/'results'),'--data-root',str(data)],
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            record(status='tracing',arm=arm,device='local CPU')
            while not (data/'complete.json').exists():time.sleep(10)
            with (out/'tracing.log').open('w') as log:
                subprocess.run([sys.executable,'-u','-c',
                                'import sys; from orthrus_comparison.tracing import run_tracing; run_tracing(sys.argv[1],sys.argv[2],workers=1)',
                                str(out/'evaluation_manifest.json'),str(out/'result.json')],
                               env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            report(root)
        record(status='complete',device='local CPU',further_training=False)
        report(root)
        ledger=Path('research/RESULTS_LEDGER.md');marker='\n## Local CPU evaluation of saved 20/18/13 epoch checkpoints\n'
        ledger.write_text(ledger.read_text().split(marker)[0]+marker+'\n'+(root/'RESULTS.md').read_text())
        print((root/'comparison.csv').read_text(),flush=True)
    except Exception as error:
        record(status='failed',error=str(error));raise


def report(root):
    from orthrus_comparison.report import report as build
    studies=build(root)
    for study in studies:
        source=root/'results'/study['arm']/'result.json'
        if source.exists():
            result=json.loads(source.read_text())
            study.update({key:result[key] for key in
                          ['checkpoint_epoch','checkpoint_sha256','evaluation_device','torch_version']})
    (root/'comparison.json').write_text(json.dumps(studies,indent=2))
    path=root/'RESULTS.md';text=path.read_text().replace('the final 20-epoch checkpoint',
        'the saved checkpoints (ORTHRUS: 20 epochs; V3: 18; V4: 13)')
    text=text.replace('this study selects the final epoch without test labels',
                      'this comparison selects the latest saved checkpoints without test labels')
    text+='\n## Evaluation conditions\n\nGPU credits ended during training. These are local CPU evaluations of the unchanged saved weights, with no further training. All arms share the full test and validation data, cold state, scoring rules, and tracing algorithm. Epoch counts and training times differ, so this is not an equal-training-budget comparison. PR-AP and ROC-AUC on tracing rows retain the detector ranking; tracing changes binary node predictions.\n\n'
    text+='| Model | Epoch | Last training loss | Training minutes | CPU evaluation minutes |\n|---|---:|---:|---:|---:|\n'
    for p in sorted((root/'results').glob('*/progress.json')):
        history=json.loads(p.read_text());last=history[-1];result=p.parent/'result.json'
        seconds=json.loads(result.read_text())['evaluation_seconds']/60 if result.exists() else None
        text+=f"| {p.parent.name} | {last['epoch']} | {last['loss']:.6f} | {sum(e['seconds'] for e in history)/60:.1f} | {seconds:.1f} |\n" if seconds is not None else f"| {p.parent.name} | {last['epoch']} | {last['loss']:.6f} | {sum(e['seconds'] for e in history)/60:.1f} | pending |\n"
    path.write_text(text)


if __name__=='__main__':run()
