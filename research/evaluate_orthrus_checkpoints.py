"""Evaluate saved CADETS weights on CPU without changing frozen training code.

The AST adapter removes only the epoch loop and adds progress reporting. The
runner's original checkpoint/source signature checks remain enabled. Relocation
changes runtime paths, not the saved manifest or data bytes. Only evaluation
inputs need to be downloaded locally, and every used file is checksum checked.
"""
import argparse
import ast
import json
import shutil
import time
from pathlib import Path
import torch


def relocate(manifest, data_root):
    original_root=str(Path(manifest['node_attributes']).parent.parent)
    def convert(value):
        if isinstance(value,str) and value.startswith(original_root+'/'):
            return str(Path(data_root)/value[len(original_root)+1:])
        if isinstance(value,list):return [convert(v) for v in value]
        if isinstance(value,dict):return {convert(k):convert(v) for k,v in value.items()}
        return value
    result=convert(manifest)
    required=set(result['files']['val']+result['files']['test'])
    required.update(result['graph_files']['test'])
    required.update(result[k] for k in ['node_attributes','ground_truth_nodes','attack_nodes'])
    result['file_sha256']={p:h for p,h in result['file_sha256'].items() if p in required}
    return result


def evaluate(manifest_path, checkpoint, output, data_root=None):
    from orthrus_comparison import run as runner
    manifest_path=Path(manifest_path);checkpoint=Path(checkpoint);output=Path(output)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    signature=state['signature'];epoch=state['epoch'];arm=signature['arm']
    if not 1<=epoch<=signature['epochs']:raise ValueError('Invalid saved epoch')
    if runner.sha256(manifest_path)!=signature['manifest_sha256']:
        raise ValueError('Original manifest checksum differs from training')
    dest=output/arm;dest.mkdir(parents=True,exist_ok=True)
    journal=dest/'resume.pt'
    if checkpoint.resolve()!=journal.resolve():shutil.copy2(checkpoint,journal)
    original_manifest=json.loads(manifest_path.read_text())
    mapped=relocate(original_manifest,data_root) if data_root else original_manifest
    used=set(mapped['files']['val']+mapped['files']['test'])
    used.update(mapped[k] for k in ['node_attributes','ground_truth_nodes','attack_nodes'])
    mapped=dict(mapped,file_sha256={p:h for p,h in mapped['file_sha256'].items() if p in used})
    relocated_path=dest/'evaluation_manifest.json'
    relocated_path.write_text(json.dumps(mapped,indent=2))

    class JSONAdapter:
        dumps=staticmethod(json.dumps)
        @staticmethod
        def loads(value,*args,**kwargs):
            loaded=json.loads(value,*args,**kwargs)
            return mapped if isinstance(loaded,dict) and loaded.get('dataset')=='CADETS_E3' else loaded

    namespace=dict(vars(runner));namespace['json']=JSONAdapter
    last_progress=0
    def progress(split,count):
        nonlocal last_progress
        now=time.time()
        if now-last_progress>30:
            last_progress=now
            info=dict(arm=arm,epoch=epoch,split=split,edges_scored=count,observed_at=now)
            temp=dest/'evaluation_progress.tmp';temp.write_text(json.dumps(info))
            temp.replace(dest/'evaluation_progress.json');print(json.dumps(info),flush=True)
    namespace['_evaluation_progress']=progress
    tree=ast.parse(Path(runner.__file__).read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    loops=[n for n in ast.walk(function) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='epoch']
    if len(loops)!=1 or ast.unparse(loops[0].iter)!='range(first, epochs)':
        raise ValueError('Frozen training loop changed; inspect adapter')
    loops[0].iter=ast.parse('range(first, first)',mode='eval').body
    # Preserve stream computation and instrument only completed batches.
    for node in ast.walk(function):
        if isinstance(node,ast.FunctionDef) and node.name=='stream':
            for batch_loop in ast.walk(node):
                if isinstance(batch_loop,ast.For) and isinstance(batch_loop.target,ast.Name) and batch_loop.target.id=='lo':
                    batch_loop.body.append(ast.parse('_evaluation_progress(split, count)').body[0])
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(module,str(Path(__file__).resolve()),'exec'),namespace)
    def forbidden_step(*args,**kwargs):raise RuntimeError('Evaluation must not optimize weights')
    torch.optim.Adam.step=forbidden_step
    result=namespace['run'](str(manifest_path),str(output),epochs=signature['epochs'],
                            batch=signature['batch'],device='cpu',arm=arm,seed=signature['seed'])
    result.update(checkpoint_epoch=epoch,checkpoint_selection=f'latest saved epoch {epoch}; GPU credits exhausted',
                  evaluation_device='cpu',checkpoint_sha256=runner.sha256(checkpoint),
                  evaluation_adapter_sha256=runner.sha256(__file__),
                  evaluation_manifest_sha256=runner.sha256(relocated_path),
                  torch_version=torch.__version__,
                  evaluation_note='No further training. Frozen source and original manifest signatures verified; all used evaluation inputs checksum verified.')
    (dest/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['node_scores','signature']},indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--data-root')
    args=parser.parse_args();evaluate(args.manifest,args.checkpoint,args.output,args.data_root)
