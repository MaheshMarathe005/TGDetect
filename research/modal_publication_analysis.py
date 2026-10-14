"""Read completed cloud outputs and score diagnostics without a GPU."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import image as base_image,volume
root=Path(__file__).parent
image=base_image
if modal.is_local():
    for name in ('publication','sliding_node_mem_v3'):
        image=image.add_local_dir(root/name,'/root/research/'+name,ignore=['__pycache__','sources','figures'])
    image=image.add_local_file(root/'modal_sliding_v2_benchmark.py','/root/research/modal_sliding_v2_benchmark.py')
app=modal.App('tg-detect-study-cpu-analysis')


@app.function(image=image,cpu=4,memory=16384,timeout=1800,volumes={'/study':volume},max_containers=4)
def inspect_run(path):
    import json
    from publication.analyze import analyze
    volume.reload();directory=Path(path)
    manifest=json.loads(Path('/study/prepared/manifest.json').read_text())
    truth=set(map(int,json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    result=json.loads((directory/'result.json').read_text())
    diagnostic=analyze(directory,truth)
    diagnostic['metric_audit']={key:dict(saved=result['metrics'][key],recomputed=diagnostic['primary'][key],
        matches=abs(diagnostic['primary'][key]-result['metrics'][key])<=1e-9)
        for key in ('tp','fp','tn','fn','pr_ap','f1')}
    return dict(result=result,diagnostic=diagnostic)


@app.local_entrypoint()
def main():
    import json
    from concurrent.futures import ThreadPoolExecutor,as_completed
    output=root/'artifacts/publication';output.mkdir(exist_ok=True)
    paths={a+'_s0_e5':'/study/v3_vs_orthrus_20261010/'+a for a in ('ORTHRUS','V3')}
    for ledger in output.glob('budget*.json'):
        for name,value in json.loads(ledger.read_text())['jobs'].items():
            if value.get('status')=='complete' and value.get('epochs',0)>0:
                paths[name]='/study/publication_20261010_v1/'+name
    def one(name,path):
        target=output/(name+'_diagnostic.json')
        if target.exists():return name+' already analyzed'
        result=inspect_run.remote(path)
        target.write_text(json.dumps(result['diagnostic'],indent=2))
        # Historical result records retain their original location and metadata.
        return name+' analyzed'
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in as_completed([pool.submit(one,k,v) for k,v in paths.items()]):print(f.result(),flush=True)
