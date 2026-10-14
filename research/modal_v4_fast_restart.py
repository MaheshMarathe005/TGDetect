"""One-GPU V4 execution-only recovery within the existing timeout allowance."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import volume

root=Path(__file__).parent
image=(modal.Image.debian_slim(python_version='3.10')
       .pip_install('torch==2.4.1','torch-geometric==2.5.3','numpy==1.26.4',
                    'scipy==1.10.1','scikit-learn==1.2.0','networkx==2.8.7',
                    'pytest==8.3.5','tqdm==4.67.1')
       .env({'PYTHONPATH':'/root/research','PYTHONHASHSEED':'0'})
       .pip_install('numba==0.61.2'))
PREPARED='/study/v3_vs_orthrus_20261010/prepared_v3_local'
ROOT='/study/v4_20261011_v1'
if modal.is_local():
    for name in ('sliding_node_mem_v2','sliding_node_mem_upgrade','sliding_node_mem',
                 'orthrus_comparison','orthrus_official','sliding_node_mem_v3','sliding_node_mem_v4'):
        image=image.add_local_dir(root/name,'/root/research/'+name,ignore=['__pycache__','.git'])
    for name in ('node_memory_v3.py','node_memory_models.py','tgnn_components.py','temporal_cnn.py',
                 'causal_tgn.py','pomsathit_baseline.py','test_sliding_v3.py','test_sliding_v4.py',
                 'modal_sliding_v2_benchmark.py'):
        image=image.add_local_file(root/name,'/root/research/'+name)
app=modal.App('tg-detect-v4-fast-restart')
common=dict(image=image,volumes={'/study':volume},cpu=4,memory=32768,retries=0,scaledown_window=2)
DEST=ROOT+'/V4_s0_e5_fast'


def verification():
    import itertools,json,os,subprocess,time,torch
    from sliding_node_mem_v3.tokens import TokenBank
    from sliding_node_mem_v3.prepare import read_batches
    from sliding_node_mem_v4.model import TGNNSlidingNodeMem
    from sliding_node_mem_v4.run import make_optimizer,train_unroll
    volume.reload();torch.set_num_threads(4)
    tests=subprocess.run(['python','-m','pytest','-q','/root/research/test_sliding_v4.py'],
        capture_output=True,text=True,env=dict(os.environ,SLIDING_TEST_DEVICE='cuda'))
    print(tests.stdout,tests.stderr,flush=True)
    if tests.returncode:raise RuntimeError('V4 parity tests failed')
    meta=json.loads((Path(PREPARED)/'manifest.json').read_text());bank=TokenBank(Path(PREPARED)/'tokens')
    batches=list(itertools.islice(read_batches(Path(PREPARED)/'train.bin',1024),8))
    result=dict(tests=tests.stdout,scope='Disposable real-prefix training timing, not accuracy',arms={})
    for compiled in ('0','1'):
        os.environ['TG_V4_COMPILED_MEMORY']=compiled;torch.manual_seed(0)
        model=TGNNSlidingNodeMem(meta['n_nodes'],bank,num_relations=meta['relations'],signatures=meta['signatures'],memory_device='array').cuda()
        opt=make_optimizer(model);durations=[]
        for lo in range(0,8,2):
            torch.cuda.synchronize();t=time.perf_counter();train_unroll(model,opt,batches[lo:lo+2]);torch.cuda.synchronize()
            durations.append(time.perf_counter()-t)
        result['arms'][compiled]=durations;print('Compiled',compiled,durations,flush=True)
        del model,opt;torch.cuda.empty_cache()
    import numpy as np
    result['speedup']=float(np.median(result['arms']['0'][1:])/np.median(result['arms']['1'][1:]))
    result['status']='complete';result['gpu']=torch.cuda.get_device_name()
    (Path(ROOT)/'fast_acceptance.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result


@app.function(gpu='L4',timeout=600,max_containers=1,**common)
def verify():
    return verification()


@app.function(gpu='L4',timeout=4800,max_containers=1,**common)
def resume():
    import json,os,shutil,time,torch
    from concurrent.futures import ThreadPoolExecutor
    from sliding_node_mem_v4.run import run
    volume.reload()
    acceptance_path=Path(ROOT)/'fast_acceptance.json'
    accepted=json.loads(acceptance_path.read_text()) if acceptance_path.exists() else verification()
    if accepted['speedup']<1.2:raise RuntimeError('Measured speedup insufficient; refusing costly restart')
    os.environ['TG_V4_COMPILED_MEMORY']='1'
    previous=Path(ROOT)/'V4_s0_e5';durable=Path(DEST);scratch=Path('/tmp/v4_fast_training')
    if durable.exists():raise FileExistsError('Fast restart already exists')
    scratch.mkdir();durable.mkdir()
    saved=torch.load(previous/'resume.pt',map_location='cpu',weights_only=False)
    if saved['signature']['model']['memory_device']!='array':raise ValueError('Expected array-memory checkpoint')
    print('Recovering event cursor',saved['stream']['count'] if saved['stream'] else 'epoch boundary',flush=True)
    del saved
    # Retain the initialization path in the signature; the saved file is reused.
    initialization=previous/'initial_embeddings.pt'
    shutil.copyfile(previous/'resume.pt',scratch/'resume.pt')
    if (previous/'best.pt').exists():shutil.copyfile(previous/'best.pt',scratch/'best.pt')
    shutil.copyfile(initialization,scratch/'initial_embeddings.pt')
    # run() records this initialization's digest, not its pathname.
    (durable/'recovery.json').write_text(json.dumps(dict(parent=str(previous),checkpoint_seconds=900,
        execution_only=True,checkpoint_io='local serialization; asynchronous atomic volume copy'),indent=2));volume.commit()
    pool=ThreadPoolExecutor(max_workers=1);pending=None
    def persist(snapshot):
        started=time.perf_counter()
        for name in ('best.pt','progress.json','execution_migration.json','resume.pt'):
            p=snapshot/name
            if p.exists():
                tmp=durable/(name+'.upload');shutil.copyfile(p,tmp);tmp.replace(durable/name)
        volume.commit();print('Background checkpoint upload seconds',round(time.perf_counter()-started,1),flush=True)
        shutil.rmtree(snapshot)
    def checkpoint():
        nonlocal pending
        if pending is not None:
            if not pending.done():return
            pending.result()
        import tempfile
        snapshot=Path(tempfile.mkdtemp(prefix='v4_checkpoint_',dir='/tmp'))
        for name in ('best.pt','progress.json','execution_migration.json','resume.pt'):
            p=scratch/name
            if p.exists():
                if name.endswith('.pt'):os.link(p,snapshot/name)
                else:shutil.copyfile(p,snapshot/name)
        pending=pool.submit(persist,snapshot)
    started=time.perf_counter()
    try:
        result=run(PREPARED,scratch,epochs=5,batch=1024,unroll_batches=2,device='cuda',seed=0,
            initialization=initialization,frozen_eval=True,model_options={'memory_device':'array'},
            checkpoint_seconds=900,on_checkpoint=checkpoint,resume_execution_changes=True)
        if pending is not None:pending.result()
        for p in scratch.iterdir():
            if p.is_file():shutil.copyfile(p,durable/p.name)
            elif p.name=='source':shutil.copytree(p,durable/p.name,dirs_exist_ok=True)
        result.update(gpu=torch.cuda.get_device_name(),recovery_parent=str(previous),
            total_seconds_this_attempt=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        (durable/'result.json').write_text(json.dumps(result,indent=2));volume.commit();return result
    finally:
        pool.shutdown(wait=True)


@app.local_entrypoint()
def main(action:str='verify'):
    import json,time
    directory=root/'artifacts/sliding_v4';ledger_path=directory/'ledger.json'
    ledger=json.loads(ledger_path.read_text());name='fast_'+action;timeout={'verify':600,'resume':4800}[action]
    if name in ledger['jobs']:raise RuntimeError('Attempt already reserved')
    if sum(j['timeout'] for j in ledger['jobs'].values())+timeout>ledger['gpu_timeout_limit']:
        raise RuntimeError('Existing V4 timeout budget exceeded')
    ledger['jobs'][name]=dict(status='reserved',timeout=timeout,started=time.time());ledger_path.write_text(json.dumps(ledger,indent=2))
    call=(verify if action=='verify' else resume).spawn()
    ledger['jobs'][name].update(status='running',call_id=call.object_id);ledger_path.write_text(json.dumps(ledger,indent=2))
    try:
        result=call.get();(directory/(name+'.json')).write_text(json.dumps(result,indent=2))
        ledger['jobs'][name]['status']='complete'
    except Exception as exc:
        ledger['jobs'][name].update(status='failed_or_timed_out',error=repr(exc));raise
    finally:ledger_path.write_text(json.dumps(ledger,indent=2))
