"""Train the new sliding model and unchanged ORTHRUS on fukatkaamsathi."""
from pathlib import Path
import modal

app = modal.App('tg-detect-sliding-node-mem-e3')
volume = modal.Volume.from_name('tg-detect-sliding-node-mem-e3-v1', create_if_missing=True)
root = Path(__file__).parent
image = (modal.Image.debian_slim(python_version='3.10')
         .pip_install('torch==2.4.1', 'torch-geometric==2.5.3', 'numpy==1.26.4',
                      'scipy==1.10.1', 'scikit-learn==1.2.0', 'networkx==2.8.7',
                      'pytest==8.3.5', 'tqdm==4.67.1')
         .env({'PYTHONPATH': '/root/research', 'PYTHONHASHSEED': '0',
               'SLIDING_REFERENCE_PATH':'/root/research/performance_reference_model.py'})
         .add_local_dir(root/'sliding_node_mem', '/root/research/sliding_node_mem', ignore=['__pycache__'])
         .add_local_dir(root/'orthrus_comparison', '/root/research/orthrus_comparison', ignore=['__pycache__'])
         .add_local_dir(root/'orthrus_official', '/root/research/orthrus_official', ignore=['.git', '__pycache__']))
for name in ('node_memory_v3.py', 'node_memory_models.py', 'tgnn_components.py', 'temporal_cnn.py',
             'causal_tgn.py', 'pomsathit_baseline.py', 'test_sliding_node_mem.py', 'conftest.py'):
    image = image.add_local_file(root/name, '/root/research/'+name)
image=image.add_local_file(root/'artifacts/sliding_node_mem_e3/performance_before/model.py',
                           '/root/research/performance_reference_model.py')


@app.function(image=image,cpu=4,memory=4096,timeout=3600,volumes={'/study':volume})
def import_data(manifest, entries):
    """Import user-authorized source data using temporary read-only block URLs."""
    import hashlib,json,time,requests
    from concurrent.futures import ThreadPoolExecutor,as_completed
    volume.reload()
    Path('/study/prepared').mkdir(parents=True,exist_ok=True)
    def fetch(entry):
        dest=Path(entry['path']);dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==entry['sha256']:
            return dest.stat().st_size
        for attempt in range(4):
            try:
                h=hashlib.sha256();tmp=dest.with_suffix(dest.suffix+'.part')
                with tmp.open('wb') as f:
                    for url in entry['urls']:
                        response=requests.get(url,stream=True,timeout=120)
                        if response.status_code!=200:raise RuntimeError(f'Block download HTTP {response.status_code}')
                        for block in response.iter_content(1024*1024):h.update(block);f.write(block)
                if h.hexdigest()!=entry['sha256']:raise RuntimeError('Input checksum mismatch')
                tmp.replace(dest);return dest.stat().st_size
            except Exception:
                if attempt==3:raise RuntimeError('Data transfer failed for '+entry['path']) from None
                time.sleep(attempt+1)
    total=0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for i,future in enumerate(as_completed([pool.submit(fetch,e) for e in entries]),1):
            total+=future.result()
            if i%100==0 or i==len(entries):
                value=dict(stage='copying',files=i,total_files=len(entries),bytes=total)
                Path('/study/transfer.json').write_text(json.dumps(value));volume.commit();print(value,flush=True)
    Path('/study/prepared/manifest.json').write_text(json.dumps(manifest,indent=2))
    Path('/study/transfer.json').write_text(json.dumps(dict(stage='complete',files=len(entries),bytes=total)))
    volume.commit()
    return dict(files=len(entries),bytes=total)


@app.function(image=image, gpu='L4', cpu=8, memory=32768, timeout=1800, volumes={'/study': volume})
def preflight():
    import json, subprocess, sys, time, os
    import torch
    from sliding_node_mem.model import TGNNSlidingNodeMem
    from sliding_node_mem.budget import verify_budget
    from orthrus_comparison.protocol import reconstruction_query
    volume.reload()
    budget = verify_budget()
    Path('/study/parameter_budget.json').write_text(json.dumps(budget,indent=2)); volume.commit()
    check = subprocess.run([sys.executable, '-m', 'pytest', '-q', '/root/research/test_sliding_node_mem.py'],
                           capture_output=True, text=True, env=dict(os.environ, SLIDING_TEST_DEVICE='cuda'))
    print(check.stdout, check.stderr, flush=True)
    if check.returncode: raise RuntimeError('Architecture acceptance tests failed')
    manifest = json.loads(Path('/study/prepared/manifest.json').read_text())
    attrs = json.loads(Path(manifest['node_attributes']).read_text())
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    m = TGNNSlidingNodeMem(manifest['n_nodes']).cuda()
    opt = torch.optim.Adam(m.parameters(), lr=1e-5, weight_decay=1e-5)
    elapsed = []; seen = 0
    for path in manifest['files']['train']:
        g = torch.load(path, map_location='cpu', weights_only=False)
        for lo in range(0, len(g.src), 1024):
            hi = min(lo+1024, len(g.src)); s=g.src[lo:hi].tolist();d=g.dst[lo:hi].tolist()
            queries=[reconstruction_query(attrs.get(str(u),{}),attrs.get(str(v),{})) for u,v in zip(s,d)]
            targets=g.msg[lo:hi,131:141].argmax(-1)
            torch.cuda.synchronize(); started=time.perf_counter()
            opt.zero_grad(set_to_none=True)
            logits,_=m.forward_events(queries,s,d,(g.t[lo:hi].double()/1e9).tolist(),targets)
            loss=torch.nn.functional.cross_entropy(logits,targets.cuda());loss.backward();opt.step();m.detach_state()
            torch.cuda.synchronize();elapsed.append(time.perf_counter()-started);seen+=len(s)
            if len(elapsed)>=8:break
        if len(elapsed)>=8:break
    result=dict(status='passed', synthetic_tests=check.stdout.strip(),device=torch.cuda.get_device_name(),
                training_events=seen,seconds=sum(elapsed),events_per_second=seen/sum(elapsed),
                estimated_epoch_seconds=3526541/(seen/sum(elapsed)),
                parameters=sum(p.numel() for p in m.parameters()),parameter_budget=budget,peak_bytes=torch.cuda.max_memory_allocated(),
                note='Real benign training prefix benchmark; disposable weights, excluded from research results.')
    del m,opt
    torch.cuda.empty_cache()
    from sliding_node_mem.performance import compare
    performance=compare(manifest,attrs)
    Path('/study/performance.json').write_text(json.dumps(performance,indent=2))
    result['implementation']='performance-v4'
    result['performance']={k:v for k,v in performance.items() if k not in ('arms',)}
    Path('/study/preflight.json').write_text(json.dumps(result,indent=2));volume.commit()
    print(json.dumps(result),flush=True)
    return result


@app.function(image=image, gpu='L4', cpu=8, memory=32768, timeout=86400,
              volumes={'/study': volume}, max_containers=2)
def train(arm, epochs=20):
    import json, traceback, torch
    from sliding_node_mem.run import run, ARM
    from orthrus_comparison.run import run as reference
    from sliding_node_mem.budget import verify_budget
    volume.reload()
    try:
        budget=verify_budget()
        failure=Path('/study/results')/arm/'failure.json'
        if failure.exists():failure.rename(failure.with_name('failure_before_recovery.json'))
        completed=Path('/study/results')/arm/'result.json'
        if completed.exists():
            result=json.loads(completed.read_text())
            if result.get('status')=='complete' and result.get('parameter_count')==budget['trainable_parameters'][arm] and result['signature']['epochs']==epochs:
                return {k:v for k,v in result.items() if k!='node_scores'}
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        fn = run if arm == ARM else reference
        if arm not in (ARM, 'ORTHRUS_Reference'):raise ValueError(arm)
        original_set_rng=torch.cuda.set_rng_state_all
        torch.cuda.set_rng_state_all=lambda states:original_set_rng([s.cpu() for s in states])
        try:
            result=fn('/study/prepared/manifest.json','/study/results',epochs=epochs,
                      batch=1024,seed=0,device='cuda',on_checkpoint=volume.commit)
        finally:torch.cuda.set_rng_state_all=original_set_rng
        if result['parameter_count'] != budget['trainable_parameters'][arm]:
            raise ValueError('Trained detector parameter count differs from matched budget')
        result['parameter_budget']=budget
        (Path('/study/results')/arm/'result.json').write_text(json.dumps(result,indent=2))
        return {k:v for k,v in result.items() if k!='node_scores'}
    except Exception:
        dest=Path('/study/results')/arm;dest.mkdir(parents=True,exist_ok=True)
        (dest/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2))
        raise
    finally:volume.commit()


@app.function(image=image, gpu='L4', cpu=8, memory=32768, timeout=86400,
              volumes={'/study': volume}, max_containers=2)
def train_epoch14(arm, epochs=14):
    """Fresh matched 14-epoch training, checkpoint and test evaluation."""
    import json, traceback, torch
    from sliding_node_mem.run import run, ARM
    from orthrus_comparison.run import run as reference
    from sliding_node_mem.budget import verify_budget
    volume.reload()
    out=Path('/study/results_epoch14')/arm;out.mkdir(parents=True,exist_ok=True)
    try:
        budget=verify_budget()
        original_set_rng=torch.cuda.set_rng_state_all
        torch.cuda.set_rng_state_all=lambda states:original_set_rng([s.cpu() for s in states])
        try:
            fn=run if arm==ARM else reference
            if arm not in (ARM,'ORTHRUS_Reference'):raise ValueError(arm)
            result=fn('/study/prepared/manifest.json','/study/results_epoch14',epochs=epochs,
                      batch=1024,seed=0,device='cuda',on_checkpoint=volume.commit)
        finally:torch.cuda.set_rng_state_all=original_set_rng
        if result['parameter_count']!=budget['trainable_parameters'][arm]:
            raise ValueError('Epoch-14 model parameter count differs from matched budget')
        result['parameter_budget']=budget
        (out/'result.json').write_text(json.dumps(result,indent=2));volume.commit()
        return {k:v for k,v in result.items() if k!='node_scores'}
    except Exception:
        (out/'failure.json').write_text(json.dumps(dict(error=traceback.format_exc()),indent=2));volume.commit()
        raise


@app.function(image=image, gpu='L4', cpu=8, memory=32768, timeout=36000,
              volumes={'/study': volume})
def evaluate_saved_sliding():
    """Evaluate the existing sliding checkpoint only; never enters a train loop."""
    import json
    from sliding_node_mem.evaluate_saved import evaluate
    volume.reload()
    checkpoint = Path('/study/results/TGNN_Sliding_Node_Mem/resume.pt')
    output = Path('/study/evaluations_existing/TGNN_Sliding_Node_Mem/result.json')
    result = evaluate('/study/prepared/manifest.json', checkpoint, output, device='cuda')
    volume.commit()
    return result


@app.function(image=image,cpu=.25,memory=512,timeout=86400,volumes={'/study':volume})
def epoch14_pipeline(epochs=14):
    import json,traceback
    status=Path('/study/epoch14_status.json');jobs=Path('/study/epoch14_jobs.json')
    def record(**value):status.write_text(json.dumps(value,indent=2));volume.commit()
    try:
        volume.reload();old=json.loads(status.read_text()) if status.exists() else {}
        if old.get('stage')=='complete':return old['results']
        calls=json.loads(jobs.read_text()) if jobs.exists() else {}
        record(stage='training',epochs=epochs,calls=calls)
        names=['ORTHRUS_Reference','TGNN_Sliding_Node_Mem'];active={}
        results={}
        for arm in names:
            saved_path=Path('/study/results_epoch14')/arm/'result.json'
            saved=json.loads(saved_path.read_text()) if saved_path.exists() else {}
            if saved.get('status')=='complete' and saved.get('checkpoint_epoch')==epochs:
                results[arm]={k:v for k,v in saved.items() if k!='node_scores'};continue
            active[arm]=modal.FunctionCall.from_id(calls[arm]) if arm in calls else train_epoch14.spawn(arm,epochs)
            calls[arm]=active[arm].object_id;jobs.write_text(json.dumps(calls));volume.commit()
        record(stage='training',epochs=epochs,calls=calls,reused_completed=list(results))
        results.update({arm:call.get() for arm,call in active.items()})
        volume.reload();record(stage='complete',epochs=epochs,results=results)
        return results
    except Exception:
        record(stage='failed',error=traceback.format_exc());raise


@app.function(image=image, cpu=8, memory=32768, timeout=86400,
              volumes={'/study':volume}, max_containers=2)
def trace(arm):
    from orthrus_comparison.tracing import run_tracing
    volume.reload()
    try:
        result=run_tracing('/study/prepared/manifest.json',f'/study/results/{arm}/result.json',workers=4)
        return {k:v for k,v in result.items() if k!='traced_nodes'}
    finally:volume.commit()


@app.function(image=image,cpu=.25,memory=512,timeout=86400,volumes={'/study':volume})
def pipeline(epochs=20):
    import json, traceback
    path=Path('/study/status.json')
    def record(**value):
        path.write_text(json.dumps(value,indent=2));volume.commit()
    try:
        volume.reload()
        existing=json.loads(path.read_text()) if path.exists() else {}
        if existing.get('stage')=='complete':return existing['results']
        jobs_path=Path('/study/jobs.json')
        ids=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
        record(stage='preflight',epochs=epochs,calls=ids)
        measured=json.loads(Path('/study/preflight.json').read_text()) if Path('/study/preflight.json').exists() else {}
        if measured.get('implementation')!='performance-v4':measured=preflight.remote()
        arms=['ORTHRUS_Reference','TGNN_Sliding_Node_Mem']
        # Persist call IDs and reuse them after orchestrator preemption. Never
        # spawn duplicate writers against a live training checkpoint.
        calls={};results={}
        for arm in arms:
            completed=Path('/study/results')/arm/'result.json'
            cached=json.loads(completed.read_text()) if completed.exists() else {}
            if cached.get('status')=='complete' and cached.get('parameter_count')==764202 and cached['signature']['epochs']==epochs:
                results[arm]={k:v for k,v in cached.items() if k!='node_scores'}
                continue
            calls[arm]=modal.FunctionCall.from_id(ids[arm]) if arm in ids else train.spawn(arm,epochs)
            ids[arm]=calls[arm].object_id
            jobs_path.write_text(json.dumps(ids));volume.commit()
        record(stage='training',epochs=epochs,calls=ids,reused_completed=list(results),preflight=measured)
        results.update({arm:call.get() for arm,call in calls.items()})
        volume.reload();record(stage='tracing',epochs=epochs,results=results)
        traces=list(trace.map(arms))
        volume.reload();record(stage='complete',epochs=epochs,results=results,tracing=dict(zip(arms,traces)))
        return results
    except Exception:
        record(stage='failed',error=traceback.format_exc());raise


@app.local_entrypoint()
def main(epochs:int=20):
    import json
    call=pipeline.spawn(epochs)
    out=Path('artifacts/sliding_node_mem_e3');out.mkdir(parents=True,exist_ok=True)
    info=dict(profile='fukatkaamsathi',app='tg-detect-sliding-node-mem-e3',
              volume='tg-detect-sliding-node-mem-e3-v1',pipeline_call_id=call.object_id,epochs=epochs)
    (out/'launch.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
