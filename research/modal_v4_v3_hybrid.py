"""One detached, bounded L4 run: V4 temporal/episodic model + V3 encoder/state."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import volume

app=modal.App('tg-detect-v4-v3-encoder-memory')
root=Path(__file__).parent
PREPARED='/study/v3_vs_orthrus_20261010/prepared_v3_local'
image=(modal.Image.debian_slim(python_version='3.10')
       .pip_install('torch==2.4.1','torch-geometric==2.5.3','numpy==1.26.4',
                    'scipy==1.10.1','scikit-learn==1.2.0','networkx==2.8.7',
                    'pytest==8.3.5','tqdm==4.67.1')
       .env({'PYTHONPATH':'/root/research','PYTHONHASHSEED':'0'})
       .pip_install('numba==0.61.2'))
if modal.is_local():
    for name in ('sliding_node_mem_v2','sliding_node_mem_upgrade','sliding_node_mem',
                 'orthrus_comparison','orthrus_official','sliding_node_mem_v3','sliding_node_mem_v4'):
        image=image.add_local_dir(root/name,'/root/research/'+name,ignore=['__pycache__','.git'])
    for name in ('node_memory_v3.py','node_memory_models.py','tgnn_components.py','temporal_cnn.py',
                 'causal_tgn.py','pomsathit_baseline.py','test_sliding_v3.py','test_sliding_v4.py',
                 'modal_sliding_v2_benchmark.py'):
        image=image.add_local_file(root/name,'/root/research/'+name)
OUTPUT='/study/v4_v3_hybrid_20261011_v1/V4_V3EncoderMemory_s0_e5'
INITIALIZATION='/study/v3_vs_orthrus_20261010/V3/initial_embeddings.pt'
TIMEOUT=3600
JOB='V4_V3EncoderMemory_s0_e5'


@app.function(image=image,gpu='L4',cpu=4,memory=32768,timeout=TIMEOUT,
              max_containers=1,retries=0,scaledown_window=2,volumes={'/study':volume})
def train():
    import itertools,json,os,subprocess,time,torch
    from orthrus_comparison.run import sha256
    from sliding_node_mem_v3.tokens import TokenBank
    from sliding_node_mem_v3.prepare import read_batches
    from sliding_node_mem_v4.model import TGNNSlidingNodeMem
    from sliding_node_mem_v4.run import run,make_optimizer,train_unroll
    volume.reload();started=time.perf_counter()
    output=Path(OUTPUT)
    if output.exists():raise FileExistsError('Hybrid output already exists; refusing duplicate training')
    if sha256(INITIALIZATION)!='6e60820a5b91a6a5dc11bcb64228bcf92d442cea04137e10b0ecb97379817970':
        raise ValueError('Expected the original V3 pretrained initialization')
    tests=subprocess.run(['python','-m','pytest','-q','/root/research/test_sliding_v4.py'],
        capture_output=True,text=True,env=dict(os.environ,SLIDING_TEST_DEVICE='cuda'))
    print(tests.stdout,tests.stderr,flush=True)
    if tests.returncode:raise RuntimeError('CUDA acceptance failed; training not started')
    torch.set_num_threads(4);torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    meta=json.loads((Path(PREPARED)/'manifest.json').read_text())
    options=dict(encoder_dim=64,node_memory='affine-v3',memory_device='model')
    model=TGNNSlidingNodeMem(meta['n_nodes'],TokenBank(Path(PREPARED)/'tokens'),
        num_relations=meta['relations'],signatures=meta['signatures'],**options).cuda()
    optimizer=make_optimizer(model)
    batches=list(itertools.islice(read_batches(Path(PREPARED)/'train.bin',1024),8))
    durations=[]
    for lo in range(0,8,2):
        torch.cuda.synchronize();t=time.perf_counter()
        loss,n=train_unroll(model,optimizer,batches[lo:lo+2])
        torch.cuda.synchronize();durations.append(time.perf_counter()-t)
        print('Hybrid smoke step',lo//2+1,'loss',float(loss),'seconds',durations[-1],flush=True)
    import statistics
    rate=2048/statistics.median(durations[1:])
    estimate=5*meta['events']['train']/rate
    output.parent.mkdir(parents=True,exist_ok=True)
    acceptance=dict(tests=tests.stdout,step_seconds=durations,events_per_second=rate,
        estimated_five_epoch_training_seconds=estimate,
        parameter_count=sum(p.numel() for p in model.parameters()),model=model.config,
        scope='Disposable prefix correctness/runtime planning, not final accuracy')
    (output.parent/'acceptance.json').write_text(json.dumps(acceptance,indent=2));volume.commit()
    print('Hybrid estimated five-epoch training seconds',estimate,flush=True)
    del model,optimizer,batches;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    # Keep a substantial margin for mature-state execution and final evaluation.
    if 1.3*estimate+300>TIMEOUT-(time.perf_counter()-started):
        raise RuntimeError('Prefix estimate exceeds the bounded run allowance; refusing partial costly training')
    try:
        result=run(PREPARED,output,epochs=5,batch=1024,unroll_batches=2,device='cuda',seed=0,
            initialization=INITIALIZATION,frozen_eval=True,prefetch_batches=True,
            model_options=options,checkpoint_seconds=900,on_checkpoint=volume.commit)
        result.update(gpu=torch.cuda.get_device_name(),total_seconds=time.perf_counter()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            job=dict(arm=JOB,timeout_seconds=TIMEOUT,initialization='Original V3 benign-only 64D pretrained table',
                     restart='Fresh weights/optimizer/state; not a V4 checkpoint resume'))
        (output/'result.json').write_text(json.dumps(result,indent=2));volume.commit();return result
    except Exception as exc:
        output.mkdir(parents=True,exist_ok=True)
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-started)))
        volume.commit();raise


@app.local_entrypoint()
def main(attempt:int=1):
    import json,math,time
    from datetime import datetime
    directory=root/'artifacts/sliding_v4'
    path=directory/'ledger.json';ledger=json.loads(path.read_text())
    job=JOB if attempt==1 else JOB+f'_attempt{attempt}'
    if job in ledger['jobs']:raise RuntimeError('Hybrid attempt already reserved; refusing duplicate')
    if any(j.get('status') in ('running','reserved') for k,j in ledger['jobs'].items() if k.startswith(JOB)):
        raise RuntimeError('Another hybrid attempt remains active')
    # Verified Modal app listing: old interrupted app created 12:23:37 and
    # stopped 13:14:13 IST. Retain original reservation and settle conservatively.
    old=ledger['jobs']['V4_s0_e5']
    if old['status']!='interrupted_client_disconnect':raise RuntimeError('Cannot release an active reservation')
    created=datetime.fromisoformat('2026-10-11T12:23:37+05:30').timestamp()
    stopped=datetime.fromisoformat(old['app_stopped_at']).timestamp()
    bound=math.ceil(stopped-created)+120
    if bound<0 or bound>old['timeout']:raise ValueError('Invalid settlement interval')
    old['settled_upper_bound_seconds']=bound
    old['settlement_evidence']=dict(app_id='ap-FuC1m552GonD2ETETYdhBD',
        created_at='2026-10-11T12:23:37+05:30',stopped_at=old['app_stopped_at'],
        teardown_margin_seconds=120,original_timeout_preserved=True)
    reserved=sum(j.get('settled_upper_bound_seconds',j['timeout']) for j in ledger['jobs'].values())
    if reserved+TIMEOUT>ledger['gpu_timeout_limit']:raise RuntimeError('Existing GPU-hour allowance exceeded')
    ledger['jobs'][job]=dict(status='reserved',timeout=TIMEOUT,started=time.time())
    path.write_text(json.dumps(ledger,indent=2))
    print('Accounted upper-bound GPU seconds including new run',reserved+TIMEOUT,
          'of',ledger['gpu_timeout_limit'],flush=True)
    try:
        call=train.spawn()
        ledger['jobs'][job].update(status='running',call_id=call.object_id)
        path.write_text(json.dumps(ledger,indent=2))
        result=call.get()
        (directory/(JOB+'.json')).write_text(json.dumps(result,indent=2))
        ledger['jobs'][job]['status']='complete'
    except Exception as exc:
        ledger['jobs'][job].update(status='failed_or_timed_out',error=repr(exc));raise
    finally:path.write_text(json.dumps(ledger,indent=2))
