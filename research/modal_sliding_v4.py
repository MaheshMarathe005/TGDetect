"""V4-only training on the exact saved AttnShort corpus; bounded, unique jobs."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import image as base_image, volume

root = Path(__file__).parent
image = base_image
if modal.is_local():
    for name in ('sliding_node_mem_v3', 'sliding_node_mem_v4'):
        image = image.add_local_dir(root/name, '/root/research/'+name, ignore=['__pycache__'])
    for name in ('test_sliding_v3.py','test_sliding_v4.py','modal_sliding_v2_benchmark.py'):
        image = image.add_local_file(root/name, '/root/research/'+name)
app = modal.App('tg-detect-v4-attnshort-matched')
PREPARED = '/study/v3_vs_orthrus_20261010/prepared_v3_local'
ROOT = '/study/v4_20261011_v1'
COMMON = dict(image=image, volumes={'/study': volume}, cpu=8, memory=32768,
              retries=0, scaledown_window=2)


@app.function(timeout=600, **COMMON)
def inspect():
    import json
    from orthrus_comparison.run import sha256
    volume.reload()
    manifest = Path(PREPARED)/'manifest.json'
    original = json.loads(Path('/study/prepared/manifest.json').read_text())
    return dict(prepared=json.loads(manifest.read_text()), prepared_sha256=sha256(manifest),
        original_manifest=original, existing_v4=Path(ROOT).exists(),
        volume_roots=[str(p) for p in Path('/study').iterdir()])


@app.function(gpu='L4', timeout=900, **COMMON)
def acceptance(memory_device:str='model'):
    import itertools, json, os, subprocess, time
    import torch
    from sliding_node_mem_v3.tokens import TokenBank
    from sliding_node_mem_v3.prepare import read_batches
    from sliding_node_mem_v4.model import TGNNSlidingNodeMem
    from sliding_node_mem_v4.run import train_unroll, make_optimizer
    volume.reload()
    tests = subprocess.run(['python','-m','pytest','-q','/root/research/test_sliding_v4.py'],
        capture_output=True,text=True,env=dict(os.environ,SLIDING_TEST_DEVICE='cuda'))
    print(tests.stdout, tests.stderr, flush=True)
    if tests.returncode: raise RuntimeError('V4 GPU acceptance failed')
    torch.set_num_threads(1 if memory_device=='cpu' else 4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    meta=json.loads((Path(PREPARED)/'manifest.json').read_text())
    bank=TokenBank(Path(PREPARED)/'tokens')
    model=TGNNSlidingNodeMem(meta['n_nodes'],bank,num_relations=meta['relations'],signatures=meta['signatures'],memory_device=memory_device).cuda()
    optimizer=make_optimizer(model)
    batches=list(itertools.islice(read_batches(Path(PREPARED)/'train.bin',1024),8))
    durations=[];waves=[]
    for lo in range(0,len(batches),2):
        torch.cuda.synchronize();started=time.perf_counter()
        loss,n=train_unroll(model,optimizer,batches[lo:lo+2])
        torch.cuda.synchronize();durations.append(time.perf_counter()-started)
        print('Smoke',lo,'loss',float(loss),'seconds',durations[-1],flush=True)
    rate=sum(len(b['src']) for b in batches[4:])/sum(durations[2:])
    result=dict(status='complete',tests=tests.stdout,parameter_count=sum(p.numel() for p in model.parameters()),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,model=model.config,
        optimizer_steps_seconds=durations,training_events_per_second=rate,
        estimated_five_epoch_train_seconds=5*meta['events']['train']/rate,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        note='Disposable random-weight prefix used only for correctness and runtime planning; not an accuracy result.')
    out=Path(ROOT);out.mkdir(parents=True,exist_ok=True)
    (out/'acceptance.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result


@app.function(gpu='L4', timeout=18000, max_containers=1, **COMMON)
def train(seed:int=0):
    import json, time, torch
    from sliding_node_mem_v4.run import run
    volume.reload()
    acceptance_path=Path(ROOT)/'acceptance.json'
    if not acceptance_path.exists():raise RuntimeError('GPU acceptance must pass before training')
    accepted=json.loads(acceptance_path.read_text())
    options={'memory_device':accepted['model']['memory_device']}
    output=Path(ROOT)/f'V4_s{seed}_e5'
    if output.exists() and (output/'result.json').exists():raise FileExistsError('Job already complete: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    (output/'job.json').write_text(json.dumps(dict(arm='V4',seed=seed,epochs=5,timeout=18000)))
    volume.commit();started=time.perf_counter()
    try:
        result=run(PREPARED,output,epochs=5,batch=1024,unroll_batches=2,device='cuda',seed=seed,
                   frozen_eval=True,model_options=options,on_checkpoint=volume.commit)
        result.update(job=dict(arm='V4',seed=seed,epochs=5),gpu=torch.cuda.get_device_name(),
                      total_seconds=time.perf_counter()-started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        (output/'result.json').write_text(json.dumps(result,indent=2));volume.commit()
        return result
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-started)))
        volume.commit();raise


@app.local_entrypoint()
def main(action:str='inspect',seed:int=0,attempt:int=1,memory_device:str='model'):
    import json,time
    destination=root/'artifacts/sliding_v4';destination.mkdir(parents=True,exist_ok=True)
    if action=='inspect':
        result=inspect.remote();name='inspection'
    elif action in ('acceptance','train'):
        if seed not in (0,1,2):raise ValueError('Only baseline seeds 0,1,2 are registered')
        ledger_path=destination/'ledger.json'
        ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'gpu_timeout_limit':28800,'jobs':{}}
        name=('acceptance' if attempt==1 else f'acceptance_attempt{attempt}') if action=='acceptance' else f'V4_s{seed}_e5'
        if name in ledger['jobs']:raise RuntimeError('Job already reserved: '+name)
        timeout=900 if action=='acceptance' else 18000
        if sum(x['timeout'] for x in ledger['jobs'].values())+timeout>ledger['gpu_timeout_limit']:
            raise RuntimeError('V4 timeout budget exhausted')
        ledger['jobs'][name]=dict(status='reserved',timeout=timeout,started=time.time())
        ledger_path.write_text(json.dumps(ledger,indent=2))
        call=acceptance.spawn(memory_device) if action=='acceptance' else train.spawn(seed)
        ledger['jobs'][name].update(status='running',call_id=call.object_id)
        ledger_path.write_text(json.dumps(ledger,indent=2))
        try:
            result=call.get()
        except Exception as exc:
            ledger['jobs'][name].update(status='failed_or_timed_out',error=repr(exc))
            ledger_path.write_text(json.dumps(ledger,indent=2));raise
        ledger['jobs'][name].update(status='complete')
        ledger_path.write_text(json.dumps(ledger,indent=2))
    else:raise ValueError('action: inspect, acceptance, train')
    (destination/(name+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('original_manifest','predicted_nodes','signature')},indent=2))
