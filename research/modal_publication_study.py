"""Explicit timeout reservations for the authorized eight-GPU-hour study."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import image as base_image,volume

root=Path(__file__).parent
image=base_image
if modal.is_local():
    for name in ('sliding_node_mem_v3','publication','kairos_comparison'):
        image=image.add_local_dir(root/name,'/root/research/'+name,
                                  ignore=['__pycache__','sources','figures'])
    image=image.add_local_file(root/'modal_sliding_v2_benchmark.py','/root/research/modal_sliding_v2_benchmark.py')
app=modal.App('tg-detect-publication-eight-hour')
ROOT='/study/publication_20261010_v1'


@app.function(image=image,cpu=8,memory=32768,timeout=7200,volumes={'/study':volume})
def prepare_kairos():
    import shutil
    from publication.data import prepare
    volume.reload();target=Path(ROOT)/'prepared_kairos'
    if (target/'manifest.json').exists():return str(target)
    scratch=Path('/tmp/prepared_kairos')
    prepare('/study/prepared/manifest.json',scratch)
    shutil.copytree(scratch,target,dirs_exist_ok=True);volume.commit();return str(target)


def execute(job):
    from publication.run import run
    volume.reload()
    return run(job,ROOT,on_checkpoint=volume.commit)


common=dict(image=image,gpu='L4',cpu=8,memory=32768,volumes={'/study':volume},
            max_containers=4,scaledown_window=2,retries=0)
train900=app.function(timeout=900,**common)(execute)
# Distinct Python functions are needed for distinct registered Modal endpoints.
def execute1500(job):return execute(job)
def execute1800(job):return execute(job)
def execute5400(job):return execute(job)
train1500=app.function(timeout=1500,**common)(execute1500)
train1800=app.function(timeout=1800,**common)(execute1800)
train5400=app.function(timeout=5400,**common)(execute5400)


@app.function(timeout=600,**common)
def acceptance(job):
    import time,torch
    from types import SimpleNamespace
    from publication.models import GraphSAGEReference
    from kairos_comparison.model import Kairos
    started=time.perf_counter();torch.set_num_threads(4)
    s=torch.tensor([0,1,0,2],device='cuda');d=torch.tensor([1,2,2,3],device='cuda')
    t=torch.arange(4,device='cuda');one=torch.nn.functional.one_hot(t%10,10).float()
    m=GraphSAGEReference(8,device='cuda')
    y=m(s,d,t.double(),torch.randn(4,131,device='cuda'),torch.randn(4,131,device='cuda'),
        one,SimpleNamespace(edge_type=one.cpu()),inference=True)
    y.mean().backward();assert torch.isfinite(y).all()
    k=Kairos(8,msg_dim=42,relations=10).cuda();msg=torch.randn(4,42,device='cuda')
    y=k.score(s,d,t,msg);k.observe(s,d,t,msg);y.square().mean().backward();k.detach_state()
    assert torch.isfinite(k.score(s,d,t.repeat(2),msg.repeat(2,1))).all()
    torch.cuda.synchronize()
    return dict(status='complete',checks=['GraphSAGE CUDA forward/backward','KAIROS CUDA history and backward'],
                total_seconds=time.perf_counter()-started,gpu=torch.cuda.get_device_name())


@app.function(timeout=600,**common)
def trained_latency(job):
    import time,json
    from publication.latency import run
    volume.reload();started=time.perf_counter();result=run(ROOT)
    result['total_seconds']=time.perf_counter()-started
    (Path(ROOT)/'trained_latency.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result


@app.local_entrypoint()
def main(wave:str='core'):
    import json,time,threading,traceback
    from concurrent.futures import ThreadPoolExecutor,as_completed
    protocol=json.loads((root/'publication/protocol.json').read_text())
    if sum(j['timeout'] for j in protocol['jobs'])>protocol['gpu_limit_seconds']:
        raise RuntimeError('Registered timeout reservations exceed the authorized budget')
    directory=root/'artifacts/publication';directory.mkdir(parents=True,exist_ok=True)
    # Independent, disjoint waves have separate ledgers to avoid lost updates
    # when CPU preparation and baseline training overlap in separate processes.
    ledger_path=directory/('budget_kairos.json' if wave=='kairos' else 'budget_latency.json' if wave=='latency' else 'budget_ledger.json')
    ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else {'limit_seconds':28800,'jobs':{}}
    lock=threading.Lock()
    def save():
        tmp=ledger_path.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));tmp.replace(ledger_path)
    if wave=='prepare':
        print(prepare_kairos.remote());return
    allowed={'acceptance':{'gpu_acceptance'},'latency':{'trained_latency'},'core':{'ORTHRUS','V3','V3_short_attention','GraphSAGE','V3_no_pair','V3_no_long_read'},
             'kairos':{'KAIROS'}}
    if wave not in allowed:raise ValueError('wave must be core, kairos, or prepare')
    def launch(job):
        key=f"{job['arm']}_s{job['seed']}_e{job['epochs']}"
        with lock:
            if key in ledger['jobs']:return key+' already reserved; not relaunched'
            reserved=sum(v['timeout'] for v in ledger['jobs'].values())
            if reserved+job['timeout']>ledger['limit_seconds']:raise RuntimeError('GPU budget exhausted')
            ledger['jobs'][key]=dict(job,status='reserved',reserved_at=time.time());save()
        function=trained_latency if job['arm']=='trained_latency' else {600:acceptance,900:train900,1500:train1500,1800:train1800,5400:train5400}[job['timeout']]
        try:
            call=function.spawn(job)
            with lock:
                ledger['jobs'][key].update(status='running',call_id=call.object_id);save()
            result=call.get()
            (directory/(key+'.json')).write_text(json.dumps(result,indent=2))
            with lock:
                ledger['jobs'][key].update(status='complete',measured_seconds=result['total_seconds']);save()
        except Exception as exc:
            with lock:
                ledger['jobs'][key].update(status='failed_or_timed_out',error=repr(exc));save()
            traceback.print_exc()
        return key
    jobs=[j for j in protocol['jobs'] if j['arm'] in allowed[wave] and j['arm'] not in protocol.get('withdrawn_arms',[])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(launch,j) for j in jobs]):print(future.result(),flush=True)
