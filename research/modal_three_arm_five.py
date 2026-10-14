"""Three isolated five-epoch CADETS arms, concurrently on the authorized profile."""
from pathlib import Path
import modal
root = Path(__file__).parent
volume = modal.Volume.from_name('tg-detect-sliding-node-mem-e3-v1')
image = (modal.Image.debian_slim(python_version='3.10')
         .pip_install('torch==2.4.1','torch-geometric==2.5.3','numpy==1.26.4',
                      'scipy==1.10.1','scikit-learn==1.2.0','networkx==2.8.7',
                      'pytest==8.3.5','tqdm==4.67.1')
         .env({'PYTHONPATH':'/root/research','PYTHONHASHSEED':'0'}))
for name in ('sliding_node_mem','sliding_node_mem_upgrade','orthrus_comparison','orthrus_official'):
    image=image.add_local_dir(root/name,'/root/research/'+name,ignore=['__pycache__','.git'])
for name in ('node_memory_v3.py','node_memory_models.py','tgnn_components.py',
             'temporal_cnn.py','causal_tgn.py','pomsathit_baseline.py'):
    image=image.add_local_file(root/name,'/root/research/'+name)

app = modal.App('tg-detect-three-arm-five-epochs')
ROOT = '/study/comparison_upgrade_5ep_v1'
ARMS = ('ORTHRUS_Reference','TGNN_Sliding_Node_Mem_Upgrade','TGNN_Sliding_ORTHRUS_Encoder')

@app.function(image=image, gpu='L4', cpu=8, memory=32768, timeout=86400,
              volumes={'/study':volume}, max_containers=3)
def train_arm(arm):
    import json, shutil, traceback, torch
    from orthrus_comparison.run import run as reference
    from sliding_node_mem_upgrade.run import run
    if arm not in ARMS: raise ValueError(arm)
    volume.reload()
    out=Path(ROOT)/arm;out.mkdir(parents=True,exist_ok=True)
    def checkpoint():
        progress=out/'progress.json'
        if progress.exists() and (out/'resume.pt').exists():
            history=json.loads(progress.read_text())
            if arm == ARMS[0] and history:
                shutil.copy2(out/'resume.pt',out/f"epoch_{history[-1]['epoch']:03d}.pt")
        volume.commit()
    existing=out/'result.json'
    if existing.exists():
        saved=json.loads(existing.read_text())
        if saved.get('status')=='complete' and saved['signature']['epochs']==5:
            return {k:v for k,v in saved.items() if k not in ('node_scores','calibrated_node_scores')}
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    original_rng=torch.cuda.set_rng_state_all
    torch.cuda.set_rng_state_all=lambda states:original_rng([s.cpu() for s in states])
    try:
        (out/'job_status.json').write_text(json.dumps({'stage':'training','epochs':5,'batch':1024,'seed':0,'gpu':torch.cuda.get_device_name()}));volume.commit()
        args=dict(manifest_path='/study/prepared/manifest.json',output=ROOT,epochs=5,batch=1024,
                  device='cuda',seed=0,on_checkpoint=checkpoint)
        result=reference(**args,arm=arm) if arm==ARMS[0] else run(**args,frontend='kv' if arm==ARMS[1] else 'orthrus')
        result['gpu']=torch.cuda.get_device_name()
        result['cuda_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
        (out/'result.json').write_text(json.dumps(result,indent=2))
        (out/'job_status.json').write_text(json.dumps({'stage':'complete','epochs':5}));volume.commit()
        return {k:v for k,v in result.items() if k not in ('node_scores','calibrated_node_scores')}
    except Exception:
        (out/'failure.json').write_text(json.dumps({'error':traceback.format_exc()}));volume.commit();raise
    finally:torch.cuda.set_rng_state_all=original_rng

@app.function(image=image,cpu=.25,memory=1024,timeout=86400,volumes={'/study':volume})
def pipeline():
    import json,traceback
    volume.reload();dest=Path(ROOT);dest.mkdir(exist_ok=True)
    jobs_path=dest/'jobs.json'
    jobs=json.loads(jobs_path.read_text()) if jobs_path.exists() else {}
    calls={}
    try:
        for arm in ARMS:
            calls[arm]=modal.FunctionCall.from_id(jobs[arm]) if arm in jobs else train_arm.spawn(arm)
            jobs[arm]=calls[arm].object_id;jobs_path.write_text(json.dumps(jobs,indent=2));volume.commit()
        (dest/'status.json').write_text(json.dumps({'stage':'training','calls':jobs}));volume.commit()
        results={arm:call.get() for arm,call in calls.items()}
        volume.reload()
        (dest/'comparison.json').write_text(json.dumps(results,indent=2))
        (dest/'status.json').write_text(json.dumps({'stage':'complete'}));volume.commit()
        return results
    except Exception:
        (dest/'status.json').write_text(json.dumps({'stage':'failed','error':traceback.format_exc()}));volume.commit();raise

@app.local_entrypoint()
def main():
    import json
    result=pipeline.remote()
    output=Path(__file__).parent/'artifacts/three_arm_five_epochs'
    output.mkdir(parents=True,exist_ok=True)
    (output/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
