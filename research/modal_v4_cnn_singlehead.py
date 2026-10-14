"""Deployed bounded L4 training: causal CNN + single-head V4/V3 hybrid."""
from pathlib import Path
import modal
from modal_sliding_v2_benchmark import volume

app=modal.App('tg-detect-v4-cnn-singlehead')
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
OUTPUT='/study/v4_cnn_singlehead_20261011_v1/V4_CNN_SingleHead_s0_e5'
INITIALIZATION='/study/v3_vs_orthrus_20261010/V3/initial_embeddings.pt'
TIMEOUT=3600
JOB='V4_CNN_SingleHead_s0_e5'


def compare_latency(output,options):
    """Paired trained-weight inference timing, same prefix and frozen FP32 cache."""
    import gc,itertools,json,time,numpy as np,torch
    from orthrus_comparison.run import sha256
    from sliding_node_mem_v3.tokens import TokenBank
    from sliding_node_mem_v3.prepare import read_batches
    from sliding_node_mem_v3.model import TGNNSlidingNodeMem as V3
    from sliding_node_mem_v4.model import TGNNSlidingNodeMem as Candidate
    meta=json.loads((Path(PREPARED)/'manifest.json').read_text())
    batches=list(itertools.islice(read_batches(Path(PREPARED)/'train.bin',1024),40))
    arms={'V3':Path('/study/v3_vs_orthrus_20261010/V3/best.pt'),'CNN_single_head':output/'best.pt'}
    observations={k:[] for k in arms}
    for repeat in range(3):
        order=list(arms) if repeat%2==0 else list(reversed(arms))
        for arm in order:
            gc.collect();torch.cuda.empty_cache()
            cls=V3 if arm=='V3' else Candidate
            model=cls(meta['n_nodes'],TokenBank(Path(PREPARED)/'tokens'),
                num_relations=meta['relations'],signatures=meta['signatures'],
                **({} if arm=='V3' else options)).cuda()
            checkpoint=torch.load(arms[arm],map_location='cpu',weights_only=False)
            model.load_state_dict(checkpoint['model']);del checkpoint
            model.eval();model.reset_state()
            torch.cuda.synchronize();setup=time.perf_counter()
            table_bytes=model.event_encoder.precompute()
            torch.cuda.synchronize();setup=time.perf_counter()-setup
            torch.cuda.reset_peak_memory_stats();durations=[]
            with torch.no_grad():
                for i,batch in enumerate(batches):
                    torch.cuda.synchronize();start=time.perf_counter()
                    model.forward_batch(batch);model.detach_state()
                    torch.cuda.synchronize()
                    if i>=8:durations.append(1000*(time.perf_counter()-start))
            row=dict(repeat=repeat,batch_ms=durations,p50_ms=float(np.median(durations)),
                p95_ms=float(np.quantile(durations,.95)),p99_ms=float(np.quantile(durations,.99)),
                events_per_second=32*1024*1000/sum(durations),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                frozen_table_bytes=table_bytes,precompute_seconds=setup,checkpoint_sha256=sha256(arms[arm]))
            observations[arm].append(row)
            print('Trained inference',arm,repeat,'p50 ms',row['p50_ms'],flush=True)
            del model
    return dict(status='complete',gpu=torch.cuda.get_device_name(),batch=1024,warmup=8,
        measured=32,repetitions=3,arms=observations,
        scope='Trained checkpoints, same chronological train prefix, alternating arm order. Includes host input transfer, forward and state updates. Excludes cache setup, preprocessing and batching wait. Prefix memory is not mature full-dataset memory.')


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
    options=dict(encoder_dim=64,node_memory='affine-v3',memory_device='model',temporal_heads=1,temporal_cnn=True)
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
        training_peak=torch.cuda.max_memory_allocated()
        latency=compare_latency(output,options)
        (output/'latency_vs_v3.json').write_text(json.dumps(latency,indent=2))
        result.update(gpu=torch.cuda.get_device_name(),total_seconds=time.perf_counter()-started,
            peak_cuda_bytes=training_peak,latency_vs_v3=latency,
            job=dict(arm=JOB,timeout_seconds=TIMEOUT,initialization='Original V3 benign-only 64D pretrained table',
                     restart='Fresh weights/optimizer/state; not a V4 checkpoint resume'))
        (output/'result.json').write_text(json.dumps(result,indent=2));volume.commit();return result
    except Exception as exc:
        output.mkdir(parents=True,exist_ok=True)
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-started)))
        volume.commit();raise


