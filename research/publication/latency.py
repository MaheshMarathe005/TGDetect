"""Trained-checkpoint inference on identical chronological 1024-event batches."""
import gc,itertools,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from orthrus_comparison.run import files,sha256
from orthrus_comparison.official import OrthrusReference
from publication.models import GraphSAGEReference
from kairos_comparison.model import Kairos
from sliding_node_mem_v3.model import TGNNSlidingNodeMem
from sliding_node_mem_v3.tokens import TokenBank
from sliding_node_mem_v3.prepare import read_batches
from sliding_node_mem_v2.runtime import stream_gc


def run(root):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(root);old=Path('/study/v3_vs_orthrus_20261010');prepared=old/'prepared_v3_local'
    manifest=json.loads(Path('/study/prepared/manifest.json').read_text())
    meta=json.loads((prepared/'manifest.json').read_text());bank=TokenBank(prepared/'tokens')
    batches=list(itertools.islice(read_batches(prepared/'train.bin',1024),72));n=72*1024
    graphs=[];count=0
    for graph in files(manifest,'train'):
        graphs.append(graph);count+=len(graph.src)
        if count>=n:break
    data={key:torch.cat([getattr(g,key) for g in graphs])[:n] for key in ('src','dst','t','msg')};del graphs
    assert torch.equal(data['src'],torch.cat([b['src'] for b in batches]))
    assert torch.equal(data['dst'],torch.cat([b['dst'] for b in batches]))
    assert torch.equal(data['msg'][:,131:141].argmax(-1),torch.cat([b['relation'] for b in batches]))
    full=SimpleNamespace(edge_type=data['msg'][:,131:141])
    paths={'ORTHRUS':old/'ORTHRUS','V3':old/'V3','V3_short_attention':root/'V3_short_attention_s0_e5'}
    available={k:p for k,p in paths.items() if (p/'result.json').exists()}
    missing=sorted(set(paths)-set(available));results={k:[] for k in available}
    for repeat in range(3):
        order=list(available);order=order[repeat:]+order[:repeat]
        for arm in order:
            gc.collect();torch.cuda.empty_cache();p=available[arm];checkpoint=torch.load(p/'best.pt',map_location='cpu',weights_only=False)
            if arm=='ORTHRUS':model=OrthrusReference(manifest['n_nodes'],device='cuda')
            elif arm=='GraphSAGE':model=GraphSAGEReference(manifest['n_nodes'],device='cuda')
            elif arm.startswith('V3'):
                model=TGNNSlidingNodeMem(meta['n_nodes'],bank,signatures=meta['signatures'],
                    short_readout='attention' if arm=='V3_short_attention' else 'cnn').cuda()
            else:
                km=json.loads((root/'prepared_kairos/manifest.json').read_text());model=Kairos(km['n_nodes'],msg_dim=42,relations=10).cuda()
            if arm.startswith('V3'):model.load_state_dict(checkpoint['model'])
            elif arm=='KAIROS':
                mismatch=model.load_state_dict(checkpoint,strict=False)
                assert set(mismatch.missing_keys)=={'memory.memory','memory.last_update','memory._assoc'} and not mismatch.unexpected_keys
            else:model.load_state_dict(checkpoint)
            model.eval();model.reset_state();torch.cuda.synchronize();setup=time.perf_counter()
            table=model.event_encoder.precompute() if arm.startswith('V3') else 0
            torch.cuda.synchronize();setup=time.perf_counter()-setup
            kairos=None
            if arm=='KAIROS':
                kairos={k:torch.from_numpy(np.load(root/'prepared_kairos/train'/(k+'.npy'),mmap_mode='r')[:n].copy())
                        for k in ('src','dst','time','messages')}
                kt=kairos['time'].cuda();kmessages=kairos['messages'].cuda()
            torch.cuda.reset_peak_memory_stats();durations=[]
            with stream_gc(),torch.no_grad():
                for i,batch in enumerate(batches):
                    lo=i*1024;hi=lo+1024;torch.cuda.synchronize();started=time.perf_counter()
                    if arm in ('ORTHRUS','GraphSAGE'):
                        msg=data['msg'][lo:hi].cuda()
                        model(data['src'][lo:hi].cuda(),data['dst'][lo:hi].cuda(),data['t'][lo:hi].double().cuda()/1e9,
                            msg[:,:131],msg[:,141:],msg[:,131:141],full,inference=True)
                    elif arm=='KAIROS':
                        s=kairos['src'][lo:hi].cuda();d=kairos['dst'][lo:hi].cuda()
                        model.score(s,d,kt,kmessages)
                        model.observe(s,d,kairos['time'][lo:hi].cuda(),kairos['messages'][lo:hi].cuda());model.detach_state()
                    else:model.forward_batch(batch);model.detach_state()
                    torch.cuda.synchronize()
                    if i>=8:durations.append(1000*(time.perf_counter()-started))
            results[arm].append(dict(repeat=repeat,batch_ms=durations,p50_ms=float(np.median(durations)),
                p95_ms=float(np.quantile(durations,.95)),p99_ms=float(np.quantile(durations,.99)),
                events_per_second=64*1024*1000/sum(durations),peak_cuda_bytes=torch.cuda.max_memory_allocated(),
                frozen_table_bytes=table,precompute_seconds=setup,checkpoint_sha256=sha256(p/'best.pt')))
            print('Trained latency',arm,repeat,results[arm][-1]['p50_ms'],flush=True)
            del model,checkpoint,kairos
            if arm=='KAIROS':del kt,kmessages
    return dict(status='complete',gpu=torch.cuda.get_device_name(),batch=1024,warmup=8,measured=64,repetitions=3,
        arms=results,missing_checkpoints=missing,source_sha256=sha256(__file__),
        scope='Trained seed-zero five-epoch checkpoints; identical first 73728 train events; prepared input H2D, forward and state updates. Excludes offline preparation, cache setup, scoring and batch-formation wait. KAIROS historical messages resident on GPU.')
