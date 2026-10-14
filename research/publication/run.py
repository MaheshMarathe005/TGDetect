"""Run registered detector-level comparisons; preserve measured provenance."""
import importlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from orthrus_comparison.run import sha256, measure
from sliding_node_mem_v2.scoring import EdgeWriter, read_edges, calibrated_scores


def kairos_run(prepared, source, output, epochs, seed, on_checkpoint):
    from kairos_comparison.model import Kairos
    root=Path(prepared); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((root/'manifest.json').read_text())
    for name,h in meta['checksums'].items():
        if sha256(root/name)!=h: raise ValueError('Prepared checksum mismatch: '+name)
    node_ids=np.load(root/'node_ids.npy')
    model=Kairos(meta['n_nodes'],msg_dim=42,relations=10).cuda()
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-5,eps=1e-8,weight_decay=.01)
    selection=meta['events']['val']//3
    def stream(split,training=False,limit=None,write=False):
        model.train(training); model.reset_state()
        data={k:torch.from_numpy(np.load(root/split/(k+'.npy'))).cuda()
              for k in ('src','dst','time','relation','messages')}
        origin=int(np.load(root/split/'time_origin.npy'))
        n=len(data['src']) if limit is None else min(limit,len(data['src']))
        writer=EdgeWriter(out/(split+'_edges.bin')) if write else None
        total=torch.zeros((),device='cuda')
        try:
            for lo in range(0,n,1024):
                hi=min(lo+1024,n); s=data['src'][lo:hi]; d=data['dst'][lo:hi]
                with torch.set_grad_enabled(training):
                    logits=model.score(s,d,data['time'],data['messages'])
                    loss=torch.nn.functional.cross_entropy(logits,data['relation'][lo:hi],reduction='none')
                    model.observe(s,d,data['time'][lo:hi],data['messages'][lo:hi])
                    if training:
                        optimizer.zero_grad(set_to_none=True);loss.mean().backward();optimizer.step()
                    model.detach_state()
                if not torch.isfinite(loss).all(): raise FloatingPointError('Nonfinite KAIROS loss')
                total+=loss.detach().sum()
                if writer:
                    writer.append(node_ids[s.cpu().numpy()],node_ids[d.cpu().numpy()],
                        (data['time'][lo:hi].cpu().numpy()+origin).astype('float64')/1e9,
                        data['relation'][lo:hi].cpu().numpy(),loss.detach().cpu().numpy())
        finally:
            if writer:writer.close()
        return float(total)/n
    history=[];best=float('inf');best_epoch=0
    for epoch in range(epochs):
        started=time.perf_counter();loss=stream('train',training=True)
        seconds=time.perf_counter()-started; val=stream('val',limit=selection)
        if val<best:
            best=val;best_epoch=epoch+1;torch.save(model.learned_state(),out/'best.pt')
        history.append(dict(epoch=epoch+1,loss=loss,validation_ce=val,seconds=seconds))
        (out/'progress.json').write_text(json.dumps(history,indent=2))
        on_checkpoint();print('KAIROS',history[-1],flush=True)
    model.load_state_dict(torch.load(out/'best.pt',weights_only=False),strict=False)
    val=stream('val',write=True);test=stream('test',write=True)
    threshold,scores,calibration=calibrated_scores(read_edges(out/'val_edges.bin')[selection:],read_edges(out/'test_edges.bin'),.001)
    manifest=json.loads(Path(source).read_text())
    truth=set(map(int,json.loads(Path(manifest['ground_truth_nodes']).read_text())))
    return dict(status='complete',history=history,checkpoint_epoch=best_epoch,
        parameter_count=sum(p.numel() for p in model.parameters()),validation_ce=val,test_ce=test,
        metrics=measure(scores,truth,{n for n,s in scores.items() if s>threshold}),
        unseen_ground_truth=sorted(truth-set(scores)),calibration=calibration,
        adaptations=meta['adaptations'],source_manifest_sha256=sha256(source),
        note='Authors-traced KAIROS neural detector with shared node scoring; not native queue reconstruction.')


def run(job, root, on_checkpoint=lambda:None):
    torch.set_num_threads(4);torch.manual_seed(job['seed']);np.random.seed(job['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(root);arm=job['arm'];output=root/f"{arm}_s{job['seed']}_e{job['epochs']}"
    if output.exists():raise FileExistsError('Refusing to overwrite study run: '+str(output))
    output.mkdir(parents=True)
    (output/'job.json').write_text(json.dumps(job,indent=2));on_checkpoint()
    started=time.perf_counter()
    try:
        if arm in ('ORTHRUS','GraphSAGE'):
            runner=importlib.import_module('sliding_node_mem_v3.compare_orthrus')
            from orthrus_comparison.official import OrthrusReference
            runner.OrthrusReference=OrthrusReference
            if arm=='GraphSAGE':
                from publication.models import GraphSAGEReference
                runner.OrthrusReference=GraphSAGEReference
            result=runner.run('/study/prepared/manifest.json',output,epochs=job['epochs'],seed=job['seed'],on_checkpoint=on_checkpoint)
            if arm=='GraphSAGE':result['note']='Our two-layer GraphSAGE baseline, shared ORTHRUS features/decoder and calibrated node scoring; not Wazuh reproduction.'
        elif arm.startswith('V3'):
            runner=importlib.import_module('sliding_node_mem_v3.run')
            from sliding_node_mem_v3.model import TGNNSlidingNodeMem
            runner.TGNNSlidingNodeMem=TGNNSlidingNodeMem
            options={};kwargs={}
            if arm=='V3_short_attention':options['short_readout']='attention'
            elif arm=='V3_no_pair':
                options['pair_head']=False;kwargs.update(pair_weight=0.,pair_score_weight=0.)
            elif arm=='V3_no_long_read':
                from publication.models import StudyV3
                runner.TGNNSlidingNodeMem=StudyV3;options['long_readout']=False
            elif arm!='V3':raise ValueError(arm)
            result=runner.run('/study/v3_vs_orthrus_20261010/prepared_v3_local',output,
                epochs=job['epochs'],seed=job['seed'],device='cuda',frozen_eval=True,
                model_options=options,on_checkpoint=on_checkpoint,**kwargs)
        elif arm=='KAIROS':
            result=kairos_run(root/'prepared_kairos','/study/prepared/manifest.json',output,
                job['epochs'],job['seed'],on_checkpoint)
        else:raise ValueError(arm)
        result.update(job=job,gpu=torch.cuda.get_device_name(),total_seconds=time.perf_counter()-started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            study_sources={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')})
        (output/'result.json').write_text(json.dumps(result,indent=2));on_checkpoint();return result
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-started)))
        on_checkpoint();raise
