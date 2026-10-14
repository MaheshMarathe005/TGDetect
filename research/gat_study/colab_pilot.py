"""Small real-CADETS old-vs-role4 pilot, with frozen common KV encoder."""
import hashlib
import json
from dataclasses import dataclass,asdict
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score,confusion_matrix

from .model import StudyV3
from .diagnostics import DegreeAudit
from sliding_node_mem_v3.tokens import TokenBank
from sliding_node_mem_v2.scoring import EdgeWriter,read_edges,calibrated_scores

DTYPE=np.dtype([(k,'<i8') for k in ('src','dst','relation','query','observed','signature','src_identity','dst_identity','negative_identity')]+[('time','<f8'),('negative_valid','?')])


@dataclass
class Config:
    prepared:str='/content/tgdetect_gat_data'
    output:str='/content/tgdetect_gat_results'
    epochs:int=16
    batch:int=256
    seeds:tuple=(0,1,2)
    device:str='cuda'
    false_alert_rate:float=.01
    unroll:int=2
    timing_repetitions:int=3
    timing_warmup:int=8
    timing_batches:int=64
    bootstrap:int=300


def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sync(cfg):
    if cfg.device.startswith('cuda'):torch.cuda.synchronize()


def batches(cfg,split):
    rows=np.memmap(Path(cfg.prepared)/(split+'.bin'),mode='r',dtype=DTYPE)
    for start in range(0,len(rows),cfg.batch):
        yield {k:torch.from_numpy(np.array(rows[k][start:start+cfg.batch])) for k in DTYPE.names}


def build(cfg,arm,seed):
    torch.manual_seed(seed)
    root=Path(cfg.prepared);manifest=json.loads((root/'manifest.json').read_text())
    bank=TokenBank(root/'tokens').to_device(cfg.device)
    model=StudyV3(manifest['n_nodes'],bank,signatures=manifest['signatures'],num_relations=manifest['relations'],operator=arm).to(cfg.device)
    weights=np.load(root/'encoder_weights.npz',allow_pickle=False)
    model.event_encoder.load_state_dict({k:torch.from_numpy(weights[k]) for k in weights.files})
    for p in model.event_encoder.parameters():p.requires_grad_(False)
    return model,manifest


def evaluate(model,cfg,split,path=None,limit=None):
    model.eval();model.reset_state()
    if model.event_encoder._frozen_table is None:model.event_encoder.precompute()
    writer=EdgeWriter(path) if path else None
    total=0.;n=0
    try:
        with torch.no_grad():
            for data in batches(cfg,split):
                if limit is not None:
                    if n>=limit:break
                    data={k:v[:limit-n] for k,v in data.items()}
                logits,aux=model.forward_batch(data);target=data['relation'].to(cfg.device)
                ce=F.cross_entropy(logits,target,reduction='none')
                total+=float(ce.sum());n+=len(logits)
                if writer:
                    value=ce+.1*aux['pair_anomaly']
                    writer.append(data['src'].numpy(),data['dst'].numpy(),data['time'].numpy(),data['relation'].numpy(),value.cpu().numpy())
                model.detach_state()
    finally:
        if writer:writer.close()
    return total/max(n,1)


def node_metrics(scores,truth,threshold):
    ids=np.array(sorted(scores));values=np.array([scores[int(n)] for n in ids])
    y=np.isin(ids,list(truth)).astype(int);prediction=values>threshold
    tn,fp,fn,tp=confusion_matrix(y,prediction,labels=[0,1]).ravel()
    p=tp/max(tp+fp,1);r=tp/max(tp+fn,1)
    return {'ap':float(average_precision_score(y,values)) if y.any() else None,
            'roc_auc':float(roc_auc_score(y,values)) if len(np.unique(y))==2 else None,
            'precision':float(p),'recall':float(r),'f1':float(2*p*r/max(p+r,1e-12)),
            'test_node_fpr':float(fp/max(tn+fp,1)),
            'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'unscored_truth':sorted(truth-set(ids.tolist()))},ids,y,values


def train(cfg,arm,seed):
    root=Path(cfg.output)/f'{arm}_s{seed}';root.mkdir(parents=True,exist_ok=False)
    model,manifest=build(cfg,arm,seed)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-5)
    best=float('inf');history=[];selection=manifest['events']['val']//3
    if cfg.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    sync(cfg);started=time.perf_counter()
    for epoch in range(cfg.epochs):
        model.train();model.reset_state();pending=[];total=0.;count=0
        def update(items):
            optimizer.zero_grad(set_to_none=True);losses=[];n=0
            for data in items:
                logits,aux=model.forward_batch(data);size=len(logits);n+=size
                loss=F.cross_entropy(logits,data['relation'].to(cfg.device))+.01*aux['compression_loss']+.1*aux['pair_loss']
                losses.append(loss*size)
            loss=torch.stack(losses).sum()/n
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();model.detach_state();return float(loss.detach())*n,n
        for data in batches(cfg,'train'):
            pending.append(data)
            if len(pending)==cfg.unroll:
                loss,n=update(pending);total+=loss;count+=n;pending=[]
        if pending:
            loss,n=update(pending);total+=loss;count+=n
        val=evaluate(model,cfg,'val',limit=selection)
        history.append({'epoch':epoch+1,'training_loss':total/count,'benign_selection_ce':val})
        if val<best:
            best=val;torch.save({'model':model.state_dict(),'epoch':epoch+1},root/'best.pt')
        print(arm,seed,history[-1],flush=True)
    sync(cfg);train_seconds=time.perf_counter()-started
    saved=torch.load(root/'best.pt',map_location=cfg.device,weights_only=False)
    model.load_state_dict(saved['model'])
    # Degree measurement is a separate stream, never a timing hook.
    audit=DegreeAudit();hook=model.gat.register_forward_pre_hook(audit.hook)
    evaluate(model,cfg,'train',limit=min(manifest['events']['train'],4096));hook.remove()
    save(root/'degree.json',audit.result())
    evaluate(model,cfg,'val',root/'val_edges.bin')
    evaluate(model,cfg,'test',root/'test_edges.bin')
    truth=set(json.loads((Path(cfg.prepared)/'ground_truth_nodes.json').read_text()))
    threshold,scores,calibration=calibrated_scores(read_edges(root/'val_edges.bin')[selection:],read_edges(root/'test_edges.bin'),cfg.false_alert_rate)
    if not np.isfinite(threshold):
        print('Calibration sample cannot support requested alert budget; threshold metrics are unavailable for interpretation.',flush=True)
    metric,ids,y,values=node_metrics(scores,truth,threshold)
    np.savez_compressed(root/'node_scores.npz',ids=ids,y=y,scores=values)
    result={'arm':arm,'seed':seed,'checkpoint_epoch':saved['epoch'],'history':history,
            'metrics':metric,'threshold':float(threshold) if np.isfinite(threshold) else None,'calibration':calibration,
            'training_seconds_including_selection':train_seconds,
            'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'peak_cuda_bytes':torch.cuda.max_memory_allocated() if cfg.device.startswith('cuda') else None,
            'checkpoint_sha256':sha(root/'best.pt'),'encoder_frozen':True}
    save(root/'result.json',result);return result


def timing(cfg):
    samples=[]
    for data in batches(cfg,'train'):
        samples.append(data)
        if len(samples)>=cfg.timing_warmup+cfg.timing_batches:break
    if len(samples)<cfg.timing_warmup+cfg.timing_batches:raise ValueError('Insufficient timing prefix')
    result=[]
    for seed in cfg.seeds:
        for rep in range(cfg.timing_repetitions):
            arms=['gat2','role4'] if rep%2==0 else ['role4','gat2']
            for arm in arms:
                model,_=build(cfg,arm,seed)
                saved=torch.load(Path(cfg.output)/f'{arm}_s{seed}/best.pt',map_location=cfg.device,weights_only=False)
                model.load_state_dict(saved['model']);model.eval();model.event_encoder.precompute();model.reset_state()
                durations=[]
                with torch.no_grad():
                    for index,data in enumerate(samples):
                        sync(cfg);start=time.perf_counter();model.forward_batch(data);model.detach_state();sync(cfg)
                        if index>=cfg.timing_warmup:durations.append(1000*(time.perf_counter()-start))
                result.append({'arm':arm,'seed':seed,'repetition':rep,'p50_ms':float(np.median(durations)),
                               'p95_ms':float(np.quantile(durations,.95)),'samples_ms':durations,
                               'scope':'full V3 forward plus prepared host transfer/state updates; excludes preparation, lookup setup, alert scoring, arrival wait'})
                del model
    save(Path(cfg.output)/'paired_latency.json',result);return result


def run(cfg=Config()):
    root=Path(cfg.output);root.mkdir(parents=True,exist_ok=True)
    if (root/'summary.json').exists():raise FileExistsError('Choose a fresh output directory')
    torch.set_num_threads(2)
    if cfg.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('Select GPU runtime.')
        torch.cuda.set_per_process_memory_fraction(min(1.,15*2**30/torch.cuda.get_device_properties(0).total_memory))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest=json.loads((Path(cfg.prepared)/'manifest.json').read_text())
    for relative,value in manifest['checksums'].items():
        if sha(Path(cfg.prepared)/relative)!=value:raise ValueError('Data checksum mismatch: '+relative)
    save(root/'config.json',asdict(cfg));save(root/'data_manifest.json',manifest)
    save(root/'environment.json',{'gpu':torch.cuda.get_device_name() if cfg.device.startswith('cuda') else 'CPU smoke',
                                'torch':torch.__version__,'source_sha256':sha(__file__)})
    results=[]
    for seed in cfg.seeds:
        for arm in (['gat2','role4'] if seed%2==0 else ['role4','gat2']):
            try:
                results.append(train(cfg,arm,seed))
            except Exception as exc:
                save(root/'failure.json',{'arm':arm,'seed':seed,'error':repr(exc)})
                raise
            save(root/'partial_summary.json',results)
    pairs=[]
    for seed in cfg.seeds:
        a=np.load(root/f'gat2_s{seed}/node_scores.npz');b=np.load(root/f'role4_s{seed}/node_scores.npz')
        np.testing.assert_array_equal(a['ids'],b['ids']);np.testing.assert_array_equal(a['y'],b['y'])
        if len(np.unique(a['y']))<2:continue
        rng=np.random.default_rng(seed+2026);groups=[np.flatnonzero(a['y']==k) for k in (0,1)];d=[]
        for _ in range(cfg.bootstrap):
            ix=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
            d.append(average_precision_score(a['y'][ix],b['scores'][ix])-average_precision_score(a['y'][ix],a['scores'][ix]))
        pairs.append({'seed':seed,'delta_ap':float(average_precision_score(a['y'],b['scores'])-average_precision_score(a['y'],a['scores'])),
                      'ci95':np.quantile(d,[.025,.975]).tolist(),'scope':'conditional stratified node bootstrap; not independent graph/capture uncertainty'})
    save(root/'summary.json',results);save(root/'paired_ap.json',pairs);timing(cfg)
    return results
