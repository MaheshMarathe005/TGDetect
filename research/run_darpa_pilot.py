"""Matched DARPA E3 association-label pilot; new weights and saved artifacts."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,time,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from windows_apt_models import make_model
from causal_data import causal_egos,fit_scaler
from temporal_cnn import rehearsal_schedule
from run_causal_pilot import select_threshold,metrics

def features(row):
    x=np.zeros(66,dtype=np.float32)
    for key in ('type','name','exec'):
        h=hashlib.sha256((key+'='+str(row.get(key) or '')).encode()).digest()
        x[int.from_bytes(h[:8],'big')%64]+=1. if h[8]&1 else -1.
    size=row.get('size')
    if size is None:x[65]=1
    else:
        if not np.isfinite(float(size)) or float(size)<0:raise ValueError('Invalid event size')
        x[64]=np.log1p(float(size))
    return x

def tensors(root,dataset):
    folder=Path(root)/'prepared'/dataset
    audit=json.loads((folder/'audit.json').read_text());assert audit['training_gate'],'Class coverage gate failed'
    rows=[json.loads(line) for line in (folder/'events.jsonl').open()]
    ns=np.array([r['time_ns'] for r in rows],dtype=np.int64)
    assert np.all(np.diff(ns)>=0)
    # Subtract before float conversion to preserve nanosecond timestamp precision.
    times=(ns-ns[0]).astype(np.float64)/1e9
    nodes={}
    src=np.array([nodes.setdefault(r['src'],len(nodes)) for r in rows]);dst=np.array([nodes.setdefault(r['dst'],len(nodes)) for r in rows])
    y=np.array([r['label'] for r in rows]);x=np.stack([features(r) for r in rows])
    tr=audit['partitions']['train']['end'];va=audit['partitions']['validation']['end']
    mean,std=fit_scaler(x,tr);x=(x-mean)/std
    ego,valid,ages=causal_egos(src,dst,times,4)
    history=valid.copy();history[:,0]=False
    assert np.all(ns[ego][history]<np.broadcast_to(ns[:,None],ego.shape)[history])
    assert np.isfinite(x).all() and np.isfinite(ages).all()
    return {'x':x,'y':y,'ego':ego,'valid':valid,'ages':ages,'train_end':tr,'val_end':va,
            'ids':np.array([r['id'] for r in rows]),'scaler_mean':mean,'scaler_std':std},audit

def run(root,dataset,commit=lambda:None):
    completed=Path(root)/'results'/dataset
    if (completed/'summary.json').exists():
        for arm in ['GNN_H','TGNN_Q_A_C']:
            for seed in range(3):
                p=completed/f'{arm}_seed{seed}.json';saved=json.loads(p.read_text())
                assert hashlib.sha256(p.with_suffix('.pt').read_bytes()).hexdigest()==saved['checkpoint_sha256']
        return json.loads((completed/'summary.json').read_text())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    assert torch.cuda.is_available() and 'L4' in torch.cuda.get_device_name()
    data,audit=tensors(root,dataset);out=Path(root)/'results'/dataset;out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'prepared.npz',**data)
    n=len(data['y']);tr=data['train_end'];va=data['val_end']
    x=torch.from_numpy(data['x']).cuda();windows=x[torch.from_numpy(data['ego']).cuda()]
    valid=torch.from_numpy(data['valid']).cuda();ages=torch.from_numpy(data['ages']).cuda();y=torch.from_numpy(data['y']).cuda()
    def forward(model,ix):return model.forward_events(windows[ix],valid[ix],ages[ix],None,None,None)[0]
    def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    meta={'dataset':dataset,'label_scope':audit['label_scope'],'epochs':20,'seeds':[0,1,2],'batch_size':256,
          'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
          'partitions':audit['partitions'],'data_sha256':digest(out/'prepared.npz'),
          'source_hashes':{p.name:digest(p) for p in Path('/root').glob('*.py')},
          'nvidia_smi':subprocess.check_output(['nvidia-smi'],text=True)}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2));commit()
    results=[]
    for seed in range(3):
        rid,sizes=rehearsal_schedule(tr,20,256,seed);assert np.all((rid>=0)&(rid<tr))
        for arm in ['GNN_H','TGNN_Q_A_C']:
            existing=out/(arm+'_seed'+str(seed)+'.json')
            if existing.exists():
                saved=json.loads(existing.read_text())
                assert digest(out/(arm+'_seed'+str(seed)+'.pt'))==saved['checkpoint_sha256']
                results.append(saved);continue
            torch.manual_seed(seed);np.random.seed(seed)
            model=make_model(arm,x.shape[1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
            # Check deterministic eval and a valid backward pass before sustained work.
            model.eval()
            with torch.no_grad():assert torch.equal(forward(model,slice(0,8)),forward(model,slice(0,8)))
            torch.manual_seed(seed+10000);start=time.monotonic();step=0;losses=[]
            torch.cuda.reset_peak_memory_stats()
            for epoch in range(20):
                model.train();total=0.
                for lo in range(0,tr,256):
                    hi=min(lo+256,tr);opt.zero_grad(set_to_none=True)
                    loss=torch.nn.functional.cross_entropy(forward(model,slice(lo,hi)),y[lo:hi])
                    size=int(sizes[step])
                    if size:
                        ix=torch.as_tensor(rid[step,:size],device='cuda');loss=loss+torch.nn.functional.cross_entropy(forward(model,ix),y[ix])
                    assert torch.isfinite(loss);loss.backward();opt.step();total+=loss.item();step+=1
                losses.append(total)
                print(dataset,arm,seed,'epoch',epoch+1,'loss',total,flush=True)
                (out/'progress.json').write_text(json.dumps({'arm':arm,'seed':seed,'epoch':epoch+1,'epochs':20}));commit()
            torch.cuda.synchronize();seconds=time.monotonic()-start
            model.eval();prob=[]
            with torch.no_grad():
                for lo in range(0,n,256):prob.append(forward(model,slice(lo,lo+256)).softmax(-1)[:,1].cpu().numpy())
            p=np.concatenate(prob);yy=data['y'];threshold=select_threshold(yy[tr:va],p[tr:va])
            stem=arm+'_seed'+str(seed)
            torch.save({'state_dict':model.cpu().state_dict(),'arm':arm,'input_dim':66,'seed':seed,'epochs':20,'label_scope':audit['label_scope']},out/(stem+'.pt'))
            np.savez_compressed(out/(stem+'_predictions.npz'),probability=p,y=yy,ids=data['ids'])
            result={'arm':arm,'seed':seed,'threshold':threshold,'validation':metrics(yy[tr:va],p[tr:va],threshold),
                    'test':metrics(yy[va:],p[va:],threshold),'train_seconds':seconds,'epoch_losses':losses,
                    'checkpoint_sha256':digest(out/(stem+'.pt')),'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            (out/(stem+'.json')).write_text(json.dumps(result,indent=2));results.append(result);commit()
            del model,opt;torch.cuda.empty_cache()
    summary={arm:{k:{'mean':float(np.mean([r['test'][k] for r in results if r['arm']==arm])),
                           'sd':float(np.std([r['test'][k] for r in results if r['arm']==arm],ddof=1))}
                  for k in ['f1','precision','recall','fpr','pr_ap','roc_auc']} for arm in ['GNN_H','TGNN_Q_A_C']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));commit();return summary
