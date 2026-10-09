"""Reproducible bounded CPU pilot. No legacy/Modal app imports or test tuning."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from torch.nn import functional as F

from causal_data import sha256
from causal_tgn import CausalTGNN, Reservoir

ROOT = Path(__file__).resolve().parent
ARMS = {'B': (), 'B_T': ('T',), 'B_M': ('M',), 'B_MR': ('M','R'),
        'B_TM': ('T','M'), 'B_TMR': ('T','M','R'), 'B_H': ('H',), 'B_TMRH': ('T','M','R','H')}
SOURCES = ['pomsathit_baseline.py', 'causal_data.py', 'causal_tgn.py', 'run_causal_pilot.py',
           'test_causal_pilot.py', 'fetch_pilot_data.py', 'ADAPTED_PROTOCOL.md']


def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def select_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def metrics(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {'f1': float(f1_score(y,pred,zero_division=0)),
            'precision': float(precision_score(y,pred,zero_division=0)),
            'recall': float(recall_score(y,pred,zero_division=0)),
            'pr_ap': float(average_precision_score(y,p)), 'roc_auc': float(roc_auc_score(y,p)),
            'fpr': float(fp / max(tn+fp,1)), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}


class PilotData:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix('.json').read_text())
        if sha256(path) != self.meta['processed_sha256']:
            raise ValueError('Processed data hash mismatch')
        raw_path = self.path.with_name('scenario06.binetflow')
        if sha256(raw_path) != self.meta['raw_source']['sha256']:
            raise ValueError('Raw source changed')
        a = np.load(path, allow_pickle=False)
        self.x = torch.from_numpy(a['x'])
        self.y = torch.from_numpy(a['y'])
        self.ego = torch.from_numpy(a['ego'])
        self.valid = torch.from_numpy(a['valid'])
        self.ages = torch.from_numpy(a['ages'])
        self.src, self.dst, self.times = a['src'], a['dst'], a['times']
        self.row_ids = a['row_ids']
        self.train_end, self.val_end = int(a['train_end']), int(a['val_end'])
        self.n = len(self.y)
        for name, lo,hi,minimum in [('train',0,self.train_end,20), ('validation',self.train_end,self.val_end,10), ('test',self.val_end,self.n,10)]:
            positives = int(self.y[lo:hi].sum())
            if min(positives, hi-lo-positives) < minimum:
                raise ValueError(f'{name} fails preregistered class support gate: {positives}/{hi-lo}')

    def batch(self, ids, ages=None):
        return (self.x[self.ego[ids]], self.valid[ids], (self.ages if ages is None else ages)[ids],
                self.src[ids], self.dst[ids], self.times[ids])


def predict(model, data, intervention=None, seed=0):
    model.eval(); model.reset_state()
    ages = data.ages
    if intervention == 'shuffled_ages':
        # Shuffle valid historical age channels within each partition separately.
        # This deliberately noncausal counterfactual is never used as a detector.
        ages = ages.clone()
        rng = np.random.default_rng(seed + 771)
        for lo, hi in [(0,data.train_end),(data.train_end,data.val_end),(data.val_end,data.n)]:
            mask = data.valid[lo:hi].clone(); mask[:,0] = False
            values = ages[lo:hi][mask].numpy().copy()
            rng.shuffle(values)
            ages[lo:hi][mask] = torch.from_numpy(values)
    outputs = []
    started = time.perf_counter()
    with torch.no_grad():
        for lo in range(0, data.n, 256):
            ids = slice(lo,min(lo+256,data.n))
            logits,_ = model.forward_events(*data.batch(ids,ages), intervention=intervention)
            outputs.append(logits.softmax(-1)[:,1].numpy())
    elapsed = time.perf_counter() - started
    return np.concatenate(outputs), elapsed


def train_one(data, arm, seed, output):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CausalTGNN(data.x.shape[1], ARMS[arm])
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)
    # Match dropout RNG starts across configurations after module initialization.
    torch.manual_seed(seed + 10000)
    replay = Reservoir(data.train_end//10, data.train_end, seed+937) if 'H' in ARMS[arm] else None
    epoch_losses, replay_visits = [], 0
    start = time.perf_counter()
    for epoch in range(3):
        model.train(); model.reset_state()
        losses = []
        for lo in range(0,data.train_end,256):
            hi = min(lo+256,data.train_end)
            opt.zero_grad(set_to_none=True)
            logits, context = model.forward_events(*data.batch(slice(lo,hi)))
            live_loss = F.cross_entropy(logits, data.y[lo:hi])
            loss = live_loss
            if replay is not None:
                items = replay.sample(26)
                if items:
                    ids = np.array([i for i,_ in items])
                    stored = torch.stack([c for _,c in items])
                    rlogits = model.replay(*data.batch(ids)[:3], stored)
                    loss = loss + F.cross_entropy(rlogits,data.y[ids])
                    replay_visits += len(items)
            loss.backward(); opt.step(); model.detach_state()
            losses.append(float(live_loss.detach())*(hi-lo))
            if replay is not None:
                for i in range(lo,hi): replay.admit(i,context[i-lo])
        epoch_losses.append(sum(losses)/data.train_end)
    train_seconds = time.perf_counter() - start
    run_id = f'{arm}_seed{seed}'
    checkpoint = output / f'{run_id}.pt'
    torch.save({'state_dict': model.state_dict(), 'factors': list(ARMS[arm]), 'seed': seed}, checkpoint)
    p, inference_seconds = predict(model,data)
    val = slice(data.train_end,data.val_end); test = slice(data.val_end,data.n)
    threshold = select_threshold(data.y[val].numpy(),p[val])
    result = {'run_id': run_id, 'arm': arm, 'factors': list(ARMS[arm]), 'seed': seed,
              'status': 'COMPLETE', 'scope': 'small CPU pilot; not exact-paper or full-TGDetect',
              'epoch_losses': epoch_losses, 'threshold': threshold,
              'validation': metrics(data.y[val].numpy(),p[val],threshold),
              'test': metrics(data.y[test].numpy(),p[test],threshold),
              'parameters': sum(v.numel() for v in model.parameters()),
              'parameters_by_module': {n: v.numel() for n,v in model.named_parameters()},
              'train_seconds': train_seconds, 'inference_seconds': inference_seconds,
              'inference_events_per_second': data.n / inference_seconds,
              'inference_ms_per_event_amortized': 1000*inference_seconds/data.n,
              'state_tensor_bytes': sum(v.numel()*v.element_size() for v in model.memory.values()),
              'pending_state_tensor_bytes': sum(h.numel()*h.element_size() for _,_,h in model.pending),
              'state_bytes_exclude_python_container_overhead': True,
              'replay_context_bytes': 0 if replay is None else sum(c.numel()*c.element_size()+8 for _,c in replay.items),
              'training_live_visits': data.train_end*3, 'training_replay_visits': replay_visits,
              'peak_process_rss_kib_cumulative': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'gpu': None, 'gpu_peak_bytes': None, 'flops': None,
              'checkpoint_bytes': checkpoint.stat().st_size, 'checkpoint_sha256': sha256(checkpoint),
              'data_sha256': sha256(data.path), 'source_hashes': {p:sha256(ROOT/p) for p in SOURCES}}
    controls = {}
    saved = {'probability':p, 'y':data.y.numpy(), 'raw_row_id':data.row_ids}
    if arm in {'B_TMR','B_TMRH'}:
        for mode in ['reset_memory','frozen_memory','gru_noop','constant_time','shuffled_ages']:
            altered,_ = predict(model,data,mode,seed)
            controls[mode] = metrics(data.y[test].numpy(),altered[test],threshold)
            saved[mode] = altered
    result['fixed_checkpoint_controls'] = controls
    np.savez_compressed(output/f'{run_id}_predictions.npz',**saved)
    save_json(output/f'{run_id}.json',result)
    print(json.dumps({k:result[k] for k in ['run_id','test','train_seconds']}),flush=True)
    return result


def feature_controls(data, output):
    results=[]
    train=slice(0,data.train_end); val=slice(data.train_end,data.val_end); test=slice(data.val_end,data.n)
    for seed in range(5):
        for name, model in [('LogisticRegression', LogisticRegression(max_iter=1000,random_state=seed)),
                            ('HistGradientBoosting', HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,early_stopping=False,random_state=seed))]:
            t=time.perf_counter(); model.fit(data.x[train].numpy(),data.y[train].numpy()); elapsed=time.perf_counter()-t
            p=model.predict_proba(data.x.numpy())[:,1]
            threshold=select_threshold(data.y[val].numpy(),p[val])
            r={'arm':name,'seed':seed,'threshold':threshold,'test':metrics(data.y[test].numpy(),p[test],threshold),
               'validation':metrics(data.y[val].numpy(),p[val],threshold),'train_seconds':elapsed,
               'configuration':model.get_params(),'status':'COMPLETE'}
            results.append(r)
            np.savez_compressed(output/f'{name}_seed{seed}_predictions.npz',probability=p,y=data.y.numpy(),raw_row_id=data.row_ids)
    save_json(output/'feature_controls.json',results)
    return results


def matrix(data):
    factors = ['F','G','T','M','R','L','N','H']
    groups = {(): 'baseline'}
    for k in [1,2]:
        for subset in itertools.combinations(factors,k): groups[subset] = 'single' if k==1 else 'pair'
    groups[tuple(factors)]='full'
    for excluded in factors: groups[tuple(f for f in factors if f != excluded)]='reverse'
    for active in ARMS.values(): groups.setdefault(tuple(f for f in factors if f in active),'pilot_additional')
    rows=[]
    for active, group in groups.items():
        arm = next((name for name, fs in ARMS.items() if set(fs)==set(active)),None)
        reason = 'pilot queued' if arm else 'not included in bounded pilot'
        if 'R' in active and 'M' not in active: reason='R requires M; not independently executable'
        if set(active)-{'T','M','R','H'}: reason='F/G/L/N alternatives not implemented; must first specify isolated interventions'
        for seed in range(5):
            rows.append({'experiment_id':f'{arm or "B_"+"".join(active)}_seed{seed}',
                         'group':group,'active_factors':list(active),'inactive_factors':[f for f in factors if f not in active],
                         'model_class':'CausalTGNN' if arm else None,'model_sha256':sha256(ROOT/'causal_tgn.py') if arm else None,
                         'feature_set':'STATIC_FAIR','graph_builder':'strict_past_event_ego4_v1',
                         'loss_id':'unweighted_CE','normalization_id':'train_mean_std',
                         'rehearsal_id':'reservoir10pct_context' if 'H' in active else 'none',
                         'temporal_encoder_id':'cos_log_age' if 'T' in active else 'none',
                         'memory_id':'node128' if 'M' in active else 'none','GRU_id':'GRUCell128' if 'R' in active else 'none',
                         'seed':seed,'dataset_version':sha256(data.path),'split_id':data.meta['split_id'],
                         'compute_budget':{'epochs':3,'batch':256,'cpu_threads':1},'modal_gpu_type':None,
                         'status':'PENDING' if arm else 'NOT_RUN','reason':reason})
    return {'factors':factors,'dependency':{'R':['M']},'additional_legacy_factors_deferred':['attention_operator','identity_embeddings','head','sampling','model_capacity','temporal_message_passing'],
            'rows':rows,'scope':'full F/G/T/M/R/L/N/H study remains incomplete; 8 registered pilot arms'}


def summarize(results, controls):
    all_results=results+controls
    stats={}
    fields=['f1','pr_ap','roc_auc','precision','recall','fpr']
    for name in dict.fromkeys(r['arm'] for r in all_results):
        subset=[r for r in all_results if r['arm']==name]
        stats[name]={'seeds':len(subset),**{f:{'mean':float(np.mean([r['test'][f] for r in subset])),
                                            'sd':float(np.std([r['test'][f] for r in subset],ddof=1)) if len(subset)>1 else None} for f in fields}}
    paired={}
    indexed={(r['arm'],r['seed']):r for r in results}
    def contrast(name, terms):
        seeds=[s for s in range(5) if all((a,s) in indexed for a,_ in terms)]
        if not seeds: return
        paired[name]={}
        for field in ['f1','pr_ap']:
            values=[sum(coef*indexed[(arm,s)]['test'][field] for arm,coef in terms) for s in seeds]
            paired[name][field]={'seed_deltas':values,'mean':float(np.mean(values)),
                                'sd':float(np.std(values,ddof=1)) if len(values)>1 else None}
    for name in ['B_T','B_M','B_H','B_TMR']:
        contrast(name+' minus B',[(name,1),('B',-1)])
    contrast('T_M interaction',[('B_TM',1),('B_T',-1),('B_M',-1),('B',1)])
    contrast('conditional R given M',[('B_MR',1),('B_M',-1)])
    contrast('TGNN H interaction',[('B_TMRH',1),('B_TMR',-1),('B_H',-1),('B',1)])
    contrast('remove T from pilot TGNN',[('B_MR',1),('B_TMR',-1)])
    contrast('remove R from pilot TGNN',[('B_TM',1),('B_TMR',-1)])
    negative={}
    for arm in ['B_TMR','B_TMRH']:
        runs=[r for r in results if r['arm']==arm]
        if runs:
            negative[arm]={mode:{f:{'mean':float(np.mean([r['fixed_checkpoint_controls'][mode][f] for r in runs])),
                                   'sd':float(np.std([r['fixed_checkpoint_controls'][mode][f] for r in runs],ddof=1)) if len(runs)>1 else None}
                                for f in fields} for mode in runs[0]['fixed_checkpoint_controls']}
    return {'metrics':'PR is average precision (AP), not trapezoidal area','means_and_sample_sd':stats,
            'paired_seed_effects':paired,'negative_controls_fixed_threshold':negative,
            'interpretation_scope':'one sampled capture, three epochs; not full ablation matrix or paper reproduction'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default='data/ctu13_verified/scenario06_pilot16384.npz')
    parser.add_argument('--output',default='artifacts/causal_pilot_v1')
    parser.add_argument('--arms',nargs='+',choices=list(ARMS),default=list(ARMS))
    parser.add_argument('--seeds',nargs='+',type=int,default=list(range(5)))
    args=parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    data=PilotData(args.data)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    current={'created_utc':datetime.now(timezone.utc).isoformat(),'python':sys.version,'executable':sys.executable,
             'torch':torch.__version__,'numpy':np.__version__,'sklearn':sklearn.__version__,'platform':platform.platform(),
             'source_hashes':{p:sha256(ROOT/p) for p in SOURCES},'data_manifest':data.meta,'arms':args.arms,'seeds':args.seeds,
             'device':'cpu','threads':1,'epochs':3,'batch_size':256,'cuda_available':torch.cuda.is_available()}
    manifest_path=output/'run_manifest.json'
    if manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        for key in ['source_hashes','data_manifest','arms','seeds']:
            if previous[key]!=current[key]: raise ValueError(f'Resume manifest mismatch: {key}; use a new output directory')
    else: save_json(manifest_path,current)
    save_json(ROOT/'ADAPTED_BASELINE_ARCHITECTURE.json',CausalTGNN(10).adapted_manifest())
    plan=matrix(data); save_json(ROOT/'ABLATION_MATRIX.json',plan)
    controls_path=output/'feature_controls.json'
    controls=json.loads(controls_path.read_text()) if controls_path.exists() else feature_controls(data,output)
    results=[]
    for arm in args.arms:
        for seed in args.seeds:
            result_path=output/f'{arm}_seed{seed}.json'
            r=json.loads(result_path.read_text()) if result_path.exists() else train_one(data,arm,seed,output)
            results.append(r)
            for row in plan['rows']:
                if row['experiment_id']==r['run_id']:
                    row['status']='COMPLETE'; row['reason']='bounded real-data pilot complete'; row['artifact']=str(result_path)
            save_json(ROOT/'ABLATION_MATRIX.json',plan)
            save_json(output/'summary.json',summarize(results,controls))
    print('DONE: '+str(output/'summary.json'),flush=True)


if __name__=='__main__': main()
