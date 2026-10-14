"""Longer-training and count robustness study; two encoder architectures."""
import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from experiment import (TokenBank, FieldCMS, prepare_data, load_data, pretrain, batches,
    score, metrics, bootstrap_delta, seed_all, save_json, digest, sync, timed)
from rarity_followup import FollowupConfig, count_variants, initialize, family_metrics


@dataclass
class StabilityConfig(FollowupConfig):
    root: str = '/content/tgdetect_rarity_stability'
    epochs: int = 16
    budgets: tuple = (8, 12, 16)
    split_seeds: tuple = (20261015, 20261016)
    timing_repeats: int = 20


def fingerprints(rows):
    """Hash each distinct canonical field once; keep all four uint64 hashes."""
    cache = {}; output = np.empty((*rows.shape, 4), dtype=np.uint64)
    for i, record in enumerate(rows):
        for j, raw in enumerate(record):
            if raw not in cache:
                # Same serialization and BLAKE2 as original FieldCMS.
                key, value = json.loads(raw)
                text = json.dumps([key,value],sort_keys=True,separators=(',',':'))
                cache[raw] = np.frombuffer(hashlib.blake2b(text.encode(),digest_size=32).digest(),dtype='<u8').copy()
            output[i,j] = cache[raw]
    return output


def build_table(hashes, width):
    table=np.zeros((4,width),dtype=np.int64)
    positions=(hashes % np.uint64(width)).astype(np.int64)
    for depth in range(4):
        np.add.at(table[depth],positions[...,depth].ravel(),1)
    return table


def lookup(table, hashes):
    positions=(hashes % np.uint64(table.shape[1])).astype(np.int64)
    result=table[np.arange(4),positions].min(-1)
    return result


def gpu_lookup(table, positions):
    counts=table[torch.arange(4,device=table.device),positions].amin(-1)
    rarity=counts.float().log1p()
    rarity[:,0]=0
    return rarity


def count_conditions(data, reference, cfg):
    started=time.perf_counter()
    ref=fingerprints(reference)
    hashes={split:fingerprints(values['field_identity']) for split,values in data.items()}
    compile_seconds=time.perf_counter()-started
    clean=build_table(ref,cfg.cms_width)
    conditions={'clean':clean,'stale_half_reference':build_table(ref[:max(1,len(ref)//2)],cfg.cms_width),
                'collision_small_table':build_table(ref,2048)}
    polluted=clean.copy()
    # Simulated duplicated ingestion: repeat a random benign 30% of reference.
    rng=np.random.default_rng(cfg.split_seed+99)
    ix=rng.choice(len(ref),max(1,int(.3*len(ref))),replace=False)
    polluted += build_table(ref[ix],cfg.cms_width)
    conditions['duplicate_updates_30pct']=polluted
    results={}
    for name,table in conditions.items():
        results[name]={}
        for split,values in data.items():
            rarity=np.log1p(lookup(table,hashes[split])).astype(np.float32)
            rarity[:,0]=0
            results[name][split]={**values,'rarity':rarity}
    results['missing_fields_10pct']={}
    for split,values in results['clean'].items():
        rng=np.random.default_rng(cfg.split_seed+len(split)+333)
        rarity=values['rarity'].copy();mask=rng.random(rarity.shape)<.1
        rarity[mask]=0
        results['missing_fields_10pct'][split]={**values,'rarity':rarity}
    audit={'fingerprint_preparation_seconds':compile_seconds,
           'fingerprint_bytes':sum(h.nbytes for h in hashes.values())+ref.nbytes,
           'table_bytes':clean.nbytes,'frozen_reference_size':len(ref),
           'synthetic_stress_conditions':list(results),
           'missing_semantics':'zero log-count conflates unavailable counts and never-seen values; stress test only'}
    return results,hashes,clean,audit


def count_benchmark(cfg, data, hashes, table, model):
    size=min(128,len(data['test']['ids']))
    raw=data['test']['field_identity'][:size];hp=hashes['test'][:size]
    sketch=FieldCMS(table.shape[1]);sketch.table[:]=table
    def python_counts():
        counts=np.array([[sketch.read(*json.loads(pair)) for pair in row] for row in raw])
        return counts
    result={}
    result['python_hash_and_lookup']=timed(python_counts,cfg,cfg.timing_repeats)
    result['numpy_prehashed_lookup']=timed(lambda:lookup(table,hp),cfg,cfg.timing_repeats)
    result['warm_identity_fingerprint_lookup']=True
    device=cfg.device
    positions=torch.as_tensor((hp % np.uint64(table.shape[1])).astype(np.int64),device=device)
    counts=torch.as_tensor(table,device=device)
    ids=torch.as_tensor(data['test']['ids'][:size],device=device)
    r=gpu_lookup(counts,positions)
    reference=torch.as_tensor(np.log1p(lookup(table,hp)).astype(np.float32),device=device);reference[:,0]=0
    torch.testing.assert_close(r,reference,atol=1e-6,rtol=1e-6)
    result['device_prehashed_lookup']=timed(lambda:gpu_lookup(counts,positions),cfg,cfg.timing_repeats)
    result['encoder_prepared_rarity']=timed(lambda:model.encode(ids,r),cfg,cfg.timing_repeats)
    result['device_count_plus_encoder']=timed(lambda:model.encode(ids,gpu_lookup(counts,positions)),cfg,cfg.timing_repeats)
    host_positions=torch.as_tensor((hp % np.uint64(table.shape[1])).astype(np.int64))
    host_ids=torch.as_tensor(data['test']['ids'][:size])
    def staged():
        p=host_positions.to(device);identity=host_ids.to(device)
        return model.encode(identity,gpu_lookup(counts,p))
    result['host_staging_count_plus_encoder']=timed(staged,cfg,cfg.timing_repeats)
    def deployed():
        if model.options.get('rarity'):
            return staged()
        return model.encoder(host_ids.to(device))
    result['deployed_encoder_pipeline']=timed(deployed,cfg,cfg.timing_repeats)
    # Cold preprocessing is reported separately; no unseen-text work is hidden.
    cold=[]
    for _ in range(cfg.timing_repeats):
        t=time.perf_counter();fingerprints(raw);cold.append((time.perf_counter()-t)*1000)
    result['cold_field_fingerprints_ms']={'p50':float(np.median(cold)),'samples':cold}
    result['batch_records']=size
    result['scope']='count+encoder only; no ingestion, service scoring, arrival wait, or state updates'
    return result


def train_stability(cfg, bank, data, conditions, hashes, table, classes, tokens, arm, seed):
    out=Path(cfg.root)/'results'/f'{arm}_s{seed}';out.mkdir(parents=True,exist_ok=False)
    model=initialize(bank,classes,arm,seed,tokens,cfg.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    clean=conditions['clean'];best=float('inf');best_state=None;best_epoch=0;history=[];results=[]
    if cfg.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    for epoch in range(1,max(cfg.budgets)+1):
        model.train();total=0.;n=0
        order=np.random.default_rng(seed*1000+epoch-1).permutation(len(data['train']['ids']))
        for batch in batches(clean['train'],cfg,order):
            loss=F.cross_entropy(model(batch['ids'],batch['rarity']),batch['service'])
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach())*len(batch['ids']);n+=len(batch['ids'])
        val=float(score(model,clean['val'],cfg,[],False)[:,0].mean())
        history.append({'epoch':epoch,'train_ce':total/n,'benign_val_ce':val})
        if val<best:
            best=val;best_epoch=epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch not in cfg.budgets:continue
        # Evaluate the best benign checkpoint available within THIS budget.
        current={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state);model.eval()
        prefix=out/f'budget_{epoch}';prefix.mkdir()
        torch.save({'model':best_state,'epoch':best_epoch},prefix/'best.pt')
        cal=score(model,clean['cal'],cfg,[],False)[:,0];half=len(cal)//2
        y=data['test']['y'];families=data['test']['family'];records={}
        for name,values in conditions.items():
            test=score(model,values['test'],cfg,[],False)[:,0]
            shifted_cal=cal if name=='clean' else score(model,values['cal'],cfg,[],False)[:,0]
            # Report unchanged deployed threshold and benign-only recalibration separately.
            fixed=family_metrics(y,families,test,cal[half:])
            recalibrated=family_metrics(y,families,test,shifted_cal[half:])
            records[name]={'fixed_clean_threshold':fixed,'benign_recalibrated':recalibrated}
            np.savez_compressed(prefix/f'{name}_scores.npz',y=y,family=families,service=test,
                                clean_calibration=cal,condition_calibration=shifted_cal)
        timing=count_benchmark(cfg,data,hashes,table,model)
        sync(cfg)
        result={'arm':arm,'seed':seed,'split_seed':cfg.split_seed,'budget_epochs':epoch,
                'checkpoint_epoch':best_epoch,'history':history.copy(),'conditions':records,
                'count_latency':timing,'parameters':sum(p.numel() for p in model.parameters()),
                'peak_cuda_bytes':torch.cuda.max_memory_allocated() if cfg.device.startswith('cuda') else None,
                'elapsed_seconds_including_previous_evaluation':time.perf_counter()-start,
                'checkpoint_sha256':digest(prefix/'best.pt')}
        save_json(prefix/'result.json',result);results.append(result)
        print(arm,seed,'budget',epoch,'selected',best_epoch,'AP',records['clean']['fixed_clean_threshold']['ap'],flush=True)
        model.load_state_dict(current)
    return results


def run_stability(cfg=StabilityConfig(), frame=None):
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('Select GPU in Colab.')
    root=Path(cfg.root);root.mkdir(parents=True,exist_ok=True)
    if (root/'runs.csv').exists():raise FileExistsError('Use a fresh output root.')
    torch.set_num_threads(2)
    if cfg.device.startswith('cuda'):
        total=torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.,cfg.memory_budget_gib*2**30/total))
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from dataclasses import asdict
    save_json(root/'config.json',asdict(cfg));save_json(root/'environment.json',{
        'gpu':torch.cuda.get_device_name() if cfg.device.startswith('cuda') else 'CPU smoke',
        'torch':torch.__version__,'source_sha256':digest(__file__)})
    frame=load_data(cfg) if frame is None else frame;rows=[];paired=[]
    for split in cfg.split_seeds:
        scfg=copy.copy(cfg);scfg.split_seed=split;scfg.root=str(root/f'split_{split}')
        data,keys,specs,classes,audit=prepare_data(scfg,frame)
        variants,_,reference,count_audit=count_variants(data,scfg);data=variants['frozen_log']
        conditions,hashes,table,vector_audit=count_conditions(data,reference,scfg)
        for s in data:
            np.testing.assert_allclose(data[s]['rarity'],conditions['clean'][s]['rarity'],atol=1e-6)
        save_json(Path(scfg.root)/'count_protocol.json',{**count_audit,**vector_audit})
        # SGNS uses optimizer records only, matching prior follow-up protocol.
        from orthrus_comparison.universal import value_parts
        corpus=[]
        for record in data['train']['field_identity']:
            words=[]
            for raw in record:
                _,value=json.loads(raw);kind,parts,_=value_parts(value)
                if kind not in ('number','bool','missing'):words.extend(parts)
            if len(words)>1:corpus.append(words)
        p=Path(scfg.root)/'benign_paths.jsonl';p.write_text(''.join(json.dumps(w)+'\n' for w in corpus))
        manifest=Path(scfg.root)/'manifest.json';m=json.loads(manifest.read_text());m['checksums']['benign_paths.jsonl']=digest(p);save_json(manifest,m)
        bank=TokenBank(Path(scfg.root)/'tokens')
        if cfg.device.startswith('cuda'):bank.to_device(cfg.device)
        initial=Path(scfg.root)/'initial_embeddings.pt'
        info=pretrain(scfg.root,initial,epochs=1,batch=2048,projection_steps=30,seed=split,device=cfg.device)
        save_json(Path(scfg.root)/'shared_sgns.json',info)
        tokens=torch.load(initial,map_location='cpu',weights_only=False)['tokens']
        for seed in cfg.seeds:
            for arm in (['old','frozen_log_film'] if seed%2==0 else ['frozen_log_film','old']):
                try:
                    results=train_stability(scfg,bank,data,conditions,hashes,table,classes,tokens,arm,seed)
                except Exception as exc:
                    save_json(root/'failure.json',{'split_seed':split,'arm':arm,'seed':seed,'error':repr(exc)})
                    raise
                for r in results:
                    for condition,values in r['conditions'].items():
                        for policy,metric in values.items():
                            op=metric['fpr_target_0.01'];f=metric['family_recall']
                            rows.append({'arm':arm,'seed':seed,'split_seed':split,'budget':r['budget_epochs'],
                                'checkpoint_epoch':r['checkpoint_epoch'],'condition':condition,'threshold_policy':policy,
                                'ap':metric['ap'],'f1':op['f1'],'fpr':op['test_fpr'],'non_neptune_recall':metric['non_neptune_recall'],
                                'back_recall':f.get('back.',{}).get('recall'),'warezclient_recall':f.get('warezclient.',{}).get('recall'),
                                'deployed_encoder_ms':r['count_latency']['deployed_encoder_pipeline']['p50_ms']})
            for budget in cfg.budgets:
                base=Path(scfg.root)/'results'
                a=np.load(base/f'old_s{seed}/budget_{budget}/clean_scores.npz')
                b=np.load(base/f'frozen_log_film_s{seed}/budget_{budget}/clean_scores.npz')
                paired.append({'split_seed':split,'seed':seed,'budget':budget,
                               **bootstrap_delta(a['y'],a['service'],b['service'],cfg.bootstrap)})
    df=pd.DataFrame(rows);df.to_csv(root/'runs.csv',index=False)
    save_json(root/'paired_comparisons.json',paired)
    summary=df.groupby(['arm','budget','condition','threshold_policy'])[['ap','f1','fpr','non_neptune_recall','back_recall','warezclient_recall']].agg(['mean','std'])
    summary.to_csv(root/'summary.csv');return df,summary
