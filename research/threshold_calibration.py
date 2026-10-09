"""Frozen post-hoc threshold calibration for the Windows-APT weak-label study.

This script never calls an optimizer.  It regenerates frozen-model scores,
checks them against the official prediction files, learns thresholds only on
validation calibration subsets, and evaluates those frozen thresholds once on
the untouched test partition.
"""
from pathlib import Path
import hashlib, json, math
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score)

from windows_apt_models import make_model, ARMS
from windows_apt_data import digest

ROOT=Path('artifacts/windows_apt_v1')
RESULTS=ROOT/'remote/results'
OUT=ROOT/'threshold_calibration'
SEEDS=range(5)
CONTROL_FILES={'Logistic Regression':ROOT/'control_logistic_predictions.npz',
               'Histogram Gradient Boosting':ROOT/'control_hist_gradient_boosting_predictions.npz'}
MODELS={'GNN':'GNN_H','TGNN hybrid':'TGNN_Q_A_C',**{k:k for k in CONTROL_FILES}}
CAL_SEED=20260926
N_BOOT=2000

def metric(y,p,t):
    pred=np.asarray(p)>=float(t); tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'f1':float(f1_score(y,pred,zero_division=0)),
      'precision':float(precision_score(y,pred,zero_division=0)),
      'recall':float(recall_score(y,pred,zero_division=0)),
      'pr_auc':float(average_precision_score(y,p)), 'roc_auc':float(roc_auc_score(y,p)),
      'fpr':float(fp/max(tn+fp,1)), 'specificity':float(tn/max(tn+fp,1)),
      'fnr':float(fn/max(fn+tp,1)), 'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}

def sweep(y,p,grid=None):
    """Return one row per threshold; exact candidate thresholds are score values."""
    if grid is None:
        grid=np.unique(np.r_[0.,1.,np.asarray(p,dtype=float)])
    y=np.asarray(y); p=np.asarray(p); pos=int(y.sum()); neg=len(y)-pos
    rows=[]
    for t in grid:
        pred=p>=float(t); tp=int(np.sum(pred & (y==1))); fp=int(np.sum(pred & (y==0))); fn=pos-tp; tn=neg-fp
        prec=tp/max(tp+fp,1); rec=tp/max(pos,1); f1=2*prec*rec/max(prec+rec,1e-15)
        rows.append({'threshold':float(t),'f1':f1,'precision':prec,'recall':rec,'fpr':fp/max(neg,1),'specificity':tn/max(neg,1),'fnr':fn/max(pos,1),'tn':tn,'fp':fp,'fn':fn,'tp':tp})
    ap=float(average_precision_score(y,p)); auc=float(roc_auc_score(y,p))
    for r in rows:r['pr_auc']=ap;r['roc_auc']=auc
    return pd.DataFrame(rows)

def choose(df,criterion='f1'):
    vals=df[criterion].to_numpy()
    # Largest threshold resolves ties, matching the official selector's final tie.
    best=np.flatnonzero(np.isclose(vals,vals.max(),rtol=0,atol=1e-14))
    return float(df.iloc[best[-1]]['threshold'])

def fast_f1_threshold(y,p,grid):
    y=np.asarray(y);p=np.asarray(p); pred=p[None,:]>=np.asarray(grid)[:,None]
    tp=np.sum(pred & (y[None,:]==1),axis=1); fp=np.sum(pred & (y[None,:]==0),axis=1); fn=np.sum(y==1)-tp
    pr=tp/np.maximum(tp+fp,1); re=tp/np.maximum(tp+fn,1); f=2*pr*re/np.maximum(pr+re,1e-15)
    return float(grid[np.flatnonzero(np.isclose(f,f.max(),rtol=0,atol=1e-14))[-1]])

def select_secondary(y,p):
    d=sweep(y,p, None if len(y)<=600 else np.linspace(0,1,1001)); out={'f1_optimal':choose(d,'f1')}
    out['youden_j']=choose(pd.DataFrame({'threshold':d.threshold,'youden':d.recall-d.fpr}),'youden')
    for target in (.90,.95,.99):
        ok=d[d.recall>=target]; out[f'recall_{int(target*100)}']=float(ok.threshold.max()) if len(ok) else None
    ok=d[d.precision>=.95]
    if len(ok):
        mx=ok.recall.max(); out['precision_95']=float(ok[ok.recall==mx].threshold.max())
    else: out['precision_95']=None
    return out

def frozen_scores(arm,seed,data):
    ck=RESULTS/f'{arm}_seed{seed}.pt'; official=RESULTS/f'{arm}_seed{seed}_predictions.npz'
    z=np.load(data,allow_pickle=False); x=torch.from_numpy(z['x']); y=z['y']; ego=torch.from_numpy(z['ego']); valid=torch.from_numpy(z['valid']); ages=torch.from_numpy(z['ages'])
    windows=x[ego]
    state=torch.load(ck,map_location='cpu',weights_only=True)
    model=make_model(arm,int(x.shape[1])); model.load_state_dict(state['state_dict'],strict=True); model.eval()
    logits=[]
    with torch.no_grad():
        for lo in range(0,len(y),4096):
            q=model.forward_events(windows[lo:lo+4096],valid[lo:lo+4096],ages[lo:lo+4096],None,None,None)[0]
            logits.append(q.numpy())
    logits=np.concatenate(logits); logit=(logits[:,1]-logits[:,0]); prob=torch.softmax(torch.from_numpy(logits),dim=1)[:,1].numpy()
    op=np.load(official,allow_pickle=False)['probability']
    delta=float(np.max(np.abs(prob-op)))
    if delta>2e-5: raise AssertionError(f'{arm} seed{seed} frozen score mismatch {delta}')
    return logit,prob,y,z['row_ids'],delta

def subsets(y,seed):
    v0=61206; v1=81610; vy=y[v0:v1]; pos=np.flatnonzero(vy==1); neg=np.flatnonzero(vy==0)
    rng=np.random.default_rng(CAL_SEED+seed); p=rng.choice(pos,50,replace=False); n50=rng.choice(neg,50,replace=False); n250=rng.choice(neg,250,replace=False)
    return {'50pos_50neg':np.r_[v0+p,v0+n50], '50pos_250neg':np.r_[v0+p,v0+n250], '50pos_allneg':np.r_[v0+p,v0+neg]}

def main():
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    OUT.mkdir(parents=True,exist_ok=True); data=ROOT/'prepared/data.npz'; z=np.load(data,allow_pickle=False); y=z['y']; ids=z['row_ids']; v0,v1=int(z['train_end']),int(z['val_end']); t0=len(y) if False else v1
    # Reproducibility and audit facts are written before any test metric is computed.
    manifest={'dataset_sha256':digest(data),'data_manifest_sha256':digest(ROOT/'prepared/data_manifest.json'), 'label_scope':'WEAK_LABEL_WAZUH_MITRE_PRESENCE','calibration_seed':CAL_SEED,'bootstrap_replicates':N_BOOT,'calibration_partition':[v0,v1],'test_partition':[v1,len(y)],'checkpoint_sha256':{},'score_replay_max_abs_error':{}}
    rows_cal=[]; rows_test=[]; all_sweeps=[]; stability=[]; boots=[]; original=[]; calibrated=[]; threshold_json={}
    score_cache={}
    for label,arm in MODELS.items():
        for seed in (SEEDS if label in ('GNN','TGNN hybrid') else [0]):
            if label in ('GNN','TGNN hybrid'):
                logit,p,yy,row_ids,delta=frozen_scores(arm,seed,data); manifest['checkpoint_sha256'][f'{label}_seed{seed}']=digest(RESULTS/f'{arm}_seed{seed}.pt'); manifest['score_replay_max_abs_error'][f'{label}_seed{seed}']=delta
                official=json.loads((RESULTS/f'{arm}_seed{seed}.json').read_text()); original.append({'model':label,'seed':seed,'original_threshold':official['threshold'],**{f'original_{k}':v for k,v in official['test'].items()}})
            else:
                q=np.load(CONTROL_FILES[label],allow_pickle=False);p=q['probability'];yy=q['y'];row_ids=q['row_ids'];logit=np.log(np.clip(p,1e-7,1-1e-7)/np.clip(1-p,1e-7,1)); seed=0
            score_cache[(label,seed)]={'logit':logit,'p':p,'y':yy,'ids':row_ids}
            for pool,ix in subsets(yy,seed).items():
                cy=yy[ix]; cp=p[ix]; threshold=choose(sweep(cy,cp),'f1'); secondary=select_secondary(cy,cp)
                threshold_json[f'{label}_seed{seed}_{pool}']={'threshold':threshold,**secondary,'n_positive':int(cy.sum()),'n_negative':int((cy==0).sum())}
                d=sweep(cy,cp, None if len(ix)<=600 else np.linspace(0,1,1001)); d['model']=label;d['seed']=seed;d['calibration_pool']=pool;all_sweeps.append(d)
                for j in ix: rows_cal.append({'model':label,'seed':seed,'calibration_pool':pool,'sample_id':int(row_ids[j]),'row_index':int(j),'label':int(yy[j]),'raw_logit':float(logit[j]),'probability':float(p[j]),'split':'validation'})
                test_slice=np.arange(v1,len(yy)); tm=metric(yy[test_slice],p[test_slice],threshold)
                calibrated.append({'model':label,'seed':seed,'calibration_pool':pool,'calibrated_threshold':threshold,**{f'test_{k}':v for k,v in tm.items()}})
                stability.append({'model':label,'seed':seed,'calibration_pool':pool,'threshold':threshold,'calibration_f1':metric(cy,cp,threshold)['f1']})
                if pool=='50pos_50neg':
                    rng=np.random.default_rng(CAL_SEED+100000+seed+(0 if label=='GNN' else 100 if label=='TGNN hybrid' else 200))
                    for b in range(N_BOOT):
                        bi=rng.integers(0,len(ix),len(ix)); by=cy[bi]; bp=cp[bi]; bt=fast_f1_threshold(by,bp,np.linspace(0,1,1001)); boots.append({'model':label,'seed':seed,'bootstrap':b,'threshold':bt})
            for j in np.arange(v1,len(yy)):
                rows_test.append({'model':label,'seed':seed,'sample_id':int(row_ids[j]),'row_index':int(j),'label':int(yy[j]),'raw_logit':float(logit[j]),'probability':float(p[j]),'split':'test'})
    pd.DataFrame(rows_cal).to_parquet(OUT/'calibration_predictions.parquet',index=False); pd.DataFrame(rows_test).to_parquet(OUT/'test_predictions.parquet',index=False)
    pd.concat(all_sweeps,ignore_index=True).to_csv(OUT/'threshold_sweep.csv',index=False);pd.DataFrame(stability).to_csv(OUT/'threshold_stability.csv',index=False);pd.DataFrame(boots).to_csv(OUT/'bootstrap_thresholds.csv',index=False)
    pd.DataFrame(original).to_csv(OUT/'metrics_original.csv',index=False);pd.DataFrame(calibrated).to_csv(OUT/'metrics_calibrated.csv',index=False)
    # Primary summary for the two neural models; controls remain in calibrated table.
    cal=pd.DataFrame(calibrated); org=pd.DataFrame(original); sm=[]
    for model in ('GNN','TGNN hybrid'):
        a=cal[(cal.model==model)&(cal.calibration_pool=='50pos_50neg')].merge(org[org.model==model],on=['model','seed'])
        for k in ('f1','precision','recall','fpr'):
            sm.append({'model':model,'metric':k,'original_mean':a[f'original_{k}'].mean(),'original_sd':a[f'original_{k}'].std(ddof=1),'calibrated_mean':a[f'test_{k}'].mean(),'calibrated_sd':a[f'test_{k}'].std(ddof=1),'delta_mean':(a[f'test_{k}']-a[f'original_{k}']).mean(),'delta_sd':(a[f'test_{k}']-a[f'original_{k}']).std(ddof=1)})
    pd.DataFrame(sm).to_csv(OUT/'metrics_summary.csv',index=False)
    # Bootstrap CI and threshold stability summary.
    b=pd.DataFrame(boots); stab=[]
    for (m,pool),g in pd.DataFrame(stability).groupby(['model','calibration_pool']):
        x=g.threshold; stab.append({'model':m,'calibration_pool':pool,'mean':x.mean(),'median':x.median(),'sd':x.std(ddof=1),'min':x.min(),'max':x.max(),'cv':x.std(ddof=1)/x.mean() if x.mean() else None})
    bs=[]
    for (m,s),g in b.groupby(['model','seed']): bs.append({'model':m,'seed':s,'median':g.threshold.median(),'ci95_low':g.threshold.quantile(.025),'ci95_high':g.threshold.quantile(.975)})
    pd.DataFrame(stab).to_csv(OUT/'threshold_stability.csv',index=False);pd.DataFrame(bs).to_csv(OUT/'bootstrap_threshold_summary.csv',index=False)
    (OUT/'thresholds_per_seed.json').write_text(json.dumps(threshold_json,indent=2));(OUT/'audit_manifest.json').write_text(json.dumps(manifest,indent=2))
    make_plots(cal,rows_cal,rows_test)
    report(manifest,org,cal,pd.DataFrame(stab),pd.DataFrame(bs),sm)

def make_plots(cal,rows_cal,rows_test):
    import matplotlib.pyplot as plt
    c=pd.DataFrame(rows_cal);t=pd.DataFrame(rows_test); fig,ax=plt.subplots(1,2,figsize=(12,4))
    for m,g in c[c.calibration_pool=='50pos_50neg'].groupby('model'):
        for lab,col in [(1,'tab:red'),(0,'tab:blue')]: ax[0].hist(g[g.label==lab].probability,bins=30,alpha=.25,label=f'{m} label {lab}',color=col,density=True)
    ax[0].set_xlabel('calibration probability');ax[0].set_ylabel('density');ax[0].legend(fontsize=7);ax[0].set_title('Frozen score distributions')
    for m,g in cal[cal.calibration_pool=='50pos_50neg'].groupby('model'):
        ax[1].plot(g.test_fpr,g.test_recall,'o',label=m)
    ax[1].set_xlabel('test FPR');ax[1].set_ylabel('test recall');ax[1].legend();fig.tight_layout();fig.savefig(OUT/'score_distributions.png',dpi=160);plt.close(fig)
    # Curves from the stored sweep, with calibration operating points.
    sw=pd.read_csv(OUT/'threshold_sweep.csv'); fig,axs=plt.subplots(1,3,figsize=(15,4))
    for (m,pool),g in sw.groupby(['model','calibration_pool']):
        if pool!='50pos_50neg': continue
        axs[0].plot(g.threshold,g.f1,label=m);axs[1].plot(g.threshold,g.precision,label=m);axs[1].plot(g.threshold,g.recall,'--');axs[2].plot(g.threshold,g.fpr,label=m)
    for a,ttl in zip(axs,['F1','Precision / recall','FPR']):a.set_title(ttl);a.set_xlabel('threshold');a.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'threshold_vs_f1.png',dpi=160);fig.savefig(OUT/'threshold_vs_precision_recall.png',dpi=160);plt.close(fig)
    from sklearn.metrics import ConfusionMatrixDisplay
    fig,axs=plt.subplots(1,2,figsize=(8,3))
    for ax,m in zip(axs,['GNN','TGNN hybrid']):
        g=cal[(cal.model==m)&(cal.calibration_pool=='50pos_50neg')]; y=np.r_[g.test_tn.sum()*0] if False else None
        cm=np.array([[g.test_tn.sum(),g.test_fp.sum()],[g.test_fn.sum(),g.test_tp.sum()]])
        ConfusionMatrixDisplay(cm).plot(ax=ax,colorbar=False);ax.set_title(m+' (sum of seeds)')
    fig.tight_layout();fig.savefig(OUT/'confusion_matrices.png',dpi=160);plt.close(fig)

def report(manifest,org,cal,stab,bs,summary):
    lines=['# Frozen threshold calibration report','',
      'This experiment keeps every neural-network checkpoint frozen. It predicts Wazuh MITRE annotation presence (a weak label), not verified attack/benign ground truth.', '',
      '## Protocol','',f"- Dataset SHA-256: `{manifest['dataset_sha256']}`; validation calibration rows `{manifest['calibration_partition']}`; untouched test rows `{manifest['test_partition']}`.",f"- Primary calibration: deterministic 50 positive + 50 negative validation events; sensitivity pools are 50+250 and 50+all validation negatives. Seed `{CAL_SEED}`.",'- Threshold selection uses calibration-set maximum F1. Test labels are only read after the threshold is frozen.','- Official saved probabilities were replayed from each checkpoint with the exact model/data path; maximum probability replay error is recorded in `audit_manifest.json`.','- PR-AUC and ROC-AUC do not depend on threshold and are reported unchanged in the per-seed metric table.','']
    lines += ['## Primary neural-model comparison (50+50)','', '|Model|Original test F1 mean±SD|Calibrated test F1 mean±SD|Δ F1 mean±SD|Δ recall mean±SD|Δ FPR mean±SD|','|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['metric']=='f1': f=r
        if r['metric']=='recall': q=r
        if r['metric']=='fpr': z=q2=r
        if r['metric']=='f1': lines.append(f"|{r['model']}|{r['original_mean']:.5f} ± {r['original_sd']:.5f}|{r['calibrated_mean']:.5f} ± {r['calibrated_sd']:.5f}|{r['delta_mean']:.5f} ± {r['delta_sd']:.5f}|pending|pending|")
    # Replace placeholders with paired recall/FPR deltas.
    for i, r in enumerate(summary):
        if r['metric']=='f1':
            rec=next(x for x in summary if x['model']==r['model'] and x['metric']=='recall')
            fpr=next(x for x in summary if x['model']==r['model'] and x['metric']=='fpr')
            target='pending|pending'
            for j,line in enumerate(lines):
                if line.startswith('|'+r['model']+'|'):
                    lines[j]=line.replace(target,f"{rec['delta_mean']:.5f} ± {rec['delta_sd']:.5f}|{fpr['delta_mean']:.5f} ± {fpr['delta_sd']:.5f}")
                    break
    lines += ['', 'The complete per-seed values, confusion counts, PR-AUC, ROC-AUC, calibration-pool sensitivity, threshold sweeps, and bootstrap intervals are in the CSV/Parquet artifacts in this directory.', '', '## Interpretation', '', 'The conclusion must be based on the untouched-test rows in `metrics_calibrated.csv`, not on the calibration sweep. Threshold calibration can change F1, precision, recall, and FPR; it cannot improve PR-AUC or ROC-AUC when scores are unchanged. Because this is a weak-label experiment, any gain means better prediction of annotation presence only.']
    (OUT/'THRESHOLD_CALIBRATION_REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__': main()
