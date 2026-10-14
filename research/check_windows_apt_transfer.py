"""Frozen DEDALE-checkpoint compatibility test on Windows-APT CSV telemetry.

No training, threshold fitting, destination scaling, or MITRE/rule inputs.
Metrics measure annotation presence only, never verified attack detection.
"""
import csv
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from causal_data import causal_egos
from dedale_data import features, node_ids
from web_app.backend import backend as b

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'data/Windows-APT 2025 A Dataset for APT-Inspired Attack/combined.csv'
OUT=ROOT/'artifacts/windows_apt_transfer_check'


def run():
    OUT.mkdir(exist_ok=True)
    prep=b.load_preprocessing()
    assert prep['verification']['status']=='passed'
    mean=np.asarray(prep['mean'],dtype=np.float32);std=np.asarray(prep['std'],dtype=np.float32)
    nodes={token:int(i) for i,token in prep['entities'].items()}
    rows=[];excluded=Counter();total=0
    with RAW.open(newline='',encoding='utf-8-sig') as f:
        reader=csv.DictReader(f)
        edkeys=[k for k in reader.fieldnames if k.startswith('_source.data.win.eventdata.')]
        for index,row in enumerate(reader):
            total+=1
            code=(row.get('_source.data.win.system.eventID') or '').strip()
            host=(row.get('_source.agent.name') or '').strip()
            if not code or not host:
                excluded['No Windows event code or host']+=1;continue
            dt=datetime.strptime(row['_source.@timestamp'].strip(),'%b %d, %Y @ %H:%M:%S.%f')
            # The source timezone is unspecified. Preserve displayed clock and
            # elapsed intervals; this UTC placeholder is explicitly recorded.
            t=dt.replace(tzinfo=timezone.utc).timestamp()
            ed={}
            for k in edkeys:
                value=(row.get(k) or '').strip()
                if not value:continue
                name=k.rsplit('.',1)[-1];name=name[0].upper()+name[1:]
                if name in ('Image','ParentImage','TargetFilename','TargetObject','OriginalFileName','CommandLine','ParentCommandLine'):
                    while '\\\\' in value:value=value.replace('\\\\','\\')
                ed[name]=value
            event={'@timestamp':dt.isoformat()+'Z','host':{'name':host},
                   'event':{'code':code,'provider':row.get('_source.data.win.system.providerName','').strip()},
                   'winlog':{'event_id':code,'provider_name':row.get('_source.data.win.system.providerName','').strip(),
                             'channel':row.get('_source.data.win.system.channel','').strip(),'event_data':ed},
                   'message':row.get('_source.data.win.system.message','').strip()}
            s,d=node_ids(event)
            src=nodes.setdefault(s,len(nodes));dst=nodes.setdefault(d,len(nodes))
            annotation=(row.get('_source.rule.mitre.id') or '').strip()
            y=int(bool(json.loads(annotation))) if annotation else 0
            x=(features(event)-mean)/std
            assert np.isfinite(x).all()
            rows.append((t,index,src,dst,y,x,row['_id']))
    rows.sort(key=lambda r:(r[0],r[1]))
    x=np.stack([r[5] for r in rows]);times=np.array([r[0] for r in rows]);src=np.array([r[2] for r in rows]);dst=np.array([r[3] for r in rows]);y=np.array([r[4] for r in rows])
    ids=np.array([r[6] for r in rows]);row_ids=np.array([r[1] for r in rows])
    ego,valid,ages=causal_egos(src,dst,times,4)
    report=dict(status='complete',source=str(RAW),source_sha256=hashlib.sha256(RAW.read_bytes()).hexdigest(),
                rows_total=total,windows_rows_scored=len(rows),excluded=dict(excluded),
                annotated=int(y.sum()),unannotated=int((y==0).sum()),
                protocol='Frozen DEDALE checkpoints; best original validation AP seed per arm; frozen DEDALE validation thresholds; no training or Windows-APT calibration',
                feature_schema='dedale-sysmon72-v1',scaler_verification=prep['verification'],
                limitations=['Wazuh MITRE presence is a weak annotation target, not verified attack/benign truth',
                             'Windows-APT CSV adapter lacks Winlogbeat event action/module/category fields',
                             'Source timezone is unspecified; displayed clocks are preserved with a UTC placeholder',
                             'Wazuh path backslashes are unescaped; event-data sparsity and message serialization differ from DEDALE',
                             'Only Windows event rows with a host and event code are scored; other Wazuh/Linux records excluded',
                             'All eligible rows are unseen-dataset transfer diagnostics, not a Windows-trained held-out test'],models=[])
    print(f'Prepared {len(rows)}/{total} Windows records; annotation positives={y.sum()}',flush=True)
    torch.set_num_threads(4)
    for arm in ('TGNN_Q_A_C','TGNN_Node_Mem_V3_Fixed','TGNN_Node_Mem_V3'):
        row=max((r for r in b.registry() if r['run']=='matched' and r['arm']==arm),key=lambda r:r['validation_pr_ap'])
        net,digest=b.checkpoint_model(row);net.n_nodes=len(nodes);net.reset_state();values=[];started=time.perf_counter()
        with torch.inference_mode():
            for lo in range(0,len(rows),512):
                hi=min(lo+512,len(rows))
                logits,_=net.forward_events(torch.from_numpy(x[ego[lo:hi]]),torch.from_numpy(valid[lo:hi]),torch.from_numpy(ages[lo:hi]),src[lo:hi],dst[lo:hi],times[lo:hi])
                values.append(logits.softmax(-1)[:,1].numpy());net.detach_state()
        p=np.concatenate(values);assert np.isfinite(p).all();pred=p>=row['threshold']
        tn,fp,fn,tp=map(int,confusion_matrix(y,pred,labels=[0,1]).ravel())
        metrics=dict(threshold=row['threshold'],tn=tn,fp=fp,fn=fn,tp=tp,precision=float(precision_score(y,pred,zero_division=0)),recall=float(recall_score(y,pred,zero_division=0)),f1=float(f1_score(y,pred,zero_division=0)),pr_ap=float(average_precision_score(y,p)),roc_auc=float(roc_auc_score(y,p)),alerts=int(pred.sum()),probability_min=float(p.min()),probability_max=float(p.max()),inference_seconds=time.perf_counter()-started)
        report['models'].append(dict(model_id=row['id'],checkpoint_sha256=digest,metrics_against_weak_annotations=metrics))
        np.savez_compressed(OUT/f'{arm}_predictions.npz',probability=p,annotation=y,event_ids=ids,row_ids=row_ids)
        print(row['id'],json.dumps(metrics),flush=True)
        (OUT/'report.json').write_text(json.dumps(report,indent=2))
    return report

if __name__=='__main__':run()
