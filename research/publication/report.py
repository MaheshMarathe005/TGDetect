"""Build manuscript tables and vector figures exclusively from saved results."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parent / 'research paper/publication'
LABELS={'V3':'V3-CNN','V3_short_attention':'V3-AttnShort','ORTHRUS':'ORTHRUS adapter',
        'KAIROS':'KAIROS adapter','GraphSAGE':'GraphSAGE','V3_no_pair':'V3 w/o pair',
        'V3_no_long_read':'V3 w/o long read'}


def results():
    rows=[]
    for arm in ('V3','ORTHRUS'):
        p=ROOT/'artifacts/v3_vs_orthrus'/f'{arm}_accuracy.json'
        if p.exists():rows.append((arm,0,5,json.loads(p.read_text()),str(p.relative_to(ROOT))))
    for p in sorted((ROOT/'artifacts/publication').glob('*.json')):
        value=json.loads(p.read_text());job=value.get('job',{})
        if value.get('status')=='complete' and 'metrics' in value:
            rows.append((job['arm'],job['seed'],job['epochs'],value,str(p.relative_to(ROOT))))
    return rows


def savefig(fig,name):
    for suffix in ('pdf','svg','png'):
        fig.savefig(OUT/'figures'/f'{name}.{suffix}',bbox_inches='tight',dpi=220)
    plt.close(fig)


def architecture():
    fig,ax=plt.subplots(figsize=(11,6.2));ax.set(xlim=(0,11),ylim=(0,6.2));ax.axis('off')
    positions={
        'input':(.15,4.9,2.35,.9,'Chronological event\n(u, v, time, attributes)'),
        'token':(2.85,4.9,2.35,.9,'Offline identity IDs\nTyped KV encoder, 64-D'),
        'window':(.15,3.3,2.35,1.05,'8-slot endpoint rings\n4+4 quota + current query\nStrictly causal planner'),
        'active':(2.85,3.3,2.35,1.05,'Two event-line GATs\nDilated causal CNN\nAttention pooling, 128-D'),
        'short':(5.55,3.3,2.35,1.05,'8 compressed slots/node\nNovelty + rarity admission\nAttention-only readout'),
        'fusion':(5.55,1.65,2.35,.9,'Gated active/short fusion\n128-D context'),
        'memory':(2.85,.1,2.35,1.05,'Parallel affine node scan\nFast/slow state, 128-D\nRead before event update'),
        'head':(8.25,1.65,2.35,.9,'Relation reconstruction\n+ pair plausibility score'),
        'calib':(8.25,.1,2.35,1.05,'Benign score calibration\n15-minute node evidence\nNode scores and alerts'),
        'pair':(5.55,4.9,2.35,.9,'Endpoint identity vectors\nBilinear pair head'),
    }
    for key,(x,y,w,h,label) in positions.items():
        color='#e9f2f7' if key not in ('short','memory') else '#e7f2e3'
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04',fc=color,ec='#304558',lw=1.1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9)
    def arrow(a,b,side='horizontal',style='-'):
        x,y,w,h,_=positions[a];X,Y,W,H,_=positions[b]
        if side=='vertical':start=(x+w/2,y);end=(X+W/2,Y+H)
        else:start=(x+w,y+h/2);end=(X,Y+H/2)
        ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#405567',lw=1.2,linestyle=style))
    for a,b in [('input','token'),('window','active'),('active','short'),('fusion','head'),('token','pair')]:arrow(a,b)
    for a,b in [('input','window'),('token','active'),('short','fusion'),('head','calib')]:arrow(a,b,'vertical')
    arrow('active','fusion');arrow('fusion','memory','vertical')
    def route(points):
        ax.plot(*zip(*points[:-1]),color='#405567',lw=1.2)
        ax.annotate('',xy=points[-1],xytext=points[-2],arrowprops=dict(arrowstyle='->',color='#405567',lw=1.2))
    route([(5.2,.625),(8.05,.625),(8.05,1.95),(8.25,1.95)])
    route([(7.9,5.35),(9.425,5.35),(9.425,2.55)])
    ax.text(.18,2.45,'Inference: frozen encoder lookup\nTraining: differentiable re-encoding',fontsize=8.5,color='#405567')
    ax.text(.18,.45,'V3-CNN comparator:\nshort readout also uses CNN.',fontsize=8.5,color='#405567')
    savefig(fig,'architecture')


def build():
    (OUT/'figures').mkdir(parents=True,exist_ok=True);rows=results();flat=[]
    for arm,seed,epochs,r,path in rows:
        m=r['metrics'];total=sum(m[k] for k in ('tp','tn','fp','fn'))
        flat.append(dict(arm=arm,seed=seed,epochs=epochs,**m,accuracy=(m['tp']+m['tn'])/total,
            parameters=r['parameter_count'],training_seconds=sum(x['seconds'] for x in r['history']),
            total_seconds=r.get('total_seconds'),peak_cuda_bytes=r.get('peak_cuda_bytes'),source=path))
    (OUT/'measured_results.json').write_text(json.dumps(flat,indent=2))
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with (OUT/'measured_results.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(flat)
    groups={}
    for r in flat:groups.setdefault((r['arm'],r['epochs']),[]).append(r)
    lines=[]
    for (arm,epochs),rr in groups.items():
        def stat(key):
            values=[x[key] for x in rr];mean=np.mean(values)
            return f'{mean:.4f}' if len(values)==1 else f'{mean:.4f} $\\pm$ {np.std(values,ddof=1):.4f}'
        lines.append(f"{LABELS[arm]} & {epochs} & {len(rr)} & {stat('pr_ap')} & {stat('f1')} & {stat('precision')} & {stat('recall')} \\\\")
    (OUT/'results_table.tex').write_text('\\begin{tabular}{lrrllll}\\toprule\nDetector & Epochs & $n$ & AP & F1 & Precision & Recall\\\\\\midrule\n'+'\n'.join(lines)+'\n\\bottomrule\\end{tabular}\n')
    counts=[]
    for r in flat:
        counts.append(f"{LABELS[r['arm']]} & {r['seed']} & {r['epochs']} & {r['tp']} & {r['fp']} & {r['fn']} & {r['accuracy']:.5f} \\\\")
    (OUT/'counts_table.tex').write_text('\\begin{tabular}{lrrrrrr}\\toprule\nDetector & Seed & Epochs & TP & FP & FN & Accuracy\\\\\\midrule\n'+'\n'.join(counts)+'\n\\bottomrule\\end{tabular}\n')
    costs=[]
    for r in flat:
        peak='--' if r['peak_cuda_bytes'] is None else f"{r['peak_cuda_bytes']/2**30:.2f}"
        costs.append(f"{LABELS[r['arm']]} & {r['seed']} & {r['epochs']} & {r['parameters']:,} & {r['training_seconds']:.1f} & {peak} \\\\")
    (OUT/'resources_table.tex').write_text('\\begin{tabular}{lrrrrr}\\toprule\nDetector & Seed & Epochs & Parameters & Train (s) & Peak (GiB)\\\\\\midrule\n'+'\n'.join(costs)+'\n\\bottomrule\\end{tabular}\n')
    main=[(k,v) for k,v in groups.items() if k[1]==5 and k[0] in ('ORTHRUS','V3','V3_short_attention','KAIROS','GraphSAGE')]
    fig,axes=plt.subplots(1,3,figsize=(11,3.3))
    for ax,metric,title in zip(axes,('pr_ap','f1','fp'),('Node average precision','Node F1','False-positive nodes')):
        for i,((arm,_),rr) in enumerate(main):
            values=[r[metric] for r in rr];ax.bar(i,np.mean(values),color='#347da0',alpha=.7)
            ax.scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,color='#252525',s=19,zorder=3)
            if len(values)>1:ax.errorbar(i,np.mean(values),yerr=np.std(values,ddof=1),color='#252525',capsize=3)
        ax.set_xticks(range(len(main)),[LABELS[k[0]].replace(' adapter','') for k,_ in main],rotation=30,ha='right',fontsize=8)
        ax.set_title(title,fontsize=10);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
        ax.set_ylim(bottom=0)
    fig.tight_layout();savefig(fig,'comparison')
    fig,ax=plt.subplots(figsize=(5.5,3.2))
    for arm,seed,epochs,r,_ in rows:
        if seed==0 and epochs==5 and arm in ('V3','ORTHRUS','V3_short_attention','KAIROS','GraphSAGE'):
            ax.plot([x['epoch'] for x in r['history']],[x['validation_ce'] for x in r['history']],marker='o',label=LABELS[arm])
    ax.set(xlabel='Training epoch',ylabel='Benign selection CE');ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();savefig(fig,'validation')
    diagnostics=[]
    for p in sorted((ROOT/'artifacts/publication').glob('*_diagnostic.json')):
        name=p.name.removesuffix('_diagnostic.json');d=json.loads(p.read_text())
        diagnostics.append((name,d))
    if diagnostics:
        operating=[];fig,ax=plt.subplots(figsize=(5.5,3.3))
        for name,d in diagnostics:
            arm,tail=name.rsplit('_s',1);seed,epochs=tail.split('_e');m=d['raw_max_top30_kmeans']
            operating.append(f"{LABELS[arm]} & {seed} & {epochs} & {m['tp']} & {m['fp']} & {m['f1']:.4f} & {m['pr_ap']:.4f} \\\\")
            if seed=='0' and epochs=='5' and arm in ('V3','V3_short_attention','ORTHRUS','KAIROS','GraphSAGE'):
                ax.plot(d['pr_curve']['recall'],d['pr_curve']['precision'],label=LABELS[arm],lw=1.1)
        ax.axhline(68/268153,color='gray',ls=':',label='Positive prevalence')
        ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));ax.set_yscale('symlog',linthresh=.0001)
        ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();savefig(fig,'precision_recall')
        (OUT/'operating_table.tex').write_text('\\begin{tabular}{lrrrrrr}\\toprule\nDetector & Seed & Epochs & TP & FP & F1 & Raw AP\\\\\\midrule\n'+'\n'.join(operating)+'\n\\bottomrule\\end{tabular}\n')
    ledger_jobs={}
    for p in (ROOT/'artifacts/publication').glob('budget*.json'):
        for key,value in json.loads(p.read_text())['jobs'].items():
            if key in ledger_jobs:raise ValueError('Duplicate budget reservation: '+key)
            ledger_jobs[key]=value
    protocol=json.loads((OUT/'protocol.json').read_text())
    status=dict(planned_gpu_timeout_seconds=sum(j['timeout'] for j in protocol['jobs']),
        reserved_seconds=sum(j['timeout'] for j in ledger_jobs.values()),
        recorded_completed_body_seconds=sum(j.get('measured_seconds',0) for j in ledger_jobs.values()),
        note='Body seconds exclude some startup/idle time and are not an invoice. GPU timeout reservations are not dollars.',jobs=ledger_jobs)
    (OUT/'experiment_status.json').write_text(json.dumps(status,indent=2))
    exploratory=ROOT/'artifacts/sliding_v3/encoder_profile/result.json'
    if exploratory.exists():
        d=json.loads(exploratory.read_text());labels={'baseline':'V3-CNN','lookup':'Frozen lookup',
            'lookup_static':'Lookup + static short read','lookup_graphs':'Lookup + temporal CUDA graphs',
            'lookup_attention':'Lookup + attention-only short read','lookup_no_short_read':'Lookup + no short read',
            'lookup_causal_attention':'Lookup + causal-attention temporal block'}
        lines=[f"{labels[k]} & {v['p50_ms']:.2f} & {v['p95_ms']:.2f} & {v['p99_ms']:.2f} \\\\" for k,v in d['results'].items()]
        (OUT/'exploratory_latency_table.tex').write_text('\\begin{tabular}{lrrr}\\toprule\nExploratory variant & p50 (ms) & p95 (ms) & p99 (ms)\\\\\\midrule\n'+'\n'.join(lines)+'\n\\bottomrule\\end{tabular}\n')
    latency_path=ROOT/'artifacts/publication/trained_latency_s0_e0.json'
    if latency_path.exists():
        latency=json.loads(latency_path.read_text());lines=[]
        fig,ax=plt.subplots(figsize=(5.5,3.2))
        for i,(arm,rr) in enumerate(latency['arms'].items()):
            vals=[r['p50_ms'] for r in rr];ax.bar(i,np.median(vals),color='#347da0',alpha=.7)
            ax.scatter([i]*len(vals),vals,color='#252525',s=16)
            metrics=[np.median([r[k] for r in rr]) for k in ('p50_ms','p95_ms','p99_ms','events_per_second')]
            lines.append(f"{LABELS[arm]} & {metrics[0]:.2f} & {metrics[1]:.2f} & {metrics[2]:.2f} & {metrics[3]:,.0f} \\\\")
        ax.set_xticks(range(len(latency['arms'])),[LABELS[a] for a in latency['arms']],rotation=25,ha='right',fontsize=8)
        ax.set(ylabel='Median batch latency (ms)',title='Trained inference, L4, batch 1,024');ax.grid(axis='y',alpha=.2)
        fig.tight_layout();savefig(fig,'trained_latency')
        (OUT/'latency_table.tex').write_text('\\begin{tabular}{lrrrr}\\toprule\nDetector & p50 (ms) & p95 (ms) & p99 (ms) & Events/s\\\\\\midrule\n'+'\n'.join(lines)+'\n\\bottomrule\\end{tabular}\n')
    architecture()
    print(f'Generated tables and figures from {len(rows)} completed measured runs.')


if __name__=='__main__':build()
