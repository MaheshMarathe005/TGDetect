"""CPU-only operating-rule diagnostics from preserved edge scores."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.cluster import KMeans
from sliding_node_mem_v2.scoring import read_edges,calibrated_scores
from orthrus_comparison.run import measure,sha256


def analyze(directory,truth):
    directory=Path(directory);val=read_edges(directory/'val_edges.bin');test=read_edges(directory/'test_edges.bin')
    threshold,scores,calibration=calibrated_scores(val[len(val)//3:],test,.001)
    nodes=np.asarray(sorted(scores));values=np.asarray([scores[int(n)] for n in nodes]);labels=np.isin(nodes,list(truth))
    order=np.lexsort((nodes,-values))
    precision,recall,_=precision_recall_curve(labels,values)
    # Keep every recall transition, plus evenly spaced points. AP is computed
    # from the full ranking; this reduction is only for portable plotting data.
    keep=np.unique(np.r_[0,len(recall)-1,np.where(np.diff(recall)!=0)[0],
        np.where(np.diff(recall)!=0)[0]+1,np.linspace(0,len(recall)-1,min(2000,len(recall))).astype(int)])
    raw={}
    for rows in np.array_split(test,max(1,len(test)//65536)):
        for u,v,loss in zip(rows['src'],rows['dst'],rows['loss']):
            raw[int(u)]=max(raw.get(int(u),0.),float(loss));raw[int(v)]=max(raw.get(int(v),0.),float(loss))
    top=sorted(raw,key=lambda n:(raw[n],n))[-30:]
    clusters=KMeans(n_clusters=2,random_state=0,n_init=10).fit(np.asarray([raw[n] for n in top]).reshape(-1,1))
    selected={n for n,c in zip(top,clusters.labels_) if c==clusters.cluster_centers_.argmax()}
    sensitivity={}
    for rate in (.0001,.01):
        cut,ss,cal=calibrated_scores(val[len(val)//3:],test,rate)
        sensitivity[str(rate)]=dict(threshold=cut if np.isfinite(cut) else None,calibration=cal,
            metrics=measure(ss,truth,{n for n,s in ss.items() if s>cut}))
    return dict(source=str(directory),edge_sha256={p:sha256(directory/p) for p in ('val_edges.bin','test_edges.bin')},
        primary=measure(scores,truth,{n for n,s in scores.items() if s>threshold}),
        precision_at={str(k):float(labels[order[:k]].mean()) for k in (10,30,100,1000)},
        pr_curve=dict(precision=precision[keep].tolist(),recall=recall[keep].tolist()),
        raw_max_top30_kmeans=measure(raw,truth,selected),sensitivity=sensitivity,
        raw_max_top30_note='Released-style max-edge/top-30 two-cluster operating-rule diagnostic on our checkpoints; not full native training or attribution reproduction.',
        scored_nodes=len(nodes),scored_malicious=int(labels.sum()),unscored_ground_truth=sorted(truth-set(nodes.tolist())))
