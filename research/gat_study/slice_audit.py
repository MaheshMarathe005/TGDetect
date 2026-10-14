"""Score chronological slices using exact corrected-V3 windows, without labels.

Run: PYTHONPATH=research python -m gat_study.slice_audit --prepared PATH
The scan replays all preceding events; selected slices must be audited again
with the actual training runner's reset/warmup policy before GPU training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sliding_node_mem_v3.rings import plan_ring, gather, merge_windows
from .diagnostics import DegreeAudit

DTYPE=np.dtype([(k,'<i8') for k in ('src','dst','relation','query','observed','signature','src_identity','dst_identity','negative_identity')]+[('time','<f8'),('negative_valid','?')])

@torch.no_grad()
def audit_split(path, n_nodes, segment=32768, chunk=4096):
    events=np.memmap(path,mode='r',dtype=DTYPE)
    history=np.full((n_nodes,8),-1,dtype=np.int64)
    results=[]
    for start in range(0,len(events),segment):
        end=min(start+segment,len(events));audit=DegreeAudit()
        for lo in range(start,end,chunk):
            hi=min(lo+chunk,end);n=hi-lo
            src=np.array(events['src'][lo:hi]);dst=np.array(events['dst'][lo:hi])
            nodes,inv=np.unique(np.r_[src,dst],return_inverse=True)
            old=torch.from_numpy(history[nodes])
            writes=torch.from_numpy(np.r_[np.ones(n,dtype=bool),src!=dst])
            plan=plan_ring(torch.from_numpy(inv),old>=0,writes,n)
            ids=torch.arange(lo,hi)
            raw=torch.cat((old.flatten(),ids,ids))
            incident=gather(raw,plan.before,fill=-1)
            past=merge_windows(incident[:n],incident[n:],8,True)
            window=torch.cat((past,ids[:,None]),1);valid=window>=0
            safe=window.clamp_min(0).numpy()
            audit.observe(torch.from_numpy(np.array(events['src'][safe])),
                          torch.from_numpy(np.array(events['dst'][safe])),valid)
            history[nodes]=gather(raw,plan.final,fill=-1).numpy()
        result=audit.result();roles=result['query_role_nonself_degree']
        hit={r:1-h.get(0,0)/(end-start) for r,h in roles.items()}
        edges={r:sum(k*v for k,v in h.items()) for r,h in roles.items()}
        # Require both transfer orientations; a high one-sided score is insufficient.
        supported=all(hit[r]>=.01 and edges[r]>=100 for r in ('sd','ds'))
        results.append({'start':start,'end':end,'start_time':float(events['time'][start]),
                        'end_time':float(events['time'][end-1]),'role_hit_fraction':hit,
                        'role_nonself_edges':edges,'both_transfer_roles_supported':supported,
                        'rank_score':min(hit['sd'],hit['ds']),'degree_audit':result})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent / 'next_run/slice_audit.json')
    parser.add_argument('--segment',type=int,default=32768)
    args=parser.parse_args();torch.set_num_threads(1)
    if args.segment<1:parser.error('segment must be positive')
    manifest_path=args.prepared/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest.get('compatible'):raise ValueError('Corrected prepared data required')
    report={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'selection_uses':'endpoints and chronological history only; no labels or model scores',
            'history_policy':'each split starts empty; full split prefix replayed before every candidate',
            'thresholds':{'each_sd_ds_hit_fraction':.01,'each_sd_ds_nonself_edges':100},
            'training_ready':False,
            'unmet_requirements':['documented download/chmod/exec chains mapped into test region',
                                  'source-backed attack-family taxonomy and node mapping',
                                  'selected-slice audit under final reset/warmup policy'], 'splits':{}}
    for split in ('train','val','test'):
        path=args.prepared/(split+'.bin')
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest!=manifest['checksums'][path.name]:raise ValueError('Checksum mismatch: '+str(path))
        rows=audit_split(path,manifest['n_nodes'],args.segment)
        report['splits'][split]={'source_sha256':digest,'segments':rows,
            'ranked_starts':[r['start'] for r in sorted(rows,key=lambda r:r['rank_score'],reverse=True)]}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False))
        print(split,len(rows),'segments;',sum(r['both_transfer_roles_supported'] for r in rows),'supported',flush=True)

if __name__=='__main__':main()
