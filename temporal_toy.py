"""Learning check: current query is identical; only previous cue predicts label.

No real-data claims. Graph history disabled in BOTH arms to isolate memory.
"""
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from causal_tgn import CausalTGNN


def stream(seed, pairs=64):
    rng=np.random.default_rng(seed)
    labels=rng.integers(0,2,pairs)
    x=torch.zeros(pairs*2,1,10)
    x[::2,0,0]=torch.from_numpy(labels.astype(np.float32)*2-1)
    # New endpoint pairs make each query depend solely on its own preceding cue.
    src=np.repeat(np.arange(pairs)*2,2); dst=src+1
    args=(x,torch.ones(pairs*2,1,dtype=torch.bool),torch.zeros(pairs*2,1),src,dst,np.arange(pairs*2,dtype=float))
    return args,torch.from_numpy(labels)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    output=[]
    for seed in range(5):
        train,y=stream(100+seed); test,yy=stream(1000+seed,128)
        for name,factors in [('B',()),('B_MR',('M','R'))]:
            torch.manual_seed(seed)
            model=CausalTGNN(10,factors)
            opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.00001)
            for epoch in range(60):
                model.train(); model.reset_state(); opt.zero_grad()
                logits,_=model.forward_events(*train)
                F.cross_entropy(logits[1::2],y).backward(); opt.step()
            row={'model':name,'seed':seed,'epochs':60,'pairs_train':64,'pairs_test':128}
            for mode in [None,'reset_memory']:
                model.eval(); model.reset_state()
                with torch.no_grad(): p=model.forward_events(*test,intervention=mode)[0][1::2].softmax(-1)[:,1].numpy()
                row[mode or 'normal']={'accuracy':float(accuracy_score(yy,p>=.5)),'f1':float(f1_score(yy,p>=.5,zero_division=0))}
            output.append(row)
            print(json.dumps(row),flush=True)
    path=Path('artifacts/temporal_toy.json'); path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'scope':'synthetic learning/correctness only; no CTU or APT claim','threshold':.5,'results':output},indent=2)+'\n')


if __name__=='__main__': main()
