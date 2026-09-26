import os
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from causal_data import causal_egos
from causal_tgn import Reservoir
from tgnn_components import ComponentTGNN
from temporal_cnn import CNN_ARMS, CausalTemporalCNN, make_model, rehearsal_schedule

torch.set_num_threads(1)
DEVICE = os.environ.get("TGNN_TEST_DEVICE", "cpu")


def fixture():
    torch.manual_seed(82)
    t = np.array([0.,0.,1.,2.,3.,4.,4.,5.])
    s, d = np.zeros(8,dtype=int), np.ones(8,dtype=int)
    ego, valid, age = causal_egos(s,d,t)
    raw = torch.randn(8,10,device=DEVICE)
    return raw[torch.as_tensor(ego,device=DEVICE)], torch.as_tensor(valid,device=DEVICE), torch.as_tensor(age,device=DEVICE), s,d,t


def test_convolution_strictly_causal_and_padding():
    torch.manual_seed(1)
    m = CausalTemporalCNN(4,0).to(DEVICE).eval()
    x = torch.randn(3,9,4,device=DEVICE,requires_grad=True)
    valid = torch.ones(3,9,device=DEVICE,dtype=torch.bool)
    y = m(x,valid)
    changed = x.detach().clone()
    changed[:,4:] += 100
    torch.testing.assert_close(y[:,:4],m(changed,valid)[:,:4])
    y[:,3].sum().backward()
    assert torch.count_nonzero(x.grad[:,4:]) == 0
    valid[:,:2] = False
    before = m(x,valid)
    changed = x.detach().clone()
    changed[:,:2] = 10000
    torch.testing.assert_close(before,m(changed,valid))


def test_cnn_event_axis_and_current_last_order_hand_computed():
    m=make_model("CNN_H").to(DEVICE).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        m.projection.weight[0,0]=1
        m.cnn.convs[0].weight[0,0,1]=1  # previous temporal position
        m.cnn.convs[1].weight[0,0,2]=1  # current temporal position
        m.head.weight[0,0]=1
        x=torch.zeros(1,5,10,device=DEVICE)
        x[0,:,0]=torch.tensor([4.,3.,2.,1.,0.],device=DEVICE)
        valid=torch.ones(1,5,dtype=torch.bool,device=DEVICE)
        logits,_=m.forward_events(x,valid,torch.zeros(1,5,device=DEVICE))
        torch.testing.assert_close(logits,torch.tensor([[3.,0.]],device=DEVICE))


@pytest.mark.parametrize("arm", list(CNN_ARMS))
def test_every_arm_causal_batch_independent_padding_gradient_and_update(arm):
    args = fixture()
    m = make_model(arm).to(DEVICE).eval()
    with torch.no_grad():
        full = m.forward_events(*args)[0]
        parts = [m.forward_events(*(v[lo:hi] for v in args))[0] for lo,hi in ((0,2),(2,5),(5,8))]
        torch.testing.assert_close(full,torch.cat(parts),rtol=2e-5,atol=2e-6)
        modified = list(args)
        modified[0] = args[0].clone()
        modified[0][4:] += 100
        torch.testing.assert_close(full[:4],m.forward_events(*modified)[0][:4])
        modified[0] = args[0].clone()
        modified[0][~args[1]] = 10000
        torch.testing.assert_close(full,m.forward_events(*modified)[0],rtol=2e-5,atol=2e-6)
    opt = torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-5)
    before = {n:p.detach().clone() for n,p in m.named_parameters() if n.startswith("cnn")}
    m.train()
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        y,ctx = m.forward_events(*args)
        loss = F.cross_entropy(y,torch.arange(8,device=DEVICE)%2)
        replay,_ = m.forward_events(*(v[[0,2,3]] for v in args))
        loss = loss+F.cross_entropy(replay,torch.tensor([0,0,1],device=DEVICE))
        loss.backward()
        assert torch.isfinite(loss) and torch.count_nonzero(ctx) == 0
        for n,p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), n
            if n.startswith(("cnn", "history_gru")):
                assert torch.count_nonzero(p.grad), n
        opt.step()
    for n,p in m.named_parameters():
        if n in before:
            assert not torch.equal(before[n],p),n


@pytest.mark.parametrize("n,batch", [(40,8),(300,64)])
def test_precomputed_replay_exact_rng_and_training_only(n,batch):
    ids,sizes = rehearsal_schedule(n,3,batch,seed=2)
    reference = Reservoir(n//10,n,939)
    step = 0
    for _ in range(3):
        for lo in range(0,n,batch):
            expected = [i for i,_ in reference.sample(26)]
            assert ids[step,:sizes[step]].tolist() == expected
            assert all(0<=i<n for i in expected)
            for i in range(lo,min(lo+batch,n)):
                reference.admit(i,torch.zeros(128))
            step+=1


@pytest.mark.parametrize("base,extended", [("TGNN","TGNN_C"),("TGNN_Q","TGNN_Q_C"),("TGNN_Q_A","TGNN_Q_A_C")])
def test_shared_initialization_and_zero_cnn_equivalence(base,extended):
    args = fixture()
    torch.manual_seed(5)
    a = make_model(base).to(DEVICE).eval()
    torch.manual_seed(5)
    b = make_model(extended).to(DEVICE).eval()
    for n,v in a.state_dict().items():
        torch.testing.assert_close(v,b.state_dict()[n],rtol=0,atol=0)
    with torch.no_grad():
        for p in b.cnn.parameters():
            p.zero_()
        torch.testing.assert_close(a.forward_events(*args)[0],b.forward_events(*args)[0],rtol=0,atol=0)


def test_optimized_training_matches_original_replay_path():
    # Detect changes in the actual optimizer/dropout trajectory, not just index lists.
    torch.manual_seed(81)
    x=torch.randn(40,5,10,device=DEVICE)
    valid=torch.ones(40,5,dtype=torch.bool,device=DEVICE)
    ages=torch.rand(40,5,device=DEVICE)
    y=torch.arange(40,device=DEVICE)%2
    schedules,sizes=rehearsal_schedule(40,2,8,seed=3)
    outcomes=[]
    for optimized in (False,True):
        torch.manual_seed(3)
        m=make_model("TGNN_Q_A").to(DEVICE).train()
        opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=1e-5)
        r=Reservoir(4,40,940)
        torch.manual_seed(10003)
        step=0
        for _ in range(2):
            for lo in range(0,40,8):
                opt.zero_grad(set_to_none=True)
                logits,ctx=m.forward_events(x[lo:lo+8],valid[lo:lo+8],ages[lo:lo+8],None,None,None)
                loss=F.cross_entropy(logits,y[lo:lo+8])
                if optimized:
                    ids=schedules[step,:sizes[step]].tolist()
                    if ids:
                        logits,_=m.forward_events(x[ids],valid[ids],ages[ids],None,None,None)
                else:
                    samples=r.sample(26)
                    ids=[i for i,_ in samples]
                    if ids:
                        logits=m.replay(x[ids],valid[ids],ages[ids],torch.stack([v for _,v in samples]))
                if ids:
                    loss=loss+F.cross_entropy(logits,y[ids])
                loss.backward()
                opt.step()
                for i in range(lo,lo+8):
                    r.admit(i,ctx[i-lo])
                step+=1
        outcomes.append(m.state_dict())
    for n,p in outcomes[0].items():
        torch.testing.assert_close(p,outcomes[1][n],rtol=0,atol=0,msg=n)
