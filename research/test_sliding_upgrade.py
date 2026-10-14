"""Causality, robustness and trainability of the experimental sliding upgrade."""
import torch
import pytest
from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem
from sliding_node_mem_upgrade.encoding import UpgradedUniversalEventEncoder
from sliding_node_mem_upgrade.scoring import RelationCalibrator, NodeEvidence
from sliding_node_mem_upgrade.pipeline import QueryCache


def make():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    return TGNNSlidingNodeMem(16, num_relations=3, buckets=256)


def args(n=24):
    return ([{'attributes': {'src': {'path': '/bin/a'}, 'dst': {'type': 'file'}}} for _ in range(n)],
            [i % 3 for i in range(n)], [4 + i % 2 for i in range(n)],
            [float(i) for i in range(n)], [i % 3 for i in range(n)])


def test_parameter_budget_unchanged():
    from sliding_node_mem.model import TGNNSlidingNodeMem as Original
    a, b = Original(16), TGNNSlidingNodeMem(16)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters()) == 764202


def test_missing_fields_and_empty_windows_finite():
    m = make().eval()
    with torch.no_grad():
        out, _ = m.forward_events([{}, {'attributes': {'value': None}}, {}],
                                 [0, 1, 2], [0, 4, 3], [0., 1., 2.], [0, 1, 2])
    assert out.shape == (3, 3) and torch.isfinite(out).all()


def test_path_order_and_field_order():
    torch.manual_seed(2)
    m = UpgradedUniversalEventEncoder(buckets=256)
    a = m([{'attributes': {'path': '/a/b'}}, {'attributes': {'path': '/b/a'}}])
    assert not torch.allclose(a[0], a[1])
    x = m([{'attributes': {'a': 1, 'b': None}}, {'attributes': {'b': None, 'a': 1}}])
    torch.testing.assert_close(x[0], x[1])


def test_current_and_future_relation_causality():
    m = make().eval(); a = args(24)
    with torch.no_grad():
        out, _ = m.forward_events(*a)
        m.reset_state(); a[-1][-1] = (a[-1][-1] + 1) % 3
        changed, _ = m.forward_events(*a)
    torch.testing.assert_close(out, changed)


def test_multiple_windows_equal_single_event_inference():
    m = make().eval(); a = args(24)
    with torch.no_grad():
        batch, _ = m.forward_events(*a)
        m.reset_state()
        steps = [m.forward_events(*[v[i:i+1] for v in a])[0] for i in range(24)]
    torch.testing.assert_close(batch, torch.cat(steps), rtol=1e-4, atol=1e-5)


def test_gradients_compression_and_memory_and_resume():
    m = make(); a = args(24)
    out, aux = m.forward_events(*a)
    loss = torch.nn.functional.cross_entropy(out, torch.tensor(a[-1])) + .01 * aux['compression_loss']
    loss.backward()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert m.compress[0].weight.grad.abs().sum() > 0
    m.detach_state(); m.eval()
    state = m.stream_state_dict()
    other = make().eval(); other.load_state_dict(m.state_dict()); other.load_stream_state_dict(state)
    future = ([{}], [0], [4], [25.], [1])
    with torch.no_grad():
        x, _ = m.forward_events(*future); y, _ = other.forward_events(*future)
    torch.testing.assert_close(x, y)


def test_local_rarity_not_global():
    m = make().eval()
    with torch.no_grad():
        m.forward_events([{}]*4, [0]*4, [1]*4, [0.,1.,2.,3.], [0]*4)
        _, aux = m.forward_events([{}], [10], [11], [4.], [0])
    assert aux['rare'].item()


def test_scoring_monotone_and_hub_penalty():
    import numpy as np
    cal = RelationCalibrator().fit(np.zeros(100), np.linspace(0, 1, 100))
    values = cal.transform([0,0,0], [.1,1.1,4.])
    assert values[0] < values[1] < values[2]
    scores = NodeEvidence()
    scores.update([1]*100+[2]*3, [1]*100+[2]*3, [5.]*103)
    result = scores.scores()
    assert result[1] < result[2]


def test_query_cache_canonical_and_target_masked():
    cache = QueryCache({'0': {'path': '/bin/x', 'action': 'attack'}, '1': {'type': 'file'}})
    a = cache.query(0, 1)
    assert a is cache.query(0, 1) and a['type'] is None
    assert 'action' not in a['_canonical_attributes']


def test_prefetch_preserves_order_and_empty_stream(tmp_path):
    from types import SimpleNamespace
    from sliding_node_mem_upgrade.pipeline import prepared_batches
    graph = SimpleNamespace(src=torch.tensor([0,1,2,3]), dst=torch.tensor([1,2,3,4]),
                            t=torch.arange(4)*1_000_000_000,
                            msg=torch.zeros(4,141))
    graph.msg[:,131] = 1
    path=tmp_path/'g.pt';torch.save(graph,path)
    batches=list(prepared_batches([path],{},batch=2))
    assert [u for b in batches for u in b[3]] == [0,1,2,3]
    assert [b[1] for b in batches] == [0,2]
    assert list(prepared_batches([],{})) == []


@pytest.mark.parametrize("frontend", ["kv", "orthrus"])
def test_end_to_end_runner_and_calibration(tmp_path, frontend):
    import json
    from types import SimpleNamespace
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from sliding_node_mem_upgrade.run import run
    from orthrus_comparison.run import sha256
    paths={}
    for split, day in [('train',3),('val',2),('test',6)]:
        n=192
        base=datetime(2018,4,day,tzinfo=ZoneInfo('America/New_York')).timestamp()
        msg=torch.zeros(n,272);msg[torch.arange(n),131+torch.arange(n)%3]=1
        graph=SimpleNamespace(src=torch.arange(n)%64,dst=(torch.arange(n)+1)%64,
                              t=(torch.arange(n,dtype=torch.float64)+base)*1e9,msg=msg)
        path=tmp_path/(split+'.pt');torch.save(graph,path);paths[split]=[str(path)]
    attrs=tmp_path/'attrs.json';attrs.write_text(json.dumps({str(i):{'type':'process'} for i in range(64)}))
    truth=tmp_path/'truth.json';truth.write_text('[1,2]')
    manifest={'dataset':'CADETS_E3','official_preprocessing':True,'days':list(range(2,14)),
              'n_nodes':64,'files':paths,'node_attributes':str(attrs),'ground_truth_nodes':str(truth),
              'file_sha256':{str(p):sha256(p) for p in [attrs,truth]+[tmp_path/(s+'.pt') for s in paths]}}
    mp=tmp_path/'manifest.json';mp.write_text(json.dumps(manifest))
    result=run(mp,tmp_path/'results',epochs=1,batch=64,device='cpu',frontend=frontend)
    assert result['checkpoint_epoch']==1 and result['status']=='complete'
    assert 0 <= result['calibrated']['f1'] <= 1
    assert result['parameter_count']==(764202 if frontend == 'kv' else 1113034)


def test_sparse_memory_large_identity_space():
    m=TGNNSlidingNodeMem(3_000_000, buckets=256).eval()
    with torch.no_grad():
        m.forward_events([{}]*2,[2_000_000,12],[2_500_000,12],[0.,1.],[0,1])
    assert len(m._memory)==64 and len(m._entity_ids)==3
    state=m.stream_state_dict()
    assert state['format']==4 and len(state['memory_values'])==3
    other=TGNNSlidingNodeMem(3_000_000,buckets=256).eval()
    other.load_state_dict(m.state_dict());other.load_stream_state_dict(state)
    with torch.no_grad():
        future=([{}],[2_000_000],[2_500_000],[2.],[1])
        x,_=m.forward_events(*future);y,_=other.forward_events(*future)
    torch.testing.assert_close(x,y)


def test_vector_window_planner_matches_sequential_with_self_edges():
    import numpy as np
    from collections import defaultdict,deque
    from sliding_node_mem_upgrade.windows import plan_windows
    rng=np.random.default_rng(17)
    history=defaultdict(deque)
    for first in range(0,512,64):
        src=rng.integers(0,11,64).tolist();dst=rng.integers(0,11,64).tolist()
        result=plan_windows(history,src,dst,first,9)
        for j,(u,v) in enumerate(zip(src,dst)):
            expected=sorted(set(history[u])|set(history[v]))[-8:]
            assert result[j][result[j]>=0].tolist()==expected
            for node in {u,v}:
                history[node].append(first+j)
                if len(history[node])>=9:history[node].popleft()


def test_exact_orthrus_frontend_and_graph_feed():
    from types import SimpleNamespace
    from orthrus_comparison.official import OrthrusReference
    from sliding_node_mem_upgrade.orthrus_frontend import OrthrusSlidingNodeMem
    torch.manual_seed(31)
    reference=OrthrusReference(16).eval()
    model=OrthrusSlidingNodeMem(16).eval()
    model.frontend.load_state_dict(reference.encoder.state_dict())
    full=SimpleNamespace(edge_type=torch.eye(10)[torch.arange(8)%10])
    for batch in range(2):
        src=torch.tensor([0,1,2,3]);dst=torch.tensor([4,5,6,7]);times=torch.arange(4)+batch*4
        msg=torch.randn(4,272)
        with torch.no_grad():
            hs,hd=reference.encoder(torch.stack((src,dst)),times,None,
                (msg[:,:131],msg[:,141:]),full,inference=True)
            events=model.encode_graph_batch([{}]*4,src,dst,times,msg,full)
        torch.testing.assert_close(torch.stack([e['_graph_features'] for e in events]),torch.cat((hs,hd),-1))
        torch.testing.assert_close(model.frontend.neighbor_loader.e_id,reference.encoder.neighbor_loader.e_id)
