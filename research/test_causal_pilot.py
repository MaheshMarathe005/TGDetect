"""Acceptance tests for the new path; legacy failures remain separate."""
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from causal_data import causal_egos, fit_scaler, transport_port
from causal_tgn import CausalTGNN, Reservoir
from pomsathit_baseline import CausalGATLayer, PomsathitAdapted

torch.set_num_threads(1)


def test_non_transport_fields_cannot_become_ports():
    assert transport_port('esp','0x2020597d') == -1
    assert transport_port('icmp','0x0800') == -1
    assert transport_port('tcp','443') == 443
    assert transport_port('udp','0x35') == 53
    with pytest.raises(ValueError): transport_port('tcp','65536')


def fixture(n=8):
    torch.manual_seed(42)
    x = torch.randn(n, 10)
    s, d = np.zeros(n, dtype=int), np.ones(n, dtype=int)
    t = np.arange(n, dtype=float)
    idx, mask, ages = causal_egos(s, d, t)
    return x, (x[torch.from_numpy(idx)], torch.from_numpy(mask), torch.from_numpy(ages), s, d, t)


def cut(args, start, stop):
    return tuple(a[start:stop] for a in args)


def test_hand_computed_attention_and_padding():
    layer = CausalGATLayer(1, hidden=1, dropout=0)
    with torch.no_grad():
        layer.projection.weight.fill_(1)
        layer.projection.bias.zero_()
        layer.a_src.zero_(); layer.a_dst.zero_()
    x = torch.tensor([[[2.], [4.], [1000.]]])
    adjacency = torch.tensor([[[1,1,0], [0,1,0], [0,0,1]]], dtype=torch.bool)
    torch.testing.assert_close(layer(x, adjacency), torch.tensor([[[3.], [4.], [1000.]]]))


def test_baseline_architecture_and_no_temporal_modules():
    model = PomsathitAdapted(10)
    manifest = model.adapted_manifest()
    assert len(model.layers) == 3
    assert [l.projection.out_features for l in model.layers] == [128]*3
    assert sum(p.numel() for p in model.parameters()) == 35458
    assert manifest['parameter_count'] == sum(v['parameters'] for v in manifest['trainable_modules'].values())
    assert all(l.dropout.p == .2 for l in model.layers)
    assert manifest['optimizer']['weight_decay'] == 1e-5
    assert manifest['batch_size'] == 256 and manifest['seeds'] == list(range(5))
    assert not any('cell' in n or 'time' in n for n, _ in model.named_parameters())


@pytest.mark.parametrize('factors', [(), ('T',), ('M',), ('M','R'), ('T','M','R')])
def test_future_invariance_and_eval_batch_partition(factors):
    raw, args = fixture()
    model = CausalTGNN(10, factors).eval()
    with torch.no_grad():
        full = model.forward_events(*args)[0]
        model.reset_state()
        pieces = [model.forward_events(*cut(args, a, b))[0] for a,b in [(0,1),(1,3),(3,8)]]
        torch.testing.assert_close(full, torch.cat(pieces), atol=1e-6, rtol=1e-5)
        model.reset_state()
        changed = list(args)
        changed[0] = args[0].clone(); changed[0][4:] += 100
        after = model.forward_events(*changed)[0]
        torch.testing.assert_close(full[:4], after[:4])
        model.reset_state()
        torch.testing.assert_close(full, model.forward_events(*args)[0])


def test_graph_prefix_equal_times_and_new_nodes():
    s, d, t = np.array([0,0,0,99]), np.array([1,2,3,100]), np.array([1.,1.,2.,3.])
    idx, mask, _ = causal_egos(s,d,t)
    assert not mask[1,1:].any()  # same-time peer is not history
    assert set(idx[2,1:3]) == {0,1}
    assert not mask[3,1:].any()  # new nodes start empty
    a = causal_egos(s[:3],d[:3],t[:3])
    b = causal_egos(s,d,t)
    for before, after in zip(a,b):
        np.testing.assert_array_equal(before, after[:3])


def test_equal_time_memory_commit_and_self_endpoint():
    _, args = fixture(4)
    args = list(args)
    args[-1] = np.array([0.,0.,1.,2.])
    args[-3] = np.zeros(4, dtype=int); args[-2] = np.zeros(4, dtype=int)
    m = CausalTGNN(10, ['M']).eval()
    with torch.no_grad():
        h = m.encode(args[0],args[1])
        _, context = m.forward_events(*args)
    torch.testing.assert_close(context[:2], torch.zeros_like(context[:2]))
    torch.testing.assert_close(context[2], .25*h[0]+.5*h[1])


def test_gru_all_tensors_receive_detection_gradient_and_update():
    _, args = fixture()
    model = CausalTGNN(10, ['M','R'])
    before = {n: p.detach().clone() for n,p in model.cell.named_parameters()}
    logits, _ = model.forward_events(*args)
    F.cross_entropy(logits, torch.tensor([0,1,0,1,1,0,1,0])).backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) for p in model.cell.parameters())
    torch.optim.Adam(model.parameters(), lr=.001).step()
    assert all(not torch.equal(before[n],p) for n,p in model.cell.named_parameters())
    model.detach_state()
    assert all(v.grad_fn is None for v in model.memory.values())
    assert all(h.grad_fn is None for _,_,h in model.pending)


def test_scaling_excludes_eval_and_memory_interface_excludes_labels():
    x = np.arange(100, dtype=float).reshape(10,10)
    before = fit_scaler(x,6)
    x[6:] += 1e6
    after = fit_scaler(x,6)
    for a,b in zip(before,after): np.testing.assert_array_equal(a,b)
    import inspect
    assert not {'y','labels','targets'} & set(inspect.signature(CausalTGNN.forward_events).parameters)


def test_replay_does_not_mutate_live_state_and_rejects_eval():
    _, args = fixture()
    model = CausalTGNN(10, ['T','M','R','H']).eval()
    with torch.no_grad():
        _, ctx = model.forward_events(*args)
        old = {k:v.clone() for k,v in model.memory.items()}
        pending = list(model.pending)
        model.replay(*args[:3], ctx)
    for k,v in old.items(): torch.testing.assert_close(v, model.memory[k])
    assert pending == model.pending
    r = Reservoir(3, 8, 0)
    for i in range(8): r.admit(i,ctx[i])
    with pytest.raises(ValueError): r.admit(8,ctx[0])
    assert len(r.items) == 3 and all(c.grad_fn is None for _,c in r.items)


def test_reservoir_inclusion_frequency():
    counts = np.zeros(10)
    for seed in range(1200):
        r = Reservoir(2,10,seed)
        for i in range(10): r.admit(i,torch.zeros(1))
        for i,_ in r.items: counts[i] += 1
    assert np.max(abs(counts/1200 - .2)) < .05


def test_disabled_factors_cannot_activate_and_memory_negative_controls():
    with pytest.raises(ValueError): CausalTGNN(10,['F'])
    with pytest.raises(ValueError): CausalTGNN(10,['R'])
    _, args = fixture()
    baseline = CausalTGNN(10).eval()
    assert baseline.cell is None and baseline.time_encoder is None
    model = CausalTGNN(10,['M','R']).eval()
    # Matching inherited weights makes each intervention exactly a static path.
    model.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        expected = baseline.forward_events(*args)[0]
        for mode in ['reset_memory','frozen_memory','gru_noop']:
            model.reset_state()
            actual, context = model.forward_events(*args, intervention=mode)
            torch.testing.assert_close(actual,expected)
            assert torch.count_nonzero(context) == 0


def test_threshold_is_validation_only_and_tie_rule():
    from run_causal_pilot import select_threshold, metrics
    y = np.array([0,1,0,1])
    p = np.array([.1,.8,.3,.7])
    threshold = select_threshold(y,p)
    assert threshold == .7
    assert metrics(y,p,threshold)['f1'] == 1
    # Test metrics cannot change the already selected scalar threshold.
    metrics(1-y,p,threshold)
    assert threshold == .7
