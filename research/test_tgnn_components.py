"""Acceptance gates run locally and again on the actual L4 before training."""
import os

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from causal_data import causal_egos
from causal_tgn import CausalTGNN, Reservoir
from tgnn_components import ARMS, ComponentTGNN, EfficientReservoir

torch.set_num_threads(1)
DEVICE = os.environ.get("TGNN_TEST_DEVICE", "cpu")


def batch():
    torch.manual_seed(741)
    s = np.array([0, 0, 0, 2, 2, 9, 9, 0, 0, 0, 9, 99])
    d = np.array([0, 1, 1, 1, 0, 0, 9, 1, 2, 0, 0, 100])
    t = np.array([0., 0., 1., 2., 2., 3., 4., 4., 4., 5., 6., 7.])
    ego, valid, age = causal_egos(s, d, t)
    x = torch.randn(len(s), 10, device=DEVICE)
    return x[torch.as_tensor(ego, device=DEVICE)], torch.as_tensor(valid, device=DEVICE), torch.as_tensor(age, device=DEVICE), s, d, t


def cut(args, lo, hi):
    return tuple(a[lo:hi] for a in args)


@pytest.mark.parametrize("factors", [("H",), ("T", "H"), ("T", "H", "M")])
def test_reference_equivalence_outputs_gradients_and_update(factors):
    args = batch()
    torch.manual_seed(0)
    old = CausalTGNN(10, factors).to(DEVICE).eval()
    torch.manual_seed(0)
    new = ComponentTGNN(10, factors).to(DEVICE).eval()
    assert old.state_dict().keys() == new.state_dict().keys()
    for k in old.state_dict():
        torch.testing.assert_close(old.state_dict()[k], new.state_dict()[k], rtol=0, atol=0)
    for lo, hi in [(0, 1), (1, 4), (4, 8), (8, 12)]:
        a, ca = old.forward_events(*cut(args, lo, hi))
        b, cb = new.forward_events(*cut(args, lo, hi))
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(ca, cb, atol=2e-6, rtol=2e-5)
    a.sum().backward()
    b.sum().backward()
    for (name, x), (_, y) in zip(old.named_parameters(), new.named_parameters()):
        torch.testing.assert_close(x.grad, y.grad, atol=3e-5, rtol=3e-4, msg=name)
    for node in old.memory:
        torch.testing.assert_close(old.memory[node], new.memory[node], atol=2e-6, rtol=2e-5)
    torch.optim.SGD(old.parameters(), lr=.001).step()
    torch.optim.SGD(new.parameters(), lr=.001).step()
    for k in old.state_dict():
        torch.testing.assert_close(old.state_dict()[k], new.state_dict()[k], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("arm", list(ARMS))
def test_all_arms_causality_replay_padding_gradients(arm):
    args = batch()
    model = ComponentTGNN(10, ARMS[arm]).to(DEVICE).eval()
    with torch.no_grad():
        full, context = model.forward_events(*args)
        assert torch.isfinite(full).all()
        model.reset_state()
        pieces = [model.forward_events(*cut(args, lo, hi))[0] for lo, hi in [(0, 1), (1, 4), (4, 8), (8, 12)]]
        torch.testing.assert_close(full, torch.cat(pieces), atol=3e-6, rtol=3e-5)
        changed = list(args)
        changed[0] = args[0].clone()
        changed[0][6:] += 100
        model.reset_state()
        altered = model.forward_events(*changed)[0]
        torch.testing.assert_close(full[:6], altered[:6], atol=2e-6, rtol=2e-5)
        changed[0] = args[0].clone()
        changed[0][~args[1]] = 10000
        model.reset_state()
        torch.testing.assert_close(full, model.forward_events(*changed)[0], atol=2e-6, rtol=2e-5)
        memory = {k: v.clone() for k, v in model.memory.items()}
        pending = model.pending.copy()
        last = model.last_time
        model.replay(*args[:3], context)
        assert model.last_time == last and model.pending == pending
        for k in memory:
            torch.testing.assert_close(memory[k], model.memory[k])
    model.reset_state()
    output, _ = model.forward_events(*args)
    F.cross_entropy(output, torch.arange(12, device=DEVICE) % 2).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if name.startswith(("history_gru", "query", "key", "value", "norm", "time_encoder")):
            assert torch.count_nonzero(p.grad), name
    model.detach_state()
    assert all(v.grad_fn is None for v in model.memory.values())
    assert all(v.grad_fn is None for _, _, v in model.pending)


def test_exact_reservoir_rng_and_snapshots():
    a, b = Reservoir(7, 31, 123), EfficientReservoir(7, 31, 123)
    for epoch in range(3):
        for i in range(31):
            ca = a.sample(4)
            cb = b.sample(4)
            assert [k for k, _ in ca] == [k for k, _ in cb]
            for (_, x), (_, y) in zip(ca, cb):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            context = torch.full((128,), float(epoch * 100 + i), device=DEVICE, requires_grad=True)
            a.admit(i, context)
            b.admit(i, context)
        assert [k for k, _ in a.items] == [k for k, _ in b.items]
    with pytest.raises(ValueError):
        b.admit(31, context)


def test_memory_controls_and_backwards_guard():
    args = batch()
    m = ComponentTGNN(10, ("T", "H", "M")).to(DEVICE).eval()
    _, context = m.forward_events(*args)
    assert torch.count_nonzero(context[2:])
    assert torch.count_nonzero(context[:2]) == 0
    assert torch.count_nonzero(context[-1]) == 0  # both endpoints unseen
    with pytest.raises(ValueError):
        m.forward_events(*args)
    _, reset = m.forward_events(*args, intervention="reset_memory")
    assert torch.count_nonzero(reset) == 0
    with pytest.raises(ValueError):
        ComponentTGNN(10, ("R",))  # do not silently conflate two different GRUs


@pytest.mark.parametrize("arm", list(ARMS))
def test_training_across_optimizer_steps_and_rehearsal(arm):
    args = batch()
    model = ComponentTGNN(10, ARMS[arm]).to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.00001)
    reservoir = EfficientReservoir(4, 12, 937)
    y = torch.arange(12, device=DEVICE) % 2
    weights = torch.tensor([1., 3.], device=DEVICE) if "W" in ARMS[arm] else None
    for _ in range(2):
        model.reset_state()
        for lo, hi in [(0,4), (4,8), (8,12)]:
            opt.zero_grad(set_to_none=True)
            logits, context = model.forward_events(*cut(args,lo,hi))
            loss = F.cross_entropy(logits, y[lo:hi], weight=weights)
            items = reservoir.sample(2)
            if items:
                ids = [i for i, _ in items]
                replay = model.replay(*(a[ids] for a in args[:3]), torch.stack([v for _,v in items]))
                loss = loss + F.cross_entropy(replay, y[ids], weight=weights)
            loss.backward()
            assert torch.isfinite(loss)
            opt.step()
            model.detach_state()
            for i in range(lo,hi):
                reservoir.admit(i, context[i-lo])
    assert all(torch.isfinite(p).all() for p in model.parameters())
