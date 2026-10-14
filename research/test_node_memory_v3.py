"""Acceptance tests for exact chronology, chunk independence and learning credit."""
import os

import pytest
import torch

from node_memory_v3 import TGNNNodeMemV3, admission_penalties, endpoint_scan


def device():
    return os.environ.get('TGNN_TEST_DEVICE', 'cpu')


def inputs():
    torch.manual_seed(19)
    dev = device()
    c = torch.randn(12, 128, device=dev)
    src = torch.tensor([0, 1, 0, 2, 0, 1, 2, 3, 3, 0, 1, 2], device=dev)
    dst = torch.tensor([4, 4, 5, 4, 5, 5, 4, 5, 5, 4, 4, 5], device=dev)
    return c, src, dst


def model():
    m = TGNNNodeMemV3(n_nodes=6).to(device()).eval()
    with torch.no_grad():
        m.read_gate.bias.fill_(.3)
    return m


def test_scan_matches_sequential_values_and_gradients():
    dev = device()
    s = torch.tensor([0, 0, 1, 0], device=dev)
    d = torch.tensor([1, 0, 2, 2], device=dev)
    torch.manual_seed(42)
    initial = torch.randn(3, 5, device=dev, requires_grad=True)
    w = (torch.rand(8, 5, device=dev) * .3).requires_grad_()
    x = torch.randn(8, 5, device=dev, requires_grad=True)
    ms, md, final = endpoint_scan(initial, s, d, w, x)
    table = initial
    expected_s, expected_d = [], []
    for i in range(4):
        expected_s.append(table[s[i]])
        expected_d.append(table[d[i]])
        for j, node in [(i, s[i]), (i + 4, d[i])]:
            if j >= 4 and s[i] == d[i]:
                continue
            value = (1 - w[j]) * table[node] + w[j] * x[j]
            table = table.index_copy(0, node.reshape(1), value[None])
    torch.testing.assert_close(ms, torch.stack(expected_s))
    torch.testing.assert_close(md, torch.stack(expected_d))
    torch.testing.assert_close(final, table)
    actual = torch.autograd.grad(ms.sum() + md.sum() + final.sum(), (initial, w, x), retain_graph=True)
    expected = torch.autograd.grad(torch.stack(expected_s).sum() + torch.stack(expected_d).sum() + table.sum(), (initial, w, x))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('chunks', [[1] * 12, [3, 5, 4], [6, 6]])
def test_context_batch_invariance_and_future_independence(chunks):
    c, s, d = inputs()
    m = model()
    with torch.no_grad():
        full, _ = m._context_forward(c, s, d)
        final = m.Mem.clone()
        m.reset_state()
        parts, lo = [], 0
        for size in chunks:
            hi = lo + size
            parts.append(m._context_forward(c[lo:hi], s[lo:hi], d[lo:hi])[0])
            lo = hi
        torch.testing.assert_close(torch.cat(parts), full, atol=3e-5, rtol=3e-5)
        torch.testing.assert_close(m.Mem, final, atol=3e-5, rtol=3e-5)
        m.reset_state()
        altered = c.clone()
        altered[6:] += 10
        output, _ = m._context_forward(altered, s, d)
        torch.testing.assert_close(output[:6], full[:6], atol=3e-5, rtol=3e-5)


def test_task_gradient_reaches_writes_and_controller():
    c, s, d = inputs()
    m = model()
    logits, _ = m._context_forward(c, s, d)
    logits[-4:].square().sum().backward()
    for name in ('write_src.0.weight', 'write_dst.0.weight', 'controller.score_raw', 'attn_score.weight'):
        grad = dict(m.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    m.detach_state()
    assert m.Mem.grad_fn is None and m.Reference.grad_fn is None


def test_zero_gate_baseline_and_replay_state_isolation():
    c, s, d = inputs()
    m = model()
    with torch.no_grad():
        m.read_gate.weight.zero_()
        m.read_gate.bias.zero_()
        logits, _ = m._context_forward(c, s, d)
        torch.testing.assert_close(logits, m.head(m.mem_norm(c)))
        before = m.Mem.clone()
        reference = m.Reference.clone()
        x = torch.randn(3, 5, 72, device=device())
        valid = torch.ones(3, 5, device=device(), dtype=torch.bool)
        ages = torch.zeros(3, 5, device=device())
        m.replay_logits(x, valid, ages)
        torch.testing.assert_close(before, m.Mem)
        torch.testing.assert_close(reference, m.Reference)


def test_full_encoder_batch_invariance():
    _, s, d = inputs()
    m = model()
    x = torch.randn(12, 5, 72, device=device())
    valid = torch.ones(12, 5, device=device(), dtype=torch.bool)
    ages = torch.zeros(12, 5, device=device())
    times = list(range(12))
    with torch.no_grad():
        full, _ = m.forward_events(x, valid, ages, s, d, times)
        m.reset_state()
        pieces = [m.forward_events(x[i:i+3], valid[i:i+3], ages[i:i+3],
                                   s[i:i+3], d[i:i+3], times[i:i+3])[0] for i in range(0, 12, 3)]
        torch.testing.assert_close(full, torch.cat(pieces), atol=3e-5, rtol=3e-5)


def test_penalty_counts_candidate_once():
    aux = {key: torch.tensor(value, device=device()) for key, value in
           {'cand': [.5, 1.], 'admit': [.1, .2], 'rate': [.2, .2], 'novelty': [.5, .5]}.items()}
    budget, redundancy = admission_penalties(aux)
    torch.testing.assert_close(budget, torch.zeros_like(budget), atol=1e-7, rtol=0)
    torch.testing.assert_close(redundancy, torch.full_like(redundancy, .1))
