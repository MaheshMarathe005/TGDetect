"""First-session gates and small diagnostic counterexamples; no training runs."""
import ast
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch
from torch.nn import functional as F

import modal_common as common
from pomsathit_baseline import (
    PaperSpecification, PomsathitPaperSpec, PomsathitAdapted,
    UnresolvedPaperSpecification, architecture_manifest,
)

ROOT = Path(__file__).resolve().parent


def test_paper_contract_and_persisted_manifest():
    spec = PaperSpecification()
    assert (spec.gnn_layers, spec.hidden_dim, spec.activation, spec.dropout) == (3, 128, 'ReLU', 0.2)
    assert (spec.prediction_target, spec.output_activation) == ('node', 'softmax')
    assert (spec.optimizer, spec.learning_rate, spec.weight_decay) == ('Adam', 1e-3, 1e-5)
    assert (spec.batch_size, spec.epochs_per_scenario, spec.seed_count) == (256, 200, 5)
    assert (spec.rehearsal_fraction, spec.buffer_update) == (0.1, 'reservoir')
    assert spec.generative_replay_in_proposed_model is False
    manifest = architecture_manifest()
    assert json.loads((ROOT / 'BASELINE_ARCHITECTURE.json').read_text()) == manifest
    assert manifest['parameter_count'] is None
    assert manifest['trainable_modules'] is None
    assert manifest['certified_executable_architecture'] is False
    assert manifest['dropout_locations'] is None
    assert manifest['rehearsal_settings']['lambda'] is None


@pytest.mark.parametrize('model', [PomsathitPaperSpec, PomsathitAdapted])
def test_missing_spec_never_silently_becomes_an_exact_model(model):
    with pytest.raises(UnresolvedPaperSpecification):
        model()


def test_fixture_default_is_outside_preserved_dataset():
    tree = ast.parse((ROOT / '.labtest/harness.py').read_text())
    # Evaluate only the path declarations, without invoking the legacy app or stub.
    declarations = [n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id in {'_FIXTURE_ROOT', 'ROOT', 'DATA'} for t in n.targets)]
    namespace = {'pathlib': __import__('pathlib'), 'tempfile': tempfile}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), '<fixture paths>', 'exec'), namespace)
    try:
        assert ROOT not in namespace['DATA'].resolve().parents
        assert namespace['ROOT'].is_dir()
    finally:
        namespace['_FIXTURE_ROOT'].cleanup()
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == 'sk_stub'
                   for n in ast.walk(tree))


def inputs():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    x = torch.randn(3, 22)
    mem = torch.zeros(3, 128)
    return (x, x, x[:, :0], mem, mem, torch.tensor([0, 0, 1]),
            torch.tensor([1, 2, 2]), torch.tensor([1., 2., 3.]))


def test_existing_baseline_shape_and_binary_head_equivalence():
    args = inputs()
    model = common.get_models()['Pomsathit_GCN']().eval()
    saved = []
    hook = model.head.register_forward_hook(lambda module, inp, out: saved.append(out))
    logits, memory = model(*args)
    hook.remove()
    assert logits.shape == (3,) and memory is None
    assert torch.isfinite(logits).all()
    assert sum(p.numel() for p in model.parameters()) == 168453
    torch.testing.assert_close(logits.sigmoid(), saved[0].softmax(dim=-1)[:, 1])
    F.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
    assert all(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize('name', ['Pomsathit_GCN', 'TGDetect_TGN'])
@pytest.mark.xfail(strict=True, reason='Known defect: snapshot includes later events; not causal per event (L02).')
def test_legacy_model_does_not_see_later_event_features(name):
    args = inputs()
    model = common.get_models()[name]().eval()
    before = model(*args)[0][0]
    changed = args[0].clone()
    changed[-1] += 10.0
    after = model(changed, *args[1:])[0][0]
    torch.testing.assert_close(before, after, rtol=0, atol=1e-6)


@pytest.mark.xfail(strict=True, reason='Known defect: GRU output is detached between steps and never enters a differentiable loss (L03).')
def test_legacy_gru_receives_gradient_across_two_detached_steps():
    args = inputs()
    model = common.get_models()['TGDetect_TGN']().train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    logits, new_memory = model(*args)
    loss = F.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.]))
    opt.zero_grad(); loss.backward(); opt.step()
    replayed_state = new_memory.detach()
    logits, _ = model(*args[:3], replayed_state, replayed_state, *args[5:])
    opt.zero_grad()
    F.binary_cross_entropy_with_logits(logits, torch.tensor([1., 0., 1.])).backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in model.cell.parameters())


@pytest.mark.xfail(strict=True, reason='Known defect: build_context early rate depends on final stream length (L04).')
def test_context_is_invariant_to_appending_future_events():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 22)).astype(np.float32)
    ts = np.arange(8, dtype=np.float64)
    src, dst = np.arange(8), np.arange(1, 9)
    prefix_z, prefix_c = common.build_context(x[:4], ts[:4], src[:4], dst[:4])
    full_z, full_c = common.build_context(x, ts, src, dst)
    np.testing.assert_allclose(prefix_z, full_z[:4])
    np.testing.assert_allclose(prefix_c, full_c[:4])
