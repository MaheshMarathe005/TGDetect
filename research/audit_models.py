"""Tiny CPU diagnostics of preserved models; never runs their training entrypoints."""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import modal_common as common

ROOT = Path(__file__).resolve().parent


def load_legacy_classes(filename, enclosing_function, names):
    tree = ast.parse((ROOT / filename).read_text())
    outer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == enclosing_function)
    # Only literal constants and the named class/function definitions are executed.
    nodes = []
    for n in outer.body:
        if isinstance(n, ast.Assign):
            try:
                ast.literal_eval(n.value)
            except (ValueError, TypeError):
                continue
            nodes.append(n)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names:
            nodes.append(n)
    scope = {'torch': torch, 'nn': nn, 'F': F, 'math': math, 'hashlib': hashlib, 'device': torch.device('cpu')}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), scope)
    return scope


def shapes(obj):
    if isinstance(obj, torch.Tensor):
        return list(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [shapes(x) for x in obj]
    return None


def profile(model, args):
    trace, hooks = [], []
    for name, module in model.named_modules():
        if name and not list(module.children()):
            def hook(mod, inputs, output, name=name):
                trace.append({'module': name, 'class': type(mod).__name__, 'in': shapes(inputs), 'out': shapes(output)})
            hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        output = model(*args)
    for hook in hooks:
        hook.remove()
    return {
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'parameters_by_name': {k: {'shape': list(p.shape), 'count': p.numel()} for k, p in model.named_parameters()},
        'leaf_forward_shapes': trace, 'output_shape': shapes(output),
        'unused_leaf_modules': [n for n, m in model.named_modules() if n and not list(m.children()) and n not in {r['module'] for r in trace}],
        'finite_output': bool(torch.isfinite(output[0]).all()),
    }


def audit():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    models = common.get_models()
    result = {'scope': 'CPU synthetic diagnostics; not accuracy experiments', 'seed': 0, 'models': {}, 'registry': {}, 'symbols': {}}
    for name, cls in models.items():
        model = cls()
        result['registry'][name] = {
            'class': type(model).__name__, 'parameters': sum(p.numel() for p in model.parameters()),
            'uses_mem': bool(getattr(model, 'uses_mem', False)), 'uses_ctx': bool(getattr(model, 'uses_ctx', False)),
            'uses_ex': bool(getattr(model, 'uses_ex', False)), 'flags': getattr(model, 'FLAGS', {}),
        }
    x = torch.randn(3, common.FEAT_IN)
    state = torch.zeros(3, common.MEM)
    src, dst = torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])
    dt = torch.tensor([1., 2., 3.])
    args = (x, x, x[:, :0], state, state, src, dst, dt)
    for name in ['Pomsathit_GCN', 'TGDetect_TGN']:
        torch.manual_seed(0)
        model = models[name]()
        record = profile(model, args)
        model.eval()
        first, _ = model(*args)
        changed = x.clone()
        changed[-1] += 10.0
        second, _ = model(changed, *args[1:])
        record['earliest_event_logit_change_when_later_event_features_change'] = float((first[0] - second[0]).abs().detach())
        logits, _ = model(*args)
        F.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
        record['parameters_without_gradient_from_current_batch_loss'] = [n for n, p in model.named_parameters() if p.grad is None]
        result['models']['modal_common.' + name] = record
    legacy = load_legacy_classes('modal_bench_real.py', 'bench', {'seg_softmax', 'PomsathitGCN'})
    result['models']['modal_bench_real.PomsathitGCN'] = profile(legacy['PomsathitGCN'](), (x, state, state, src, dst, dt))
    v16 = load_legacy_classes('modal_train_v16.py', 'train', {'MultiResTimeEncoder', 'CausalHTAConv', 'BaseModel'})
    st64 = torch.zeros(3, 64)
    result['models']['modal_train_v16.V13_GNN_Rehearsal'] = profile(
        v16['BaseModel'](is_apex=False), (['a', 'a', 'b'], ['b', 'c', 'c'], [[0, 0, 0]] * 3, [[0, 0, 0]] * 3, st64, st64, dt))
    for path in sorted(ROOT.glob('*.py')):
        source = path.read_text()
        tree = ast.parse(source)
        result['symbols'][path.name] = {
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'definitions': [
                {'name': n.name, 'kind': type(n).__name__, 'line': n.lineno, 'end_line': n.end_lineno}
                for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            ],
        }
    return result


if __name__ == '__main__':
    result = audit()
    (ROOT / 'EXISTING_MODEL_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, record in result['models'].items():
        print(name, record['parameters'], 'parameters; output', record['output_shape'])
        if 'earliest_event_logit_change_when_later_event_features_change' in record:
            print('  Future-feature perturbation:', record['earliest_event_logit_change_when_later_event_features_change'])
            print('  No gradient:', record['parameters_without_gradient_from_current_batch_loss'])
    print('Registry:', len(result['registry']), 'variants; wrote EXISTING_MODEL_AUDIT.json')
