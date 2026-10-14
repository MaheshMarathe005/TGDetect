"""Behavioral tests using the exact standalone source bundle shipped to Colab."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('build_encoder_notebook', HERE/'build_notebook.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
_bundle_dir = tempfile.TemporaryDirectory(prefix='tgdetect_encoder_test_bundle_')
for relative, source in builder.source_bundle().items():
    path = Path(_bundle_dir.name)/relative
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source)
# Import the standalone bundle without leaving its deliberately incomplete
# packages in the process-wide namespace used by the rest of the test suite.
_bundle_packages = ('orthrus_comparison', 'sliding_node_mem_upgrade',
                    'sliding_node_mem_v2', 'sliding_node_mem_v3')
_saved_packages = {name: module for name, module in list(sys.modules.items())
                   if name.split('.')[0] in _bundle_packages}
for name in _saved_packages:
    del sys.modules[name]
sys.path.insert(0, _bundle_dir.name)

from experiment import (Config, FieldCMS, CandidateEncoder, TokenWriter, TokenBank,
                        PackedEncoder, FEATURES, prepare_data, run, calibrated_threshold,
                        bootstrap_delta)
from optimizations import field_cache, schema_probes, execution_study, distill

followup_spec = importlib.util.spec_from_file_location('rarity_followup', HERE/'rarity_followup.py')
followup = importlib.util.module_from_spec(followup_spec)
sys.modules['rarity_followup'] = followup
followup_spec.loader.exec_module(followup)
stability_spec = importlib.util.spec_from_file_location('rarity_stability', HERE/'rarity_stability.py')
stability = importlib.util.module_from_spec(stability_spec)
sys.modules['rarity_stability'] = stability
stability_spec.loader.exec_module(stability)

sys.path.remove(_bundle_dir.name)
for name in list(sys.modules):
    if name.split('.')[0] in _bundle_packages:
        del sys.modules[name]
sys.modules.update(_saved_packages)

torch.set_num_threads(2)


def make_bank(path, events):
    writer = TokenWriter(path)
    ids = [writer.add(event) for event in events]
    writer.close()
    return TokenBank(path), torch.tensor(ids)


def synthetic_frame():
    """Pipeline fixture, deliberately not used as a detection benchmark."""
    rows = []
    for i in range(180):
        row = {key: (i*7 + j) % 11 for j, key in enumerate(FEATURES)}
        row.update(duration=i, protocol_type=('tcp' if i % 2 else 'udp'),
                   service=('http' if i % 2 else 'domain_u'), flag=('SF' if i % 3 else 'REJ'),
                   attack='normal.' if i < 140 else 'attack.')
        rows.append(row)
    # Exact duplicate inputs must never leak across splits.
    rows.extend([rows[0].copy(), rows[1].copy()])
    return pd.DataFrame(rows)


def test_original_parity_and_shared_gradients(tmp_path):
    bank, ids = make_bank(tmp_path/'tokens', [
        {'attributes': {'path': '/usr/bin/curl', 'n': 12}},
        {'attributes': {'path': '/tmp/a'}},
        {'attributes': {'path': '/usr/bin/curl', 'n': 12}},
    ])
    torch.manual_seed(4)
    old = PackedEncoder(bank)
    new = CandidateEncoder(bank)
    new.load_state_dict(old.state_dict(), strict=False)
    weights = torch.randn(len(ids), 64)
    a, b = old(ids), new(ids)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    (a*weights).sum().backward(); (b*weights).sum().backward()
    for name, parameter in old.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(new.named_parameters())[name].grad,
                                   atol=2e-5, rtol=2e-4)


def test_mask_hides_value_kind_number_and_rarity(tmp_path):
    bank, ids = make_bank(tmp_path/'tokens', [
        {'attributes': {'a': 1, 'b': 'same'}},
        {'attributes': {'a': '/different/value', 'b': 'same'}},
    ])
    model = CandidateEncoder(bank, rarity=True, interaction=True).eval()
    with torch.no_grad():
        model.film.weight.normal_(0, .1)
    hide = torch.zeros((2, 3), dtype=torch.bool); hide[:, 1] = True
    rarity = torch.zeros(2, 3); rarity[1, 1] = 8
    with torch.no_grad():
        embeddings, fields = model(ids, rarity, hide=hide, return_fields=True)
    torch.testing.assert_close(embeddings[0], embeddings[1])
    torch.testing.assert_close(fields[0], fields[1])


def test_dynamic_rarity_never_deduplicates_final_embedding(tmp_path):
    bank, ids = make_bank(tmp_path/'tokens', [{'attributes': {'a': 'same'}}]*2)
    model = CandidateEncoder(bank, rarity=True).eval()
    with torch.no_grad():
        model.film.weight.normal_(0, .2)
        out = model(ids, torch.tensor([[0., 0.], [0., 9.]]))
    assert not torch.allclose(out[0], out[1])


def test_static_field_cache_and_compact_parity(tmp_path):
    bank, ids = make_bank(tmp_path/'tokens', [
        {'attributes': {'a': 'shared', 'b': '/tmp/a'}},
        {'attributes': {'a': 'shared'}},
    ])
    model = CandidateEncoder(bank, rarity=True, interaction=True).eval()
    r = torch.ones(2, 3)
    with torch.no_grad():
        reference = model(ids, r)
        model.compact = True
        torch.testing.assert_close(model(ids, r), reference, atol=2e-6, rtol=2e-5)
        model.compact = False
        info = field_cache(model, Config(device='cpu', batch=2))
        torch.testing.assert_close(model(ids, r), reference, atol=2e-6, rtol=2e-5)
    assert info['unique_field_records'] < info['stored_field_records']


def test_counts_read_before_insert_and_eval_frozen(tmp_path):
    cms = FieldCMS(width=1024)
    assert cms.read('a', 'x') == 0
    cms.add('a', 'x')
    assert cms.read('a', 'x') == 1
    cfg = Config(root=str(tmp_path/'prepared'), train=32, val=8, calibration=16, test=16)
    data, keys, _, _, _ = prepare_data(cfg, synthetic_frame())
    np.testing.assert_array_equal(data['train']['rarity'][:, 0], np.log1p(np.arange(32)).astype(np.float32))
    for split in ('val', 'cal', 'test'):
        np.testing.assert_array_equal(data[split]['rarity'][:, 0], np.full(len(data[split]['ids']), np.log1p(32), np.float32))
    assert len(keys) == 41 and all('service' not in k for k in keys)


def test_threshold_is_strict_and_bootstrap_is_paired():
    values = np.arange(100)
    threshold = calibrated_threshold(values, .01)
    assert np.mean(values > threshold) <= .01
    y = np.array([0, 0, 1, 1]); score = np.array([.1, .2, .8, .9])
    interval = bootstrap_delta(y, score, score, 10)
    assert interval['ci95_low'] == interval['ci95_high'] == 0


def test_schema_probes_cover_tail_and_order():
    result = schema_probes()
    assert result['idf_retains_rare_key']
    assert result['retained_fields'] == 64
    assert result['list_permutation_max_error'] < 1e-4
    assert result['late_element_change_detected']


def test_all_arms_end_to_end_and_optional_cpu_paths(tmp_path):
    cfg = Config(root=str(tmp_path/'smoke'), train=32, val=8, calibration=16, test=16,
                 batch=16, epochs=1, pretrain_epochs=1, seeds=(0,), bootstrap=5,
                 latency_repeats=2, device='cpu', score_fields=2)
    result = run(cfg, synthetic_frame())
    assert len(result['table']) == 10
    assert result['table']['ap'].between(0, 1).all()
    execution = execution_study(result)
    for entry in execution:
        assert entry['max_abs_embedding_error'] < 1e-4
    distilled = distill(result, epochs=1)
    assert 0 <= distilled['metrics']['ap'] <= 1
    saved = json.loads((Path(cfg.root)/'results/full_s0/result.json').read_text())
    assert saved['optimizer_updates'] == 4


def test_notebook_cells_compile_and_baseline_snapshot_is_verbatim():
    notebook = builder.build()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<colab-cell>', 'exec')
    bundle = builder.source_bundle()
    for relative in builder.SOURCE_FILES:
        assert bundle[relative] == (builder.RESEARCH/relative).read_text()


def test_frozen_reference_counts_are_disjoint_and_shuffling_preserves_marginals(tmp_path):
    cfg = followup.FollowupConfig(root=str(tmp_path/'counts'), train=32, reference_rows=8,
                                 val=8, calibration=16, test=16, device='cpu')
    data, _, _, _, _ = prepare_data(cfg, synthetic_frame())
    variants, surprise, reference, audit = followup.count_variants(data, cfg)
    assert audit['task_rows'] == 24
    # Each synthetic duration is unique: optimizer records cannot count themselves.
    first = variants['frozen_log']['train']['field_identity'][0]
    position = next(i for i, raw in enumerate(first) if json.loads(raw)[0].endswith(':duration'))
    assert (variants['frozen_log']['train']['rarity'][:, position] == 0).all()
    assert all((v['rarity'][:, 0] == 0).all() for d in variants.values() for v in d.values())
    for split in data:
        np.testing.assert_array_equal(np.sort(variants['frozen_surprise'][split]['rarity'], axis=0),
                                      np.sort(variants['shuffled_surprise'][split]['rarity'], axis=0))
        assert surprise[split].min() >= 0 and surprise[split].max() <= 1


def test_tail_scaling_does_not_cap_extreme_evidence():
    reference = np.arange(10.)
    scaled = followup.tail_scale(reference, np.array([10., 100., 1000.]))
    assert scaled[2] > scaled[1] > scaled[0]


def test_followup_all_arms_two_seeds_and_split_exports(tmp_path):
    cfg = followup.FollowupConfig(root=str(tmp_path/'followup'), train=32, reference_rows=8,
             val=8, calibration=16, test=16, device='cpu', batch=16, epochs=1,
             seeds=(0,1), split_seeds=(20261013,20261014), bootstrap=3, latency_repeats=1)
    table, summary = followup.run_followup(cfg, synthetic_frame())
    assert len(table) == 24
    for split in cfg.split_seeds:
        for seed in cfg.seeds:
            counts = []
            for arm in followup.MATRIX:
                result = json.loads((Path(cfg.root)/f'split_{split}/results/{arm}_s{seed}/result.json').read_text())
                counts.append(result['updates'])
            assert len(set(counts)) == 1


def test_followup_notebook_compiles():
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location('build_rarity_notebook', HERE/'build_rarity_notebook.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    notebook = module.build()
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), '<followup-cell>', 'exec')


def test_vector_counts_match_python_and_collision_overcount():
    raw=np.array([[json.dumps(['a','x']),json.dumps(['b',1])],
                  [json.dumps(['a','x']),json.dumps(['b',2])]])
    hashes=stability.fingerprints(raw)
    table=stability.build_table(hashes,64)
    cms=FieldCMS(width=64)
    for row in raw:
        for pair in row:cms.add(*json.loads(pair))
    np.testing.assert_array_equal(table,cms.table)
    counts=stability.lookup(table,hashes)
    expected=np.array([[cms.read(*json.loads(pair)) for pair in row] for row in raw])
    np.testing.assert_array_equal(counts,expected)
    positions=torch.as_tensor((hashes % np.uint64(64)).astype(np.int64))
    output=stability.gpu_lookup(torch.as_tensor(table),positions)
    expected=np.log1p(expected).astype(np.float32);expected[:,0]=0
    np.testing.assert_allclose(output.numpy(),expected,rtol=1e-6)
    collided=stability.lookup(stability.build_table(hashes,1),hashes)
    assert (collided>=counts).all()


def test_stability_two_budgets_and_count_invariance_for_original(tmp_path):
    cfg=stability.StabilityConfig(root=str(tmp_path/'stability'),train=32,reference_rows=8,
        val=8,calibration=16,test=16,batch=16,seeds=(0,),split_seeds=(20261015,),
        budgets=(1,2),bootstrap=3,latency_repeats=1,timing_repeats=1,device='cpu')
    table,summary=stability.run_stability(cfg,synthetic_frame())
    assert len(table)==40
    for budget in cfg.budgets:
        original=table[(table.arm=='old') & (table.budget==budget)]
        assert original.ap.nunique()==1
        assert original.f1.nunique()==1
        assert (original.checkpoint_epoch<=budget).all()


def test_stability_notebook_compiles():
    sys.path.insert(0,str(HERE))
    spec=importlib.util.spec_from_file_location('build_stability_notebook',HERE/'build_stability_notebook.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    for cell in module.build()['cells']:
        if cell['cell_type']=='code':compile(''.join(cell['source']),'<stability-cell>','exec')
