"""Behavioral gates for the new architecture; synthetic metrics are not results."""
import copy
import os
import pytest
import torch
from torch import nn
from sliding_node_mem.model import TGNNSlidingNodeMem, TemporalCNN, provenance_links


def sample(n=28):
    events = [{'attributes': {'src': {'type': 'process', 'path': f'/bin/p{i%3}'},
                              'dst': {'type': 'file'}, 'optional': None}} for i in range(n)]
    return events, [i % 3 for i in range(n)], [3 + i % 2 for i in range(n)], [float(i//2) for i in range(n)], [i % 3 for i in range(n)]


def model():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(10)
    return TGNNSlidingNodeMem(8, num_relations=3, buckets=256, short_slots=3).to(os.environ.get('SLIDING_TEST_DEVICE', 'cpu'))


def test_default_detector_budget_matches_unmodified_orthrus_exactly():
    from sliding_node_mem.budget import verify_budget, TARGET
    report = verify_budget()
    assert report['trainable_parameters'] == {'ORTHRUS_Reference': TARGET, 'TGNN_Sliding_Node_Mem': TARGET}
    m = TGNNSlidingNodeMem(1)
    assert m.event_encoder.tokens.num_embeddings == 5303
    assert m.head[0].out_features == 130
    assert m.gat.projection.out_features == 128 and m.cnn.convs[0].out_channels == 128


def test_cached_tokenization_preserves_values_and_encoder_gradients():
    from orthrus_comparison.universal import UniversalEventEncoder
    from sliding_node_mem.encoding import CachedUniversalEventEncoder
    a=UniversalEventEncoder(buckets=256); b=CachedUniversalEventEncoder(buckets=256)
    b.load_state_dict(a.state_dict())
    events=[{}, {'type':'network','attributes':{'ip':'10.0.0.1','missing':None,'n':float('nan'),
             'list':[1,'/tmp/x',False], 'nested':{'name':'a'}}}, sample(1)[0][0]]
    x=a(events); y=b(events)
    torch.testing.assert_close(x,y,rtol=0,atol=0)
    x.sum().backward(); y.sum().backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0,msg=name)


def test_packed_snapshot_is_cpu_only_and_independent():
    m=model().eval()
    with torch.no_grad():m.forward_events(*sample(32))
    state=m.stream_state_dict()
    assert state['format']==3
    assert state['compressed_values'].device.type=='cpu' and state['memory_values'].device.type=='cpu'
    assert len(state['memory_nodes']) < m.n_nodes
    previous=state['memory_values'].clone()
    with torch.no_grad():m.forward_events([{}],[0],[1],[20.],[0])
    torch.testing.assert_close(state['memory_values'],previous,rtol=0,atol=0)


def test_memory_commit_updates_only_touched_rows_without_full_copy():
    m=model();args=sample(24)
    logits,_=m.forward_events(*args)
    assert m._pending_memory is not None
    pointer=m._memory.data_ptr()
    nn.functional.cross_entropy(logits,torch.tensor(args[-1],device=logits.device)).backward()
    m.detach_state()
    assert m._pending_memory is None and m._memory.data_ptr()==pointer
    assert not m.Mem.requires_grad and m.Mem.abs().sum()>0


def test_optimized_forward_gradients_and_updates_match_original():
    import importlib.util,sys
    from pathlib import Path
    path=Path(os.environ.get('SLIDING_REFERENCE_PATH',
              str(Path(__file__).resolve().parent/'artifacts/sliding_node_mem_e3/performance_before/model.py')))
    spec=importlib.util.spec_from_file_location('sliding_original_fixture',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    b=model();a=module.TGNNSlidingNodeMem(8,num_relations=3,buckets=256,short_slots=3).to(b.project.weight.device)
    a.load_state_dict(b.state_dict())
    oa=torch.optim.Adam(a.parameters(),lr=1e-5);ob=torch.optim.Adam(b.parameters(),lr=1e-5)
    args=sample(48)
    for lo in range(0,48,16):
        oa.zero_grad();ob.zero_grad()
        subset=[v[lo:lo+16] for v in args]
        x,_=a.forward_events(*subset);y,_=b.forward_events(*subset)
        torch.testing.assert_close(x,y,rtol=3e-5,atol=3e-5)
        target=torch.tensor(subset[-1],device=x.device)
        nn.functional.cross_entropy(x,target).backward();nn.functional.cross_entropy(y,target).backward()
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-6,msg=name)
        oa.step();ob.step();a.detach_state();b.detach_state()
        torch.testing.assert_close(a.Mem,b.Mem,rtol=3e-5,atol=3e-5)


def test_no_gru_missing_fields_and_gradient_paths():
    m = model()
    assert not any(isinstance(layer, (nn.GRU, nn.GRUCell)) for layer in m.modules())
    args = sample(32)
    args[0][0] = {}
    logits, aux = m.forward_events(*args)
    assert torch.isfinite(logits).all()
    assert aux['short_count'].max() > 0
    nn.functional.cross_entropy(logits, torch.tensor(args[-1], device=logits.device)).backward()
    for name in ('event_encoder.tokens.weight', 'gat.projection.weight', 'cnn.convs.0.weight',
                 'compress.0.weight', 'write_src.0.weight', 'admission.weight', 'importance_query.weight'):
        grad = dict(m.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    m.detach_state()
    assert not m.Mem.requires_grad


def test_stream_predictions_independent_of_batch_boundaries():
    a = model().eval(); b = copy.deepcopy(a)
    args = sample(33)
    with torch.no_grad():
        expected = a.forward_events(*args)[0]
        actual = torch.cat([b.forward_events(*(v[i:i+1] for v in args))[0] for i in range(33)])
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    torch.testing.assert_close(a.Mem, b.Mem, atol=3e-5, rtol=3e-5)
    assert {k: list(q) for k, q in a.history.items()} == {k: list(q) for k, q in b.history.items()}


def test_current_relation_and_future_events_cannot_change_prediction():
    a = model().eval(); b = copy.deepcopy(a); c = copy.deepcopy(a)
    args = sample(24)
    altered = copy.deepcopy(args)
    altered[-1][12] = (altered[-1][12] + 1) % 3
    future = copy.deepcopy(args)
    for i in range(13, 24): future[0][i] = {'attributes': {'future': 'changed'}}
    with torch.no_grad():
        x = a.forward_events(*args)[0]
        y = b.forward_events(*altered)[0]
        z = c.forward_events(*future)[0]
    torch.testing.assert_close(x[:13], y[:13], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x[:13], z[:13], atol=2e-5, rtol=2e-5)


def test_provenance_roles_and_cnn_are_causal():
    s, d = torch.tensor([[0, 2, 1]]), torch.tensor([[1, 3, 4]])
    links, roles = provenance_links(s, d, torch.ones(1, 3, dtype=torch.bool))
    assert links[0, 2, 0] and not links[0, 2, 1] and not links[0, 0, 2]
    assert roles[0, 2, 0, 1]
    cnn = TemporalCNN().eval()
    x = torch.randn(1, 9, 128); changed = x.clone(); changed[:, 5:] += 100
    valid = torch.ones(1, 9, dtype=torch.bool)
    torch.testing.assert_close(cnn(x, valid)[:, :5], cnn(changed, valid)[:, :5])


def test_top_half_novel_retention_bounds_and_snapshot_resume():
    a = model().eval(); args = sample(36)
    with torch.no_grad():
        _, aux = a.forward_events(*(v[:18] for v in args))
        # Nine active slots eventually select exactly five before novelty union.
        assert aux['selected'][-1].sum() == 5
        assert all(len(q) <= 8 for q in a.history.values())
        assert all(len(q) <= 3 for q in a.short.values())
        assert all(len({i for i, _, _ in q}) == len(q) for q in a.short.values())
        b = model().eval(); b.load_state_dict(a.state_dict()); b.load_stream_state_dict(a.stream_state_dict())
        x = a.forward_events(*(v[18:] for v in args))[0]
        y = b.forward_events(*(v[18:] for v in args))[0]
    torch.testing.assert_close(x, y)


def test_invalid_inputs_do_not_advance_stream_and_self_edges_work():
    m = model(); args = sample(12)
    args[2][:] = args[1]
    m.forward_events(*args)
    count = m.next_index
    with pytest.raises(ValueError): m.forward_events([{}], [0], [1], [-1.], [0])
    assert m.next_index == count
    with pytest.raises(ValueError): m.forward_events([{}], [None], [1], [20.], [0])
    m.reset_state()
    assert m.Mem is None and not m.history and not m.short


def test_rare_event_survives_even_when_attention_does_not_select_it():
    m = model().eval()
    m.novelty_threshold = 3.  # isolate structural rarity from semantic novelty
    nn.init.zeros_(m.importance_query.weight)
    nn.init.zeros_(m.importance_key.weight)
    args = sample(16); args[1][:] = [0]*16; args[2][:] = [1]*16; args[-1][:] = [0]*15+[2]
    with torch.no_grad(): _, aux = m.forward_events(*args)
    assert not aux['selected'][-1, -1] and aux['rare'][-1]
    assert m.records[15].selected


def test_training_detach_and_reset_across_optimizer_steps():
    m = model(); opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    args = sample(40)
    for lo in range(0, 40, 10):
        opt.zero_grad(set_to_none=True)
        logits, _ = m.forward_events(*(v[lo:lo+10] for v in args))
        loss = nn.functional.cross_entropy(logits, torch.tensor(args[-1][lo:lo+10], device=logits.device))
        loss.backward(); opt.step(); m.detach_state()
        assert torch.isfinite(loss)


def test_runner_mid_epoch_resume_matches_uninterrupted_weights(tmp_path):
    import json
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from torch_geometric.data import TemporalData
    from sliding_node_mem.run import run, ARM
    paths = {}
    for split, day in [('train', 3), ('val', 2), ('test', 6)]:
        start = int(datetime(2018, 4, day, 12, tzinfo=ZoneInfo('America/New_York')).timestamp()*1e9)
        src = torch.arange(80) % 40; dst = src + 40
        msg = torch.zeros(80, 272); msg[:, 131:141] = torch.eye(10)[torch.arange(80) % 10]
        path = tmp_path/(split+'.pt')
        torch.save(TemporalData(src=src, dst=dst, t=torch.arange(80)*1000000000+start, msg=msg), path)
        paths[split] = [str(path)]
    attrs = tmp_path/'attrs.json'; attrs.write_text(json.dumps({str(i): {'type': 'subject', 'path': f'/bin/{i%3}'} for i in range(80)}))
    truth = tmp_path/'truth.json'; truth.write_text('[1, 2]')
    manifest = tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(dataset='CADETS_E3', official_preprocessing=True,
        days=list(range(2, 14)), files=paths, n_nodes=80, node_attributes=str(attrs),
        ground_truth_nodes=str(truth), file_sha256={})))
    run(manifest, tmp_path/'whole', epochs=1, batch=16)
    class SimulatedInterruption(Exception): pass
    def interrupt(): raise SimulatedInterruption()
    with pytest.raises(SimulatedInterruption):
        run(manifest, tmp_path/'resumed', epochs=1, batch=16, checkpoint_seconds=0, on_checkpoint=interrupt)
    result = run(manifest, tmp_path/'resumed', epochs=1, batch=16)
    assert result['status'] == 'complete'
    a = torch.load(tmp_path/'whole'/ARM/'resume.pt', weights_only=False)
    b = torch.load(tmp_path/'resumed'/ARM/'resume.pt', weights_only=False)
    for key in a['model']: torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
