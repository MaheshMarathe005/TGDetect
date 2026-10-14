"""Acceptance checks; synthetic fixtures are never research performance evidence."""
import copy
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import pytest
import torch

from node_memory_v4 import TGNNNodeMemV4, decayed_endpoint_scan
from kairos_comparison.data import convert, message, process, prepare, LABEL_MEMBER
from kairos_comparison.model import Kairos
from kairos_comparison.detection import detect, fit_rareness


def test_decay_scan_matches_sequential_values_and_gradients():
    torch.manual_seed(1)
    src = torch.tensor([0, 1, 0, 2, 1])
    dst = torch.tensor([1, 1, 2, 0, 0])
    times = torch.tensor([0., 2., 5., 9., 15.], dtype=torch.float64)
    initial = torch.randn(3, 4, requires_grad=True)
    weights = torch.rand(10, 4, requires_grad=True)
    content = torch.randn(10, 4, requires_grad=True)
    half = torch.tensor([3., 3., 20., 20.], requires_grad=True)
    last = torch.full((3,), float('nan'), dtype=torch.float64)
    ms, md, final, _ = decayed_endpoint_scan(initial, last, src, dst, times, weights, content, half)
    table = list(initial.unbind()); clock = [None]*3; ss, dd = [], []
    for i, (s, d) in enumerate(zip(src.tolist(), dst.tolist())):
        before = {}
        for node in {s, d}:
            dt = 0 if clock[node] is None else times[i]-clock[node]
            before[node] = table[node] * torch.exp(-np.log(2)*dt/half)
        ss.append(before[s]); dd.append(before[d])
        for row, node in [(i, s)] + ([(i+5, d)] if s != d else []):
            table[node] = (1-weights[row])*before[node] + weights[row]*content[row]
            clock[node] = times[i]
    reference = (torch.stack(ss), torch.stack(dd), torch.stack(table))
    for actual, expected in zip((ms, md, final), reference):
        torch.testing.assert_close(actual, expected)
    inputs = (initial, weights, content, half)
    actual_grad = torch.autograd.grad(sum(v.square().sum() for v in (ms, md, final)), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(sum(v.square().sum() for v in reference), inputs)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_v4_causality_chunk_invariance_and_time_decay():
    torch.manual_seed(3)
    model = TGNNNodeMemV4(n_nodes=3).eval()
    with torch.no_grad():
        model.read_gate.bias.fill_(.4)
    c = torch.randn(9, 128)
    src, dst = torch.arange(9)%3, (torch.arange(9)+1)%3
    times = torch.arange(9, dtype=torch.float64)*100
    with torch.no_grad():
        full = model._context_forward(c, src, dst, times)[0]
        model.reset_state()
        split = torch.cat([model._context_forward(c[lo:hi], src[lo:hi], dst[lo:hi], times[lo:hi])[0]
                           for lo,hi in [(0,2),(2,6),(6,9)]])
        torch.testing.assert_close(full, split, atol=2e-6, rtol=2e-5)
        model.reset_state()
        perturbed = c.clone(); perturbed[5:] += 100
        future = model._context_forward(perturbed, src, dst, times)[0]
        torch.testing.assert_close(full[:5], future[:5])
        model.reset_state()
        late_times = times.clone(); late_times[5:] += 86400
        late = model._context_forward(c, src, dst, late_times)[0]
        assert not torch.allclose(late[5:], full[5:])
    assert not any('last_seen' in k or k in {'Mem', 'Reference'} for k in model.state_dict())


def test_v4_task_gradient_reaches_decay_and_endpoint_gates():
    torch.manual_seed(5)
    model = TGNNNodeMemV4(n_nodes=2)
    with torch.no_grad():
        model.read_gate.bias.fill_(.5)
    c = torch.randn(8, 128)
    logits, aux = model._context_forward(c, torch.zeros(8,dtype=torch.long),
        torch.ones(8,dtype=torch.long), torch.arange(8,dtype=torch.float64)*1000)
    logits[-1].square().sum().backward()
    assert model.log_half_lives.grad.abs().sum() > 0
    assert (model.attn_score.weight.grad.abs().sum(1) > 0).all()
    assert aux['admit'].shape == (16,)


def test_v4_self_edge_admission_budget_counts_actual_writes():
    model = TGNNNodeMemV4(n_nodes=2).eval()
    with torch.no_grad():
        _, aux = model._context_forward(torch.randn(2,128), torch.tensor([0,1]),
            torch.tensor([0,0]), torch.tensor([0.,1.],dtype=torch.float64))
    assert aux['cand'][2] == 0 and aux['admit'][2] == 0
    assert aux['cand'][0] > 0 and aux['cand'][3] > 0


def event(code=11):
    return {'@timestamp':'2025-01-01T00:00:00Z', 'host':{'name':'CLIENT1'},
            'event':{'code':code, 'provider':'Microsoft-Windows-Sysmon'},
            'process':{'entity_id':'guid-a', 'pid':42, 'executable':'C:\\Windows\\a.exe'},
            'file':{'path':'C:\\tmp\\x.txt'},
            'winlog':{'record_id':1, 'channel':'Sysmon', 'event_data':{}}}


def test_ecs_guid_no_label_features_and_no_pid_merging():
    e = event()
    edge, reason = convert(e)
    assert reason == 'included' and '|guid|guid-a' in edge[0][0]
    x = message(edge)
    assert x.shape == (41,) and x[16:25].sum() == 1
    e['attack'] = {'label': 1}; e['rule'] = {'mitre':'arbitrary'}
    np.testing.assert_array_equal(x, message(convert(e)[0]))
    del e['process']['entity_id']
    first = process(e, {})[0]
    e['winlog']['record_id'] = 2
    assert process(e, {})[0] != first
    assert convert(event(22))[0] is None


def test_kairos_no_current_or_future_message_leak_and_training():
    torch.manual_seed(8)
    model = Kairos(4).eval(); model.reset_state()
    s, d = torch.tensor([0,1,2,0]), torch.tensor([1,2,3,3])
    t, x = torch.arange(4, dtype=torch.long)*1000000, torch.randn(4,41)
    with torch.no_grad():
        first = model.score(s[:2], d[:2], t, x)
        changed = x.clone(); changed[:] += 1000
        torch.testing.assert_close(first, model.score(s[:2], d[:2], t, changed))
        model.observe(s[:2], d[:2], t[:2], x[:2])
        second = model.score(s[2:],d[2:],t,x)
        changed = x.clone(); changed[2:] += 1000
        torch.testing.assert_close(second, model.score(s[2:],d[2:],t,changed))
    model.train(); model.reset_state()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for lo in (0,2):
        opt.zero_grad()
        out = model.score(s[lo:lo+2], d[lo:lo+2], t,x)
        model.observe(s[lo:lo+2],d[lo:lo+2],t[lo:lo+2],x[lo:lo+2])
        torch.nn.functional.cross_entropy(out, torch.tensor([1,4])).backward()
        assert model.decoder.lin_src.weight.grad is not None
        opt.step(); model.detach_state()
    assert 'memory.memory' not in model.learned_state()


def test_queue_online_never_changes_after_future_window():
    # One large RE outlier in each window, connected through rare node 9.
    src=np.array([0,0,0,9,0,0,0,9]); dst=np.array([1,1,1,8,1,1,1,7])
    w=np.array([1]*4+[2]*4); loss=np.array([.1,.1,.1,4,.1,.1,.1,4])
    rare=dict(counts={0:10,1:10},n=10,alpha=np.log(9))
    prefix=detect(src[:4],dst[:4],w[:4],loss[:4],rare)
    full=detect(src,dst,w,loss,rare)
    assert full[0]['online_score']==prefix[0]['online_score']
    assert full[0]['retrospective_score']>prefix[0]['retrospective_score']


def test_full_pipeline_smoke(tmp_path):
    from datetime import datetime, timedelta, timezone
    from kairos_comparison.run import train_one, report
    raw, labels = tmp_path/'raw.zip', tmp_path/'labels.zip'
    rows=[]; positives=[]
    for i in range(180):
        e=event(11 if i%2 else 7)
        e['@timestamp']=(datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(seconds=300*i)).isoformat()
        e['winlog']['record_id']=i
        e['process']['entity_id']=f'guid-{i%7}'
        rows.append(json.dumps(e))
        if (i//3)%5==3:
            positives.append(json.dumps(e))
    with zipfile.ZipFile(raw,'w') as z:
        z.writestr('events.jsonl','\n'.join(rows))
    with zipfile.ZipFile(labels,'w') as z:
        z.writestr(LABEL_MEMBER,'\n'.join(positives))
        z.writestr(LABEL_MEMBER.replace('class_1.jsonl','class_1_and_2.jsonl'),'\n'.join(positives))
    prepared=tmp_path/'prepared'
    meta=prepare(raw,labels,prepared)
    assert meta['counts']['unique_events']==180
    output=tmp_path/'synthetic_smoke_only'
    results=[train_one(prepared/'prepared.npz',output,arm,epochs=1,batch=16,
                      device=os.environ.get('TGNN_COMPARISON_TEST_DEVICE', 'cpu'))
             for arm in ['KAIROS','TGNN_Node_Mem_V3','TGNN_Node_Mem_V4']]
    assert len({r['feature_sha256'] for r in results})==1
    assert len({r['event_ids_sha256'] for r in results})==1
    assert results[0]['train_events_per_epoch']<results[1]['train_events_per_epoch']
    report(output)
    assert (output/'RESULTS.md').exists()
    # Reload learned weights and regenerate evaluation from epoch checkpoints.
    # No runtime memory is restored; chronological warming must reproduce scores.
    for result in results:
        arm = result['arm']
        prior = np.load(output/f'{arm}_seed0_predictions.npz')['score'].copy()
        (output/f'{arm}_seed0.json').unlink()
        train_one(prepared/'prepared.npz',output,arm,epochs=1,batch=16,
                  device=os.environ.get('TGNN_COMPARISON_TEST_DEVICE', 'cpu'))
        np.testing.assert_allclose(np.load(output/f'{arm}_seed0_predictions.npz')['score'],prior,
                                   rtol=1e-5,atol=1e-6)
    from kairos_comparison.audit import audit
    assert audit(prepared/'prepared.npz',output,expected_seeds=1)['status']=='passed'
