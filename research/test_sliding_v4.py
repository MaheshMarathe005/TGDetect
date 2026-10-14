import copy
import json
import os
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from test_sliding_v3 import fixture, take
from sliding_node_mem_v4.model import TGNNSlidingNodeMem
from sliding_node_mem_v4.temporal import TemporalAttention,ConvTemporalAttention
from sliding_node_mem_v4.memory import dependency_waves
from sliding_node_mem_v4.archive import EventArchive


@pytest.fixture(autouse=True)
def setup():
    torch.set_num_threads(1)
    torch.manual_seed(42)


def model(bank):
    return TGNNSlidingNodeMem(8, bank, num_relations=3, signatures=3,
        exact_counts=True, initial_capacity=2).to(os.environ.get('SLIDING_TEST_DEVICE','cpu'))


def test_temporal_exact_backends_masks_time_and_gradients():
    a = TemporalAttention(backend='manual')
    b = TemporalAttention(backend='sdpa'); b.load_state_dict(a.state_dict())
    x = torch.randn(3,9,128,requires_grad=True)
    valid = torch.tensor([[False]*9,[False]*4+[True]*5,[True]*9])
    time = torch.arange(9,dtype=torch.float64).repeat(3,1) + 1_700_000_000
    y = a(x,valid,time)
    torch.testing.assert_close(y,b(x,valid,time),rtol=2e-5,atol=2e-6)
    assert torch.isfinite(y).all() and (y[~valid]==0).all()
    altered=x.detach().clone();altered[:,7:]+=100
    changed=time.clone();changed[:,7:]+=10000
    torch.testing.assert_close(y[:,:7],a(altered,valid,changed)[:,:7])
    assert not torch.allclose(y[2,-1],a(x,valid,time*10)[2,-1])
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert a.gap[-1].weight.grad.abs().sum()>0


def test_dependency_schedule():
    src=[0,2,0,5,2,8];dst=[1,3,2,5,4,9]
    seen={}
    for level,wave in enumerate(dependency_waves(src,dst)):
        nodes=set()
        for i in wave:
            assert not nodes.intersection({src[i],dst[i]})
            nodes.update((src[i],dst[i]))
            for n in {src[i],dst[i]}:
                assert n not in seen or seen[n][0]<level and seen[n][1]<i
                seen[n]=(level,i)


def hybrid(bank):
    return TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,exact_counts=True,
        initial_capacity=2,encoder_dim=64,node_memory='affine-v3').to(os.environ.get('SLIDING_TEST_DEVICE','cpu'))


def test_cnn_single_head_masking_causality_and_gradients():
    device=os.environ.get('SLIDING_TEST_DEVICE','cpu')
    module=ConvTemporalAttention().to(device)
    x=torch.randn(3,9,128,device=device,requires_grad=True)
    valid=torch.tensor([[False]*9,[False]*4+[True]*5,[True]*9],device=device)
    times=torch.arange(9,device=device,dtype=torch.float64).repeat(3,1)+1_700_000_000
    y=module(x,valid,times)
    assert module.attention.heads==1 and torch.isfinite(y).all() and (y[~valid]==0).all()
    changed=x.detach().clone();changed[:,7:]+=100
    torch.testing.assert_close(y[:,:7],module(changed,valid,times)[:,:7])
    torch.testing.assert_close(y[:,-1],module.forward_last(x,valid,times),rtol=3e-5,atol=3e-6)
    y.square().mean().backward()
    for p in module.parameters(): assert p.grad is not None and torch.isfinite(p.grad).all()


def test_cnn_single_head_hybrid_batch_parity_and_training(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    options=dict(num_relations=3,signatures=3,exact_counts=True,initial_capacity=2,
                 encoder_dim=64,node_memory='affine-v3',temporal_heads=1,temporal_cnn=True)
    device=os.environ.get('SLIDING_TEST_DEVICE','cpu')
    a=TGNNSlidingNodeMem(8,bank,**options).to(device).eval()
    b=TGNNSlidingNodeMem(8,bank,**options).to(device).eval();b.load_state_dict(a.state_dict())
    with torch.no_grad():
        expected=a.forward_batch(data)[0]
        actual=torch.cat([b.forward_batch(take(data,i,i+1))[0] for i in range(96)])
        torch.testing.assert_close(actual,expected,rtol=6e-4,atol=6e-5)
    a.reset_state();a.train();losses=[]
    for lo,hi in ((0,48),(48,96)):
        logits,aux=a.forward_batch(take(data,lo,hi))
        losses.append(F.cross_entropy(logits,data['relation'][lo:hi].to(device))+.1*aux['pair_loss'])
    sum(losses).backward()
    for name in ('cnn.depthwise.weight','cnn.pointwise.weight','cnn.attention.qkv.weight','write_src.0.weight'):
        grad=dict(a.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name


def test_hybrid_v3_components_causality_sequential_and_resume(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    a,b=hybrid(bank).eval(),hybrid(bank).eval();b.load_state_dict(a.state_dict())
    assert a.event_encoder.tokens.weight.shape==(256,64)
    assert isinstance(a.pair_adapter,torch.nn.Identity)
    assert a.project.in_features==64 and a.episodic_dim==256
    assert not hasattr(a,'entity_memory') and 'surprise_trace' not in a.state.spec
    with torch.no_grad():
        expected=a.forward_batch(data)[0]
        actual=torch.cat([b.forward_batch(take(data,i,i+1))[0] for i in range(len(data['src']))])
        torch.testing.assert_close(actual,expected,rtol=6e-4,atol=6e-5)
        a.reset_state();a.forward_batch(take(data,0,47));saved=a.stream_state_dict()
        b.load_stream_state_dict(saved)
        torch.testing.assert_close(a.forward_batch(take(data,47,96))[0],b.forward_batch(take(data,47,96))[0])
        a.reset_state();b.reset_state()
        changed={k:v.clone() for k,v in data.items()};at=40
        changed['relation'][at]=(changed['relation'][at]+1)%3
        changed['observed'][at]=changed['query'][at]
        changed['src'][at+1:]=0;changed['dst'][at+1:]=0
        x=a.forward_batch(data)[0];y=b.forward_batch(changed)[0]
        torch.testing.assert_close(x[:at+1],y[:at+1],rtol=6e-4,atol=6e-5)
    with pytest.raises(ValueError):model(bank).load_stream_state_dict(saved)


def test_hybrid_gradients_through_parallel_memory_and_episodic_unroll(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    m=hybrid(bank).train();losses=[]
    for lo,hi in ((0,48),(48,96)):
        logits,aux=m.forward_batch(take(data,lo,hi))
        assert 'surprise' not in aux and 'dependency_waves' not in aux
        losses.append(F.cross_entropy(logits,data['relation'][lo:hi].to(logits.device))+.1*aux['pair_loss'])
    sum(losses).backward()
    for name in ('event_encoder.tokens.weight','project.weight','cnn.qkv.weight','cnn.gap.2.weight',
                 'write_src.0.weight','write_dst.0.weight','admission.weight','importance_key.weight','short_value.weight'):
        grad=dict(m.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
    opt=torch.optim.Adam(m.parameters(),lr=1e-5);opt.step();m.detach_state()
    assert m.state.pending is None


def test_cpu_prefetch_preserves_batch_columns_order_and_start(tmp_path):
    from sliding_node_mem_v3.prepare import EVENT_DTYPE, read_batches
    from sliding_node_mem_v4.prefetch import prefetch_batches
    _, data, _ = fixture(tmp_path, corrected=True)
    rows = np.zeros(len(data['src']), dtype=EVENT_DTYPE)
    for k, v in data.items(): rows[k] = v.numpy()
    path = tmp_path/'events.bin'; rows.tofile(path)
    expected = list(read_batches(path, 13, start=17))
    actual = list(prefetch_batches(read_batches(path, 13, start=17)))
    assert len(expected) == len(actual)
    for a, b in zip(expected, actual):
        for key in a: torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_cpu_prefetch_propagates_failure_and_closes_producer():
    from sliding_node_mem_v4.prefetch import prefetch_batches
    closed = []
    def producer():
        try:
            yield 1
            raise ValueError('reader failure')
        finally:
            closed.append(True)
    reader = prefetch_batches(producer())
    assert next(reader) == 1
    with pytest.raises(ValueError, match='reader failure'): next(reader)
    assert closed == [True]
    closed.clear()
    def endless():
        try:
            while True: yield 1
        finally: closed.append(True)
    reader = prefetch_batches(endless())
    assert next(reader) == 1
    reader.close()
    assert closed == [True]


@pytest.mark.parametrize('backend', ['manual', 'sdpa'])
def test_temporal_last_read_outputs_and_all_gradients(backend):
    device = os.environ.get('SLIDING_TEST_DEVICE', 'cpu')
    a = TemporalAttention(backend=backend).to(device)
    b = copy.deepcopy(a)
    x = torch.randn(5, 9, 128, device=device, requires_grad=True)
    z = x.detach().clone().requires_grad_()
    valid = torch.tensor([[False]*9, [False]*4+[True]*5, [True]*9,
                          [True]*8+[False], [True,False]*4+[True]], device=device)
    times = torch.arange(9, dtype=torch.float64, device=device).repeat(5, 1) * .001 + 1_700_000_000
    times[2] = times[2, 0]  # Equal timestamps are allowed.
    expected = a(x, valid, times)[:, -1]
    actual = b.forward_last(z, valid, times)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(z.grad, x.grad, rtol=3e-4, atol=2e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        torch.testing.assert_close(q.grad, p.grad, rtol=3e-4, atol=2e-5, msg=name)


def test_execution_migration_rejects_training_and_model_changes():
    from sliding_node_mem_v4.run import execution_migration
    previous={'seed':0,'batch':1024,'model':{'memory_device':'array'},
              'sources':{'sliding_node_mem_v4/model.py':'same','sliding_node_mem_v4/array_memory.py':'old'}}
    current=copy.deepcopy(previous);current['sources']['sliding_node_mem_v4/array_memory.py']='new'
    assert execution_migration(previous,current)==['sliding_node_mem_v4/array_memory.py']
    for key,value in (('batch',256),('seed',1),('model',{'memory_device':'cpu'})):
        changed=copy.deepcopy(current);changed[key]=value
        with pytest.raises(ValueError):execution_migration(previous,changed)
    current['sources']['sliding_node_mem_v4/model.py']='different'
    with pytest.raises(ValueError):execution_migration(previous,current)


def test_sequential_batch_parity_self_edges_and_resume(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    a,b=model(bank).eval(),model(bank).eval();b.load_state_dict(a.state_dict())
    with torch.no_grad():
        expected=a.forward_batch(data)[0];a.detach_state()
        actual=torch.cat([b.forward_batch(take(data,i,i+1))[0] for i in range(len(data['src']))])
        b.detach_state()
    torch.testing.assert_close(actual,expected,rtol=4e-4,atol=4e-5)
    for name in a.state.spec:
        x,y=a.state.read(torch.arange(5,device=expected.device))[name],b.state.read(torch.arange(5,device=expected.device))[name]
        torch.testing.assert_close(x,y,rtol=5e-4,atol=5e-5,equal_nan=True,msg=name)
    a.reset_state()
    with torch.no_grad(): a.forward_batch(take(data,0,47))
    checkpoint=a.stream_state_dict();b.load_stream_state_dict(checkpoint)
    with torch.no_grad():
        torch.testing.assert_close(a.forward_batch(take(data,47,96))[0],b.forward_batch(take(data,47,96))[0])


def test_target_and_future_causality(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    a,b=model(bank).eval(),model(bank).eval();b.load_state_dict(a.state_dict())
    changed={k:v.clone() for k,v in data.items()}
    # Alter target AND post-observation identity; current logits must remain identical.
    at=40
    changed['relation'][at]=(changed['relation'][at]+1)%3
    changed['signature'][at]=(changed['signature'][at]+1)%3
    changed['observed'][at]=changed['observed'][at+1]
    with torch.no_grad():
        x=a.forward_batch(data)[0];y=b.forward_batch(changed)[0]
    torch.testing.assert_close(x[:at+1],y[:at+1],rtol=3e-4,atol=3e-5)


def test_cpu_memory_partition_preserves_scores_and_gradients(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    a=model(bank).eval()
    b=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3,exact_counts=True,
        initial_capacity=2,memory_device='cpu').to(os.environ.get('SLIDING_TEST_DEVICE','cpu')).eval()
    b.load_state_dict(a.state_dict())
    x=a.forward_batch(data)[0];y=b.forward_batch(data)[0]
    torch.testing.assert_close(x,y,rtol=5e-4,atol=5e-5)
    x.square().mean().backward();y.square().mean().backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad.cpu(),q.grad.cpu(),rtol=3e-3,atol=3e-5,msg=name)


def test_array_recurrence_matches_reference(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    a,b=model(bank).eval(),model(bank).eval();b.load_state_dict(a.state_dict())
    b.memory_device='array'
    b.to(os.environ.get('SLIDING_TEST_DEVICE','cpu'))
    x=a.forward_batch(data)[0];y=b.forward_batch(data)[0]
    torch.testing.assert_close(x,y,rtol=5e-4,atol=5e-5)
    x.square().mean().backward();y.square().mean().backward()
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad.cpu(),q.grad.cpu(),rtol=3e-3,atol=3e-5,msg=name)


def test_compiled_array_training_matches_interpreted_unroll(tmp_path,monkeypatch):
    bank,data,_=fixture(tmp_path,n=48,corrected=True)
    a,b=model(bank).train(),model(bank).train();b.load_state_dict(a.state_dict())
    a.memory_device=b.memory_device='array'
    # Repartition recurrence/decoder parameters after selecting the CPU backend.
    a.to(os.environ.get('SLIDING_TEST_DEVICE','cpu'))
    b.to(os.environ.get('SLIDING_TEST_DEVICE','cpu'))
    def execute(m,compiled):
        monkeypatch.setenv('TG_V4_COMPILED_MEMORY',compiled)
        torch.manual_seed(123);losses=[];outputs=[]
        for lo,hi in ((0,24),(24,48)):
            logits,aux=m.forward_batch(take(data,lo,hi))
            outputs.append(logits)
            losses.append(F.cross_entropy(logits,data['relation'][lo:hi].to(logits.device))+.1*aux['pair_loss'])
        sum(losses).backward()
        return torch.cat(outputs)
    x=execute(a,'0');y=execute(b,'1')
    torch.testing.assert_close(x,y,rtol=5e-4,atol=5e-5)
    for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad.cpu(),q.grad.cpu(),rtol=3e-3,atol=3e-5,msg=name)


def test_last_read_matches_full_model_training_unroll(tmp_path):
    bank, data, _ = fixture(tmp_path, n=48, corrected=True)
    a, b = model(bank).train(), model(bank).train()
    b.load_state_dict(a.state_dict())
    a.memory_device = b.memory_device = 'array'
    a.to(os.environ.get('SLIDING_TEST_DEVICE', 'cpu'))
    b.to(os.environ.get('SLIDING_TEST_DEVICE', 'cpu'))
    # Reference execution uses the original complete attention output.
    a.cnn.forward_last = lambda x, valid, times: a.cnn(x, valid, times)[:, -1]
    def execute(m):
        torch.manual_seed(123)
        outputs, losses = [], []
        for lo, hi in ((0, 24), (24, 48)):
            logits, aux = m.forward_batch(take(data, lo, hi))
            outputs.append(logits)
            losses.append(F.cross_entropy(logits, data['relation'][lo:hi].to(logits.device)) + .1*aux['pair_loss'])
        sum(losses).backward()
        return torch.cat(outputs)
    torch.testing.assert_close(execute(b), execute(a), rtol=5e-4, atol=5e-5)
    for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(q.grad.cpu(), p.grad.cpu(), rtol=3e-3, atol=3e-5, msg=name)


def test_gradients_encoder_memory_and_repeated_optimizer_steps(tmp_path):
    bank,data,_=fixture(tmp_path,corrected=True)
    m=model(bank);opt=torch.optim.Adam(m.parameters(),lr=1e-4)
    # Eval disables dropout but retains all gradients and chronological training state.
    m.eval()
    for lo,hi in [(0,48),(48,96)]:
        opt.zero_grad();logits,aux=m.forward_batch(take(data,lo,hi))
        loss=F.cross_entropy(logits,data['relation'][lo:hi].to(logits.device))+.1*aux['pair_loss']
        assert 'compression_loss' not in aux
        loss.backward()
        for name in ('event_encoder.tokens.weight','cnn.qkv.weight','cnn.gap.2.weight',
                     'entity_memory.source.weight','entity_memory.gate.2.weight',
                     'entity_memory.decay_raw','short_value.weight','pair_adapter.weight'):
            grad=dict(m.named_parameters())[name].grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0,name
        opt.step();m.detach_state()
    assert m.event_encoder.tokens.weight.shape==(256,16)
    assert m.state.hist_c.shape[-1]==m.state.short_c.shape[-1]==256
    assert not hasattr(m,'compress') and not hasattr(m,'importance_query')


def test_archive_exact_bytes_idempotence_and_causal_retrieval(tmp_path):
    archive=EventArchive(tmp_path/'events.sqlite')
    payload=b'{ "value": "\\u0061", "unknown": [1, 2] }\n'
    archive.append(0,100.,1,1,payload)
    archive.append(0,100.,1,1,payload)
    archive.append(1,100.,1,2,b'next')
    archive.append(2,200.,1,3,b'future')
    rows=archive.retrieve(1,before_id=1,before_time=100.)
    assert len(rows)==1 and rows[0][-1]==payload
    with pytest.raises(ValueError):archive.append(0,100.,1,1,b'changed')
    archive.close()


def test_training_checkpoint_resume_matches_uninterrupted(tmp_path):
    from sliding_node_mem_v3.prepare import EVENT_DTYPE
    from sliding_node_mem_v4.run import run
    from orthrus_comparison.run import sha256
    prepared=tmp_path/'prepared';prepared.mkdir()
    bank,data,_=fixture(prepared,corrected=True)
    rows=np.zeros(len(data['src']),dtype=EVENT_DTYPE)
    for k,v in data.items():rows[k]=v.numpy()
    for split in ('train','val','test'):rows.tofile(prepared/(split+'.bin'))
    (prepared/'ground_truth_nodes.json').write_text('[1]')
    files=[p for p in prepared.rglob('*') if p.is_file()]
    meta=dict(n_nodes=8,relations=3,signatures=3,compatible=False,
              events={s:len(rows) for s in ('train','val','test')},negative_eligible={'train':len(rows)},
              checksums={str(p.relative_to(prepared)):sha256(p) for p in files})
    (prepared/'manifest.json').write_text(json.dumps(meta))
    options=dict(epochs=2,batch=32,unroll_batches=2,random_init=True,seed=7,
                 checkpoint_seconds=0.,frozen_eval=True,
                 model_options={'exact_counts':True,'initial_capacity':2})
    expected=run(prepared,tmp_path/'full',**options)
    class Interrupted(Exception):pass
    def stop():raise Interrupted()
    with pytest.raises(Interrupted):run(prepared,tmp_path/'resumed',on_checkpoint=stop,prefetch_batches=True,**options)
    actual=run(prepared,tmp_path/'resumed',prefetch_batches=True,**options)
    assert expected['metrics']==actual['metrics']
    for x,y in zip(expected['history'],actual['history']):
        assert x['loss']==pytest.approx(y['loss'],rel=2e-6,abs=2e-7)
        assert x['validation_ce']==pytest.approx(y['validation_ce'],rel=2e-6,abs=2e-7)
    x=torch.load(tmp_path/'full/best.pt',weights_only=False)['model']
    y=torch.load(tmp_path/'resumed/best.pt',weights_only=False)['model']
    # Scatter/gather reduction layouts can differ after restoring compact rows.
    for key in x:torch.testing.assert_close(x[key],y[key],rtol=2e-5,atol=2e-7)
