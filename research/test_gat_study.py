"""Causality, role isolation and full-model integration for proposed operators."""
import copy
from pathlib import Path
import json

import pytest
import torch

from gat_study.operators import (links,RoleFactoredGAT,DotProductGAT,MeanAggregator,NoMessagePassing)
from gat_study.diagnostics import DegreeAudit
from gat_study.model import StudyV3,ARMS
from sliding_node_mem_upgrade.model import provenance_links
from sliding_node_mem_v3.model import TGNNSlidingNodeMem
from test_sliding_v3 import fixture,take


@pytest.fixture(autouse=True)
def configure():
    torch.manual_seed(42);torch.set_num_threads(1)


def test_masks_match_existing_provenance_graph():
    src=torch.tensor([[1,2,1,4,4],[1,1,1,1,1]])
    dst=torch.tensor([[2,3,4,4,8],[2,2,2,2,2]])
    valid=torch.tensor([[False,True,True,True,True],[False]*5])
    old,roles=provenance_links(src,dst,valid)
    new,r,_,_=links(src,dst,valid)
    assert torch.equal(new,old) and torch.equal(r,roles)


@pytest.mark.parametrize('model',[DotProductGAT,RoleFactoredGAT,lambda:RoleFactoredGAT(deviation=True),MeanAggregator,NoMessagePassing])
def test_causal_padding_and_finite_gradients(model):
    module=model()
    x=torch.randn(2,9,128,requires_grad=True)
    src=torch.tensor([[1]*9,[2]*9]);dst=torch.arange(18).reshape(2,9)
    valid=torch.tensor([[False]*9,[False]*2+[True]*7])
    y=module(x,src,dst,valid)
    assert torch.isfinite(y).all() and (y[~valid]==0).all()
    changed=x.detach().clone();changed[:,6:]+=100
    torch.testing.assert_close(y[:,:6],module(changed,src,dst,valid)[:,:6],rtol=0,atol=0)
    y.square().mean().backward();assert torch.isfinite(x.grad).all()


def test_role_head_content_isolated_before_output_projection():
    # Query 2 has an ss neighbor 0 and a dd neighbor 1, with no other cross role.
    module=RoleFactoredGAT()
    with torch.no_grad():
        module.output.weight.copy_(torch.eye(128));module.output.bias.zero_()
    x=torch.randn(1,3,128)
    src=torch.tensor([[1,8,1]]);dst=torch.tensor([[7,9,9]])
    valid=torch.ones(1,3,dtype=torch.bool)
    reference=module.message(x,src,dst,valid)
    changed=x.clone();changed[:,0]+=torch.randn(128)
    updated=module.message(changed,src,dst,valid)
    # Changing ss-only neighbor changes ss head; cannot change sd/ds/dd query heads.
    assert not torch.allclose(reference[:,2,:32],updated[:,2,:32])
    torch.testing.assert_close(reference[:,2,32:],updated[:,2,32:],rtol=0,atol=0)


def test_degree_audit_separates_query_and_history():
    audit=DegreeAudit()
    src=torch.tensor([[1,1,1],[1,8,11]])
    dst=torch.tensor([[2,3,4],[2,9,12]])
    valid=torch.ones(2,3,dtype=torch.bool)
    audit.observe(src,dst,valid);result=audit.result()
    assert result['query_degree_including_self']['histogram']=={1:1,3:1}
    assert result['query_degree_including_self']['self_only_fraction']==.5
    assert result['history_degree_including_self']['histogram']=={1:3,2:1}


def test_gat2_full_model_is_existing_v3_and_common_weights_match(tmp_path):
    bank,batch,_=fixture(tmp_path,corrected=True)
    torch.manual_seed(5);reference=TGNNSlidingNodeMem(8,bank,num_relations=3,signatures=3).eval()
    common=reference.state_dict()
    for arm in ARMS:
        torch.manual_seed(5);model=StudyV3(8,bank,num_relations=3,signatures=3,operator=arm).eval()
        for name,weight in model.state_dict().items():
            if not name.startswith(('gat.','gat_second.')):
                torch.testing.assert_close(weight,common[name],rtol=0,atol=0)
        if arm=='gat2':
            torch.testing.assert_close(reference.forward_batch(batch)[0],model.forward_batch(batch)[0],rtol=0,atol=0)


@pytest.mark.parametrize('arm',ARMS)
def test_full_model_batch_replay_gradients_and_degree_hook(tmp_path,arm):
    bank,batch,_=fixture(tmp_path,corrected=True)
    model=StudyV3(8,bank,num_relations=3,signatures=3,operator=arm).eval()
    chunked=StudyV3(8,bank,num_relations=3,signatures=3,operator=arm).eval()
    chunked.load_state_dict(model.state_dict())
    audit=DegreeAudit();hook=model.gat.register_forward_pre_hook(audit.hook)
    whole=model.forward_batch(batch)[0];hook.remove()
    parts=torch.cat([chunked.forward_batch(take(batch,a,b))[0] for a,b in [(0,1),(1,35),(35,96)]])
    torch.testing.assert_close(whole,parts,rtol=3e-4,atol=3e-5)
    whole.square().mean().backward()
    assert model.project.weight.grad is not None and torch.isfinite(model.project.weight.grad).all()
    degree=audit.result()
    # Active-window records are incident to a query endpoint: query sees every retained slot.
    assert degree['query_shared_history_fraction']==1


def test_training_copy_only_changes_documented_imports_and_source_list():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'gat_study/TRAINER_SOURCE.json').read_text())
    text=(root/manifest['source']).read_text()
    for a,b in manifest['changes'].items():text=text.replace(a,b)
    assert text==(root/'gat_study/training.py').read_text()


def test_full_training_calibration_minority_and_timing_runner(tmp_path):
    import hashlib
    import numpy as np
    from gat_study.training import run as train
    from gat_study.run import audit,minority_metrics,paired_timing
    from sliding_node_mem_v3.prepare import EVENT_DTYPE
    prepared=tmp_path/'prepared';prepared.mkdir()
    bank,batch,_=fixture(prepared,corrected=True)
    hashes={}
    for split in ('train','val','test'):
        rows=np.zeros(len(batch['src']),dtype=EVENT_DTYPE)
        for key in EVENT_DTYPE.names:rows[key]=batch[key].numpy()
        path=prepared/(split+'.bin');rows.tofile(path)
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (prepared/'ground_truth_nodes.json').write_text('[0]')
    manifest={'compatible':False,'n_nodes':8,'relations':3,'signatures':3,
              'events':{'train':96,'val':96,'test':96},'negative_eligible':{'train':96,'val':0,'test':0},'checksums':hashes}
    (prepared/'manifest.json').write_text(json.dumps(manifest))
    init=tmp_path/'initial.pt'
    from sliding_node_mem_v3.tokens import PackedEncoder
    torch.save({'tokens':PackedEncoder(bank).tokens.weight.detach(),
                'token_config':bank.config,'source_manifest_sha256':hashlib.sha256((prepared/'manifest.json').read_bytes()).hexdigest()},init)
    degree=audit(prepared,tmp_path/'audit.json',device='cpu',batches=2,batch=16)
    assert degree['query_degree_including_self']['rows']==32
    outputs=tmp_path/'outputs'
    for arm in ('gat2','role4'):
        result=train(prepared,outputs/arm/'0',epochs=1,batch=16,device='cpu',seed=0,
                     initialization=init,frozen_eval=True,model_options={'operator':arm})
        assert result['status']=='complete'
        subset=minority_metrics(prepared,outputs/arm/'0',result,[0])
        assert subset['scored_minority_nodes']==1
    timing=paired_timing(prepared,outputs,['gat2','role4'],[0],device='cpu',batch=16,
                         repetitions=1,warmup=1,measured=2)
    assert len(timing)==2 and all(len(r['samples_ms'])==2 for r in timing)


def test_colab_pilot_full_run_with_frozen_encoder(tmp_path):
    import hashlib
    import numpy as np
    from gat_study.colab_pilot import Config,run,build,DTYPE
    prepared=tmp_path/'pilot';prepared.mkdir()
    bank,batch,_=fixture(prepared,corrected=True)
    checksums={}
    for split in ('train','val','test'):
        rows=np.zeros(len(batch['src']),dtype=DTYPE)
        for key in DTYPE.names:rows[key]=batch[key].numpy()
        path=prepared/(split+'.bin');rows.tofile(path);checksums[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (prepared/'ground_truth_nodes.json').write_text('[0]')
    from sliding_node_mem_v3.tokens import PackedEncoder
    state=PackedEncoder(bank).state_dict()
    np.savez_compressed(prepared/'encoder_weights.npz',**{k:v.numpy() for k,v in state.items()})
    checksums['encoder_weights.npz']=hashlib.sha256((prepared/'encoder_weights.npz').read_bytes()).hexdigest()
    (prepared/'manifest.json').write_text(json.dumps({'n_nodes':8,'relations':3,'signatures':3,
        'events':{'train':96,'val':96,'test':96},'checksums':checksums}))
    cfg=Config(prepared=str(prepared),output=str(tmp_path/'results'),device='cpu',epochs=1,batch=16,
               seeds=(0,),timing_repetitions=1,timing_warmup=1,timing_batches=2,bootstrap=3)
    model,_=build(cfg,'gat2',0)
    assert all(not p.requires_grad for p in model.event_encoder.parameters())
    result=run(cfg)
    assert len(result)==2 and all(r['encoder_frozen'] for r in result)
    assert (Path(cfg.output)/'paired_latency.json').exists()
