import os
import numpy as np
import pytest
import torch
from causal_data import causal_egos,fit_scaler
from windows_apt_data import features,endpoints,boundaries,FEATURE_FIELDS
from windows_apt_models import make_model,ARMS
DEVICE=os.environ.get('TGNN_TEST_DEVICE','cpu')

def test_features_exclude_annotations_and_identity():
 row={k:'4' for k in FEATURE_FIELDS}; a=features(row)
 row.update({'_source.rule.mitre.id':'["T1003"]','_source.rule.description':'attack',
 '_source.rule.id':'42','_source.data.win.eventdata.ruleName':'T1003','_source.agent.id':'new',
 '_source.@timestamp':'future','_source.full_log':'malicious','_source.data.win.system.message':'malicious'})
 np.testing.assert_array_equal(a,features(row))
 assert a.shape==(72,) and np.isfinite(a).all()

def test_missing_fields_ports_and_endpoint_isolation():
 x=features({});assert np.isfinite(x).all() and x[70]==x[71]==1
 a=endpoints({'_source.agent.id':'1'},0);b=endpoints({'_source.agent.id':'1'},1)
 assert a[0]==b[0] and a[1]!=b[1]
 assert endpoints({'_source.agent.id':'1','_source.data.win.eventdata.processGuid':'A'},0)[1]!=endpoints({'_source.agent.id':'2','_source.data.win.eventdata.processGuid':'A'},0)[1]

def test_prefix_and_tie_safety():
 src=np.zeros(10,dtype=int);dst=np.arange(10)+1; times=np.array([0,1,1,2,3,4,5,5,6,7.])
 a=causal_egos(src,dst,times);b=causal_egos(src[:6],dst[:6],times[:6])
 for x,y in zip(a,b):np.testing.assert_array_equal(x[:6],y)
 assert 1 not in a[0][2,a[1][2]][1:]
 ends=boundaries(times)
 for end in ends: assert times[end]>times[end-1]

def test_scaler_train_only():
 x=np.arange(40,dtype=np.float32).reshape(10,4);a=fit_scaler(x,6);x[6:]=1e8
 b=fit_scaler(x,6)
 for v,w in zip(a,b):np.testing.assert_array_equal(v,w)

@pytest.mark.parametrize('arm',ARMS)
def test_models_gradients_padding_and_batch(arm):
 torch.manual_seed(2);model=make_model(arm).to(DEVICE)
 x=torch.randn(6,5,72,device=DEVICE);valid=torch.ones(6,5,dtype=torch.bool,device=DEVICE);valid[0,3:]=False
 ages=torch.rand(6,5,device=DEVICE);ages[:,0]=0
 def f(xx,vv,aa):return model.forward_events(xx,vv,aa,None,None,None)[0]
 model.eval();a=f(x,valid,ages)
 changed=x.clone();changed[0,3:]=1e4
 torch.testing.assert_close(a,f(changed,valid,ages))
 torch.testing.assert_close(a,torch.cat([f(x[:2],valid[:2],ages[:2]),f(x[2:],valid[2:],ages[2:])]),rtol=1e-4,atol=1e-5)
 model.train();opt=torch.optim.Adam(model.parameters(),lr=.001)
 before={n:p.detach().clone() for n,p in model.named_parameters()}
 loss=torch.nn.functional.cross_entropy(f(x,valid,ages),torch.arange(6,device=DEVICE)%2);loss.backward()
 for name,p in model.named_parameters():
  assert p.grad is not None and torch.isfinite(p.grad).all(),name
 for prefix in (['cnn','history_gru','query','time_encoder'] if arm!='GNN_H' else ['layers']):
  assert sum(float(p.grad.abs().sum()) for n,p in model.named_parameters() if n.startswith(prefix))>0
 opt.step()
 assert any(not torch.equal(before[n],p) for n,p in model.named_parameters())

def test_shared_initialization():
 torch.manual_seed(4);a=make_model('GNN_H')
 torch.manual_seed(4);b=make_model('TGNN_Q_A_C')
 for n,p in a.state_dict().items(): torch.testing.assert_close(p,b.state_dict()[n],rtol=0,atol=0)
