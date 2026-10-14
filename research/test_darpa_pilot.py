import numpy as np
from run_darpa_pilot import features
from causal_data import causal_egos

def test_label_uuid_time_do_not_enter_features():
    row={'type':'EVENT_READ','name':'read','exec':'python','size':4}
    altered={**row,'label':1,'src':'malicious','dst':'other','time_ns':99,'attack':'x'}
    np.testing.assert_array_equal(features(row),features(altered))

def test_strictly_past_history():
    src=np.array([1,1,1]);dst=np.array([2,2,2]);times=np.array([0.,0.,.1])
    ego,valid,ages=causal_egos(src,dst,times,4)
    assert valid[0].sum()==valid[1].sum()==1
    assert set(ego[2][valid[2]])=={0,1,2}
    assert np.all(ages[2,1:3]>0)
