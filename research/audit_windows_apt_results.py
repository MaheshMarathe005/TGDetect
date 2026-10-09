"""Independent local saved-prediction, split, checkpoint and lineage audit."""
from pathlib import Path
import json
import numpy as np
import torch
from windows_apt_models import make_model,ARMS
from windows_apt_data import digest,write_json
from run_windows_apt import verify

def audit():
 torch.set_num_threads(1)
 root=Path('artifacts/windows_apt_v1');out=root/'remote/results'
 meta=json.loads((root/'prepared/data_manifest.json').read_text());data=np.load(root/'prepared/data.npz',allow_pickle=False)
 manifest_hash=digest(root/'bundle_manifest.json');runs=[]
 for arm in ARMS:
  for seed in range(5):
   path=out/f'{arm}_seed{seed}.json'
   if not path.exists():continue
   row=json.loads(path.read_text());verify(row,out,meta)
   assert row['bundle_manifest_sha256']==manifest_hash
   pred=np.load(out/f'{arm}_seed{seed}_predictions.npz',allow_pickle=False)
   np.testing.assert_array_equal(pred['y'],data['y']);np.testing.assert_array_equal(pred['row_ids'],data['row_ids'])
   model=make_model(arm,meta['input_dim']);ck=torch.load(out/f'{arm}_seed{seed}.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['state_dict']);model.eval()
   indices=np.linspace(0,len(data['y'])-1,64,dtype=int)
   with torch.no_grad():p=model.forward_events(torch.from_numpy(data['x'][data['ego'][indices]]),torch.from_numpy(data['valid'][indices]),torch.from_numpy(data['ages'][indices]),None,None,None)[0].softmax(-1)[:,1].numpy()
   np.testing.assert_allclose(p,pred['probability'][indices],rtol=1e-4,atol=2e-5)
   runs.append(row['run_id'])
 result={'status':'PASS' if len(runs)==10 else 'PARTIAL','verified_runs':runs,'expected_runs':10,'label_scope':meta['label_scope'],'saved_labels_ids_metrics_checkpoints':'PASS','sampled_checkpoint_inference':'PASS'}
 write_json(root/'local_result_verification.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':audit()
