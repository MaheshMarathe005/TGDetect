"""ZIP-on-Volume staging, CPU verification, then one L4 allocation."""
import modal
VOLUME='tg-detect-windows-apt-v1'
image=(modal.Image.debian_slim(python_version='3.10')
 .uv_pip_install('torch==2.7.1','numpy==2.2.6','scikit-learn==1.7.2','pytest==9.0.2')
 .env({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8'}))
app=modal.App('tg-detect-windows-apt-weak-label',image=image)
volume=modal.Volume.from_name(VOLUME,create_if_missing=True)

@app.function(cpu=4,memory=16384,timeout=1800,volumes={'/study':volume})
def stage_zip(zip_sha):
 from pathlib import Path,PurePosixPath
 import hashlib,json,zipfile,sys
 import numpy as np
 archive=Path('/study/upload/windows_apt_v1.zip')
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==zip_sha,'ZIP changed in transit'
 bundle=Path('/study/windows_apt_v1/bundle');bundle.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(archive) as z:
  names=z.namelist();assert len(names)==len(set(names))
  for name in names:
   p=PurePosixPath(name)
   if p.is_absolute() or '..' in p.parts or '\\' in name:raise ValueError('Unsafe ZIP member')
  z.extractall(bundle)
 manifest=json.loads((bundle/'bundle_manifest.json').read_text())
 for name,expected in manifest['files'].items():
  assert hashlib.sha256((bundle/name).read_bytes()).hexdigest()==expected,name
 assert manifest['label_scope']=='WEAK_LABEL_WAZUH_MITRE_PRESENCE' and manifest['user_authorized'] is True
 sys.path.insert(0,str(bundle/'source'))
 from windows_apt_data import prepare,write_json,digest
 meta=prepare(bundle/'raw',Path('/tmp/windows_apt_rebuilt'))
 original=np.load(bundle/'prepared/data.npz',allow_pickle=False);rebuilt=np.load('/tmp/windows_apt_rebuilt/data.npz',allow_pickle=False)
 assert set(original.files)==set(rebuilt.files)
 for key in original.files:np.testing.assert_array_equal(original[key],rebuilt[key],err_msg=key)
 # Validate every stored rehearsal schedule against the training boundary.
 n=int(original['train_end']);steps=200*((n+255)//256)
 for seed in range(5):
  a=np.load(bundle/f'prepared/replay_seed{seed}.npz',allow_pickle=False)
  ids,sizes=a['ids'],a['sizes']
  assert ids.shape==(steps,26) and sizes.shape==(steps,)
  assert sizes[0]==0 and np.all(sizes[1:]==26)
  for index in range(steps):
   used=ids[index,:int(sizes[index])]
   assert np.all((used>=0)&(used<n))
   if index<(steps//200):assert np.all(used<index*256)
 report={'status':'PASS','zip_sha256':zip_sha,'bundle_manifest_sha256':digest(bundle/'bundle_manifest.json'),
  'member_hashes':'PASS','raw_to_tensor_exact_reproduction':'PASS','replay_training_only':'PASS',
  'rows':meta['rows'],'partitions':meta['partitions'],'label_scope':manifest['label_scope']}
 write_json(bundle/'remote_verification.json',report);volume.commit();return report

@app.function(gpu='L4',cpu=16,memory=49152,timeout=21600,max_containers=1,volumes={'/study':volume})
def train():
 import sys
 from pathlib import Path
 bundle=Path('/study/windows_apt_v1/bundle');sys.path.insert(0,str(bundle/'source'))
 from run_windows_apt import orchestrate
 return orchestrate(bundle,Path('/study/windows_apt_v1/results'),volume.commit)

@app.local_entrypoint()
def main(stage: str='prepare',zip_sha: str=''):
 import json
 from pathlib import Path
 if stage=='prepare':
  result=stage_zip.remote(zip_sha)
  Path('artifacts/windows_apt_v1/remote_verification.json').write_text(json.dumps(result,indent=2)+'\n')
  print(json.dumps(result,indent=2))
 elif stage=='train':print(json.dumps(train.remote(),indent=2))
 else:raise ValueError(stage)
