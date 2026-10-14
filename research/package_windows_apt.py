"""Freeze exact code/protocol/prepared tensors and raw canonical CSV into uploaded ZIP."""
from pathlib import Path
import json,zipfile,subprocess,sys
import numpy as np
from windows_apt_data import RAW,digest,write_json
from temporal_cnn import rehearsal_schedule
ROOT=Path(__file__).resolve().parent
out=ROOT/'artifacts/windows_apt_v1';prepared=out/'prepared'
meta=json.loads((prepared/'data_manifest.json').read_text())
assert meta['status']=='PASS' and meta['user_authorized']
result=subprocess.run([sys.executable,'-m','pytest','-q','-p','no:cacheprovider','test_windows_apt.py','test_temporal_cnn.py','test_tgnn_components.py','test_causal_pilot.py'],text=True,capture_output=True)
(out/'local_tests.txt').write_text(result.stdout+result.stderr)
assert result.returncode==0,result.stdout+result.stderr
for seed in range(5):
 ids,sizes=rehearsal_schedule(meta['partitions']['train']['end'],200,256,seed)
 np.savez_compressed(prepared/f'replay_seed{seed}.npz',ids=ids,sizes=sizes)
files={}
for name in ['combined.csv','README.md','scenario_manifest.csv','validation_summary.csv','checksums.sha256']:files['raw/'+name]=RAW/name
for p in prepared.iterdir():
 if p.is_file():files['prepared/'+p.name]=p
sources=['windows_apt_data.py','windows_apt_models.py','run_windows_apt.py','modal_windows_apt.py',
 'test_windows_apt.py','pomsathit_baseline.py','causal_data.py','causal_tgn.py','tgnn_components.py',
 'temporal_cnn.py','run_causal_pilot.py','test_temporal_cnn.py','test_tgnn_components.py','test_causal_pilot.py']
for name in sources:files['source/'+name]=ROOT/name
files['PLAN.md']=ROOT/'windows_apt/PLAN.md';files['local_tests.txt']=out/'local_tests.txt'
manifest={'label_scope':'WEAK_LABEL_WAZUH_MITRE_PRESENCE','user_authorized':True,
 'files':{name:digest(path) for name,path in files.items()},'note':'All original period checksums verified locally; upload canonical combined CSV only to prevent double counting.'}
write_json(out/'bundle_manifest.json',manifest)
with zipfile.ZipFile(out/'windows_apt_v1.zip','w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for name,path in files.items():z.write(path,name)
 z.write(out/'bundle_manifest.json','bundle_manifest.json')
archive=out/'windows_apt_v1.zip'
write_json(out/'package.json',{'zip_sha256':digest(archive),'bytes':archive.stat().st_size,'members':len(files)+1,'local_tests':'PASS'})
print((out/'package.json').read_text())
