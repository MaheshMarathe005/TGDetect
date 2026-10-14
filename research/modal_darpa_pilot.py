"""Download a bounded DARPA E3 sample directly into a dedicated Modal Volume."""
import modal

app = modal.App('tg-detect-darpa-e3-pilot')
volume = modal.Volume.from_name('tg-detect-darpa-e3-pilot-v1', create_if_missing=True)
image = modal.Image.debian_slim(python_version='3.10').pip_install('gdown==5.2.0', 'requests==2.32.3','fastavro==1.12.0').add_local_file('darpa_pilot_data.py','/root/darpa_pilot_data.py')
gpu_image=(modal.Image.debian_slim(python_version='3.10')
 .uv_pip_install('torch==2.7.1','numpy==2.2.6','scikit-learn==1.7.2','pytest==9.0.2')
 .env({'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','TGNN_TEST_DEVICE':'cuda:0'}))
for source in ['run_darpa_pilot.py','test_darpa_pilot.py','windows_apt_models.py','windows_apt_data.py','tgnn_components.py','temporal_cnn.py','causal_tgn.py','causal_data.py','pomsathit_baseline.py','run_causal_pilot.py','test_windows_apt.py','test_temporal_cnn.py']:
    gpu_image=gpu_image.add_local_file(source,'/root/'+source)

@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={'/study': volume})
def acquire():
    import hashlib, json, os, traceback
    from pathlib import Path
    import gdown, requests
    root = Path('/study/raw'); root.mkdir(parents=True, exist_ok=True)
    status = {'stage':'acquisition', 'datasets':{}, 'training_started':False}
    for name, file_id in [('cadets_e3','1AcWrYiBmgAqp7DizclKJYYJJBQbnDMfb'), ('theia_e3','1yksk4I9DBnl0Tca_Y0eaiNcpRNccB9sF')]:
        dest=root/name; dest.mkdir(exist_ok=True)
        try:
            path=gdown.download(id=file_id,output=str(dest)+'/',quiet=False,use_cookies=False)
            if not path: raise RuntimeError('Download returned no file')
            h=hashlib.sha256()
            with open(path,'rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            status['datasets'][name]={'path':path,'bytes':os.path.getsize(path),'sha256':h.hexdigest(),'drive_id':file_id}
        except Exception as exc:
            status['datasets'][name]={'error':str(exc),'drive_id':file_id}
        (Path('/study')/'acquisition.json').write_text(json.dumps(status,indent=2));volume.commit()
    return status

@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={'/study': volume})
def inspect():
    import tarfile, json, requests
    from pathlib import Path
    volume.reload()
    result={}
    for dataset in ['cadets_e3','theia_e3']:
        archives=list((Path('/study/raw')/dataset).glob('*.tar.gz'))
        result[dataset]={'archives':[], 'examples':{}}
        for path in archives:
            with tarfile.open(path,'r|gz') as tar:
                for member in tar:
                    result[dataset]['archives'].append({'member':member.name,'bytes':member.size})
                    if not member.isfile():continue
                    if not member.name.endswith('.json'):
                        result[dataset]['format']='binary archive; inspect Avro reader before parsing'
                        break
                    f=tar.extractfile(member)
                    for i,line in enumerate(f):
                        obj=json.loads(line); datum=obj.get('datum',{})
                        for kind,value in datum.items():
                            result[dataset]['examples'].setdefault(kind,value)
                        if i>=50000:break
                    break
    gt=requests.get('https://raw.githubusercontent.com/ProvenanceAnalytics/ground-truth/main/README.md',timeout=60);gt.raise_for_status()
    result['ground_truth_readme']=gt.text
    (Path('/study')/'inspection.json').write_text(json.dumps(result,indent=2));volume.commit()
    return result

@app.local_entrypoint()
def main(stage: str='acquire'):
    import json
    from pathlib import Path
    result=acquire.remote() if stage=='acquire' else prepare.remote() if stage=='prepare' else train.remote() if stage=='train' else finalize.remote() if stage=='finalize' else inspect.remote()
    out=Path('artifacts/darpa_e3_pilot_v1');out.mkdir(parents=True,exist_ok=True)
    (out/(stage+'.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

@app.function(image=image,cpu=4,memory=16384,timeout=3600,volumes={'/study':volume})
def prepare():
    from darpa_pilot_data import scan
    volume.reload();reports={}
    for name in ['cadets_e3','theia_e3']:
        reports[name]=scan('/study',name);volume.commit();print(reports[name],flush=True)
    return reports

@app.function(image=gpu_image,gpu='L4',cpu=4,memory=16384,timeout=10800,volumes={'/study':volume})
def train():
    import subprocess,sys,json
    from pathlib import Path
    from run_darpa_pilot import run
    volume.reload();reports={}
    tests=subprocess.run([sys.executable,'-m','pytest','-q','test_darpa_pilot.py','test_windows_apt.py','test_temporal_cnn.py'],cwd='/root',capture_output=True,text=True)
    Path('/study/gpu_tests.txt').write_text(tests.stdout+tests.stderr);volume.commit()
    assert tests.returncode==0,tests.stdout+tests.stderr
    for dataset in ['cadets_e3','theia_e3']:
        volume.reload()
        audit_path=Path('/study/prepared')/dataset/'audit.json'
        if not audit_path.exists():
            reports[dataset]={'status':'PENDING_PREPROCESSING'};continue
        audit=json.loads(audit_path.read_text())
        if not audit['training_gate']:
            reports[dataset]={'status':'BLOCKED_CLASS_COVERAGE','partitions':audit['partitions']};continue
        reports[dataset]=run('/study',dataset,volume.commit)
    Path('/study/training_summary.json').write_text(json.dumps(reports,indent=2));volume.commit();return reports

@app.function(image=image,cpu=1,memory=1024,timeout=300,volumes={'/study':volume})
def finalize():
    """Finalize durable completed results without allocating a GPU or retraining."""
    import json,hashlib
    from pathlib import Path
    volume.reload();reports={}
    for dataset in ['cadets_e3','theia_e3']:
        folder=Path('/study/results')/dataset
        audit_path=Path('/study/prepared')/dataset/'audit.json'
        if not audit_path.exists():reports[dataset]={'status':'PENDING_PREPROCESSING'};continue
        audit=json.loads(audit_path.read_text())
        if not audit['training_gate']:
            reports[dataset]={'status':'BLOCKED_CLASS_COVERAGE','partitions':audit['partitions']};continue
        verified=[]
        for arm in ['GNN_H','TGNN_Q_A_C']:
            for seed in range(3):
                path=folder/f'{arm}_seed{seed}.json'
                if not path.exists():continue
                row=json.loads(path.read_text())
                assert hashlib.sha256(path.with_suffix('.pt').read_bytes()).hexdigest()==row['checkpoint_sha256']
                verified.append(path.stem)
        reports[dataset]={'status':'COMPLETE' if len(verified)==6 else 'INCOMPLETE','verified_checkpoints':verified}
        if len(verified)==6:reports[dataset]['metrics']=json.loads((folder/'summary.json').read_text())
    Path('/study/training_summary.json').write_text(json.dumps(reports,indent=2));volume.commit();return reports
