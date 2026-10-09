"""Durable acquisition of the official full E3 database, isolated from prior pilots."""
from pathlib import Path
import json
import modal

app = modal.App('tg-detect-orthrus-e3')
volume = modal.Volume.from_name('tg-detect-orthrus-e3-v1', create_if_missing=True)
image = (modal.Image.from_registry('postgres:17-bookworm',add_python='3.10')
         .pip_install('gdown==5.2.0','psycopg2-binary==2.9.10')
         .add_local_file(Path(__file__).parent/'orthrus_comparison'/'acquisition.py','/root/acquisition.py',copy=True))

@app.function(image=image,cpu=4,memory=16384,timeout=86400,volumes={'/study':volume})
def acquire():
    import subprocess, traceback
    from acquisition import acquire as download
    root=Path('/study'); raw=root/'raw'; raw.mkdir(exist_ok=True)
    status=root/'acquisition.json'
    def record(**values):
        status.write_text(json.dumps(values,indent=2)); volume.commit()
    try:
        record(status='downloading',source='PIDSMaker authors database bundle',method='validated HTTP ranges')
        last_commit=0
        def progress(**info):
            nonlocal last_commit
            if info['bytes']-last_commit >= 128*1024*1024:
                record(status='downloading',**info);last_commit=info['bytes']
        extracted=download(raw,progress)
        for item in extracted:
            dest=Path(item['file'])
            toc=subprocess.run(['pg_restore','--list',str(dest)],capture_output=True,text=True)
            (raw/(dest.name+'.toc')).write_text(toc.stdout+toc.stderr)
            item['restore_list_returncode']=toc.returncode
            if toc.returncode: raise RuntimeError('Downloaded dump cannot be read by pg_restore')
        record(status='downloaded',datasets=extracted,training_started=False)
        return extracted
    except Exception:
        record(status='failed',error=traceback.format_exc(),training_started=False)
        raise

if __name__=='__main__':
    call=modal.Function.from_name('tg-detect-orthrus-e3','acquire').spawn()
    path=Path('artifacts/orthrus_e3_study');path.mkdir(parents=True,exist_ok=True)
    value=dict(app='tg-detect-orthrus-e3',call_id=call.object_id,volume='tg-detect-orthrus-e3-v1',stage='acquisition')
    (path/'launch.json').write_text(json.dumps(value,indent=2));print(json.dumps(value,indent=2))

# Kept separate so acquisition does not require downloading the GPU image.
study_image=(image.pip_install('torch==2.4.1','torch-geometric==2.5.3','numpy==1.26.4',
    'scipy==1.10.1','scikit-learn==1.2.0','networkx==2.8.7','xxhash==3.2.0',
    'graphviz==0.20.1','psutil','matplotlib==3.8.4','wandb==0.16.6','chardet==5.2.0',
    'nltk==3.8.1','igraph==0.11.5','cairocffi==1.7.0','gensim==4.3.1',
    'pytz==2024.1','pandas==2.2.2','yacs==0.1.8','pytest==8.3.5')
    .run_commands("python -c \"import os,certifi; os.environ['SSL_CERT_FILE']=certifi.where(); import nltk; nltk.download('punkt', raise_on_error=True)\"")
    .env({'PYTHONHASHSEED':'0','WANDB_MODE':'disabled','PYTHONPATH':'/root/research',
          'SSL_CERT_FILE':'/usr/local/lib/python3.10/site-packages/certifi/cacert.pem'})
    .add_local_dir(Path(__file__).parent/'orthrus_official','/root/research/orthrus_official',ignore=['.git'])
    .add_local_dir(Path(__file__).parent/'orthrus_comparison','/root/research/orthrus_comparison',ignore=['__pycache__']))
for source in ['node_memory_v3.py','node_memory_models.py','tgnn_components.py','temporal_cnn.py',
               'causal_tgn.py','causal_data.py','pomsathit_baseline.py']:
    study_image=study_image.add_local_file(Path(__file__).parent/source,'/root/research/'+source)

@app.function(image=study_image,cpu=16,memory=65536,timeout=86400,volumes={'/study':volume})
def prepare():
    import subprocess,os,glob,sys,platform,tempfile
    from orthrus_comparison.prepare_official import prepare as build
    volume.reload()
    manifest=Path('/study/prepared/manifest.json')
    if manifest.exists(): return str(manifest)
    dumps=list(Path('/study/raw').glob('*cadets*.dump'))
    if len(dumps)!=1: raise RuntimeError(f'Expected one complete CADETS dump; found {len(dumps)}')
    pg=Path(sorted(glob.glob('/usr/lib/postgresql/*/bin'))[-1])
    # A fresh private directory also permits a warm-container retry after a
    # failed build, without reusing an incomplete database.
    dbdir=tempfile.mkdtemp(prefix='orthrus-pg-',dir='/tmp')
    subprocess.run(['install','-d','-o','postgres','-g','postgres',dbdir],check=True)
    subprocess.run(['runuser','-u','postgres','--',str(pg/'initdb'),'-D',dbdir,'-A','trust'],check=True)
    subprocess.run(['runuser','-u','postgres','--',str(pg/'pg_ctl'),'-D',dbdir,'-l','/tmp/orthrus-postgres.log',
                    '-o','-p 55432 -h 127.0.0.1','-w','start'],check=True)
    subprocess.run([str(pg/'createdb'),'-h','127.0.0.1','-p','55432','-U','postgres','cadets_e3'],check=True)
    restored=subprocess.run([str(pg/'pg_restore'),'-h','127.0.0.1','-p','55432','-U','postgres',
                    '--no-owner','--no-privileges','-d','cadets_e3',str(dumps[0])],capture_output=True,text=True)
    Path('/study/restore.log').write_text(restored.stdout+restored.stderr);volume.commit()
    if restored.returncode: raise RuntimeError('Database restore failed; see restore.log')
    Path('/study/environment.json').write_text(json.dumps(dict(python=sys.version,
        platform=platform.platform(),postgres=subprocess.check_output([str(pg/'postgres'),'--version'],text=True).strip(),
        packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()),indent=2))
    os.environ.update(ORTHRUS_DB_HOST='127.0.0.1',ORTHRUS_DB_PORT='55432',ORTHRUS_DB_USER='postgres',ORTHRUS_DB_NAME='cadets_e3')
    try: build('/study/prepared')
    finally:
        subprocess.run(['runuser','-u','postgres','--',str(pg/'pg_ctl'),'-D',dbdir,'-m','fast','stop'])
        volume.commit()
    return str(manifest)

@app.function(image=study_image,gpu='L4',cpu=8,memory=65536,timeout=86400,volumes={'/study':volume},max_containers=3)
def train(arm):
    from orthrus_comparison.run import run
    volume.reload()
    try:
        result=run('/study/prepared/manifest.json','/study/results',epochs=20,device='cuda',arm=arm,
                   on_checkpoint=volume.commit)
        return {k:v for k,v in result.items() if k!='node_scores'}
    finally: volume.commit()

@app.function(image=study_image,cpu=8,memory=65536,timeout=86400,volumes={'/study':volume},max_containers=3)
def trace(arm):
    from orthrus_comparison.tracing import run_tracing
    volume.reload()
    try:
        return run_tracing('/study/prepared/manifest.json',f'/study/results/{arm}/result.json',
                           workers=8,on_checkpoint=volume.commit)
    finally:volume.commit()

@app.function(image=image,cpu=1,memory=1024,timeout=86400,volumes={'/study':volume})
def pipeline():
    import traceback
    path=Path('/study/status.json')
    def state(**value):
        path.write_text(json.dumps(value,indent=2));volume.commit()
    try:
        volume.reload()
        state(status='acquiring_or_validating');acquire.remote();volume.reload()
        state(status='preparing');prepare.remote();volume.reload()
        state(status='training',epochs=20)
        arms=['ORTHRUS_Reference','TGNN_Node_Mem_V3_Reconstruction','TGNN_Node_Mem_V4_KV']
        results=list(train.map(arms))
        volume.reload();state(status='tracing',results=results)
        traced=list(trace.map(arms))
        volume.reload();state(status='complete',results=results,
                              tracing=[{k:v for k,v in r.items() if k!='traced_nodes'} for r in traced])
    except Exception:
        state(status='failed',error=traceback.format_exc());raise
