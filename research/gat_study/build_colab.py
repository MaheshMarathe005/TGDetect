"""Embed a real prepared CADETS prefix and standalone code into one notebook."""
import base64
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;ROOT=RESEARCH.parent
DTYPE=np.dtype([(k,'<i8') for k in ('src','dst','relation','query','observed','signature','src_identity','dst_identity','negative_identity')]+[('time','<f8'),('negative_valid','?')])


def package(source,checkpoint,counts=(32768,16384,262144)):
    import torch
    files={};rows={}
    original=json.loads((source/'manifest.json').read_text())
    for split,count in zip(('train','val','test'),counts):
        path=source/(split+'.bin')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=original['checksums'][path.name]:raise ValueError('Source events checksum mismatch')
        rows[split]=np.array(np.memmap(path,mode='r',dtype=DTYPE)[:count])
    node_ids=np.unique(np.concatenate([r[k] for r in rows.values() for k in ('src','dst')]))
    identity_keys=('query','observed','src_identity','dst_identity','negative_identity')
    identities=np.unique(np.r_[0,np.concatenate([r[k] for r in rows.values() for k in identity_keys])])
    old_config=json.loads((source/'tokens/tokens.json').read_text())
    for relative in ('tokens/tokens.json','tokens/offsets.npy','tokens/fields.bin'):
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=original['checksums'][relative]:
            raise ValueError('Source token-bank checksum mismatch: '+relative)
    dtype=np.dtype([('key','<i4',(2,)),('word','<i4',(64,2)),('kind','<i8'),('number','<f4')])
    field=np.memmap(source/'tokens/fields.bin',mode='r',dtype=dtype)
    offsets=np.load(source/'tokens/offsets.npy');parts=[];new_offsets=[0]
    for i in identities:
        value=np.array(field[offsets[i]:offsets[i+1]]);parts.append(value);new_offsets.append(new_offsets[-1]+len(value))
    files['tokens/fields.bin']=np.concatenate(parts).tobytes()
    buffer=io.BytesIO();np.save(buffer,np.array(new_offsets,dtype=np.int64));files['tokens/offsets.npy']=buffer.getvalue()
    config={**old_config,'identities':len(identities),'fields':new_offsets[-1]}
    files['tokens/tokens.json']=json.dumps(config,indent=2).encode()
    truth=set(json.loads((source/'ground_truth_nodes.json').read_text()))
    mapped_truth=[int(np.searchsorted(node_ids,n)) for n in sorted(truth&set(node_ids.tolist()))]
    files['ground_truth_nodes.json']=json.dumps(mapped_truth).encode()
    for split,r in rows.items():
        for k in ('src','dst'):r[k]=np.searchsorted(node_ids,r[k])
        for k in identity_keys:r[k]=np.searchsorted(identities,r[k])
        files[split+'.bin']=r.tobytes()
    # Trusted local artifact: export only learned encoder tensors, no pickle in the data package.
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['signature']['model']['token_config']!=old_config:
        raise ValueError('Encoder checkpoint token configuration does not match source bank')
    weights={k.removeprefix('event_encoder.'):v.detach().cpu().numpy()
             for k,v in saved['model'].items() if k.startswith('event_encoder.')}
    buffer=io.BytesIO();np.savez_compressed(buffer,**weights);files['encoder_weights.npz']=buffer.getvalue()
    test_nodes=set(rows['test']['src'].tolist())|set(rows['test']['dst'].tolist())
    manifest={'dataset':'CADETS E3 prepared chronological prefix pilot','n_nodes':len(node_ids),
              'relations':original['relations'],'signatures':original['signatures'],
              'events':{s:len(r) for s,r in rows.items()},
              'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
              'encoder_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'encoder_scope':'existing V3 encoder trained on full official benign training; graph study uses prefixes; frozen across arms',
              'source_truth_count':len(truth),'prefix_test_truth_count':len(set(mapped_truth)&test_nodes),
              'prefix_truth_count_all_splits':len(mapped_truth),
              'minority_family_annotation':'unavailable; report malicious-node recall, not claimed minority-family recall',
              'sampling':'fixed earliest prefixes; no attack-centered slicing; historical test-informed development disclosed',
              'checksums':{p:hashlib.sha256(v).hexdigest() for p,v in files.items()}}
    files['manifest.json']=json.dumps(manifest,indent=2).encode()
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p,v in files.items():z.writestr(p,v)
    return buffer.getvalue(),manifest


def sources():
    names=['node_memory_v3.py','node_memory_models.py','tgnn_components.py','temporal_cnn.py','causal_tgn.py','pomsathit_baseline.py',
           'orthrus_comparison/universal.py','sliding_node_mem_upgrade/model.py','sliding_node_mem_upgrade/encoding.py',
           'sliding_node_mem_upgrade/windows.py','sliding_node_mem_upgrade/scoring.py','sliding_node_mem_v2/encoding.py',
           'sliding_node_mem_v2/layers.py','sliding_node_mem_v2/scoring.py',
           'sliding_node_mem_v3/model.py','sliding_node_mem_v3/memory.py','sliding_node_mem_v3/rings.py',
           'sliding_node_mem_v3/tokens.py','sliding_node_mem_v3/temporal.py','sliding_node_mem_v3/graphs.py',
           'gat_study/model.py','gat_study/operators.py','gat_study/diagnostics.py','gat_study/colab_pilot.py']
    result={name:(RESEARCH/name).read_text() for name in names}
    for p in ('orthrus_comparison','sliding_node_mem_upgrade','sliding_node_mem_v2','sliding_node_mem_v3','gat_study'):
        result[p+'/__init__.py']=''
    return result


def cell(kind,text):
    value={'cell_type':kind,'metadata':{},'source':text.splitlines(True)}
    if kind=='code':value.update(execution_count=None,outputs=[])
    return value


def build():
    payload,manifest=package(ROOT/'datasets/application/cadets_v3/prepared',ROOT/'datasets/application/cadets_v3/best.pt')
    bundle=sources();hashes={p:hashlib.sha256(v.encode()).hexdigest() for p,v in bundle.items()}
    cells=[cell('markdown','''# TGDetect: old GAT versus new role-factored GAT

Select GPU runtime (T4 or L4), then Run all. This single notebook includes a
compressed real CADETS prefix and all model code; no repo upload, Drive mount,
API key, or data download is needed.

**Old:** two original additive provenance GAT layers. **New:** one role-factored
four-head layer (ss/sd/ds/dd, separate masks, softmax and value subspaces).
Total output width 128. Both use the same frozen original V3 KV encoder,
nine-slot causal planner, CNN, short/long memory, decoder and losses. Common
non-GAT initial weights match by seed. The new operator changes depth AND
attention/message structure, so this pair cannot isolate their individual effects.

This is a small real-data pilot: earliest 32,768 training, 16,384 validation,
262,144 test events. The included encoder was trained on the full original benign
training corpus; this is not from-scratch prefix-only learning. Historical test
results have informed development. The prefix includes only a subset of ground
truth nodes and has no minority-family annotation: do not claim minority-family
recall or complete CADETS performance. Check exact coverage in the manifest.
Ground truth is evaluated at node level, not per-event attack labels.
'''),cell('code','''import importlib.util,subprocess,sys
packages={'numpy':'numpy','sklearn':'scikit-learn'}
missing=[name for module,name in packages.items() if importlib.util.find_spec(module) is None]
if missing:subprocess.check_call([sys.executable,'-m','pip','install',*missing])
import torch
assert torch.cuda.is_available(),'Choose Runtime → Change runtime type → GPU.'
print(torch.__version__,torch.cuda.get_device_name())
assert torch.cuda.mem_get_info()[0]>4*2**30,'Restart runtime to free GPU memory.'
''')]
    setup='''import base64,hashlib,io,json,sys,zipfile
from pathlib import Path
BUNDLE=Path('/content/tgdetect_gat_bundle');BUNDLE.mkdir(parents=True,exist_ok=True)
'''
    setup+='SOURCE_BUNDLE='+repr(bundle)+'\n'
    setup+='''for name,source in SOURCE_BUNDLE.items():
    p=BUNDLE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(source)
sys.path.insert(0,str(BUNDLE))
'''
    setup+='(BUNDLE/"source_hashes.json").write_text('+repr(json.dumps(hashes,indent=2))+')\n'
    setup+='DATA_B64='+repr(base64.b64encode(payload).decode())+'\n'
    setup+='DATA_SHA256='+repr(hashlib.sha256(payload).hexdigest())+'\n'
    setup+='''raw=base64.b64decode(DATA_B64)
assert hashlib.sha256(raw).hexdigest()==DATA_SHA256
DATA=Path('/content/tgdetect_gat_data');DATA.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(io.BytesIO(raw)) as z:
    for name in z.namelist():
        assert '..' not in Path(name).parts and not Path(name).is_absolute()
        p=DATA/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(name))
print((DATA/'manifest.json').read_text())
'''
    cells += [cell('code',setup),cell('markdown','''## Train paired seeds

Default 16 epochs, seeds 0/1/2, batch 256, two-batch unroll, 15-GiB allocator
budget. All learned components except KV encoder train jointly. The first third
of benign validation selects checkpoint; remaining thirds fit relation calibration
and the alert threshold (target 1% per node-window, not guaranteed node-level FPR).
If calibration lacks enough windows, the saved threshold is null and
`calibration.supported` is false; AP remains measurable, but threshold recall
cannot be interpreted as a valid operating point.
No test labels select checkpoints or thresholds. Test missing truth nodes are
reported. Each run's separate degree audit logs query/history and per-role degrees.

The degree audit is disabled during timing. After training, both models replay
the same chronological prefix with fresh state, alternating order over three
repetitions per seed. Full-model timing includes prepared host input transfer,
model forward and state updates, but excludes raw preparation, lookup build,
alert scoring and arrival/batch waiting. Cached KV lookup is identical in both.
'''),cell('code','''from gat_study.colab_pilot import Config,run
CFG=Config()
# Optional debug: CFG.epochs=1;CFG.seeds=(0,)
# CFG.timing_repetitions=1;CFG.timing_batches=8
print(CFG)
RESULTS=run(CFG)
'''),cell('code','''for r in RESULTS:
    print(r['arm'],r['seed'],'selected epoch',r['checkpoint_epoch'],r['metrics'])
print('Paired AP intervals:',(Path(CFG.output)/'paired_ap.json').read_text())
for seed in CFG.seeds:
    for arm in ['gat2','role4']:
        print('Degree:',arm,seed)
        print((Path(CFG.output)/f'{arm}_s{seed}/degree.json').read_text())
'''),cell('markdown','''## Export for review

The archive contains source hashes/snapshots, data sampling manifest, checkpoint
weights/hashes, histories, raw edge and node scores, degree diagnostics and paired
timings. Confidence intervals condition on this small node population and do
not establish capture-level uncertainty. This comparison does not add the new
GAT to TGNN Final automatically. Do not choose an operator by AP alone: compare
recall, false alerts, coverage, memory, training and full-model timing.
'''),cell('code','''import shutil
shutil.copy(BUNDLE/'source_hashes.json',Path(CFG.output)/'source_hashes.json')
shutil.copytree(BUNDLE,Path(CFG.output)/'source_snapshot',dirs_exist_ok=True)
archive=shutil.make_archive('/content/tgdetect_old_vs_new_gat_results','zip',CFG.output)
from google.colab import files
files.download(archive)
''')]
    for i,c in enumerate(cells):c['id']=f'gat-{i:03d}'
    notebook={'cells':cells,'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'},
               'language_info':{'name':'python'},'accelerator':'GPU','colab':{'name':'TGDetect_Old_vs_New_GAT.ipynb'}}}
    p=HERE/'TGDetect_Old_vs_New_GAT.ipynb';p.write_text(json.dumps(notebook,indent=1))
    print(p,'payload MB',len(payload)/1e6,'test truth',manifest['prefix_test_truth_count']);return notebook


if __name__=='__main__':build()
