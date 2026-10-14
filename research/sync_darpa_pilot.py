"""Fetch durable pilot results and verify saved checkpoint hashes."""
from pathlib import Path
import hashlib,json
import modal
import numpy as np
from run_causal_pilot import metrics,select_threshold

root=Path('artifacts/darpa_e3_pilot_v1/remote')
volume=modal.Volume.from_name('tg-detect-darpa-e3-pilot-v1')
for sub in ['results','prepared','gpu_tests.txt','training_summary.json']:
    try:entries=list(volume.iterdir(sub,recursive=True))
    except Exception as exc:print(sub,type(exc).__name__);continue
    for item in entries:
        if item.type!=modal.volume.FileEntryType.FILE:continue
        rel=Path(item.path.lstrip('/'))
        if rel.name=='events.jsonl':continue
        target=root/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==item.size and target.suffix not in ['.json','.txt']:continue
        tmp=target.with_suffix(target.suffix+'.download')
        with tmp.open('wb') as f:
            for block in volume.read_file(item.path):f.write(block)
        tmp.replace(target)
for path in sorted(root.glob('results/*/*_seed*.json')):
    row=json.loads(path.read_text());checkpoint=path.with_suffix('.pt')
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==row['checkpoint_sha256']
    prepared=np.load(path.parent/'prepared.npz',allow_pickle=False)
    prediction=np.load(path.with_name(path.stem+'_predictions.npz'),allow_pickle=False)
    np.testing.assert_array_equal(prepared['y'],prediction['y'])
    np.testing.assert_array_equal(prepared['ids'],prediction['ids'])
    tr,va=int(prepared['train_end']),int(prepared['val_end'])
    assert select_threshold(prediction['y'][tr:va],prediction['probability'][tr:va])==row['threshold']
    assert metrics(prediction['y'][va:],prediction['probability'][va:],row['threshold'])==row['test']
    print(path.parent.name,path.stem,'F1',row['test']['f1'],'checkpoint hash PASS')
for path in sorted(root.glob('results/*/progress.json')):print(path.parent.name,path.read_text())
