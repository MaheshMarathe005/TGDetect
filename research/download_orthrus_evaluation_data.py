"""Download only checksum-verified evaluation inputs from the study volume."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import time
import modal


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def download():
    root=Path('artifacts/orthrus_e3_study');data=root/'local_evaluation_data'
    manifest=json.loads((root/'prepared/manifest.json').read_text())
    prefix=str(Path(manifest['node_attributes']).parent.parent)+'/'
    paths=set(manifest['files']['val']+manifest['files']['test']+manifest['graph_files']['test'])
    paths.update(manifest[k] for k in ['node_attributes','ground_truth_nodes','attack_nodes'])
    volume=modal.Volume.from_name('tg-detect-orthrus-e3-v1')
    def fetch(path):
        relative=path.removeprefix(prefix);dest=data/relative
        expected=manifest['file_sha256'][path]
        if dest.exists() and sha(dest)==expected:return dest.stat().st_size
        dest.parent.mkdir(parents=True,exist_ok=True)
        for attempt in range(5):
            try:
                tmp=dest.with_suffix(dest.suffix+'.part')
                with tmp.open('wb') as handle:
                    for chunk in volume.read_file('/'+relative):handle.write(chunk)
                if sha(tmp)!=expected:raise ValueError('Downloaded input checksum mismatch: '+relative)
                tmp.replace(dest);return dest.stat().st_size
            except Exception:
                if attempt==4:raise
                time.sleep(attempt+1)
    started=time.time();total=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for index,future in enumerate(as_completed([pool.submit(fetch,p) for p in sorted(paths)]),1):
            total+=future.result()
            if index%20==0 or index==len(paths):
                progress=dict(verified_files=index,total_files=len(paths),verified_bytes=total,seconds=time.time()-started)
                (root/'local_download_progress.json').write_text(json.dumps(progress,indent=2))
                print(json.dumps(progress),flush=True)
    (data/'complete.json').write_text(json.dumps(dict(files=len(paths),bytes=total),indent=2))


if __name__=='__main__':download()
