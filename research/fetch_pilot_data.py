"""Fetch the two registered textual CTU captures."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import requests

SOURCES = {
    '06': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/detailed-bidirectional-flow-labels/capture20110816.binetflow',
    '11': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow',
}


def fetch(scenario):
    folder=Path('data/ctu13_verified'); folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'scenario{scenario}.binetflow'
    manifest_path=path.with_suffix('.source.json')
    if path.exists() or manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['url'] != SOURCES[scenario] or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:
            raise ValueError('Existing source/manifest mismatch; investigate before replacing it')
        print(f'Already verified {path}')
        return
    with requests.get(SOURCES[scenario],stream=True,timeout=(30,90)) as response:
        response.raise_for_status()
        temporary=path.with_suffix('.part')
        with temporary.open('wb') as f:
            for chunk in response.iter_content(1<<20): f.write(chunk)
        temporary.replace(path)
        manifest={'url':SOURCES[scenario],'final_url':response.url,
                  'retrieved_utc':datetime.now(timezone.utc).isoformat(),
                  'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                  'headers':{k:response.headers.get(k) for k in ['Content-Length','Last-Modified','ETag']},
                  'authentication':'HTTPS authoritative source; local checksum, no published reference hash'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--scenario',choices=SOURCES,default='06')
    fetch(parser.parse_args().scenario)
