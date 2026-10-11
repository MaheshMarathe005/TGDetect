"""Verify the standalone package against its recorded SHA-256 inventory."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'PACKAGE_MANIFEST.json').read_text())
failures=[]
for name,record in manifest['files'].items():
    path=root/name
    if not path.is_file():
        failures.append(name+': missing');continue
    checksum=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            checksum.update(block)
    if path.stat().st_size!=record['bytes'] or checksum.hexdigest()!=record['sha256']:
        failures.append(name+': changed')
if failures:
    raise SystemExit('\n'.join(failures))
print(f"Verified {len(manifest['files'])} packaged files")
