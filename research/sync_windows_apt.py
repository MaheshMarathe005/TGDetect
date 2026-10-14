"""Download only new/changed durable Windows-APT artifacts from Modal Volume."""
from pathlib import Path
import json,modal
from windows_apt_data import write_json
ROOT=Path('artifacts/windows_apt_v1/remote')
volume=modal.Volume.from_name('tg-detect-windows-apt-v1')
for sub in ['windows_apt_v1/results','windows_apt_v1/bundle/remote_verification.json']:
 try:entries=list(volume.iterdir(sub,recursive=True))
 except Exception as exc:
  print(f'{sub}: {type(exc).__name__}: {exc}');continue
 for item in entries:
  if item.type!=modal.volume.FileEntryType.FILE:continue
  rel=Path(item.path).relative_to('windows_apt_v1');target=ROOT/rel
  if target.exists() and target.stat().st_size==item.size and not target.name.endswith(('.json','.jsonl','.log','.txt')):continue
  target.parent.mkdir(parents=True,exist_ok=True)
  tmp=target.with_suffix(target.suffix+'.download')
  with tmp.open('wb') as f:
   for block in volume.read_file(item.path):f.write(block)
  tmp.replace(target)
p=ROOT/'results/progress.json'
if p.exists():print(p.read_text())
else:print('Training progress not yet committed; inspect gates/calibration.')
