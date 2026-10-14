"""Resume the complete DEDALE Windows host dataset and verify publisher checksums."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'data/real/dedale'
BASE = 'https://entrepot.recherche.data.gouv.fr'
NAMES = ['README.md', 'system_logs_labels.zip', 'userbehaviour_logs.zip', 'system_logs_winlogbeat.zip']


def write_status(data):
    temp = DEST / 'status.tmp'
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(DEST / 'status.json')


def md5(path):
    digest = hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    if args.status:
        state = json.loads((DEST / 'status.json').read_text()) if (DEST / 'status.json').exists() else {'status': 'not_started'}
        if state.get('current_file'):
            part = DEST / (state['current_file'] + '.part')
            if part.exists():
                state['downloaded_bytes'] = part.stat().st_size
                state['percent'] = round(100 * part.stat().st_size / state['expected_bytes'], 2)
        print(json.dumps(state, indent=2))
        return
    DEST.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (DEST / 'download.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'starting', 'pid': os.getpid(), 'started_at_unix': time.time(), 'verified_files': {}}
    write_status(state)
    try:
        url = BASE + '/api/datasets/:persistentId/?persistentId=doi:10.57745/Y5JLDG'
        subprocess.run(['curl', '-fsSL', '--retry', '5', '--connect-timeout', '30', '--max-time', '180', url, '-o', str(DEST/'publisher_metadata.json')], check=True)
        metadata = json.loads((DEST/'publisher_metadata.json').read_text())
        files = {f['dataFile']['filename']: f['dataFile'] for f in metadata['data']['latestVersion']['files']}
        manifest = {name: dict(id=files[name]['id'], bytes=files[name]['filesize'], md5=files[name]['checksum']['value'], url=BASE+'/api/access/datafile/'+str(files[name]['id'])) for name in NAMES}
        (DEST/'manifest.json').write_text(json.dumps(manifest, indent=2))
        for name, info in manifest.items():
            target = DEST/name
            if target.exists():
                if target.stat().st_size != info['bytes'] or md5(target) != info['md5']:
                    raise RuntimeError(f'Existing {name} failed verification; preserved for inspection')
            else:
                cached = ROOT/'data/dedale_meta'/name
                if cached.exists() and cached.stat().st_size == info['bytes'] and md5(cached) == info['md5']:
                    shutil.copyfile(cached, target)
                else:
                    part = DEST/(name+'.part')
                    remaining = max(0, info['bytes']-(part.stat().st_size if part.exists() else 0))
                    if shutil.disk_usage(DEST).free < remaining + 5*1024**3:
                        raise RuntimeError('Insufficient free disk space for download plus 5 GiB reserve')
                    state.update(status='downloading', current_file=name, expected_bytes=info['bytes'])
                    write_status(state)
                    print(f'Downloading {name}: {info["bytes"]:,} bytes', flush=True)
                    subprocess.run(['curl', '-fL', '--retry', '20', '--retry-delay', '10', '--connect-timeout', '30', '--speed-limit', '1024', '--speed-time', '180', '-C', '-', info['url'], '-o', str(part)], check=True)
                    state['status'] = 'verifying'; write_status(state)
                    if part.stat().st_size != info['bytes'] or md5(part) != info['md5']:
                        raise RuntimeError(f'{name} failed publisher size/MD5 verification')
                    part.replace(target)
            if name.endswith('.zip'):
                with zipfile.ZipFile(target) as z:
                    entries = [dict(name=i.filename, bytes=i.file_size, compressed_bytes=i.compress_size) for i in z.infolist()]
                (DEST/(name+'.members.json')).write_text(json.dumps(entries, indent=2))
            state['verified_files'][name] = info
            write_status(state)
            print(f'Verified {name}', flush=True)
        state.update(status='complete', completed_at_unix=time.time(), current_file=None)
        write_status(state)
    except Exception as exc:
        state.update(status='failed', error=str(exc))
        write_status(state)
        raise


if __name__ == '__main__':
    main()
