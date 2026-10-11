"""Download or install the checksum-verified CADETS/V3 release data bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def install(archive=None):
    record = json.loads((ROOT / 'DATA_BUNDLE.json').read_text())
    expected = json.loads((ROOT / 'PACKAGE_MANIFEST.json').read_text())['files']
    data_files = {k: v for k, v in expected.items() if k.startswith('dashboard/data/')}
    if all((ROOT / k).is_file() and checksum(ROOT / k) == v['sha256'] for k, v in data_files.items()):
        print('Local CADETS/V3 files already verified.')
        return
    if archive is None:
        cache = ROOT / '.cache'
        cache.mkdir(exist_ok=True)
        archive = cache / 'cadets-v3-local-data.tar.gz'
        if not archive.exists() or checksum(archive) != record['sha256']:
            partial = archive.with_suffix('.part')
            request = urllib.request.Request(record['url'], headers={'User-Agent': 'TGDetect-data-installer'})
            print('Downloading CADETS/V3 data bundle...', flush=True)
            with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
                while chunk := response.read(4 * 1024 * 1024):
                    output.write(chunk)
            if partial.stat().st_size != record['bytes'] or checksum(partial) != record['sha256']:
                raise RuntimeError('Data archive checksum mismatch')
            partial.replace(archive)
    archive = Path(archive)
    if archive.stat().st_size != record['bytes'] or checksum(archive) != record['sha256']:
        raise RuntimeError('Data archive checksum mismatch')
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unsafe archive entry: ' + member.name)
            if member.name != 'dashboard/data/cadets_v3' and not member.name.startswith('dashboard/data/cadets_v3/'):
                raise RuntimeError('Unexpected archive entry: ' + member.name)
        # Membership is checked before extracting; links and traversal are rejected.
        bundle.extractall(ROOT, members=members)
    for name, entry in data_files.items():
        path = ROOT / name
        if not path.is_file() or checksum(path) != entry['sha256']:
            raise RuntimeError('Extracted data checksum mismatch: ' + name)
    print('Verified and installed local CADETS E3/V3 data: 5,537,964 events.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Install an already-downloaded archive')
    install(parser.parse_args().archive)
