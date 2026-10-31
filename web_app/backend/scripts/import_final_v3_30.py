"""Import the saved model for /inference without copying intermediate epochs.

python -m web_app.backend.scripts.import_final_v3_30 /path/to/combined_s0
"""
import argparse
from pathlib import Path
import shutil
import zipfile

from web_app.backend.final_v3_30 import RUN, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    for name in ('model.pt', 'bundle_manifest.json', 'config.json', 'evaluation.json',
                 'result.json', 'history.json', 'latency.json',
                 'test_relation_confusion.json', 'val_relation_confusion.json'):
        shutil.copy2(args.run / name, RUN / name)
    shutil.copytree(args.run / 'source_snapshot', RUN / 'source_snapshot', dirs_exist_ok=True)
    with zipfile.ZipFile(args.run / 'model_bundle.zip') as archive:
        for name in ('tokens/tokens.json', 'tokens/fields.bin', 'tokens/offsets.npy',
                     'typed_ce_calibration.json', 'typed_ce_pair_calibration.json', 'node_types.npy'):
            target = RUN / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    verify()
    print('Verified Final V3 30-epoch model:', RUN)


if __name__ == '__main__':
    main()
