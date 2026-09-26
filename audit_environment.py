"""Read-only source/data audit; generated comparison fixtures live in a temp dir."""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def command_version(command):
    if not shutil.which(command[0]):
        return None
    result = subprocess.run(command, text=True, capture_output=True, timeout=20)
    return {'returncode': result.returncode, 'output': (result.stdout + result.stderr).strip()}


def audit():
    packages = {}
    for name in ['pip', 'torch', 'numpy', 'scipy', 'scikit-learn', 'torch-geometric', 'dgl', 'modal', 'pytest']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(ROOT.rglob('*.py')) if '__pycache__' not in p.parts}
    # Parse every retained/new Python file without generating bytecode or importing apps.
    for name in source_hashes:
        compile((ROOT / name).read_bytes(), name, 'exec')
    reset = json.loads((ROOT / 'RESET_INVENTORY.json').read_text())
    original_hashes = {e['path']: e['sha256'] for e in reset['pre_reset_inventory']}
    expected_edits = {'.labtest/harness.py'}
    unchanged = [name for name in reset['preserved_and_hash_verified'] if name not in expected_edits]
    mismatches = [name for name in unchanged if not (ROOT / name).exists() or sha256(ROOT / name) != original_hashes[name]]
    if mismatches:
        raise RuntimeError('Preserved inputs/source changed unexpectedly: ' + ', '.join(mismatches))
    result = {
        'executable': sys.executable, 'prefix': sys.prefix, 'python': sys.version,
        'os': platform.platform(), 'machine': platform.machine(), 'packages': packages,
        'git_commit': None, 'git_note': 'Workspace has no .git repository.',
        'source_sha256': source_hashes,
        'syntax_checked': sorted(source_hashes),
        'preservation': {'unchanged_files_verified': unchanged, 'intentional_edits': sorted(expected_edits), 'unexpected_changes': mismatches},
        'source_bundle_sha256': hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        'nvcc': command_version(['nvcc', '--version']),
        'compiler': command_version(['gcc', '--version']),
        'gpu': command_version(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader']),
        'seed_policy': {'local_smoke_seed': 0, 'paper_seed_count': 5, 'paper_seed_values': None},
        'datasets': [], 'paper_sha256': {p.name: sha256(p) for p in sorted(ROOT.glob('*.pdf'))},
    }
    if packages['torch']:
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(0)
        first = torch.randn(3, 3)
        torch.manual_seed(0)
        result['torch_smoke'] = {
            'cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'cudnn': torch.backends.cudnn.version(),
            'seed_repeat_equal': torch.equal(first, torch.randn(3, 3)),
            'cpu_matmul_finite': bool(torch.isfinite(first @ first.T).all()),
        }
    if packages['numpy']:
        import numpy as np
        # Extract only the inspected fixture generator, avoiding sklearn stubs and Modal.
        tree = ast.parse((ROOT / '.labtest/harness.py').read_text())
        generator = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'make_fixtures')
        with tempfile.TemporaryDirectory(prefix='tgdetect-provenance-') as tmp:
            scope = {'np': np, 'DATA': Path(tmp)}
            exec(compile(ast.Module(body=[generator], type_ignores=[]), '<isolated fixture generator>', 'exec'), scope)
            with contextlib.redirect_stdout(io.StringIO()):
                scope['make_fixtures'](n=30000, seed=0)
            for p in sorted((ROOT / '.labtest/data').rglob('*.npz')):
                entry = {'path': str(p.relative_to(ROOT)), 'sha256': sha256(p), 'bytes': p.stat().st_size}
                with np.load(p, allow_pickle=False) as z:
                    entry['arrays'] = {k: {'shape': list(z[k].shape), 'dtype': str(z[k].dtype)} for k in z.files}
                    if 'ts' in z:
                        ts = z['ts']
                        entry['time_range'] = [float(ts.min()), float(ts.max())]
                        entry['nondecreasing_time'] = bool(np.all(np.diff(ts.astype(np.float64)) >= 0))
                    if 'y' in z:
                        entry['positive_count'] = int((z['y'] > 0).sum())
                    fixture = Path(tmp) / p.name
                    if fixture.exists():
                        with np.load(fixture, allow_pickle=False) as f:
                            same = set(z.files) == set(f.files) and all(np.array_equal(z[k], f[k]) for k in z.files)
                        entry['exact_match_to_legacy_synthetic_fixture_seed0'] = same
                        entry['research_eligible'] = False
                        entry['provenance'] = 'synthetic fixture (exact array match)' if same else 'unverified; no raw-source manifest'
                    else:
                        entry['provenance'] = 'derived context cache; source lineage unverified'
                        entry['research_eligible'] = False
                result['datasets'].append(entry)
    return result


if __name__ == '__main__':
    output = ROOT / 'ENVIRONMENT_AUDIT.json'
    result = audit()
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['executable', 'python', 'packages', 'torch_smoke'] if k in result}, indent=2))
    for entry in result['datasets']:
        print(entry['path'], entry['provenance'])
    print('Wrote', output)
