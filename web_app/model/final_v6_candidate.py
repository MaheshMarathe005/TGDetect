"""Isolated V6 candidate loader; never replaces the application's V3 runtime."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'final_v6_results/v6_s0'
BUNDLE = ROOT / 'datasets/application/final_v6_candidate/bundle'
PACKAGE = '_tgdetect_final_v6_candidate'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_artifacts():
    manifest = json.loads((RUN / 'bundle_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if digest(RUN / name) != expected:
            raise ValueError('V6 artifact checksum mismatch: ' + name)
    for name, expected in manifest['token_files'].items():
        if digest(BUNDLE / 'tokens' / name) != expected:
            raise ValueError('V6 token checksum mismatch: ' + name)
    config = json.loads((RUN / 'config.json').read_text())
    for name, expected in config['sources'].items():
        if digest(RUN / 'source_snapshot/tgnn_final' / name) != expected:
            raise ValueError('V6 source checksum mismatch: ' + name)
    return manifest


def modules():
    """Use a private package name so older research imports cannot collide."""
    verify_artifacts()
    if PACKAGE not in sys.modules:
        source = RUN / 'source_snapshot/tgnn_final'
        spec = importlib.util.spec_from_file_location(
            PACKAGE, source / '__init__.py', submodule_search_locations=[str(source)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(PACKAGE, None)
            raise
    return PACKAGE


def load_model(device='cpu'):
    from importlib import import_module
    package = modules()
    bank = import_module(package + '.encoder').TokenBank(BUNDLE / 'tokens')
    bank.to_device(device)
    model = import_module(package + '.v3').TGNNFinalV6.from_checkpoint(
        bank, RUN / 'model.pt', device=device)
    model.eval()
    model.reset_state()
    return model
