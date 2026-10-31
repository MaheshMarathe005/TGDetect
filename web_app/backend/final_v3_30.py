"""Application adapter for the saved 30-epoch Final V3 combined model."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np
from fastapi import APIRouter, HTTPException

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'datasets/application/final_v3_30/combined_s0'
PREPARED = ROOT / 'datasets/application/cadets_v3/prepared'
PACKAGE = '_tgdetect_final_v3_30'
router = APIRouter(prefix='/api/inference', tags=['Final V3 · 30 epochs'])
lock = threading.Lock()
INDEX = RUN / 'investigation.sqlite'
CHANNEL_DTYPE = np.dtype([(k, '<i8') for k in ('src', 'dst', 'relation', 'src_identity', 'dst_identity')]
                         + [('time', '<f8'), ('ce', '<f4'), ('pair', '<f4'), ('src_type', 'i1'), ('dst_type', 'i1')])


def active():
    return INDEX.exists()


def dashboard_result():
    r = read('result.json')
    evaluation = read('evaluation.json')['paper_threshold']
    metrics = dict(evaluation['metrics'])
    metrics['pr_ap'] = metrics['ap']
    return dict(metrics=metrics, threshold=evaluation['threshold'],
                checkpoint_epoch=r['checkpoint_epoch'], parameter_count=r['parameters'],
                checkpoint_sha256=r['checkpoint_sha256'],
                unseen_ground_truth=metrics['unscored_truth'],
                signature=dict(epochs=r['signature']['epochs'], batch=r['signature']['batch'],
                               seed=r['seed'], model=r['model_config']))


def read(name):
    try:
        return json.loads((RUN / name).read_text())
    except FileNotFoundError:
        raise HTTPException(503, 'Import the Final V3 30-epoch model artifacts first')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify():
    manifest = read('bundle_manifest.json')
    for name, expected in manifest['files'].items():
        if digest(RUN / name) != expected:
            raise ValueError('Artifact checksum mismatch: ' + name)
    for name, expected in manifest['token_files'].items():
        if digest(RUN / 'tokens' / name) != expected:
            raise ValueError('Token checksum mismatch: ' + name)
    for name, expected in read('config.json')['sources'].items():
        if digest(RUN / 'source_snapshot/tgnn_final' / name) != expected:
            raise ValueError('Source checksum mismatch: ' + name)
    return manifest


@router.get('/model')
def summary():
    r = read('result.json')
    if digest(RUN / 'model.pt') != r['checkpoint_sha256']:
        raise HTTPException(503, 'Checkpoint checksum mismatch')
    return dict(id='final-v3-30:combined:0', name='TGNN Final V3 · Combined · 30 epochs',
                epochs=r['signature']['epochs'], checkpoint_epoch=r['checkpoint_epoch'],
                checkpoint_sha256=r['checkpoint_sha256'], parameters=r['parameters'],
                batch_size=r['signature']['batch'], seed=r['seed'],
                model_config=r['model_config'], events=r['signature']['events'],
                evaluation=read('evaluation.json'), history=r['history'],
                relation_confusion=read('test_relation_confusion.json'),
                relations=read('bundle_manifest.json')['relations'],
                limitation=r['limitation'])


@router.post('/replay')
def replay():
    """Bounded, isolated neural replay; never interpret event CE as a probability."""
    if not lock.acquire(blocking=False):
        raise HTTPException(409, 'A replay is already running')
    try:
        import torch
        manifest = verify()
        # The old full CADETS preparation has the same mapping. Verify it rather
        # than using the smaller V6 pilot, which has a different node vocabulary.
        for name, expected in manifest['token_files'].items():
            if digest(PREPARED / 'tokens' / name) != expected:
                raise ValueError('Replay token mapping mismatch')
        prepared_manifest = json.loads((PREPARED / 'manifest.json').read_text())
        if digest(PREPARED / 'test.bin') != prepared_manifest['checksums']['test.bin']:
            raise ValueError('Replay event checksum mismatch')
        if PACKAGE not in sys.modules:
            source = RUN / 'source_snapshot/tgnn_final'
            spec = importlib.util.spec_from_file_location(PACKAGE, source / '__init__.py',
                                                         submodule_search_locations=[str(source)])
            module = importlib.util.module_from_spec(spec)
            sys.modules[PACKAGE] = module
            try:
                spec.loader.exec_module(module)
            except Exception:
                sys.modules.pop(PACKAGE, None)
                raise
        bank = importlib.import_module(PACKAGE + '.encoder').TokenBank(RUN / 'tokens')
        model = importlib.import_module(PACKAGE + '.v3').TGNNFinalV3.from_checkpoint(bank, RUN / 'model.pt')
        dtype = np.dtype([(k, '<i8') for k in ('src', 'dst', 'relation', 'query', 'observed',
                           'signature', 'src_identity', 'dst_identity', 'negative_identity')]
                         + [('time', '<f8'), ('negative_valid', '?')])
        rows = np.memmap(PREPARED / 'test.bin', dtype=dtype, mode='r')[:1024]
        batch = {k: torch.from_numpy(np.array(rows[k])) for k in dtype.names}
        # No sampled negatives are needed for inference.
        batch['negative_identity'] = batch['dst_identity']
        batch['negative_valid'] = torch.zeros(len(rows), dtype=torch.bool)
        started = time.monotonic()
        with torch.inference_mode():
            scored = model.score_batch(batch)
        ce = scored['relation_nll'].cpu().numpy()
        pair = scored['pair_anomaly'].cpu().numpy()
        logits = scored['logits'].argmax(1).cpu().numpy()
        return dict(checkpoint_sha256=manifest['files']['model.pt'], events=len(rows),
                    inference_ms=(time.monotonic() - started) * 1000,
                    mean_relation_ce=float(ce.mean()),
                    scope='First 1,024 prepared test events, one batch, fresh state. This preview does not reproduce full-capture evaluation or calibrated node alerts.',
                    items=[dict(index=i, src=int(rows[i]['src']), dst=int(rows[i]['dst']),
                                relation=int(rows[i]['relation']), predicted_relation=int(logits[i]),
                                relation_ce=float(ce[i]), pair_anomaly=float(pair[i])) for i in range(32)])
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(503, str(exc))
    finally:
        lock.release()
