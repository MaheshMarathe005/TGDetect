"""Build the main dashboard index from the saved Final V3 30-epoch channels.

python -m web_app.backend.scripts.prepare_final_v3_30 /path/to/combined_s0
"""
import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import numpy as np

from web_app.backend.cadets import DATA, EVENT_DTYPE
from web_app.backend.final_v3_30 import RUN, INDEX, PREPARED, CHANNEL_DTYPE, read, verify


def build(source):
    verify()
    evaluation = read('evaluation.json')['paper_threshold']
    threshold = evaluation['threshold']
    metrics = evaluation['metrics']
    manifest = json.loads((PREPARED / 'manifest.json').read_text())
    assert manifest['events'] == read('config.json')['events']
    temporary = RUN / 'investigation.building.sqlite'
    # Preserve the real entity attributes and original event IDs. Replace every
    # scored event and node prediction before atomically activating the index.
    shutil.copy2(DATA / 'investigation.sqlite', temporary)
    db = sqlite3.connect(temporary)
    try:
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        for split in ('val', 'test'):
            shutil.copy2(source / (split + '_channels.bin'), RUN / (split + '_channels.bin'))
            channels = np.memmap(RUN / (split + '_channels.bin'), dtype=CHANNEL_DTYPE, mode='r')
            rows = np.memmap(PREPARED / (split + '.bin'), dtype=EVENT_DTYPE, mode='r')
            assert len(channels) == len(rows) == manifest['events'][split]
            for key in ('src', 'dst', 'relation', 'time', 'src_identity', 'dst_identity'):
                assert np.array_equal(channels[key], rows[key]), 'Event mapping mismatch: ' + key
            assert np.isfinite(channels['ce']).all()
            db.executemany('UPDATE events SET loss=?, evidence=? WHERE split=? AND idx=?',
                           ((float(row['ce']), float(row['ce']), split, i) for i, row in enumerate(channels)))
            db.commit()
            print('Updated event scores:', split, len(rows), flush=True)
        ids = np.unique(np.concatenate([channels['src'], channels['dst']]))
        scores = np.zeros(read('result.json')['model_config']['n_nodes'])
        np.maximum.at(scores, channels['src'], channels['ce'])
        np.maximum.at(scores, channels['dst'], channels['ce'])
        truth = set(json.loads((PREPARED / 'ground_truth_nodes.json').read_text()))
        predicted = set(map(int, ids[scores[ids] > threshold]))
        assert predicted == set(evaluation['predicted_nodes']), 'Saved predictions differ'
        actual = dict(tp=len(predicted & truth), fp=len(predicted - truth),
                      fn=len((truth & set(map(int, ids))) - predicted))
        actual['tn'] = len(ids) - sum(actual.values())
        assert all(actual[k] == metrics[k] for k in actual), 'Saved confusion matrix differs'
        existing = {row[0] for row in db.execute('SELECT id FROM nodes')}
        assert existing == set(map(int, ids)), 'Node universe mismatch'
        db.executemany('UPDATE nodes SET score=?, alert=?, benchmark_member=? WHERE id=?',
                       ((float(scores[n]), int(n in predicted), int(n in truth), int(n)) for n in ids))
        metadata = dict(events=manifest['events'], total_events=sum(manifest['events'].values()),
                        scored_nodes=len(ids), alert_nodes=len(predicted), positive_scored_nodes=metrics['positive_support'],
                        index_version=2, recomputed_metrics=metrics,
                        checkpoint_sha256=read('result.json')['checkpoint_sha256'],
                        scoring_method='paper_threshold', score_scope='Maximum raw relation CE per capture node',
                        metric_reproduction_max_error=0)
        db.execute('DELETE FROM metadata')
        db.executemany('INSERT INTO metadata VALUES (?,?)', ((k, json.dumps(v)) for k, v in metadata.items()))
        db.commit()
    finally:
        db.close()
    temporary.replace(INDEX)
    print('Activated epoch 30 dashboard:', actual, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    build(parser.parse_args().source)
