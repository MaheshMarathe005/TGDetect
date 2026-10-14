"""Recompute saved held-out metrics and require matching data/features/event IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .data import sha256
from .detection import detect, fit_rareness
from .run import benign_windows, window_values, metrics, finite_scores, report, f1_threshold


def audit(prepared, results, expected_seeds=3, expected_arms=None):
    prepared, results = Path(prepared), Path(results)
    with np.load(prepared, allow_pickle=False) as z:
        a = {k:z[k] for k in z.files}
    tr, va = int(a['train_end']), int(a['val_end'])
    common = dict(dataset_sha256=sha256(prepared),
        feature_sha256=hashlib.sha256(a['x'].tobytes()).hexdigest(),
        event_ids_sha256=hashlib.sha256(a['ids'].tobytes()).hexdigest())
    checked = []
    for path in sorted(results.rglob('*_seed*.json')):
        result = json.loads(path.read_text())
        if not isinstance(result,dict) or result.get('status') != 'complete':
            continue
        assert all(result[k] == v for k,v in common.items()), f'Unmatched inputs: {path}'
        assert result['train_end'] == tr and result['val_end'] == va
        with np.load(path.with_name(path.stem+'_predictions.npz')) as pred:
            np.testing.assert_array_equal(pred['y'], a['y'])
            np.testing.assert_array_equal(pred['ids'], a['ids'])
            np.testing.assert_array_equal(pred['window'], a['window'])
            score = pred['score'].copy()
        assert len(score)==len(a['y']) and np.isfinite(score).all()
        _, wy, ws = window_values(a['window'][va:],a['y'][va:],score[va:])
        recalculated = {}
        if result['arm']=='KAIROS':
            ix = np.flatnonzero(benign_windows(a['window'][:tr],a['y'][:tr]))
            rs, rd = a['rarity_ids'][a['src']],a['rarity_ids'][a['dst']]
            rarity = fit_rareness(rs[ix],rd[ix],a['window'][ix])
            vi=np.arange(tr,va)[benign_windows(a['window'][tr:va],a['y'][tr:va])]
            validation=detect(rs[vi],rd[vi],a['window'][vi],score[vi],rarity)
            beta=max([r['retrospective_score'] for r in validation if np.isfinite(r['retrospective_score'])],default=0.)
            assert result['window_online']['threshold']==beta
            assert result['window_retrospective']['threshold']==beta
            assert result['event_diagnostic']['threshold']==float(score[vi].max())
            rows = detect(rs[va:],rd[va:],a['window'][va:],score[va:],rarity)
            for key,field in [('window_online','online_score'),('window_retrospective','retrospective_score')]:
                recalculated[key] = metrics(wy,finite_scores([r[field] for r in rows]),result[key]['threshold'],strict=True)
            recalculated['event_diagnostic'] = metrics(a['y'][va:],score[va:],result['event_diagnostic']['threshold'],strict=True)
        else:
            _,vwy,vws=window_values(a['window'][tr:va],a['y'][tr:va],score[tr:va])
            assert result['event']['threshold']==f1_threshold(a['y'][tr:va],score[tr:va])
            assert result['window_online']['threshold']==f1_threshold(vwy,vws)
            recalculated['window_online'] = metrics(wy,ws,result['window_online']['threshold'])
            recalculated['event'] = metrics(a['y'][va:],score[va:],result['event']['threshold'])
        for key,values in recalculated.items():
            for metric,value in values.items():
                original=result[key][metric]
                if value is None or isinstance(value,str):
                    assert original==value
                else:
                    np.testing.assert_allclose(original,value,rtol=1e-10,atol=1e-12)
        checked.append((result['arm'],result['seed']))
    if expected_arms is None:
        expected_arms = ['KAIROS','TGNN_Node_Mem_V3','TGNN_Node_Mem_V4']
    expected={(arm,seed) for arm in expected_arms for seed in range(expected_seeds)}
    if len(checked)!=len(set(checked)) or set(checked)!=expected:
        raise ValueError(f'Incomplete or duplicate grid: found {checked}; expected {sorted(expected)}')
    output=dict(status='passed',runs=len(checked),dataset_sha256=common['dataset_sha256'],
                checks=['complete grid','identical data/features/IDs','finite scores','validation-only thresholds','recomputed event/window confusion matrices and ranking metrics'])
    (results/'audit.json').write_text(json.dumps(output,indent=2))
    report(results)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--results',required=True)
    p.add_argument('--expected-seeds',type=int,default=3)
    print(json.dumps(audit(**vars(p.parse_args())),indent=2))
