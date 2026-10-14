"""Real-checkpoint and API acceptance tests; no fabricated telemetry fixtures."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from web_app.backend import backend as b

client = TestClient(b.app)


def test_real_window_graph_and_feature_context():
    boot = client.get('/api/bootstrap').json()
    model = boot['default_model']
    response = client.get('/api/window', params={'model_id': model, 'count': 80})
    assert response.status_code == 200
    window = response.json()
    d = b.load_data()
    assert window['summary']['events'] == 80
    for e in window['events']:
        i = e['index']
        assert e['id'] == str(d['ids'][i])
        assert e['src'] == int(d['src'][i]) and e['dst'] == int(d['dst'][i])
        assert e['alert'] == (e['score'] >= window['threshold'])
    endpoints = {(str(e['src']), str(e['dst'])) for e in window['events']}
    assert all((edge['source'], edge['target']) in endpoints for edge in window['graph']['edges'])
    i = window['events'][-1]['index']
    detail = client.get(f'/api/events/{i}', params={'model_id': model}).json()
    np.testing.assert_allclose(detail['normalized_features'], d['x'][i])
    assert all(h['index'] < i for h in detail['history'])


@pytest.mark.parametrize('arm', ['TGNN_Node_Mem_V3', 'TGNN_Q_A_C', 'GNN_H'])
def test_actual_checkpoint_inference_matches_saved_predictions_and_sessions_isolate(arm):
    candidates = [r for r in b.registry() if r['arm'] == arm and r['checkpoint_available']]
    assert candidates
    row = max(candidates, key=lambda r: r['validation_pr_ap'])
    expected = b.saved_predictions(row)
    one = client.post('/api/sessions', json={'model_id': row['id']}).json()
    two = client.post('/api/sessions', json={'model_id': row['id']}).json()
    assert one['checkpoint_sha256']
    try:
        a = client.post(f'/api/sessions/{one["id"]}/next?count=16')
        assert a.status_code == 200, a.text
        a = a.json()
        first = client.post(f'/api/sessions/{two["id"]}/next?count=8').json()
        last = client.post(f'/api/sessions/{two["id"]}/next?count=8').json()
        values = [e['score'] for e in a['events']]
        np.testing.assert_allclose(values, expected[:16], atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(values, [e['score'] for e in first['events'] + last['events']], atol=3e-5, rtol=3e-5)
    finally:
        client.delete(f'/api/sessions/{one["id"]}')
        client.delete(f'/api/sessions/{two["id"]}')


def test_invalid_windows_models_and_thresholds():
    assert client.get('/api/window?start=-1').status_code == 422
    assert client.get('/api/window?threshold=1.5').status_code == 422
    assert client.get('/api/window?model_id=unknown').status_code == 404
    assert client.post('/api/sessions', json={'model_id': 'unknown'}).status_code == 404


def test_matched_training_fields_and_exact_source_artifacts():
    results = client.get('/api/comparison').json()['results']
    assert len(results) == 9
    for seed in range(3):
        rows = [r for r in results if r['seed'] == seed]
        assert len(rows) == 3
        for field in ['warm_start_sha256', 'dataset_sha256', 'optimizer_steps',
                      'training_live_visits', 'training_replay_visits', 'replay_schedule_sha256',
                      'train_batch', 'infer_batch', 'tbptt_k', 'epochs_trained']:
            assert len({r[field] for r in rows}) == 1, field


def test_raw_ingestion_requires_verified_training_scaler_and_rejects_time_reversal():
    prep = b.load_preprocessing()
    model = b.default_model()['id']
    response = client.post('/api/sessions', json={'model_id': model, 'source': 'raw'})
    if prep is None:
        assert response.status_code == 503
        return
    assert response.status_code == 200
    sid = response.json()['id']
    event = {'@timestamp': '2026-10-02T12:00:00Z', 'host': {'name': 'UNSEEN-HOST'},
             'event': {'code': '1'}, 'winlog': {'channel': 'Sysmon', 'record_id': 'new-1',
             'event_data': {'ProcessGuid': 'unseen-process', 'Image': 'C:\\Windows\\example.exe'}}}
    try:
        response = client.post(f'/api/sessions/{sid}/events', json=[event])
        assert response.status_code == 200, response.text
        e = response.json()['events'][0]
        assert e['label'] is None and len(e['normalized_features']) == 72
        assert 0 <= e['score'] <= 1
        assert e['src'] >= int(b.load_data()['n_nodes'])
        event['@timestamp'] = '2026-10-01T12:00:00Z'
        assert client.post(f'/api/sessions/{sid}/events', json=[event]).status_code == 422
        assert b.get_session(sid)['position'] == 1
    finally:
        client.delete(f'/api/sessions/{sid}')
