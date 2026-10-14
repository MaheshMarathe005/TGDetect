"""Integration contracts for the copied UI using real checkpoints and telemetry."""
import json
import uuid
from urllib.parse import quote

import numpy as np
from fastapi.testclient import TestClient

from web_app.backend import backend as b

client = TestClient(b.app)


def test_copied_ui_has_real_dataset_graph_and_score_mapping():
    datasets = client.get('/api/ui/datasets').json()
    ds = next(x for x in datasets if x['id'] == 'dedale-host-memory')
    assert ds['num_raw_events'] == len(b.load_data()['y']) and ds['kind'] == 'dedale'
    graph = client.get('/api/ui/graph/edges?limit=30').json()
    d = b.load_data()
    assert all(e['relation'] == 'OBSERVED_ON' for e in graph)
    assert [(e['src_id'],e['dst_id']) for e in graph] == [(f'process:{s}',f'host:{t}') for s,t in zip(d['src'][:30],d['dst'][:30])]
    for arm in ('TGNN_Q_A_C','TGNN_Node_Mem_V3'):
        row = next(r for r in b.registry() if r['run']=='matched' and r['arm']==arm and r['seed']==0)
        events = client.get('/api/ui/events',params={'model_id':row['id'],'limit':8}).json()['items']
        np.testing.assert_allclose([e['attrs']['probability'] for e in events], b.saved_predictions(row)[:8])
        event = client.get('/api/ui/events/'+quote(events[0]['event_id'],safe=''),params={'model_id':row['id']}).json()
        assert event == events[0]


def test_architecture_history_and_evaluation_match_artifacts():
    models = client.get('/api/ui/models').json()
    assert len(models)==9 and models[0]['id']==b.default_model()['id']
    row = b.get_model(models[0]['id']); net,_ = b.checkpoint_model(row)
    summary = client.get('/api/ui/model/summary',params={'model_id':row['id']}).json()
    assert summary['total_parameters']==sum(p.numel() for p in net.parameters())
    assert summary['config']['num_gnn_layers']==len(net.layers)
    assert summary['config']['num_rnn_layers']==1 and 'GRU' in summary['temporal_aggregator']
    metrics = client.get('/api/ui/model/evaluation',params={'model_id':row['id'],'split':'test'}).json()['metrics']
    for actual,field in [('f1','f1'),('auc_pr','pr_ap'),('auc_roc','roc_auc')]:
        assert abs(metrics[actual]-row['metrics'][field])<1e-9
    assert metrics['confusion_matrix']=={k:row['metrics'][k] for k in ('tn','fp','fn','tp')}


def test_uploaded_stream_uses_checkpoint_and_never_invents_ground_truth():
    did='test-'+uuid.uuid4().hex
    raw={'@timestamp':'2026-10-02T12:00:00Z','host':{'name':'TEST-NEW-HOST'},'event':{'code':'1'},
         'winlog':{'channel':'Sysmon','record_id':'1','event_data':{'ProcessGuid':'TEST-NEW-PROCESS','Image':'C:\\Windows\\test.exe'}}}
    upload=client.post('/api/ui/datasets/upload',files={'file':('test.jsonl',json.dumps(raw),'application/json')}).json()
    token=upload['saved_path']
    try:
        assert upload['status']=='valid_with_warnings' and upload['sample_events'][0]['label'] is None
        response=client.post('/api/ui/datasets/process',json={'dataset_id':did,'source_path':token,'format':'dedale'})
        assert response.status_code==200,response.text
        rows=client.get('/api/ui/events',params={'dataset_id':did}).json()['items']
        e=rows[0];assert e['label'] is None and e['tactics']==[] and e['causal_parent'] is None
        assert 0<=e['attrs']['probability']<=1 and e['attrs']['checkpoint_sha256']==b.default_model()['checkpoint_sha256']
        stats=client.get('/api/ui/graph/stats',params={'dataset_id':did}).json()
        assert stats['graph']['unlabeled_events']==1 and stats['graph']['benign_events']==0
        timeline=client.get('/api/ui/analytics/events',params={'dataset_id':did}).json()['timeline']
        assert sum(x['unlabeled'] for x in timeline)==1 and sum(x['benign'] for x in timeline)==0
        export=client.get('/api/ui/artifacts/events/download',params={'dataset_id':did})
        assert export.json()==rows and 'attachment' in export.headers['content-disposition']
    finally:
        for path in (b.DATA/'ui_uploads'/f'{token}.json',b.DATA/'ui_datasets'/f'{did}.json'):
            path.unlink(missing_ok=True)


def test_ui_upload_rejects_arbitrary_paths_and_invalid_chronology():
    assert client.post('/api/ui/datasets/validate',json={'file_path':'/etc/passwd'}).status_code==422
    event={'@timestamp':'invalid','host':{'name':'host'},'winlog':{'event_id':1}}
    report=client.post('/api/ui/datasets/upload',files={'file':('invalid.jsonl',json.dumps(event),'application/json')}).json()
    assert report['status']=='invalid' and report['saved_path'] is None
