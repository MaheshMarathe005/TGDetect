"""Import a saved Kaggle graph artifact into the existing Detection graph page.

No rescoring and no frontend rebuild. Run from repository root with a locally
created application_graph.zip; predictions remain those of the exported model.
"""
import argparse,hashlib,json,shutil,sqlite3,tempfile,time,uuid,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def install(archive,store=None):
    store=Path(store) if store else ROOT/'datasets/application/graph_uploads'
    with tempfile.TemporaryDirectory(prefix='paper_graph_') as tmp:
        tmp=Path(tmp)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                p=Path(name)
                if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path')
            z.extractall(tmp)
        manifest=json.loads((tmp/'application_manifest.json').read_text())
        if manifest['format']!='tgdetect-application-graph-v1':raise ValueError('Wrong graph artifact format')
        for name,digest in manifest['files'].items():
            p=Path(name)
            if p.is_absolute() or '..' in p.parts or sha(tmp/p)!=digest:raise ValueError('Artifact checksum mismatch: '+name)
        db=sqlite3.connect(f'file:{tmp / "graph.sqlite"}?mode=ro',uri=True)
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity failed')
        metadata={k:json.loads(v) for k,v in db.execute('SELECT key,value FROM metadata')}
        nodes=db.execute('SELECT count(*) FROM nodes').fetchone()[0];events=db.execute('SELECT count(*) FROM events').fetchone()[0]
        dangling=db.execute('SELECT count(*) FROM events e LEFT JOIN nodes s ON e.src=s.id LEFT JOIN nodes d ON e.dst=d.id WHERE s.id IS NULL OR d.id IS NULL').fetchone()[0]
        if dangling:raise ValueError('Missing event endpoint metadata')
        db.close();identifier=uuid.uuid4().hex;dest=store/identifier;dest.mkdir(parents=True)
        shutil.copy(tmp/'graph.sqlite',dest/'graph.sqlite');shutil.copy(tmp/'application_manifest.json',dest/'application_manifest.json')
        record={'id':identifier,'name':metadata['arm']+' · Kaggle saved checkpoint','status':'ready','stage':'Imported saved predictions',
                'progress':1,'created':time.time(),'events':events,'nodes':nodes,'threshold':metadata['threshold'],
                'checkpoint_sha256':metadata['checkpoint_sha256'],'prediction_rule':metadata['prediction_rule'],'scoring':metadata['prediction_rule']+'; raw relation reconstruction CE, saved predictions; benchmark membership is separate.'}
        (dest/'status.json').write_text(json.dumps(record,indent=2));return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive');a=p.parse_args();print(json.dumps(install(a.archive),indent=2))
