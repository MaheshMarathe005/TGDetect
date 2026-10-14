"""Bounded, label-blind DARPA E3 event sample and independent label audit."""
import csv, hashlib, heapq, io, json, tarfile, uuid
from pathlib import Path
from collections import Counter
import requests

GT_FILES={'cadets_e3':['node_Nginx_Backdoor_06.csv','node_Nginx_Backdoor_12.csv','node_Nginx_Backdoor_13.csv'],
          'theia_e3':['node_Browser_Extension_Drakon_Dropper.csv','node_Firefox_Backdoor_Drakon_In_Memory.csv']}

def unwrap(v):
    if isinstance(v,dict) and len(v)==1:return unwrap(next(iter(v.values())))
    return v

def uid(v):
    v=unwrap(v)
    if isinstance(v,bytes):return str(uuid.UUID(bytes=v)).upper()
    return str(v).upper() if v is not None else ''

def records(path):
    import fastavro
    with tarfile.open(path,'r|gz') as tar:
        for member in tar:
            if not member.isfile():continue
            f=tar.extractfile(member)
            if '.json' in member.name:
                for line in f:
                    obj=json.loads(line);datum=obj.get('datum',{})
                    for kind,value in datum.items():yield kind.rsplit('.',1)[-1],value
            elif '.bin' in member.name:
                for obj in fastavro.reader(f,return_record_name=True):
                    kind,value=obj['datum'];yield kind.rsplit('.',1)[-1],value

def scan(root,dataset,limit=100000):
    root=Path(root);out=root/'prepared'/dataset;out.mkdir(parents=True,exist_ok=True)
    gt=set();hashes={}
    for name in GT_FILES[dataset]:
        url='https://raw.githubusercontent.com/ubc-provenance/ground-truth/main/darpa/'+('E3-CADETS' if dataset=='cadets_e3' else 'E3-THEIA')+'/'+name
        r=requests.get(url,timeout=90);r.raise_for_status();(out/name).write_bytes(r.content)
        hashes[name]=hashlib.sha256(r.content).hexdigest()
        gt.update(row[0].upper() for row in csv.reader(io.StringIO(r.text)) if row)
    heap=[];count=Counter();matched=set();lo=None;hi=None
    for path in sorted((root/'raw'/dataset).glob('*.tar.gz')):
        for kind,e in records(path):
            count[kind]+=1
            if kind!='Event':continue
            s,d,d2=uid(e.get('subject')),uid(e.get('predicateObject')),uid(e.get('predicateObject2'))
            if not s or not d:count['missing_endpoint']+=1;continue
            t=e.get('timestampNanos');eid=uid(e.get('uuid'))
            if not isinstance(t,int) or t<=0 or not eid:raise ValueError('Invalid time/event ID')
            lo=t if lo is None else min(lo,t);hi=t if hi is None else max(hi,t)
            hit={s,d,d2}&gt;matched.update(hit);count['matched_events']+=bool(hit)
            h=int.from_bytes(hashlib.sha256(('DARPA_PILOT_20260927:'+eid).encode()).digest()[:8],'big')
            if len(heap)==limit and h>=-heap[0][0]:continue
            prop=e.get('properties') or {};prop=prop.get('map',prop)
            value={'id':eid,'src':s,'dst':d,'dst2':d2,'time_ns':t,'type':e.get('type'),
                   'name':unwrap(e.get('name')),'exec':prop.get('exec',''),
                   'size':unwrap(e.get('size')), 'label':int(bool(hit))}
            item=(-h,eid,value)
            if len(heap)<limit:heapq.heappush(heap,item)
            else:heapq.heapreplace(heap,item)
    rows=sorted((v for _,_,v in heap),key=lambda v:(v['time_ns'],v['id']))
    assert len({r['id'] for r in rows})==len(rows),'Duplicate sampled event IDs'
    ends=[int(len(rows)*.6),int(len(rows)*.8)]
    for i in range(2):
        while ends[i]<len(rows) and rows[ends[i]]['time_ns']==rows[ends[i]-1]['time_ns']:ends[i]+=1
    partitions={}
    for name,a,b in [('train',0,ends[0]),('validation',ends[0],ends[1]),('test',ends[1],len(rows))]:
        partitions[name]={'start':a,'end':b,'positive':sum(r['label'] for r in rows[a:b]),'count':b-a}
    report={'dataset':dataset,'sample_size':len(rows),'counts':dict(count),'time_ns_min':lo,'time_ns_max':hi,
            'ground_truth_nodes':len(gt),'matched_ground_truth_nodes':len(matched),'ground_truth_sha256':hashes,
            'partitions':partitions,'sample_rule':'smallest 100000 SHA256(event UUID, fixed seed); label-blind',
            'label_scope':'EVENT_INCIDENT_TO_PUBLISHED_GROUND_TRUTH_NODE; association proxy, not verified event malice',
            'training_gate':all(0<v['positive']<v['count'] for v in partitions.values())}
    with (out/'events.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    return report
