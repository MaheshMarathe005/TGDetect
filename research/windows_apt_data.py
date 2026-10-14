"""Audited Windows-APT weak-label preparation; never silently treats missing as benign."""
from pathlib import Path
from datetime import datetime
from collections import Counter
import csv, hashlib, json, time
import numpy as np
from causal_data import causal_egos, fit_scaler

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/Windows-APT 2025 A Dataset for APT-Inspired Attack'
CATEGORICAL = ['_source.data.win.system.eventID', '_source.data.win.system.providerName',
 '_source.data.win.system.channel', '_source.data.win.eventdata.image',
 '_source.data.win.eventdata.parentImage', '_source.data.win.eventdata.protocol',
 '_source.data.win.eventdata.integrityLevel', '_source.data.win.eventdata.eventType']
LENGTHS = ['_source.data.win.eventdata.'+k for k in ('image','parentImage','commandLine','targetFilename')]
PORTS = ['_source.data.win.eventdata.'+k for k in ('sourcePort','destinationPort')]
FEATURE_FIELDS = sorted(set(CATEGORICAL + LENGTHS + PORTS))
INPUT_DIM = 72

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
 return h.hexdigest()

def write_json(path, value):
 p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(p)

def clean(value): return (value or '').strip()

def features(row):
 x=np.zeros(INPUT_DIM,dtype=np.float32)
 for key in CATEGORICAL:
  value=clean(row.get(key)).lower()
  if key.endswith(('.image','.parentImage')): value=value.replace('\\','/').rsplit('/',1)[-1]
  token=(key+'='+value).encode()
  h=hashlib.sha256(token).digest()
  x[int.from_bytes(h[:8],'little')%64] += 1. if h[8]&1 else -1.
 x[64:68]=np.log1p([len(clean(row.get(k))) for k in LENGTHS])
 for i,key in enumerate(PORTS):
  try:
   p=float(clean(row.get(key)))
   if not np.isfinite(p) or not p.is_integer() or not 0<=p<=65535: raise ValueError()
   x[68+i]=p/65535
  except (ValueError,TypeError): x[70+i]=1.
 return x

def target(row):
 value=clean(row.get('_source.rule.mitre.id'))
 if not value: return 0
 values=json.loads(value)
 if not isinstance(values,list) or not all(isinstance(v,str) and v.startswith('T') for v in values):
  raise ValueError('Invalid MITRE annotations')
 return int(bool(values))

def endpoints(row,row_id):
 agent=clean(row.get('_source.agent.id')) or clean(row.get('_source.agent.name'))
 if not agent: raise ValueError('Missing agent')
 guid=clean(row.get('_source.data.win.eventdata.processGuid'))
 host='agent:'+agent
 return host, host+':process:'+guid.lower() if guid else 'event:'+str(row_id)

def boundaries(times):
 ends=[int(len(times)*.6),int(len(times)*.8)]
 for k in range(2):
  while ends[k]<len(times) and times[ends[k]]==times[ends[k]-1]: ends[k]+=1
 if not 0<ends[0]<ends[1]<len(times): raise ValueError('Invalid split')
 return ends

def prepare(raw,out):
 started=time.perf_counter(); raw=Path(raw); out=Path(out); out.mkdir(parents=True,exist_ok=True)
 rows=[]; ids=set(); nodes={}; missing=Counter(); guid_count=0; agents=Counter(); ruleids=set()
 with (raw/'combined.csv').open(newline='',encoding='utf-8-sig') as f:
  reader=csv.DictReader(f); columns=reader.fieldnames
  if not set(FEATURE_FIELDS)<=set(columns): raise ValueError('Missing schema')
  for row_id,row in enumerate(reader):
   if None in row or any(v is None for v in row.values()): raise ValueError('Malformed CSV')
   identity=clean(row['_id'])
   if not identity or identity in ids: raise ValueError('Repeated/missing _id')
   ids.add(identity)
   # Naive timestamps have an unspecified timezone; use timezone-independent epoch differences.
   t=(datetime.strptime(clean(row['_source.@timestamp']),'%b %d, %Y @ %H:%M:%S.%f')-datetime(1970,1,1)).total_seconds()
   s,d=endpoints(row,row_id)
   src=nodes.setdefault(s,len(nodes)); dst=nodes.setdefault(d,len(nodes))
   rows.append((t,row_id,src,dst,target(row),features(row)))
   guid_count+=bool(clean(row.get('_source.data.win.eventdata.processGuid')))
   agents[clean(row['_source.agent.name'])]+=1; ruleids.add(clean(row['_source.rule.id']))
   missing.update(k for k in FEATURE_FIELDS if not clean(row.get(k)))
 rows.sort(key=lambda r:(r[0],r[1]))
 times=np.array([r[0] for r in rows],dtype=np.float64)
 row_ids=np.array([r[1] for r in rows],dtype=np.int64)
 src=np.array([r[2] for r in rows],dtype=np.int64); dst=np.array([r[3] for r in rows],dtype=np.int64)
 y=np.array([r[4] for r in rows],dtype=np.int64); raw_x=np.stack([r[5] for r in rows])
 train_end,val_end=boundaries(times); mean,std=fit_scaler(raw_x,train_end)
 x=(raw_x-mean)/std
 ego,valid,ages=causal_egos(src,dst,times,4)
 assert np.isfinite(x).all() and np.isfinite(ages).all()
 assert np.array_equal(ego[:,0],np.arange(len(y))) and valid[:,0].all()
 history=valid.copy(); history[:,0]=False
 assert (times[ego][history]<np.broadcast_to(times[:,None],ego.shape)[history]).all()
 partitions={}
 for name,lo,hi in [('train',0,train_end),('validation',train_end,val_end),('test',val_end,len(y))]:
  n=hi-lo; positives=int(y[lo:hi].sum())
  assert min(positives,n-positives)>=20
  partitions[name]={'start':lo,'end':hi,'count':n,'annotated':positives,'unannotated':n-positives}
 np.savez_compressed(out/'data.npz',x=x,y=y,src=src,dst=dst,times=times,row_ids=row_ids,
  ego=ego,valid=valid,ages=ages,train_end=train_end,val_end=val_end)
 meta={'status':'PASS','label_scope':'WEAK_LABEL_WAZUH_MITRE_PRESENCE','user_authorized':True,
  'rows':len(y),'columns':len(columns),'input_dim':INPUT_DIM,'partitions':partitions,'agents':dict(agents),
  'rule_id_count':len(ruleids),'feature_fields':FEATURE_FIELDS,'missing_features':dict(missing),
  'features':'64 signed SHA256 categorical bins plus 8 numeric fields; no label/rule/ID/time input',
  'scaler_mean':mean.tolist(),'scaler_std':std.tolist(),'neighbors':4,
  'process_guid_rows':guid_count,'graph_node_count':len(nodes),'history_count_mean':float(history.sum(1).mean()),
  'history_age_seconds_quantiles':np.quantile(np.expm1(ages[history]),[0,.5,.9,.99,1]).tolist(),
  'preprocess_seconds':time.perf_counter()-started,'raw_sha256':digest(raw/'combined.csv'),
  'data_sha256':digest(out/'data.npz'),'timezone':'unspecified in CSV; relative displayed-time intervals',
  'warning':'Unannotated is not verified benign; metrics assess annotation prediction, not attack detection.'}
 write_json(out/'data_manifest.json',meta)
 return meta

def source_audit(raw,out):
 raw=Path(raw); checks={}
 for line in (raw/'checksums.sha256').read_text().splitlines():
  expected,path=line.split(maxsplit=1); name=path.rsplit('\\',1)[-1]
  actual=digest(raw/name); assert actual==expected.lower(),name
  checks[name]=actual
 def identities(path):
  with path.open(newline='',encoding='utf-8-sig') as f:
   rd=csv.DictReader(f); count=Counter(clean(r['_id']) for r in rd)
  return count
 combined=identities(raw/'combined.csv'); periods=Counter(); counts={}
 for p in sorted(raw.glob('*.csv')):
  if p.name in {'combined.csv','scenario_manifest.csv','validation_summary.csv'}: continue
  c=identities(p); periods.update(c); counts[p.name]=sum(c.values())
 assert combined==periods and all(v==1 for v in combined.values())
 report={'status':'PASS','checksum_files':checks,'period_counts':counts,'records':sum(combined.values()),
  'combined_and_period_event_id_multisets_equal':True,'unique_ids':len(combined)}
 write_json(out,report); return report

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--raw',default=str(RAW));p.add_argument('--out',default='artifacts/windows_apt_v1/prepared'); a=p.parse_args()
 source_audit(a.raw,Path(a.out)/'source_audit.json')
 m=prepare(a.raw,a.out);print(json.dumps({k:m[k] for k in ['status','rows','input_dim','partitions','preprocess_seconds']}))
