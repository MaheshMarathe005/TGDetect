"""Immutable CADETS metadata and disclosed KAIROS hierarchical-hash adapter."""
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from sklearn.feature_extraction import FeatureHasher
from orthrus_comparison.run import files,sha256


@lru_cache(maxsize=100000)
def hierarchical(kind,attribute):
    # Preserve released character-FeatureHasher semantics explicitly. Network
    # attributes in the shared manifest expose remote IP, not original full flow.
    sep='.' if kind=='netflow' and '.' in attribute else ':' if kind=='netflow' else '/'
    parts=attribute.split(sep)
    prefixes=[sep.join(parts[:i]) for i in range(1,len(parts)+1)]
    text=kind+''.join(prefixes)
    return FeatureHasher(n_features=16,input_type='string',alternate_sign=True).transform([list(text)]).toarray()[0].astype('float32')


def prepare(manifest_path,output):
    manifest=json.loads(Path(manifest_path).read_text());out=Path(output)
    if (out/'manifest.json').exists():return json.loads((out/'manifest.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    for path,h in manifest['file_sha256'].items():
        if sha256(path)!=h:raise ValueError('Source checksum mismatch: '+path)
    node_set=set()
    counts={}
    for split in ('train','val','test'):
        count=0
        for g in files(manifest,split):
            node_set.update(g.src.tolist());node_set.update(g.dst.tolist());count+=len(g.src)
        counts[split]=count
    nodes=np.asarray(sorted(node_set),dtype=np.int64);del node_set
    np.save(out/'node_ids.npy',nodes)
    attrs=json.loads(Path(manifest['node_attributes']).read_text())
    features=np.empty((len(nodes),16),dtype=np.float32)
    for i,node in enumerate(nodes):
        attr=attrs.get(str(int(node)),{});kind=str(attr.get('type','unknown')).lower()
        kind={'process':'subject','socket':'netflow'}.get(kind,kind)
        value=attr.get('remote_ip') if kind=='netflow' else attr.get('path')
        value='' if value is None else str(value)
        features[i]=hierarchical(kind,value)
    np.save(out/'node_features.npy',features);del attrs
    for split,n in counts.items():
        directory=out/split;directory.mkdir(exist_ok=True)
        arrays={name:np.lib.format.open_memmap(directory/(name+'.npy'),mode='w+',dtype=dtype,shape=shape)
                for name,dtype,shape in [('src','int64',(n,)),('dst','int64',(n,)),('time','int64',(n,)),
                    ('relation','int64',(n,)),('messages','float32',(n,42))]}
        offset=0;origin=None
        for g in files(manifest,split):
            size=len(g.src);s=np.searchsorted(nodes,g.src.numpy());d=np.searchsorted(nodes,g.dst.numpy())
            rel=g.msg[:,131:141].numpy();time=g.t.numpy().astype(np.int64)
            if origin is None:origin=int(time[0])
            arrays['src'][offset:offset+size]=s;arrays['dst'][offset:offset+size]=d
            arrays['time'][offset:offset+size]=time-origin
            arrays['relation'][offset:offset+size]=rel.argmax(-1)
            arrays['messages'][offset:offset+size]=np.concatenate((features[s],rel,features[d]),axis=1)
            offset+=size
        for a in arrays.values():a.flush()
        np.save(directory/'time_origin.npy',np.asarray(origin,dtype=np.int64))
        print('KAIROS adapter prepared',split,n,flush=True)
    result=dict(source_manifest_sha256=sha256(manifest_path),n_nodes=len(nodes),events=counts,
                feature_scope='Fixed character hash; no fitted vocabulary; shared manifest path/remote-IP attributes',
                adaptations=['Ten shared relation classes','Remote IP rather than unavailable full native flow description',
                             'Monotonic compaction of observed node IDs','Per-split nanosecond offset'],
                checksums={str(p.relative_to(out)):sha256(p) for p in out.rglob('*.npy')})
    (out/'manifest.json').write_text(json.dumps(result,indent=2));return result
