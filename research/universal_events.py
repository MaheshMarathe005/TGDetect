"""Schema-independent normalization before the shared V4-KV event encoder.

Parsers may map vendor fields into this envelope. Flat JSON and structured
envelopes share one path; no learned Windows/network/cloud encoder is selected.
Reconstruction queries must additionally hide their prediction target (see the
CADETS allowlist in orthrus_comparison.protocol).
"""
from collections.abc import Mapping
from datetime import datetime
import math
import time


def normalize_event(raw, *, source='unknown', host=None, ingestion_time=None,
                    timestamp_unit='s', event_id=None):
    if timestamp_unit not in ('s','ms','us','ns'):
        raise ValueError('timestamp_unit must be s, ms, us, or ns')
    fallback=time.time() if ingestion_time is None else float(ingestion_time)
    if not math.isfinite(fallback):raise ValueError('Finite ingestion timestamp required')
    data=dict(raw) if isinstance(raw,Mapping) else {'raw':raw}
    given=data.get('timestamp',data.get('time'))
    stamp=None
    if isinstance(given,(int,float)) and not isinstance(given,bool):
        try:stamp=float(given)/{'s':1,'ms':1e3,'us':1e6,'ns':1e9}[timestamp_unit]
        except OverflowError:pass
    elif isinstance(given,str):
        try:
            dt=datetime.fromisoformat(given.replace('Z','+00:00'))
            # Unspecified timezones cannot establish a global causal order.
            if dt.tzinfo is not None:stamp=dt.timestamp()
        except (ValueError,OverflowError):pass
    observed=stamp is not None and math.isfinite(stamp)
    attributes=data.pop('attributes',{})
    attrs=dict(attributes) if isinstance(attributes,Mapping) else {'raw_attributes':attributes}
    metadata={'type','time','timestamp','source','host','event_id','src_entity','dst_entity'}
    for key,value in data.items():
        if key in metadata:continue
        if key in attrs and attrs[key]!=value:
            # Preserve both values rather than silently choosing one.
            attrs.setdefault('@top_level',{})[key]=value
        else:attrs[key]=value
    if given is not None and not observed:attrs['@unparsed_time']=given
    source_value=str(data.get('source',source))
    host_value=data.get('host',host)
    if source_value!='unknown':attrs.setdefault('source',source_value)
    if host_value is not None:attrs.setdefault('host',host_value)
    origin='observed' if observed else 'ingestion'
    return dict(type=data.get('type'),attributes=attrs,
                timestamp=stamp if observed else fallback,timestamp_origin=origin,
                source=source_value,host=host_value,
                event_id=data.get('event_id',event_id),
                src_entity=data.get('src_entity'),dst_entity=data.get('dst_entity'))
