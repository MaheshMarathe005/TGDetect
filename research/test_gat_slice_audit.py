"""Check streaming audit against an event-by-event reference, including self edges."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from gat_study.slice_audit import audit_split,DTYPE
from gat_study.diagnostics import DegreeAudit
from sliding_node_mem_v3.rings import merge_windows

class SliceAuditTest(unittest.TestCase):
    def test_chunk_and_segment_invariance(self):
        pairs=[(0,1),(1,2),(2,3),(3,0),(0,0),(1,0),(2,1)]*4
        events=np.zeros(len(pairs),dtype=DTYPE)
        events['src'],events['dst']=np.array(pairs).T
        events['time']=np.arange(len(events))
        reference=DegreeAudit();hist=[[] for _ in range(4)]
        for i,(src,dst) in enumerate(pairs):
            a=torch.tensor([[-1]*(8-len(hist[src]))+hist[src]])
            b=torch.tensor([[-1]*(8-len(hist[dst]))+hist[dst]])
            past=merge_windows(a,b,8,True)
            window=torch.cat((past,torch.tensor([[i]])),1);safe=window.clamp_min(0).numpy()
            reference.observe(torch.from_numpy(np.array(events['src'][safe])),
                              torch.from_numpy(np.array(events['dst'][safe])),window>=0)
            for node in set((src,dst)):hist[node]=(hist[node]+[i])[-8:]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'stream.bin';events.tofile(path)
            for chunk in (1,3,16):
                rows=audit_split(path,4,segment=len(events),chunk=chunk)
                self.assertEqual(rows[0]['degree_audit'],reference.result())
            # Splitting the report must not reset history at boundaries.
            segments=audit_split(path,4,segment=7,chunk=3)
            whole=reference.result()['query_role_nonself_degree']
            for role,histogram in whole.items():
                combined={}
                for segment in segments:
                    for degree,count in segment['degree_audit']['query_role_nonself_degree'][role].items():
                        combined[degree]=combined.get(degree,0)+count
                self.assertEqual(combined,histogram)

if __name__=='__main__':unittest.main()
