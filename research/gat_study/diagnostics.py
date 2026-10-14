"""Separate untimed degree audit; avoid GPU->CPU synchronization in benchmarks."""
from collections import Counter
import torch
from .operators import links, ROLES


class DegreeAudit:
    def __init__(self):
        self.query=Counter();self.history=Counter();self.roles={r:Counter() for r in ROLES}
        self.windows=Counter();self.query_shared_history=0;self.query_nonself=0

    @torch.no_grad()
    def observe(self,src,dst,valid):
        allowed,roles,eligible,eye=links(src,dst,valid)
        degree=(allowed & valid[:,None,:]).sum(-1)
        query_valid=valid[:,-1]
        self.query.update(degree[:,-1][query_valid].cpu().tolist())
        self.history.update(degree[:,:-1][valid[:,:-1]].cpu().tolist())
        self.windows.update(valid.sum(-1)[query_valid].cpu().tolist())
        self.query_shared_history+=int((degree[:,-1][query_valid]-1).sum())
        self.query_nonself+=int((valid[:,:-1].sum(-1)[query_valid]).sum())
        for head,name in enumerate(ROLES):
            # Nonself degree for each head; mandatory self-loop excluded.
            count=(roles[...,head]&eligible&~eye).sum(-1)
            self.roles[name].update(count[:,-1][query_valid].cpu().tolist())

    def hook(self,module,args):
        _,src,dst,valid=args;self.observe(src,dst,valid)

    def result(self):
        def summary(hist):
            total=sum(hist.values())
            return {'histogram':dict(sorted(hist.items())),'rows':total,
                    'self_only_fraction':hist[1]/max(total,1),
                    'degree_le2_fraction':sum(v for k,v in hist.items() if k<=2)/max(total,1)}
        return {'query_degree_including_self':summary(self.query),
                'history_degree_including_self':summary(self.history),
                'valid_window_slots':dict(sorted(self.windows.items())),
                'query_role_nonself_degree':{r:dict(sorted(h.items())) for r,h in self.roles.items()},
                'query_shared_history_fraction':self.query_shared_history/max(self.query_nonself,1),
                'scope':'untimed separate forward stream; structural multi-role overlaps counted in every role'}
