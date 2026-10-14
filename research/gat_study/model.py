"""Substitute only the GAT region of corrected V3 for full-model experiments."""
import torch
from sliding_node_mem_v3.model import TGNNSlidingNodeMem as V3
from .operators import NoMessagePassing,DotProductGAT,RoleFactoredGAT,MeanAggregator

ARMS=('none','gat1','gat2','gat2_tied','dot_role_bias','role4','role4_deviation','mean')


class StudyV3(V3):
    def __init__(self,*args,operator='gat2',**kwargs):
        if operator not in ARMS:raise ValueError('Unknown graph operator: '+operator)
        if kwargs.get('profile','corrected') != 'corrected':raise ValueError('Corrected V3 required')
        kwargs.pop('gat_layers',None);kwargs.pop('tied_gat',None)
        # Build common V3 modules BEFORE operator-dependent parameters so common
        # initial weights are identical for a paired seed.
        super().__init__(*args,gat_layers=2,tied_gat=False,**kwargs)
        self.gat_layers=2 if operator in ('gat2','gat2_tied') else 1
        self.tied_gat=operator=='gat2_tied'
        if self.gat_layers==1 or self.tied_gat:self.gat_second=None
        # Keep graph-initialization RNG from affecting later task dropout draws.
        with torch.random.fork_rng(devices=[]):
            if operator=='none':self.gat=NoMessagePassing()
            elif operator=='dot_role_bias':self.gat=DotProductGAT()
            elif operator=='role4':self.gat=RoleFactoredGAT()
            elif operator=='role4_deviation':self.gat=RoleFactoredGAT(deviation=True)
            elif operator=='mean':self.gat=MeanAggregator()
        self.config.update(gat_operator=operator,gat_layers=self.gat_layers,tied_gat=self.tied_gat,
                           study_version='role-message-study-v1')
