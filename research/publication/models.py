"""Explicit study baselines and contribution ablations, not invented reproductions."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv
from orthrus_comparison.official import OrthrusReference
from sliding_node_mem_v3.model import TGNNSlidingNodeMem


class GraphSAGEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.first=SAGEConv(100,128)
        self.second=SAGEConv(128,64)

    def forward(self,x,edge_index,edge_feats=None,**kwargs):
        return self.second(F.dropout(F.relu(self.first(x,edge_index)),p=.25,training=self.training),edge_index)


class GraphSAGEReference(OrthrusReference):
    """Mean-neighbor GraphSAGE; same prepared features/neighbor store/decoder.

    This is our graph reconstruction baseline, not the Wazuh paper's supervised
    attention-GNN or its rehearsal learning algorithm.
    """
    def __init__(self,n_nodes,device='cpu',**kwargs):
        super().__init__(n_nodes,device=device,**kwargs)
        self.encoder.encoder=GraphSAGEEncoder().to(device)


def zero_read(module,args,output):
    return torch.zeros_like(output)


class StudyV3(TGNNSlidingNodeMem):
    def __init__(self,*args,long_readout=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.config['long_readout']=long_readout
        self.config['study_model_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if not long_readout:
            # A contribution ablation, not a claimed removal of state costs.
            self.read_src.register_forward_hook(zero_read)
            self.read_dst.register_forward_hook(zero_read)
