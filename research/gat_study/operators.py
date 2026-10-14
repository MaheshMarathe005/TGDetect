"""Dense nine-slot causal provenance operators with role-aware message content."""
import math
import torch
from torch import nn
from torch.nn import functional as F

ROLES = ('ss', 'sd', 'ds', 'dd')


def links(src, dst, valid):
    role = torch.stack((src[:,:,None] == src[:,None,:], src[:,:,None] == dst[:,None,:],
                        dst[:,:,None] == src[:,None,:], dst[:,:,None] == dst[:,None,:]), -1)
    length = src.shape[1]
    eye = torch.eye(length, dtype=torch.bool, device=src.device)[None]
    causal = torch.ones(length,length,dtype=torch.bool,device=src.device).tril()
    eligible = valid[:,:,None] & valid[:,None,:] & causal
    allowed = (role.any(-1) & eligible) | eye
    return allowed, role, eligible, eye


class NoMessagePassing(nn.Module):
    def forward(self, x, src, dst, valid):
        return x*valid[...,None]


class DotProductGAT(nn.Module):
    """One head, explicit bmm, shared values and learned scalar role bias."""
    def __init__(self, dim=128):
        super().__init__(); self.dim=dim
        self.q=nn.Linear(dim,dim,bias=False);self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim);self.role=nn.Linear(4,1,bias=False)
        self.norm=nn.LayerNorm(dim)

    def forward(self,x,src,dst,valid):
        allowed,roles,_,_=links(src,dst,valid)
        scores=torch.bmm(self.q(x),self.k(x).transpose(1,2))/math.sqrt(self.dim)
        scores=scores+self.role(roles.to(x.dtype)).squeeze(-1)
        weights=scores.masked_fill(~allowed,-torch.inf).softmax(-1)
        message=torch.bmm(weights,self.v(x))
        return self.norm(x+F.elu(message))*valid[...,None]


class RoleFactoredGAT(nn.Module):
    """Four 32D heads with ss/sd/ds/dd masks and independently projected values.

    Multi-role edges contribute to every applicable head. Self-loops are present
    in every head so empty role neighborhoods remain finite. No scalar role bias.
    """
    def __init__(self,dim=128,deviation=False):
        super().__init__()
        if dim%4:raise ValueError('Role-factored dimension must be divisible by four')
        self.dim=dim;self.head_dim=dim//4;self.deviation=deviation
        self.q=nn.Linear(dim,dim,bias=False);self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim);self.output=nn.Linear(dim,dim)
        if deviation:self.deviation_projection=nn.Linear(3*dim,dim)
        self.norm=nn.LayerNorm(dim)

    def message(self,x,src,dst,valid):
        _,roles,eligible,eye=links(src,dst,valid)
        b,l,_=x.shape
        def heads(layer):return layer(x).reshape(b,l,4,self.head_dim).transpose(1,2)
        q,k,v=heads(self.q),heads(self.k),heads(self.v)
        # [B,4,L,L], not [B,L,L,128]. Role-specific masks/normalizers.
        mask=(roles.permute(0,3,1,2)&eligible[:,None])|eye[:,None]
        scores=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.head_dim)
        attention=scores.masked_fill(~mask,-torch.inf).softmax(-1)
        aggregate=torch.matmul(attention,v).transpose(1,2).reshape(b,l,self.dim)
        return self.output(aggregate)

    def forward(self,x,src,dst,valid):
        message=self.message(x,src,dst,valid)
        if self.deviation:
            mixed=self.deviation_projection(torch.cat((x,message,x-message),-1))
        else:mixed=message
        return self.norm(x+F.elu(mixed))*valid[...,None]


class MeanAggregator(nn.Module):
    """Uniform aggregation with the same shared-endpoint mask/residual as old GAT."""
    def __init__(self,dim=128):
        super().__init__();self.projection=nn.Linear(dim,dim);self.norm=nn.LayerNorm(dim)

    def forward(self,x,src,dst,valid):
        allowed,_,_,_=links(src,dst,valid)
        weights=allowed.to(x.dtype)/allowed.sum(-1,keepdim=True).clamp_min(1)
        return self.norm(x+F.elu(weights@self.projection(x)))*valid[...,None]
