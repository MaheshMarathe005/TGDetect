"""Compiled execution of the existing CPU recurrence; no fast-math or stale state."""
import math
import numpy as np
from numba import njit, vectorize
from numba.typed import List

@vectorize(['float32(float32)', 'float64(float64)'],nopython=True,cache=True)
def erf(x):
    return math.erf(x)

@vectorize(['float32(float32)', 'float64(float64)'],nopython=True,cache=True)
def expit(x):
    if x >= 0:
        return 1. / (1. + math.exp(-x))
    v=math.exp(x)
    return v / (1. + v)

@njit(cache=True)
def gelu(x):
    return (.5*x*(1+erf(x/math.sqrt(2)))).astype(x.dtype)

@njit(cache=True)
def dgelu(x):
    return (.5*(1+erf(x/math.sqrt(2)))+x*np.exp(-.5*x*x)/math.sqrt(2*math.pi)).astype(x.dtype)

@njit(cache=True)
def logsumexp(x):
    m=x.max()
    return m+np.log(np.exp(x-m).sum())

@njit(cache=True)
def add_outer(target,a,b):
    for i in range(len(a)):
        for j in range(len(b)):
            target[i,j]+=a[i]*b[j]

@njit(cache=True)
def add_two_outer(target,a,b):
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            target[i,j]+=a[0,i]*b[0,j]+a[1,i]*b[1,j]

@njit(cache=True)
def forward(state,trace,x,msg,dt,sources,destinations,labels,gm,hm,use_trace,w):
    state=state.copy();trace=trace.copy()
    n,dim=x.shape;rate=np.logaddexp(w[0],0.).astype(x.dtype)
    outputs=np.empty((n,len(w[15])),dtype=x.dtype);surprises=np.empty(n,dtype=x.dtype);cache=List()
    for i in range(n):
        s,d=int(sources[i]),int(destinations[i]);nodes=np.array([s,d])
        gap=dt[np.array([i,i+n])].astype(x.dtype);loggap=np.log1p(gap)[:,None]
        decay=np.exp(-gap[:,None]*rate/3600.).astype(x.dtype)
        pre=state[nodes]*decay
        q=x[i]*gm[i]
        endpoint=expit(w[8]@q+w[9])
        left=w[10]@pre[0];right=w[11]@pre[1]
        decoder_in=np.concatenate((x[i]*hm[i],endpoint[0]*left,endpoint[1]*right))
        hidden_pre=w[12]@decoder_in+w[13];hidden=gelu(hidden_pre)
        logits=w[14]@hidden+w[15]
        surprise=min(10.,max(0.,(logsumexp(logits)-logits[int(labels[i])])/math.log(len(logits))))
        message=np.tanh(msg[np.array([i,i+n])]+loggap@w[1].T)
        old_trace=trace[nodes].copy()
        trace_in=np.concatenate((message,loggap),1)
        trace_decay=np.exp(-gap[:,None]/86400.).astype(x.dtype)
        beta=expit(trace_in@w[6].T+w[7])
        new_trace=(beta*old_trace*trace_decay+(1-beta)*surprise).astype(x.dtype) if use_trace else np.zeros_like(old_trace)
        gate_in=np.concatenate((pre,message,np.full((2,1),surprise,dtype=x.dtype),new_trace),1)
        gate_pre=gate_in@w[2].T+w[3];gate_hidden=gelu(gate_pre)
        gate=expit(gate_hidden@w[4].T+w[5])
        new=(1-gate)*pre+gate*message
        state[s]=new[0];trace[s]=new_trace[0]
        if s!=d:state[d]=new[1];trace[d]=new_trace[1]
        outputs[i]=logits;surprises[i]=surprise
        cache.append((gap,loggap,decay,pre,q,endpoint,left,right,decoder_in,
                      hidden_pre,hidden,message,old_trace,trace_in,trace_decay,
                      beta,gate_in,gate_pre,gate_hidden,gate))
    return outputs,state,trace,surprises,cache

@njit(cache=True)
def backward(cache,use_trace,w,x,gm,hm,sources,destinations,gs,gt,gl,gx,gmsg,gw):
    n,dim=x.shape
    for i in range(n-1,-1,-1):
        s,d=int(sources[i]),int(destinations[i])
        (gap,loggap,decay,pre,q,endpoint,left,right,decoder_in,hidden_pre,hidden,
         message,old_trace,trace_in,trace_decay,beta,gate_in,gate_pre,gate_hidden,gate)=cache[i]
        add_outer(gw[14],gl[i],hidden);gw[15][:]+= gl[i]
        gh=(gl[i]@w[14])*dgelu(hidden_pre)
        add_outer(gw[12],gh,decoder_in);gw[13][:]+= gh
        di=gh@w[12];gx[i]+=di[:dim]*hm[i]
        de=np.array([di[dim:2*dim]@left,di[2*dim:]@right],dtype=x.dtype)*endpoint*(1-endpoint)
        add_outer(gw[8],de,q);gw[9][:]+= de;gx[i]+=(de@w[8])*gm[i]
        dl=di[dim:2*dim]*endpoint[0];dr=di[2*dim:]*endpoint[1]
        add_outer(gw[10],dl,pre[0]);add_outer(gw[11],dr,pre[1])
        dpre=np.stack((dl@w[10],dr@w[11]))
        dh=np.stack((gs[s].copy(),gs[d].copy() if s!=d else np.zeros(dim,dtype=x.dtype)))
        dtrace=np.stack((gt[s].copy(),gt[d].copy() if s!=d else np.zeros(1,dtype=x.dtype)))
        dg=dh*(message-pre)*gate*(1-gate)
        dpre+=dh*(1-gate);dm=dh*gate
        add_two_outer(gw[4],dg,gate_hidden);gw[5][:]+= dg.sum(0)
        dhidden=(dg@w[4])*dgelu(gate_pre)
        add_two_outer(gw[2],dhidden,gate_in);gw[3][:]+= dhidden.sum(0)
        dgi=dhidden@w[2];dpre+=dgi[:,:dim];dm+=dgi[:,dim:2*dim];dtrace+=dgi[:,-1:]
        if use_trace:
            surprise=gate_in[:,-2:-1]
            db=dtrace*(old_trace*trace_decay-surprise)*beta*(1-beta)
            old_trace_grad=dtrace*beta*trace_decay
            add_two_outer(gw[6],db,trace_in);gw[7][:]+= db.sum(0)
            dm+=(db@w[6])[:,:dim]
        else:old_trace_grad=np.zeros_like(dtrace)
        dmessage=dm*(1-message*message)
        gmsg[i]+=dmessage[0];gmsg[i+n]+=dmessage[1]
        add_two_outer(gw[1],dmessage,loggap)
        gw[0][:]+= (-gap[:,None]/3600.*pre*dpre).sum(0)*expit(w[0])
        old_state_grad=dpre*decay
        gs[s]=old_state_grad[0];gt[s]=old_trace_grad[0]
        if s==d:gs[s]+=old_state_grad[1];gt[s]+=old_trace_grad[1]
        else:gs[d]=old_state_grad[1];gt[d]=old_trace_grad[1]
    return gs,gt,gx,gmsg
