"""Fused CPU recurrence with explicit first-order reverse-mode derivatives.

Only execution changes: the same decoder, gates, elapsed-time decay and
stop-gradient surprise run without thousands of tiny autograd/scatter nodes.
Weights remain PyTorch parameters optimized by the existing Adam optimizer.
"""
import math
import os
import numpy as np
from scipy.special import erf, expit, logsumexp
import torch


def gelu(x):
    return .5*x*(1+erf(x/math.sqrt(2)))


def dgelu(x):
    return .5*(1+erf(x/math.sqrt(2)))+x*np.exp(-.5*x*x)/math.sqrt(2*math.pi)


class ArrayRecurrence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, initial, initial_trace, fused, messages, elapsed, src, dst,
                targets, gate_mask, head_mask, use_trace, *weights):
        ctx.save_for_backward(initial, initial_trace, fused, messages, elapsed,
                              src, dst, targets, gate_mask, head_mask, *weights)
        arrays=[x.detach().numpy() for x in (initial, initial_trace, fused, messages,
                                            elapsed, src, dst, targets, gate_mask, head_mask)]
        state,trace,x,msg,dt,sources,destinations,labels,gm,hm=arrays
        state=state.copy();trace=trace.copy();w=[p.detach().numpy() for p in weights]
        ctx.compiled=os.environ.get('TG_V4_COMPILED_MEMORY','1')=='1'
        if ctx.compiled:
            from .compiled_memory import forward
            outputs,state,trace,surprises,cache=forward(state,trace,x,msg,dt,sources,destinations,labels,gm,hm,use_trace,tuple(w))
        else:
            n,dim=x.shape;rate=np.logaddexp(w[0],0.)
            outputs=[];surprises=[];cache=[]
            for i in range(n):
                s,d=int(sources[i]),int(destinations[i]);nodes=[s,d]
                gap=dt[[i,i+n]].astype(x.dtype);loggap=np.log1p(gap)[:,None]
                decay=np.exp(-gap[:,None]*rate/3600.)
                pre=state[nodes]*decay
                q=x[i]*gm[i]
                endpoint=expit(w[8]@q+w[9])
                left=w[10]@pre[0];right=w[11]@pre[1]
                decoder_in=np.concatenate((x[i]*hm[i],endpoint[0]*left,endpoint[1]*right))
                hidden_pre=w[12]@decoder_in+w[13];hidden=gelu(hidden_pre)
                logits=w[14]@hidden+w[15]
                surprise=np.clip((logsumexp(logits)-logits[int(labels[i])])/math.log(len(logits)),0.,10.)
                message=np.tanh(msg[[i,i+n]]+loggap@w[1].T)
                old_trace=trace[nodes].copy()
                trace_in=np.concatenate((message,loggap),1)
                trace_decay=np.exp(-gap[:,None]/86400.)
                beta=expit(trace_in@w[6].T+w[7])
                new_trace=beta*old_trace*trace_decay+(1-beta)*surprise if use_trace else np.zeros_like(old_trace)
                gate_in=np.concatenate((pre,message,np.full((2,1),surprise,dtype=x.dtype),new_trace),1)
                gate_pre=gate_in@w[2].T+w[3];gate_hidden=gelu(gate_pre)
                gate=expit(gate_hidden@w[4].T+w[5])
                new=(1-gate)*pre+gate*message
                state[s]=new[0];trace[s]=new_trace[0]
                if s!=d:state[d]=new[1];trace[d]=new_trace[1]
                outputs.append(logits);surprises.append(surprise)
                cache.append((gap,loggap,decay,pre,q,endpoint,left,right,decoder_in,
                              hidden_pre,hidden,message,old_trace,trace_in,trace_decay,
                              beta,gate_in,gate_pre,gate_hidden,gate))
        ctx.cache=cache;ctx.use_trace=use_trace
        surprise_tensor=torch.from_numpy(np.asarray(surprises,dtype=x.dtype))
        ctx.mark_non_differentiable(surprise_tensor)
        return torch.from_numpy(np.asarray(outputs)),torch.from_numpy(state),torch.from_numpy(trace),surprise_tensor

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_logits, grad_state, grad_trace, grad_surprise):
        saved=ctx.saved_tensors
        initial,initial_trace,fused,messages,elapsed,src,dst,targets,gate_mask,head_mask=saved[:10]
        weights=saved[10:];w=[p.detach().numpy() for p in weights]
        x=fused.detach().numpy();gm=gate_mask.numpy();hm=head_mask.numpy()
        sources=src.numpy();destinations=dst.numpy();n,dim=x.shape
        gs=np.zeros_like(initial.detach().numpy()) if grad_state is None else grad_state.numpy().copy()
        gt=np.zeros_like(initial_trace.detach().numpy()) if grad_trace is None else grad_trace.numpy().copy()
        gl=np.zeros((n,len(w[15])),dtype=x.dtype) if grad_logits is None else grad_logits.numpy()
        gx=np.zeros_like(x);gmsg=np.zeros_like(messages.detach().numpy())
        gw=[np.zeros_like(p) for p in w]
        if ctx.compiled:
            from .compiled_memory import backward
            gs,gt,gx,gmsg=backward(ctx.cache,ctx.use_trace,tuple(w),x,gm,hm,sources,destinations,
                                   gs,gt,gl,gx,gmsg,tuple(gw))
        else:
            for i in range(n-1,-1,-1):
                s,d=int(sources[i]),int(destinations[i])
                (gap,loggap,decay,pre,q,endpoint,left,right,decoder_in,hidden_pre,hidden,
                 message,old_trace,trace_in,trace_decay,beta,gate_in,gate_pre,gate_hidden,gate)=ctx.cache[i]
                gw[14]+=np.outer(gl[i],hidden);gw[15]+=gl[i]
                gh=(gl[i]@w[14])*dgelu(hidden_pre)
                gw[12]+=np.outer(gh,decoder_in);gw[13]+=gh
                di=gh@w[12];gx[i]+=di[:dim]*hm[i]
                de=np.array([di[dim:2*dim]@left,di[2*dim:]@right],dtype=x.dtype)*endpoint*(1-endpoint)
                gw[8]+=np.outer(de,q);gw[9]+=de;gx[i]+=(de@w[8])*gm[i]
                dl=di[dim:2*dim]*endpoint[0];dr=di[2*dim:]*endpoint[1]
                gw[10]+=np.outer(dl,pre[0]);gw[11]+=np.outer(dr,pre[1])
                dpre=np.stack((dl@w[10],dr@w[11]))
                dh=np.stack((gs[s].copy(),gs[d].copy() if s!=d else np.zeros(dim,dtype=x.dtype)))
                dtrace=np.stack((gt[s].copy(),gt[d].copy() if s!=d else np.zeros(1,dtype=x.dtype)))
                dg=dh*(message-pre)*gate*(1-gate)
                dpre+=dh*(1-gate);dm=dh*gate
                gw[4]+=dg.T@gate_hidden;gw[5]+=dg.sum(0)
                dhidden=(dg@w[4])*dgelu(gate_pre)
                gw[2]+=dhidden.T@gate_in;gw[3]+=dhidden.sum(0)
                dgi=dhidden@w[2];dpre+=dgi[:,:dim];dm+=dgi[:,dim:2*dim];dtrace+=dgi[:,-1:]
                if ctx.use_trace:
                    surprise=gate_in[:,-2:-1]
                    db=dtrace*(old_trace*trace_decay-surprise)*beta*(1-beta)
                    old_trace_grad=dtrace*beta*trace_decay
                    gw[6]+=db.T@trace_in;gw[7]+=db.sum(0)
                    dm+=(db@w[6])[:,:dim]
                else:old_trace_grad=np.zeros_like(dtrace)
                dmessage=dm*(1-message*message)
                gmsg[i]+=dmessage[0];gmsg[i+n]+=dmessage[1]
                gw[1]+=dmessage.T@loggap
                gw[0]+=(-gap[:,None]/3600.*pre*dpre).sum(0)*expit(w[0])
                old_state_grad=dpre*decay
                gs[s]=old_state_grad[0];gt[s]=old_trace_grad[0]
                if s==d:gs[s]+=old_state_grad[1];gt[s]+=old_trace_grad[1]
                else:gs[d]=old_state_grad[1];gt[d]=old_trace_grad[1]
        grads=[torch.from_numpy(a) for a in (gs,gt,gx,gmsg)]
        return (*grads,None,None,None,None,None,None,None,
                *(torch.from_numpy(a) for a in gw))


def run(memory, rows, src, dst, times, relations, fused, observed, owner):
    from threadpoolctl import threadpool_limits
    n=len(src);last=rows['last_time'].tolist();s=src.tolist();d=dst.tolist();clock=times.tolist()
    gaps=[[0.]*n,[0.]*n]
    for i,(u,v,t) in enumerate(zip(s,d,clock)):
        gaps[0][i]=0. if math.isnan(last[u]) else max(0.,t-last[u])
        gaps[1][i]=0. if math.isnan(last[v]) else max(0.,t-last[v])
        last[u]=last[v]=t
    gap=torch.tensor(gaps,dtype=times.dtype).flatten()
    content=torch.cat((fused,observed),-1)
    messages=torch.cat((memory.source(content),memory.destination(content)))
    gate_mask=torch.nn.functional.dropout(torch.ones_like(fused),p=.1,training=owner.training)
    head_mask=torch.nn.functional.dropout(torch.ones_like(fused),p=.1,training=owner.training)
    weights=(memory.decay_raw,memory.time_message.weight,
             memory.gate[0].weight,memory.gate[0].bias,memory.gate[2].weight,memory.gate[2].bias,
             memory.trace_gate.weight,memory.trace_gate.bias,owner.endpoint_gate.weight,owner.endpoint_gate.bias,
             owner.read_src.weight,owner.read_dst.weight,owner.head[0].weight,owner.head[0].bias,
             owner.head[2].weight,owner.head[2].bias)
    # Retain the single-thread BLAS limit through the later backward call.
    if not hasattr(memory,'_blas_controller'):
        memory._blas_controller=threadpool_limits(limits=1,user_api='blas')
    logits,state,trace,surprise=ArrayRecurrence.apply(rows['memory'],rows['surprise_trace'],fused,messages,
        gap,src,dst,relations,gate_mask,head_mask,memory.use_trace,*weights)
    rows['memory'],rows['surprise_trace'],rows['last_time']=state,trace,times.new_tensor(last)
    return logits,surprise
