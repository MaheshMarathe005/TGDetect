"""Prediction-before-write entity memory; independent entities execute together.

This is the specified vector-state adaptation inspired by Titans, not Titans'
inner-loop gradient descent on a neural associative memory. The optional scalar
surprise trace carries recent surprise into subsequent gates without adding an
auxiliary training objective or accessing attack labels.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def dependency_waves(src, dst):
    """CPU metadata schedule. Events in one wave have disjoint endpoints."""
    last, waves = {}, []
    for i, (s, d) in enumerate(zip(src, dst)):
        level = 1 + max(last.get(s, -1), last.get(d, -1))
        if level == len(waves):
            waves.append([])
        waves[level].append(i)
        last[s] = last[d] = level
    return waves


class SurpriseMemory(nn.Module):
    def __init__(self, dim=128, surprise_trace=True):
        super().__init__()
        self.use_trace = surprise_trace
        self.source = nn.Linear(256, dim)
        self.destination = nn.Linear(256, dim)
        self.time_message = nn.Linear(1, dim, bias=False)
        self.gate = nn.Sequential(nn.Linear(2 * dim + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        nn.init.constant_(self.gate[-1].bias, -2.)
        # Trainable per-dimension rate, initialized to a one-day half-life.
        self.decay_raw = nn.Parameter(torch.full((dim,), math.log(math.expm1(math.log(2) / 24))))
        self.trace_gate = nn.Linear(dim + 1, 1)

    def decay(self, state, elapsed):
        rate = F.softplus(self.decay_raw)
        return state * torch.exp(-elapsed.to(state.dtype)[:, None] * rate / 3600.)

    def update(self, old, projected_message, elapsed, surprise, previous_trace):
        dt = elapsed.log1p().to(old.dtype)[:, None]
        message = torch.tanh(projected_message + self.time_message(dt))
        # Detached, bounded observed-relation surprise cannot optimize its own gate signal.
        surprise = surprise.detach().clamp(0, 10)[:, None]
        if self.use_trace:
            beta = self.trace_gate(torch.cat((message, dt), -1)).sigmoid()
            trace = beta * previous_trace * torch.exp(-elapsed.to(old.dtype)[:, None] / 86400.) + (1-beta) * surprise
        else:
            trace = torch.zeros_like(previous_trace)
        gate = self.gate(torch.cat((old, message, surprise, trace), -1)).sigmoid()
        return torch.lerp(old, message, gate), trace

    def score_update(self, rows, src, dst, times, relations, fused, observed,
                     decode, waves):
        if fused.device.type == 'cpu':
            return self.score_update_sparse(rows, src, dst, times, relations, fused, observed, decode, waves)
        state, trace, last = rows['memory'], rows['surprise_trace'], rows['last_time']
        messages = torch.cat((self.source(torch.cat((fused, observed), -1)),
                              self.destination(torch.cat((fused, observed), -1))))
        n = len(src)
        scores, positions, surprises = [], [], []
        for wave in waves:
            at = torch.as_tensor(wave, device=src.device, dtype=torch.long)
            s, d, now = src[at], dst[at], times[at]
            endpoints = torch.cat((s, d))
            elapsed = now.repeat(2) - last[endpoints]
            elapsed = torch.where(torch.isnan(elapsed), torch.zeros_like(elapsed), elapsed).clamp_min(0)
            pre = self.decay(state[endpoints], elapsed)
            pre_s, pre_d = pre.chunk(2)
            logits = decode(fused[at], pre_s, pre_d)
            surprise = F.cross_entropy(logits, relations[at], reduction='none') / math.log(logits.shape[-1])
            new, new_trace = self.update(pre, messages[torch.cat((at, at+n))], elapsed,
                                         surprise.repeat(2), trace[endpoints])
            # A self-edge reads the same state twice and writes the source role once.
            keep = torch.cat((torch.ones_like(s, dtype=torch.bool), s != d))
            state = state.index_copy(0, endpoints[keep], new[keep])
            trace = trace.index_copy(0, endpoints[keep], new_trace[keep])
            last = last.index_copy(0, endpoints[keep], now.repeat(2)[keep])
            scores.append(logits); positions.append(at); surprises.append(surprise.detach())
        order = torch.cat(positions).argsort()
        rows['memory'], rows['surprise_trace'], rows['last_time'] = state, trace, last
        return torch.cat(scores)[order], torch.cat(surprises)[order]

    def score_update_sparse(self, rows, src, dst, times, relations, fused, observed, decode, waves):
        """Keep row references through recurrence instead of copying full state tables.

        One unbind/stack autograd node per bank avoids a full [batch, dim] scatter
        gradient for every event. This is the same transition and decoder as the
        device-table reference above; only CPU execution/storage differs.
        """
        states=list(rows['memory'].unbind(0));traces=list(rows['surprise_trace'].unbind(0))
        last=rows['last_time'].tolist();sources=src.tolist();destinations=dst.tolist();clock=times.tolist()
        messages=torch.cat((self.source(torch.cat((fused, observed), -1)),
                            self.destination(torch.cat((fused, observed), -1))))
        message_rows=messages.unbind(0);fused_rows=fused.unbind(0)
        n=len(src);scores=[];positions=[];surprises=[]
        for wave in waves:
            s=[sources[i] for i in wave];d=[destinations[i] for i in wave]
            endpoints=s+d;now=[clock[i] for i in wave]*2
            elapsed=times.new_tensor([0. if math.isnan(last[node]) else max(0.,t-last[node])
                                      for node,t in zip(endpoints,now)])
            pre=self.decay(torch.stack([states[node] for node in endpoints]),elapsed)
            ms,md=pre.chunk(2)
            at=torch.tensor(wave,dtype=torch.long)
            logits=decode(torch.stack([fused_rows[i] for i in wave]),ms,md)
            surprise=F.cross_entropy(logits.detach(),relations[at],reduction='none')/math.log(logits.shape[-1])
            selected_messages=torch.stack([message_rows[i] for i in wave]+[message_rows[i+n] for i in wave])
            new,trace=self.update(pre,selected_messages,elapsed,surprise.repeat(2),
                                  torch.stack([traces[node] for node in endpoints]))
            new_rows=new.unbind(0);trace_rows=trace.unbind(0)
            for j,node in enumerate(endpoints):
                if j>=len(wave) and s[j-len(wave)]==node:continue
                states[node]=new_rows[j];traces[node]=trace_rows[j];last[node]=now[j]
            scores.append(logits);positions.append(at);surprises.append(surprise)
        order=torch.cat(positions).argsort()
        rows['memory']=torch.stack(states);rows['surprise_trace']=torch.stack(traces)
        rows['last_time']=times.new_tensor(last)
        return torch.cat(scores)[order],torch.cat(surprises)[order]

