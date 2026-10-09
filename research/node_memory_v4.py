"""Experimental V4: independent endpoint admission and learned wall-time decay.

V3 is preserved. State is runtime-only; callers must feed chronological seconds.
Decay half-lives initialize at one hour / one day, and are learned from task loss.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from node_memory_v3 import TGNNNodeMemV3, endpoint_scan


def decayed_endpoint_scan(table, last_seen, src, dst, times, weights, content, half_lives):
    """Exact pre-event reads for M <- (1-w)*decay(dt)*M + w*content.

    Associative affine scan; chronological within each entity, self-edge writes once.
    last_seen is float64 seconds, NaN for unseen entities. No future state enters reads.
    """
    batch = len(src)
    if not batch:
        return table[src], table[dst], table, last_seen
    ar = torch.arange(batch, device=src.device)
    nodes = torch.cat((src, dst))
    order = torch.argsort(nodes * (2 * batch + 1) + torch.cat((2 * ar, 2 * ar + 1)))
    ns, ts = nodes[order], torch.cat((times, times))[order]
    first = torch.ones_like(ns, dtype=torch.bool)
    first[1:] = ns[1:] != ns[:-1]
    previous = torch.cat((ts[:1], ts[:-1]))
    previous = torch.where(first, last_seen[ns], previous)
    elapsed = torch.where(torch.isnan(previous), torch.zeros_like(ts), ts - previous)
    if bool((elapsed < 0).any()):
        raise ValueError('Memory stream moved backwards')
    decay = torch.exp(-math.log(2) * elapsed.to(table.dtype)[:, None] / half_lives)
    w, x = weights[order], content[order]
    duplicate = torch.cat((torch.zeros_like(src, dtype=torch.bool), src == dst))[order]
    a = torch.where(duplicate[:, None], torch.ones_like(w), (1 - w) * decay)
    b = torch.where(duplicate[:, None], torch.zeros_like(x), w * x)
    group = first.long().cumsum(0)
    step = 1
    while step < len(ns):
        same = (group[step:] == group[:-step])[:, None]
        aa = torch.where(same, a[step:] * a[:-step], a[step:])
        bb = torch.where(same, a[step:] * b[:-step] + b[step:], b[step:])
        a, b = torch.cat((a[:step], aa)), torch.cat((b[:step], bb))
        step *= 2
    initial = table[ns]
    post = a * initial + b
    before = torch.where(first[:, None], initial, torch.cat((initial[:1], post[:-1])))
    before = before * decay
    original = before[torch.argsort(order)]
    ms, md = original[:batch], original[batch:]
    md = torch.where((src == dst)[:, None], ms, md)
    last = torch.ones_like(first)
    last[:-1] = first[1:]
    if torch.is_grad_enabled():
        table = table.index_copy(0, ns[last], post[last])
    else:
        table.index_copy_(0, ns[last], post[last])
    return ms, md, table, last_seen.index_copy(0, ns[last], ts[last])


class TGNNNodeMemV4(TGNNNodeMemV3):
    def __init__(self, input_dim=41, n_nodes=1, gat_layers=2, memory_dim=64):
        super().__init__(input_dim, n_nodes, True, gat_layers, True, memory_dim)
        self.attn_score = nn.Linear(128, 2)
        self.log_half_lives = nn.Parameter(torch.tensor([math.log(3600.), math.log(86400.)]))

    def reset_state(self):
        super().reset_state()
        self.last_seen = None

    def _context_forward(self, c, src, dst, times):
        if self.Mem is None:
            self.Mem = c.new_zeros(self.n_nodes, 2 * self.memory_dim)
            self.Reference = c.new_zeros(self.n_nodes, self.memory_dim)
            self.last_seen = torch.full((self.n_nodes,), float('nan'), dtype=torch.float64, device=c.device)
        ws, wd = self.write_src(c), self.write_dst(c)
        content = torch.cat((ws, wd))
        with torch.no_grad():
            rs, rd, self.Reference = endpoint_scan(
                self.Reference, src, dst, torch.full_like(content, .1), content.detach())
            reference = torch.cat((rs, rd))
            novelty = 1 - F.cosine_similarity(content.detach(), reference, dim=1, eps=1e-6)
            base = self.head(self.mem_norm(c)).softmax(-1)
            unc = -(base * base.clamp_min(1e-9).log()).sum(-1) / math.log(2)
        importance = self.attn_score(c).sigmoid().transpose(0, 1).reshape(-1)
        cand = torch.maximum(((importance - self.candidate_threshold) / .1).sigmoid(),
                             (novelty >= self.nov_tau).to(c.dtype))
        other = torch.cat((rd, rs))
        feats = torch.stack((importance.detach(), novelty, torch.cat((unc, unc)),
                             reference.norm(dim=1), other.norm(dim=1),
                             (reference.norm(dim=1) > 0).float(), (other.norm(dim=1) > 0).float()), 1)
        rate = self.controller.rate(feats)
        score = self.controller.score(importance, novelty, torch.cat((unc, unc)))
        admit = cand * ((score - self.controller.threshold) /
                        self.controller.log_temp.exp().clamp(.01, 10)).sigmoid()
        # A self-edge only performs the source write. Exclude its suppressed
        # destination candidate from both admission mass and budget penalties.
        active = torch.cat((torch.ones_like(src, dtype=torch.bool), src != dst)).to(c.dtype)
        cand, admit = cand * active, admit * active
        strength = c.new_tensor([.5] * self.memory_dim + [.005] * self.memory_dim)
        half_lives = self.log_half_lives.exp().clamp(1., 365 * 86400.).repeat_interleave(self.memory_dim)
        ms, md, self.Mem, self.last_seen = decayed_endpoint_scan(
            self.Mem, self.last_seen, src, dst, times, admit[:, None] * strength,
            torch.cat((content, content), 1), half_lives)
        gates = self.read_gate(c).tanh()
        z = c + gates[:, :1] * self.read_src(ms) + gates[:, 1:] * self.read_dst(md)
        return self.head(self.mem_norm(z)), dict(admit=admit, cand=cand, novelty=novelty,
            rate=rate, read_gate=gates, effective_fast_weight=admit * .5,
            effective_slow_weight=admit * .005)

    def forward_events(self, x, valid, ages, src, dst, times, intervention=None):
        times = torch.as_tensor(times, device=x.device, dtype=torch.float64)
        if not len(times) or not bool(torch.isfinite(times).all()):
            raise ValueError('Nonempty finite event times required')
        if bool((times[1:] < times[:-1]).any()) or (self.last_time is not None and times[0] < self.last_time):
            raise ValueError('Chronological input required')
        if intervention not in (None, 'constant_time'):
            raise ValueError('Unsupported intervention')
        age = torch.zeros_like(ages) if intervention == 'constant_time' else ages
        extra = self.time_encoder(age.unsqueeze(-1)).cos() * valid.unsqueeze(-1)
        c = self.encode(x, valid, extra)
        out = self._context_forward(c, torch.as_tensor(src, device=x.device, dtype=torch.long),
                                   torch.as_tensor(dst, device=x.device, dtype=torch.long), times)
        self.last_time = float(times[-1])
        return out
