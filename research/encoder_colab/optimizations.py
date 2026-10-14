"""Optional measured execution variants and schema coverage probes."""
import copy
import math
from pathlib import Path
import time
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment import (CandidateEncoder, PilotModel, batches, sync, timed, save_json,
                        score, metrics, cdf_fusion, seed_all)


def wrapper(model):
    device = model.encoder.tokens.weight.device
    candidate = CandidateEncoder(model.encoder.bank,
        rarity=model.options.get('rarity', False), interaction=model.options.get('interaction', False)).to(device)
    candidate.load_state_dict(model.encoder.state_dict(), strict=False)
    candidate.eval()
    return candidate


@torch.no_grad()
def field_cache(encoder, cfg):
    """Deduplicate exact prepared field records, then store static learned fields."""
    bank = encoder.bank; device = encoder.tokens.weight.device
    # This is an explicit frozen-weight cache. Caller owns invalidation/rebuild.
    start = time.perf_counter(); encoder.field_table = None
    p_parts, k_parts, pads = [], [], []
    n = bank.config['identities']
    for lo in range(0, n, cfg.batch):
        ids = torch.arange(lo, min(lo+cfg.batch, n), device=device)
        p, k, pad = encoder.fields(ids)
        p_parts.append(p); k_parts.append(k); pads.append(pad)
    p_all, k_all, pads = torch.cat(p_parts), torch.cat(k_parts), torch.cat(pads)
    offsets = np.array(bank.offsets); sizes = np.diff(offsets)
    owners = np.repeat(np.arange(n), sizes)
    within = np.arange(len(bank.fields)) - np.repeat(offsets[:-1], sizes)
    raw = np.array(bank.fields)
    _, first, inverse = np.unique(raw.view(np.dtype((np.void, raw.dtype.itemsize))).ravel(),
                                  return_index=True, return_inverse=True)
    owner = torch.as_tensor(owners[first], device=device)
    pos = torch.as_tensor(within[first], device=device)
    # One extra zero entry for padding.
    p_table = torch.cat((p_all.new_zeros(1, 64), p_all[owner, pos]))
    k_table = torch.cat((k_all.new_zeros(1, 64), k_all[owner, pos]))
    width = bank.config['max_width']
    refs = torch.zeros((n, width), dtype=torch.long, device=device)
    refs[torch.as_tensor(owners, device=device), torch.as_tensor(within, device=device)] = torch.as_tensor(inverse+1, device=device)
    encoder.field_table = (p_table, k_table, refs, pads)
    encoder.field_signature = encoder._weight_signature()
    sync(cfg)
    return {'build_seconds': time.perf_counter()-start,
            'unique_field_records': len(first), 'stored_field_records': len(bank.fields),
            'cache_bytes': sum(t.numel()*t.element_size() for t in encoder.field_table),
            'cache_type': 'complete frozen field table, not bounded LRU',
            'invalidation': 'weight signature checked; training/device changes clear cache',
            'coverage': 1.0, 'build_peak_includes_temporary_per_identity_fields': True}


@torch.no_grad()
def execution_study(run):
    cfg, data, bank = run['config'], run['data'], run['bank']
    results = []
    for arm, model in run['models'].items():
        model = model.to(cfg.device).eval()
        reference_encoder = wrapper(model)
        # Actual held-out identities; KDD has few exact repeats after deduplication.
        for workload in ('actual_unique_test', 'synthetic_repeat_heavy'):
            n = min(128, len(data['test']['ids']))
            source = np.arange(n) if workload == 'actual_unique_test' else np.arange(n) % min(8, n)
            ids = torch.as_tensor(data['test']['ids'][source], device=cfg.device)
            rarity = torch.as_tensor(data['test']['rarity'][source], device=cfg.device)
            reference = reference_encoder(ids, rarity).float()
            def measure(name, encoder, fn=None, extra=None):
                fn = fn or (lambda: encoder(ids, rarity))
                with torch.inference_mode():
                    error = float((fn().float()-reference).abs().max())
                record = {'arm': arm, 'workload': workload, 'variant': name,
                          'max_abs_embedding_error': error, **timed(fn, cfg), **(extra or {})}
                results.append(record)
                return record
            measure('fp32_eager', reference_encoder)
            compact = wrapper(model); compact.compact = True
            measure('compact_bags', compact)
            cached = wrapper(model); info = field_cache(cached, cfg)
            measure('frozen_field_table', cached, extra=info)
            # Full-event caching stores this fixed snapshot only; dynamic rarity is forbidden.
            start = time.perf_counter()
            table = []
            # Cache only test identities for this workload, not arbitrary bank identities.
            all_ids = torch.as_tensor(data['test']['ids'], device=cfg.device)
            all_r = torch.as_tensor(data['test']['rarity'], device=cfg.device)
            for lo in range(0, len(all_ids), cfg.batch):
                table.append(reference_encoder(all_ids[lo:lo+cfg.batch], all_r[lo:lo+cfg.batch]))
            event_table = torch.cat(table); sync(cfg)
            event_info = {'build_seconds': time.perf_counter()-start, 'cache_bytes': event_table.numel()*event_table.element_size(),
                          'cache_type': 'test-identity frozen snapshot table; not LRU', 'coverage': 1.,
                          'streaming_rarity_compatible': False if model.options.get('rarity') else True}
            request = torch.as_tensor(source, device=cfg.device)
            measure('frozen_event_snapshot', reference_encoder, lambda: event_table[request], event_info)
            if cfg.device.startswith('cuda'):
                def bf16():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        return reference_encoder(ids, rarity)
                measure('bf16_autocast', reference_encoder, bf16)
                quantized = wrapper(model)
                w = quantized.tokens.weight.detach()
                scale = w.abs().amax(1, keepdim=True).clamp_min(1e-8)/127
                quantized.int8_tokens = (torch.round(w/scale).clamp(-127, 127).to(torch.int8), scale)
                measure('int8_token_storage_prototype', quantized,
                        extra={'packed_token_bytes': w.numel()+scale.numel()*scale.element_size(),
                               'fp32_reference_token_bytes': w.numel()*w.element_size(),
                               'reference_fp32_table_still_resident': True})
                # Compile/capture only static field-cache -> FiLM -> interaction -> pool.
                class StaticTail(nn.Module):
                    def __init__(self, encoder):
                        super().__init__(); self.encoder = encoder
                    def forward(self, identity, r):
                        p_table, k_table, refs, pads = self.encoder.field_table
                        p, key, pad = p_table[refs[identity]], k_table[refs[identity]], pads[identity]
                        if self.encoder.use_rarity:
                            gamma, beta = self.encoder.film(key).chunk(2, -1)
                            p = p*(1+.1*gamma.tanh()*r[..., None])+.1*beta.tanh()*r[..., None]
                        if self.encoder.use_interaction:
                            p = self.encoder.interact(p, src_key_padding_mask=pad)
                        pooled, _ = self.encoder.attention(self.encoder.query.expand(len(identity), -1, -1), p, p,
                                                           key_padding_mask=pad, need_weights=False)
                        return self.encoder.output(pooled[:, 0])
                tail = StaticTail(cached).eval()
                try:
                    start = time.perf_counter(); compiled = torch.compile(tail, fullgraph=True)
                    compiled(ids, rarity); sync(cfg)
                    measure('compiled_static_tail', cached, lambda: compiled(ids, rarity),
                            {'compile_seconds': time.perf_counter()-start})
                except Exception as exc:
                    results.append({'arm': arm, 'workload': workload, 'variant': 'compiled_static_tail', 'error': repr(exc)})
                try:
                    static_ids, static_r = ids.clone(), rarity.clone()
                    side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(side):
                        for _ in range(3):
                            tail(static_ids, static_r)
                    torch.cuda.current_stream().wait_stream(side)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        static_out = tail(static_ids, static_r)
                    def replay():
                        static_ids.copy_(ids); static_r.copy_(rarity); graph.replay()
                        return static_out
                    measure('cuda_graph_static_tail', cached, replay,
                            {'scope_detail': 'includes fixed input-buffer copies and replay; field-cache build excluded'})
                except Exception as exc:
                    results.append({'arm': arm, 'workload': workload, 'variant': 'cuda_graph_static_tail', 'error': repr(exc)})
            del compact, cached, event_table
        model.cpu()
    save_json(Path(cfg.root)/'results/execution_variants.json', results)
    return results


def distill(run, epochs=2):
    cfg, data = run['config'], run['data']
    if 'full' not in run['models'] or 'old' not in run['models']:
        raise ValueError('Run both old and full arms first.')
    teacher = run['models']['full'].to(cfg.device).eval()
    student = copy.deepcopy(run['models']['old'], memo={id(run['bank']): run['bank']}).to(cfg.device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-4)
    history = []; best = math.inf; best_state = None
    for epoch in range(epochs):
        student.train(); total = 0.; count = 0
        order = np.random.default_rng(cfg.split_seed+epoch).permutation(len(data['train']['ids']))
        for b in batches(data['train'], cfg, order):
            with torch.no_grad():
                target = teacher.encode(b['ids'], b['rarity'])
            embedding = student.encode(b['ids'], b['rarity'])
            loss = F.mse_loss(embedding, target) + F.cross_entropy(student.decoder(embedding), b['service'])
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.detach())*len(embedding); count += len(embedding)
        val = float(score(student, data['val'], cfg, run['positions'])[:, 0].mean())
        history.append({'epoch': epoch+1, 'loss': total/count, 'val_service_ce': val})
        if val < best:
            best = val; best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    student.load_state_dict(best_state); student.eval()
    cal = score(student, data['cal'], cfg, run['positions'])[:, 0]
    test = score(student, data['test'], cfg, run['positions'])[:, 0]
    result = {'history': history, 'metrics': metrics(data['test']['y'], test, cal[len(cal)//2:]),
              'seed_count': 1, 'scope': 'optional teacher/student pilot, not part of primary three-seed matrix'}
    torch.save(best_state, Path(cfg.root)/'results/distilled_student.pt')
    save_json(Path(cfg.root)/'results/distillation.json', result)
    teacher.cpu(); student.cpu()
    return result


def idf_select(attributes, training_documents, limit=64):
    frequency = Counter(key for doc in training_documents for key in doc)
    n = len(training_documents)
    # Deterministic tie breaking; transform uses training statistics only.
    ranked = sorted(attributes, key=lambda k: (-math.log((n+1)/(frequency[k]+1)), k))
    return {k: attributes[k] for k in ranked[:limit]}


def unordered_list_pool(values, embedding):
    # Full list, no first-32 truncation. Only valid for schema-declared sets/bags.
    ids = torch.as_tensor(values, dtype=torch.long, device=embedding.weight.device)
    total = embedding(ids).sum(0) if len(ids) else embedding.weight.new_zeros(embedding.embedding_dim)
    return torch.cat((total, total.new_tensor([math.log1p(len(values))])))


def schema_probes():
    seed_all(123)
    attrs = {f'common_{i:02}': i for i in range(70)}; attrs['zz_rare_key'] = 'evidence'
    docs = [{f'common_{i:02}': i for i in range(70)} for _ in range(20)]
    priority = idf_select(attrs, docs)
    embedding = nn.Embedding(100, 8)
    forward = unordered_list_pool(list(range(50)), embedding)
    reverse = unordered_list_pool(list(reversed(range(50))), embedding)
    changed = unordered_list_pool(list(range(49))+[99], embedding)
    return {'idf_retains_rare_key': 'zz_rare_key' in priority, 'retained_fields': len(priority),
            'list_permutation_max_error': float((forward-reverse).abs().max().detach()),
            'late_element_change_detected': bool(not torch.allclose(forward, changed)),
            'coverage': 'synthetic functional probes only; not KDD detection results',
            'list_policy': 'use only for schema-declared unordered lists; order may be meaningful otherwise'}
