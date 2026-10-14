"""TGDetect V4: causal RoPE attention, full episodic tokens, surprise entity state."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from sliding_node_mem_upgrade.model import TGNNSlidingNodeMem as Reference
from .memory import SurpriseMemory, dependency_waves
from sliding_node_mem_v3.rings import RingState,CountTable,plan_ring,gather,merge_windows
from sliding_node_mem_v3.tokens import PackedEncoder
from .temporal import TemporalAttention,ConvTemporalAttention
from sliding_node_mem_v3.memory import endpoint_scan


class TGNNSlidingNodeMem(Reference):
    def __init__(self, n_nodes, bank, num_relations=10, signatures=1,
                 profile='corrected', initial_capacity=16384, exact_counts=False,
                 encoder_dim=16, attention_backend='sdpa', surprise_trace=True, memory_device='model',
                 node_memory='surprise',temporal_heads=4,temporal_cnn=False,
                 **kwargs):
        if node_memory not in ('surprise','affine-v3'):
            raise ValueError('node_memory: surprise or affine-v3')
        if profile != 'corrected' or encoder_dim != (64 if node_memory=='affine-v3' else 16):
            raise ValueError('Corrected V4 requires 16D surprise or 64D V3-affine configuration')
        if node_memory=='affine-v3' and memory_device!='model':
            raise ValueError('V3 affine memory stays on the model device')
        self.node_memory = node_memory
        super().__init__(n_nodes, num_relations=num_relations,
                         buckets=bank.config['buckets'], event_dim=encoder_dim, **kwargs)
        self.corrected = True
        self.gat_layers, self.tied_gat = 2, False
        self.gat_second = copy.deepcopy(self.gat)
        self.event_encoder = PackedEncoder(bank, dim=encoder_dim)
        temporal_class=ConvTemporalAttention if temporal_cnn else TemporalAttention
        self.cnn = temporal_class(heads=temporal_heads,backend=attention_backend)
        self.episodic_dim = 256
        self.state = RingState(n_nodes, initial_capacity=initial_capacity)
        for name in ('hist_c', 'short_c'):
            shape, dtype, fill = self.state.spec[name]
            self.state.spec[name] = ((8, self.episodic_dim), dtype, fill)
        if node_memory=='surprise':
            self.state.spec['surprise_trace'] = ((1,), torch.float32, 0)
            self.state.register_buffer('surprise_trace', None, persistent=False)
        self.counter = CountTable(n_nodes, signatures, exact=exact_counts)
        self.short_key = nn.Linear(self.episodic_dim, 128, bias=False)
        self.short_value = nn.Linear(self.episodic_dim, 128, bias=False)
        if node_memory=='surprise':
            self.entity_memory = SurpriseMemory(surprise_trace=surprise_trace)
        else:
            self.register_buffer('half_lives',torch.tensor([3600.]*64+[86400.]*64),persistent=False)
            self.register_buffer('write_strength',torch.tensor([.5]*64+[.005]*64),persistent=False)
            self.temperature=.15
        if memory_device not in ('model','cpu','array'):
            raise ValueError('memory_device: model, cpu or array')
        self.memory_device = memory_device
        # Preserve the 64D bilinear endpoint head via an explicit shared adapter.
        self.pair_adapter = nn.Identity() if node_memory=='affine-v3' else nn.Linear(encoder_dim, 64)
        self.pair_head = nn.Bilinear(64, 64, 1)
        # These V3 modules have no role in V4; do not retain unused parameters.
        unused = ('compress',) if node_memory=='affine-v3' else ('compress', 'importance_query', 'importance_key', 'admission', 'write_src', 'write_dst')
        for name in unused:
            delattr(self, name)
        self.config.update(version='tgdetect-v4', profile=profile, encoder_dim=encoder_dim,
            episodic_dim=self.episodic_dim, gat_layers=2, tied_gat=False,
            short_readout='attention', temporal='rope-time-exact-attention',
            attention_backend=attention_backend, memory='surprise-vector-state',
            surprise_trace=surprise_trace, token_config=bank.config,
            memory_device=memory_device,
            signatures=signatures, exact_counts=exact_counts,
            initial_capacity=initial_capacity, pair_adapter='16-to-64', compression_loss=False)
        if node_memory=='affine-v3':
            self.config.update(version='tgdetect-v4-v3-encoder-affine',
                memory='affine-v3',memory_half_lives=(3600.,86400.),temperature=self.temperature,
                surprise_trace=False,pair_adapter='identity-64',
                memory_admission='v3-temporal-importance-novelty-rarity')
        self.config.update(temporal_heads=temporal_heads,temporal_cnn=bool(temporal_cnn))
        if temporal_cnn:
            self.config.update(version=self.config['version']+'-cnn',
                temporal='causal-depthwise-k3-rope-time-exact-attention')
        self.reset_state()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        if getattr(self, 'memory_device', 'model') in ('cpu','array'):
            for name in ('entity_memory','head','endpoint_gate','read_src','read_dst'):
                getattr(self, name).cpu()
        return result

    def decode(self, fused, ms, md):
        gates = self.endpoint_gate(F.dropout(fused, p=.1, training=self.training)).sigmoid()
        return self.head(torch.cat((F.dropout(fused, p=.1, training=self.training),
            gates[:, :1] * self.read_src(ms), gates[:, 1:] * self.read_dst(md)), -1))

    def reset_state(self):
        self.next_index=0;self.last_time=None
        if hasattr(self,'state'):self.state.reset();self.counter.reset()

    def detach_state(self):
        self.state.detach()

    def forward_events(self,*args,**kwargs):
        raise TypeError('V4 consumes offline token IDs: use forward_batch(batch)')

    def stream_state_dict(self):
        self.detach_state()
        return dict(config=self.config,next_index=self.next_index,last_time=None if self.last_time is None else self.last_time.cpu(),
                    used=self.state.used,slot_map=None if self.state.slot_map is None else self.state.slot_map.cpu().clone(),
                    rings={k:None if getattr(self.state,k) is None else getattr(self.state,k)[:self.state.used].detach().cpu().clone() for k in self.state.spec},
                    counts=self.counter.counts.detach().cpu().clone())

    def load_stream_state_dict(self,saved):
        if saved['config']!=self.config:raise ValueError('V4 stream configuration mismatch')
        self.reset_state();device=self.project.weight.device
        self.state.slot_map=None if saved['slot_map'] is None else saved['slot_map'].to(device).clone()
        self.state.used=self.state.capacity=saved['used']
        for k,v in saved['rings'].items():
            setattr(self.state,k,None if v is None else v.to(device).clone())
        self.counter.counts.copy_(saved['counts'].to(device))
        self.next_index=saved['next_index']
        self.last_time=None if saved['last_time'] is None else saved['last_time'].to(device)

    def forward_batch(self,batch):
        # Only dependency metadata reaches CPU; learned computations stay on device.
        waves = None if self.node_memory=='affine-v3' else dependency_waves(torch.as_tensor(batch['src']).tolist(), torch.as_tensor(batch['dst']).tolist())
        device=self.project.weight.device
        data={k:torch.as_tensor(v,device=device) for k,v in batch.items()}
        src,dst,times,relations=(data[k] for k in ('src','dst','time','relation'))
        n=len(src)
        if not n:raise ValueError('Empty batch')
        if any(len(v)!=n for v in data.values()):raise ValueError('Mismatched batch columns')
        valid_input=(src>=0).all()&(dst>=0).all()&(src<self.n_nodes).all()&(dst<self.n_nodes).all()
        valid_input &= torch.isfinite(times).all()&(times[1:]>=times[:-1]).all()
        valid_input &= (relations>=0).all()&(relations<self.num_relations).all()
        valid_input &= (data['signature']>=0).all()&(data['signature']<self.counter.signatures).all()
        if self.last_time is not None:valid_input &= times[0]>=self.last_time
        torch._assert_async(valid_input,'Invalid endpoints, relations or chronological timestamps')
        ids=torch.arange(n,device=device)+self.next_index
        endpoints=torch.cat((src,dst));nodes,inverse=torch.unique(endpoints,sorted=True,return_inverse=True)
        self.state.allocate(device,nodes);rows=self.state.read(nodes)
        old=self.state.chronological(rows,'hist')
        infinite=2**60
        current=torch.stack((ids,src,dst,data['observed'],torch.full_like(ids,infinite)),-1)
        hist_meta=torch.cat((old[''].reshape(-1,5),current,current))
        hist_time=torch.cat((old['_t'].flatten(),times,times))
        write_valid=torch.cat((torch.ones_like(src,dtype=torch.bool),src!=dst))
        hp=plan_ring(inverse,old[''][...,0]>=0,write_valid,n)
        incident=gather(hist_meta[:,0],hp.before,fill=-1)
        past=merge_windows(incident[:n],incident[n:],8,self.corrected)
        window_ids=torch.cat((past,ids[:,None]),1)
        # Unique event bank; a sentinel guarantees safe padded lookups.
        raw_bank=torch.cat((current.new_full((1,5),-1),old[''].reshape(-1,5),current))
        bank_ids,bank_inverse=torch.unique(raw_bank[:,0],sorted=True,return_inverse=True)
        first=torch.full_like(bank_ids,len(raw_bank))
        first.scatter_reduce_(0,bank_inverse,torch.arange(len(raw_bank),device=device),reduce='amin',include_self=True)
        meta=raw_bank[first]
        bank_times=torch.cat((times.new_zeros(1),old['_t'].flatten(),times))[first]
        current_at=torch.searchsorted(bank_ids,ids)
        wi=torch.searchsorted(bank_ids,window_ids)
        want=[data['query'],meta[:,3].clamp_min(0)]
        if self.pair_head is not None:want += [data['src_identity'],data['dst_identity'],data['negative_identity']]
        encoded=self.event_encoder(torch.cat(want))
        query=self.project(encoded[:n]);observed_emb=encoded[n:n+len(meta)]
        observed=self.project(observed_emb)
        actual=observed[current_at]
        raw=observed[wi];valid=window_ids>=0
        x=torch.cat((raw[:,:-1],query[:,None]),1)
        ages=(times[:,None]-bank_times[wi]).masked_fill(~valid,0)
        x=(x+self.time(ages.log1p().to(x.dtype)[...,None]).cos())*valid[...,None]
        h=self.gat(x,meta[wi,1],meta[wi,2],valid)
        if self.gat_layers==2:
            h=(self.gat if self.tied_gat else self.gat_second)(h,meta[wi,1],meta[wi,2],valid)
        window_times = torch.cat((bank_times[wi[:, :-1]], times[:, None]), 1)
        if self.node_memory=='affine-v3':
            # V3 admission uses temporal importance; keep V4's final-token context.
            h=self.cnn(h,valid,window_times)
            scores=(self.importance_query(h[:,-1])[:,None]*self.importance_key(h)).sum(-1)/math.sqrt(128)
            attention=scores.masked_fill(~valid,-torch.inf).softmax(-1)
            c=h[:,-1]
        else:
            c = self.cnn.forward_last(h, valid, window_times)
        similarity=F.cosine_similarity(actual[:,None].detach(),raw[:,:-1].detach(),dim=-1).masked_fill(~valid[:,:-1],-1)
        novelty=(1-similarity.amax(-1)).clamp(0,2)
        rare=(self.counter(src,dst,data['signature'])<self.rare_count).any(-1)
        exceptional=rare|(novelty>=self.novelty_threshold)
        selected_at=meta[:,4].clone().masked_fill(bank_ids<0,infinite)
        selected_at=selected_at.scatter_reduce(0,current_at,torch.where(exceptional,ids,infinite),reduce='amin',include_self=True)
        # Keep all 256 learned values; this is not a reconstruction of raw attributes.
        episodic = torch.cat((c, actual), -1)
        bank_c=torch.cat((episodic.new_zeros(1,self.episodic_dim),old['_c'].reshape(-1,self.episodic_dim),episodic))[first]
        # Evictions are committed AFTER the current short-memory read.
        evicted=gather(hist_meta,hp.before[:,0],fill=-1)
        ev_at=torch.searchsorted(bank_ids,evicted[:,0].contiguous())
        admitted=(evicted[:,0]>=0)&(selected_at[ev_at]<=ids.repeat(2))&write_valid
        old_short=self.state.chronological(rows,'short')
        sp=plan_ring(inverse,old_short[''][...,0]>=0,admitted,n)
        short_meta=torch.cat((old_short[''].reshape(-1,2),torch.stack((evicted[:,0],ids.repeat(2)),-1)))
        short_c=torch.cat((old_short['_c'].reshape(-1,self.episodic_dim),bank_c[ev_at]))
        short_t=torch.cat((old_short['_t'].flatten(),bank_times[ev_at]))
        short_incident=gather(short_meta[:,0],sp.before,fill=-1)
        chosen=merge_windows(short_incident[:n],short_incident[n:],9)
        candidates=torch.cat((short_incident[:n],short_incident[n:]),1)
        candidate_refs=torch.cat((sp.before[:n],sp.before[n:]),1)
        match=chosen[:,:,None]==candidates[:,None,:]
        positions=match.long().argmax(-1)
        refs=candidate_refs.gather(1,positions)
        smask=chosen>=0
        refs=refs.masked_fill(~smask,-1)
        roles=torch.where(match[:,:,:8].any(-1),0,1)
        roles=torch.where(match[:,:,:8].any(-1)&match[:,:,8:].any(-1),2,roles)
        active=smask.any(-1).nonzero().flatten()
        short_context=c.new_zeros(n,128)
        if len(active):
            sv=gather(short_c,refs[active]);st=gather(short_t,refs[active]);sm=smask[active]
            age=(times[active,None]-st).masked_fill(~sm,0)
            sx=self.short_value(sv)+self.short_time(age.log1p().to(c.dtype)[...,None]).cos()+self.short_role(roles[active])
            sh=sx*sm[...,None]
            sk=self.short_key(sv)+sh
            sa=((sk*c[active,None]).sum(-1)/math.sqrt(128)).masked_fill(~sm,-1e9).softmax(-1)*sm
            pooled=(sa[...,None]*sh).sum(1)
            short_context=short_context.index_copy(0,active,pooled)
        fused=self.fusion_norm(c+self.fusion(torch.cat((c,short_context),-1)).sigmoid()*short_context)
        if self.node_memory=='affine-v3':
            gate=self.admission(torch.stack((attention[:,-1],novelty,rare.to(c.dtype)),-1)).sigmoid().squeeze(-1)
            soft_keep=torch.where(rare,torch.ones_like(novelty),torch.sigmoid((novelty-self.novelty_threshold)/self.temperature))
            admission=gate*soft_keep
            writes=torch.cat((self.write_src(torch.cat((fused,actual),-1)),self.write_dst(torch.cat((fused,actual),-1))))
            ms,md,updated=endpoint_scan(rows['memory'],inverse[:n],inverse[n:],
                admission.repeat(2)[:,None]*self.write_strength,writes.repeat(1,2),
                times=times,last_seen=rows['last_time'],half_lives=self.half_lives)
            logits=self.decode(fused,ms,md)
            rows['memory']=updated
            rows['last_time']=torch.full_like(rows['last_time'],-torch.inf).scatter_reduce(0,inverse,times.repeat(2),reduce='amax',include_self=True)
        elif self.memory_device in ('cpu','array'):
            memory_rows = {k: rows[k].cpu() for k in ('memory','surprise_trace','last_time')}
            if self.memory_device == 'array':
                from .array_memory import run as array_run
                logits, surprise = array_run(self.entity_memory,memory_rows,inverse[:n].cpu(),inverse[n:].cpu(),
                    times.cpu(),relations.cpu(),fused.cpu(),actual.cpu(),self)
            else:
                logits, surprise = self.entity_memory.score_update(memory_rows, inverse[:n].cpu(), inverse[n:].cpu(),
                    times.cpu(), relations.cpu(), fused.cpu(), actual.cpu(), self.decode, waves)
            for key, value in memory_rows.items(): rows[key] = value.to(device)
            logits, surprise = logits.to(device), surprise.to(device)
        else:
            logits, surprise = self.entity_memory.score_update(rows, inverse[:n], inverse[n:],
                times, relations, fused, actual, self.decode, waves)
        # Commit final ring rows in one scatter per tensor, with unique node IDs.
        hist_at=torch.searchsorted(bank_ids,hist_meta[:,0].contiguous())
        hist_meta=torch.cat((hist_meta[:,:4],selected_at[hist_at,None]),1)
        self.state.write_ring(rows,'hist',{'':hist_meta,'_t':hist_time,'_c':bank_c[hist_at]},hp)
        self.state.write_ring(rows,'short',{'':short_meta,'_t':short_t,'_c':short_c},sp)
        self.state.stage(nodes,rows)
        self.next_index+=n;self.last_time=times[-1].detach()
        aux=dict(novelty=novelty, rare=rare,short_count=smask.sum(-1),window_ids=window_ids)
        if self.node_memory=='affine-v3':
            aux.update(admission=admission,attention=attention)
        else:
            aux.update(surprise=surprise,dependency_waves=len(waves))
        if self.pair_head is not None:
            left,right,negative=self.pair_adapter(encoded[n+len(meta):]).split(n)
            pos=self.pair_head(left,right).squeeze(-1);neg=self.pair_head(left,negative).squeeze(-1)
            mask=data['negative_valid'].to(pos.dtype)
            # Unknown/ambiguous negatives contribute neither positive nor negative
            # training examples, avoiding an all-positive collapse on tiny graphs.
            aux['pair_loss']=((F.softplus(-pos)+F.softplus(neg))*mask).sum()/(2*mask.sum().clamp_min(1))
            aux['pair_anomaly']=F.softplus(-pos)
        return logits,aux
