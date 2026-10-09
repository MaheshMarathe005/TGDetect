"""KAIROS encoder/decoder traced to the authors' CADETS_E3 implementation.

Upstream commit 0e0b633beb46a1117c0a6d63be5d2481b59ac0dc, model.py/train.py.
PyG TGNMemory: IdentityMessage, LastAggregator, GRU, 100 state/time dimensions.
Two UniMP TransformerConv layers (8 heads then 1); 100 per endpoint, 200 paired.
Nine output relations follow paper Table 1 (released CADETS uses seven).
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, LastNeighborLoader


class GraphAttentionEmbedding(nn.Module):
    def __init__(self, time_enc, msg_dim=41, state_dim=100, embedding_dim=100):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(state_dim, embedding_dim, heads=8, dropout=0., edge_dim=edge_dim)
        self.conv2 = TransformerConv(8 * embedding_dim, embedding_dim, heads=1,
                                     concat=False, dropout=0., edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        relative_time = last_update[edge_index[0]] - t
        attr = torch.cat((self.time_enc(relative_time.to(x.dtype)), msg), -1)
        return F.relu(self.conv2(F.relu(self.conv(x, edge_index, attr)), edge_index, attr))


class LinkPredictor(nn.Module):
    def __init__(self, embedding_dim=100, relations=9):
        super().__init__()
        d = embedding_dim
        self.lin_src, self.lin_dst = nn.Linear(d, 2*d), nn.Linear(d, 2*d)
        self.lin_seq = nn.Sequential(nn.Linear(4*d, 8*d), nn.Dropout(.5), nn.Tanh(),
            nn.Linear(8*d, 2*d), nn.Dropout(.5), nn.Tanh(),
            nn.Linear(2*d, d//2), nn.Dropout(.5), nn.Tanh(), nn.Linear(d//2, relations))

    def forward(self, src, dst):
        return self.lin_seq(torch.cat((self.lin_src(src), self.lin_dst(dst)), -1))


class Kairos(nn.Module):
    def __init__(self, n_nodes, msg_dim=41, relations=9):
        super().__init__()
        self.memory = TGNMemory(n_nodes, msg_dim, 100, 100,
            message_module=IdentityMessage(msg_dim, 100, 100), aggregator_module=LastAggregator())
        self.gnn = GraphAttentionEmbedding(self.memory.time_enc, msg_dim)
        self.decoder = LinkPredictor(relations=relations)
        self.n_nodes = n_nodes
        self.neighbors = None
        self.register_buffer('assoc', torch.empty(n_nodes, dtype=torch.long), persistent=False)

    def reset_state(self):
        self.memory.reset_state()
        self.neighbors = LastNeighborLoader(self.n_nodes, size=20, device=self.assoc.device)

    def score(self, src, dst, history_times, history_messages):
        """Current message/type is intentionally absent from the predictor input."""
        if self.neighbors is None:
            self.reset_state()
        nodes, edge_index, event_ids = self.neighbors(torch.cat((src, dst)).unique())
        self.assoc[nodes] = torch.arange(len(nodes), device=src.device)
        z, last = self.memory(nodes)
        z = self.gnn(z, last, edge_index, history_times[event_ids], history_messages[event_ids])
        return self.decoder(z[self.assoc[src]], z[self.assoc[dst]])

    def observe(self, src, dst, times, messages):
        self.memory.update_state(src, dst, times, messages)
        self.neighbors.insert(src, dst)

    def detach_state(self):
        self.memory.detach()

    def learned_state(self):
        # Runtime history is reconstructed chronologically, never imported from test.
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()
                if k not in {'memory.memory', 'memory.last_update', 'memory._assoc'}}
