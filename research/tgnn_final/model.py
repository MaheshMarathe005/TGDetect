"""Encoder-only foundation for TGNN Final."""
import torch
from torch import nn
from .encoder import PackedEncoder


class TGNNFinal(nn.Module):
    """Return original V3's 64D event vectors; later components remain pending."""
    def __init__(self, bank):
        super().__init__()
        if not bank.config['two_hash'] or not bank.config['ordered']:
            raise ValueError('TGNN Final requires original corrected-V3 two-hash ordered token rows')
        self.event_encoder = PackedEncoder(bank, dim=64)
        self.config = {'version': 'tgnn-final-encoder-v1', 'stage': 'encoder-only',
                       'output_dim': 64, 'token_config': dict(bank.config)}

    def forward(self, identities):
        """Prepared integer identity tensor [N] -> learned event vectors [N,64]."""
        return self.event_encoder(identities)

    def load_encoder_weights(self, state_dict):
        """Load encoder-only weights, or extract event_encoder.* from V3 weights."""
        prefix = 'event_encoder.'
        if any(key.startswith(prefix) for key in state_dict):
            state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
        self.event_encoder.load_state_dict(state_dict, strict=True)

    @torch.no_grad()
    def precompute_encoder(self, batch_size=1024):
        """Build the existing frozen FP32 lookup after calling model.eval()."""
        return self.event_encoder.precompute(batch_size=batch_size)
