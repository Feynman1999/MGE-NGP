import megengine.functional as F
import megengine.module as M
from ..registry import BACKBONES


# Positional encoding (section 5.1)
@BACKBONES.register_module
class PositionalEncoding(M.Module):
    def __init__(self, multires, log_sampling = True, include_input = True):
		self.input_dims = 2
		self.multires = multires
		self.log_sampling = log_sampling
		self.include_input = include_input
		self.periodic_fns = [F.sin, F.cos]
		self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.multires - 1
        N_freqs = self.multires

        if self.log_sampling:
            freq_bands = 2.0 ** F.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = F.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return F.concat([fn(inputs) for fn in self.embed_fns], -1)
