import torch.nn as nn
from fla.layers.attn import Attention
from fla.modules import GatedMLP, RMSNorm
from torch import Tensor

from gpt.config import ModelConfig


class OlmoBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(hidden_size=config.hidden_size, layer_idx=layer_idx, **vars(config.attn))
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(hidden_size=config.hidden_size)

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> tuple[Tensor, Tensor]:
        residual, x = x, self.attn_norm(x)
        x, attentions, _ = self.attn(hidden_states=x, cu_seqlens=cu_seqlens)
        x, residual = self.mlp_norm(x, residual=residual)
        x = self.mlp(x) + residual

        return x, attentions
