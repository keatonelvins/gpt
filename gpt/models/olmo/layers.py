import math

import torch.nn as nn
from fla.layers.attn import Attention
from fla.modules import GatedMLP, RMSNorm
from torch import Tensor

from gpt.config import ModelConfig


class OlmoBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.fp32_residual = config.fp32_residual
        self.layer_idx = layer_idx
        self.num_layers = config.num_layers

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(hidden_size=config.hidden_size, layer_idx=layer_idx, **vars(config.attn))
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(hidden_size=config.hidden_size)

    def init_weights(self):
        self.attn_norm.reset_parameters()
        self.mlp_norm.reset_parameters()

        std = 0.02
        o_std = std / math.sqrt(2 * self.num_layers)

        nn.init.trunc_normal_(self.attn.q_proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.attn.k_proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.attn.v_proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.attn.o_proj.weight, mean=0.0, std=o_std, a=-3 * o_std, b=3 * o_std)

        nn.init.trunc_normal_(self.mlp.gate_proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.mlp.up_proj.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        nn.init.trunc_normal_(self.mlp.down_proj.weight, mean=0.0, std=o_std, a=-3 * o_std, b=3 * o_std)

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        residual = x
        x = self.attn_norm(x)
        x, _, _ = self.attn(hidden_states=x, cu_seqlens=cu_seqlens)
        x, residual = self.mlp_norm(x, residual=residual, prenorm=True, residual_in_fp32=self.fp32_residual)
        return self.mlp(x) + residual
