from typing import Any

import torch
import torch.nn as nn
from fla.layers.attn import Attention
from fla.layers.kda import KimiDeltaAttention
from fla.modules import GatedMLP, RMSNorm

from gpt.config import ModelConfig


class KDABlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        if config.attn is not None and layer_idx in config.attn.layers:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn.num_heads,
                rope_theta=config.attn.rope_theta,
                layer_idx=layer_idx,
            )
        else:
            self.attn = KimiDeltaAttention(
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                layer_idx=layer_idx,
            )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(hidden_size=config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return (hidden_states, attentions, past_key_values)
