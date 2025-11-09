from typing import Any

import torch
import torch.nn as nn
from fla.layers.attn import Attention
from fla.modules import GatedMLP, RMSNorm

from gpt.config import ModelConfig


class Block(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attn = Attention(
            hidden_size=config.hidden_size,
            num_heads=config.attn.num_heads,
            rope_theta=config.attn.rope_theta,
            layer_idx=layer_idx,
        )
        self.mlp_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = GatedMLP(config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, _ = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        return hidden_states, attentions


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.matrix_params = [p for p in self.layers.parameters() if p.ndim == 2]
        self.vector_params = [p for p in self.layers.parameters() if p.ndim != 2]
        self.embed_params  = list(self.embeddings.parameters())
        self.lm_head_params= list(self.lm_head.parameters())

    def init_weights(self):
        sigma = 1.0 / (self.config.hidden_size ** 0.5)
        nn.init.trunc_normal_(self.embeddings.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

        for layer in self.layers:
            layer.attn_norm.reset_parameters()
            layer.mlp_norm.reset_parameters()
            if hasattr(layer.attn, 'rotary'):
                layer.attn.rotary.reset_parameters()

        self.norm.reset_parameters()
        sigma = 1.0 / (self.config.hidden_size ** 0.5)
        nn.init.trunc_normal_(self.lm_head.weight, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(
        self,
        input_ids: torch.LongTensor,
        position_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: dict[str, Any],
    ) -> torch.FloatTensor:
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask, **kwargs)
        return self.norm(hidden_states)
