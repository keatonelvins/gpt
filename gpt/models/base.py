from typing import Any

import torch
import torch.nn as nn
from fla.modules import RMSNorm

from gpt.config import ModelConfig


class GPT(nn.Module):
    layers: nn.ModuleList

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    def get_param_groups(self) -> dict[str, Any]:
        return {
            "matrix_params": [p for p in self.layers.parameters() if p.ndim == 2],
            "vector_params": [p for p in self.layers.parameters() if p.ndim != 2],
            "embed_params": list(self.embeddings.parameters()),
            "lm_head_params": list(self.lm_head.parameters()),
        }

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
