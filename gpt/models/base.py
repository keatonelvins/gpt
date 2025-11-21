import math

import torch
import torch.nn as nn
from fla.modules import RMSNorm

from gpt.config import ModelConfig


class GPT(nn.Module):
    layers: nn.ModuleList

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def init_weights(self):
        std = 0.02
        nn.init.trunc_normal_(self.embed.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

        self.norm.reset_parameters()

        for layer in self.layers:
            layer.init_weights()

        head_std = 1.0 / math.sqrt(self.config.hidden_size)
        nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=head_std, a=-3 * head_std, b=3 * head_std)

    def get_param_groups(self) -> dict[str, list[torch.Tensor]]:
        return {
            "matrix_params": [p for p in self.layers.parameters() if p.ndim == 2],
            "vector_params": [p for p in self.layers.parameters() if p.ndim != 2],
            "embed_params": list(self.embed.parameters()),
            "lm_head_params": list(self.lm_head.parameters()),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden_states = self.embed(batch["input_ids"])
        cu_seqlens = batch["cu_seqlens"].squeeze(0).to(torch.int32)  # drop batch dim
        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens=cu_seqlens)
        return self.norm(hidden_states)
