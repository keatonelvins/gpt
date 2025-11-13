from typing import Any

import torch
import torch.nn as nn

from gpt.config import ModelConfig
from gpt.models.base import GPT
from gpt.models.transformer import Block


class Transformer(GPT):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_layers)])

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
