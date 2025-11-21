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
        self.embed.reset_parameters()
        self.norm.reset_parameters()
        self.lm_head.reset_parameters()
        for layer in self.layers:
            layer.init_weights()

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
