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
        raise NotImplementedError

    def get_param_groups(self) -> dict[str, list[torch.Tensor]]:
        return {
            "matrix_params": [p for p in self.layers.parameters() if p.ndim == 2],
            "vector_params": [p for p in self.layers.parameters() if p.ndim != 2],
            "embed_params": list(self.embeddings.parameters()),
            "lm_head_params": list(self.lm_head.parameters()),
        }

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.FloatTensor:
        hidden_states = self.embeddings(batch["input_ids"])
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states)
        return self.norm(hidden_states)
