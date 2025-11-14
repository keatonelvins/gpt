import torch.nn as nn

from gpt.config import ModelConfig
from gpt.models.base import GPT
from gpt.models.transformer import Block


class Transformer(GPT):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Block(config, i) for i in range(config.num_layers)])
