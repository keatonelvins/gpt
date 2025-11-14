import torch.nn as nn

from gpt.config import ModelConfig
from gpt.models.base import GPT
from gpt.models.kda import KDABlock


class KDA(GPT):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([KDABlock(config, i) for i in range(config.num_layers)])
