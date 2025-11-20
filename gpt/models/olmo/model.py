import torch.nn as nn

from gpt.config import ModelConfig
from gpt.models.base import GPT
from gpt.models.olmo.layers import OlmoBlock


class Olmo(GPT):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([OlmoBlock(config, i) for i in range(config.num_layers)])

    def init_weights(self):
        pass
