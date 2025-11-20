import torch.nn as nn
from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp
from torch import LongTensor, Tensor

from gpt.config import LossConfig


class Loss(nn.Module):
    def __init__(self, config: LossConfig):
        super().__init__()
        if config.type == "fused_linear":
            self.criterion = FusedLinearCrossEntropyLoss(use_l2warp=config.use_l2warp)
        elif config.type == "fused":
            self.criterion = FusedCrossEntropyLoss(inplace_backward=True)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.use_l2warp = config.use_l2warp
        self.materialize_logits = config.type != "fused_linear"

    def forward(self, hidden_states: Tensor, labels: LongTensor, lm_head: nn.Linear) -> Tensor:
        if self.materialize_logits:
            logits = lm_head(hidden_states)
            loss = self.criterion(logits.view(labels.numel(), -1), labels.view(-1))
            return l2_warp(loss, logits) if self.use_l2warp else loss
        return self.criterion(hidden_states, labels, lm_head.weight, lm_head.bias)


def build_loss(loss_config: LossConfig) -> Loss:
    return Loss(loss_config)
