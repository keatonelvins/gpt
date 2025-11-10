import torch.nn as nn
from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.modules.l2warp import l2_warp
from torch import LongTensor, Tensor

from gpt.config import LossConfig


class Loss(nn.Module):

    def __init__(self, loss_config: LossConfig):
        super().__init__()
        if loss_config.type == "fused_linear":
            self.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=loss_config.num_chunks,
                use_l2warp=loss_config.use_l2warp,
            )
        elif loss_config.type == "fused":
            self.criterion = FusedCrossEntropyLoss(inplace_backward=True)
        else:
            self.criterion = nn.CrossEntropyLoss()
        self.use_l2warp = loss_config.use_l2warp
        self.materialize_logits = loss_config.type != "fused_linear"

    def forward(self, h: Tensor, labels: LongTensor, lm_head: nn.Linear) -> Tensor:
        if self.materialize_logits:
            logits = lm_head(h)
            loss = self.criterion(logits.view(labels.numel(), -1), labels.view(-1))
            return l2_warp(loss, logits) if self.use_l2warp else loss
        return self.criterion(h, labels, lm_head.weight, lm_head.bias)

def build_loss(loss_config: LossConfig) -> Loss:
    return Loss(loss_config)
