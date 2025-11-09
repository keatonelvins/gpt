import math

import torch.nn as nn
from torch.distributed.tensor import DeviceMesh

from gpt.config import Config
from gpt.optimizer.muon import Muon


def build_optimizer(model: nn.Module, config: Config, mesh: DeviceMesh) -> Muon:
    param_groups = [
        {"params": model.matrix_params},
        {"params": model.vector_params, "algorithm": "adamw"},
        {"params": model.embed_params, "algorithm": "adamw", "weight_decay": 0},
        {
            "params": model.lm_head_params,
            "algorithm": "adamw",
            "lr": config.optim.lr / math.sqrt(config.model.hidden_size),
            "weight_decay": 0,
        },
    ]

    return Muon(
        param_groups,
        distributed_mesh=mesh,
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
