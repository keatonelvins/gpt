from torch.distributed.tensor import DeviceMesh

from gpt.config import Config
from gpt.models.base import GPT
from gpt.optimizer.muon import Muon


def build_optimizer(model: GPT, config: Config, mesh: DeviceMesh) -> Muon:
    param_groups = model.get_param_groups()
    params = [
        {"params": param_groups["matrix_params"]},
        {"params": param_groups["vector_params"], "algorithm": "adamw"},
        {"params": param_groups["embed_params"], "algorithm": "adamw", "weight_decay": 0},
        {
            "params": param_groups["lm_head_params"],
            "algorithm": "adamw",
            "lr": config.optim.lr / config.model.hidden_size ** 0.5,
            "weight_decay": 0,
        },
    ]

    return Muon(
        params,
        distributed_mesh=mesh,
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )
