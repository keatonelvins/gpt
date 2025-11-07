from pathlib import Path
from typing import Any

import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.nn import Module
from torch.optim.optimizer import Optimizer

from gpt.config import Config


# ref: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class TrainerState(Stateful):
    """Stateful tracker for saving/loading checkpoints."""

    def __init__(self, model: Module, optim: Optimizer):
        self.model = model
        self.optim = optim

    def state_dict(self) -> dict[str, Any]:
        model, optim = get_state_dict(self.model, self.optim, options=StateDictOptions(cpu_offload=True))
        return {
            "model": model,
            "optim": optim,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        set_state_dict(
            self.model,
            self.optim,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
        )


def save_checkpoint(step: int, config: Config, model: Module, optim: Optimizer) -> None:
    """Save resumable DCP checkpoint."""
    path = config.ckpt.save_dir / f"step_{step}"
    logger.info(f"Saving checkpoint to {path}")
    dcp.save({"trainer": TrainerState(model, optim)}, checkpoint_id=str(path))


def load_checkpoint(path: Path, model: Module, optim: Optimizer) -> None:
    """Load trainer state from DCP checkpoint."""
    trainer_state = TrainerState(model, optim)
    dcp.load({"trainer": trainer_state}, checkpoint_id=str(path))
