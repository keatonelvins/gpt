"""Distributed checkpoint saving and loading."""

from pathlib import Path
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from loguru import logger
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from gpt.config import Config


# ref: https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class TrainerState(Stateful):
    """Stateful tracker for saving/loading checkpoints."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        dataloader: StatefulDataLoader | None = None,
        step: int = 0,
    ):
        self.step = step
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader

    def state_dict(self) -> dict[str, Any]:
        state_dict = {"step": self.step}
        state_dict["model"], state_dict["optimizer"] = get_state_dict(
            self.model, self.optimizer, options=StateDictOptions(cpu_offload=True),
        )

        if self.scheduler is not None:
            state_dict["scheduler"] = self.scheduler.state_dict()
        if self.dataloader is not None:
            state_dict["dataloader"] = self.dataloader.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.step = state_dict.get("step", 0)

        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optimizer"],
        )

        if "scheduler" in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict["scheduler"])
        if "dataloader" in state_dict and self.dataloader is not None:
            self.dataloader.load_state_dict(state_dict["dataloader"])


def save_checkpoint(
    step: int,
    config: Config,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataloader: StatefulDataLoader | None,
) -> None:
    """Save checkpoint: resumable DCP or model-only safetensors."""
    path = config.ckpt.save_dir / f"step_{step}"
    logger.info(f"Saving checkpoint to {path}")

    trainer_state = TrainerState(model, optimizer, scheduler, dataloader, step)
    dcp.save({"trainer": trainer_state}, checkpoint_id=str(path))


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    dataloader: StatefulDataLoader | None,
) -> int:
    """Load trainer state from DCP checkpoint (scheduler and dataloader can be skipped)."""
    trainer_state = TrainerState(model, optimizer, scheduler, dataloader)
    dcp.load({"trainer": trainer_state}, checkpoint_id=str(path))
    return trainer_state.step
