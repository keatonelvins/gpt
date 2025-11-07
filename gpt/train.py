import os
from pathlib import Path

import torch
from loguru import logger
from tokenizers import Tokenizer
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.utils import clip_grad_norm_ as clip
from torchtitan.tools import utils

from gpt.ckpt import load_checkpoint, save_checkpoint
from gpt.config import Config
from gpt.data import build_dataset
from gpt.models.gpt import GPT
from gpt.optimizer import Muon


class Trainer:
    def __init__(self, config: Config):
        self.step = 0
        self.config = config

        _ = utils.GarbageCollection()
        device_memory_monitor = build_device_memory_monitor()
        _ = utils.get_peak_flops(device_memory_monitor.device_name)

        self.device_module = utils.device_module
        self.device_type = utils.device_type
        self.rank = int(os.getenv('LOCAL_RANK', '0'))
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.device_module.set_device(torch.device(f"{self.device_type}:{self.rank}"))
        self.device = self.device_module.current_device()

        tokenizer: Tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
        eos_token_id = tokenizer.token_to_id("<|end_of_text|>")

        ds = build_dataset(config.data, tokenizer, eos_token_id)
        ds = ds.to_iterable_dataset(num_shards=self.world_size)
        self.dataset = ds.shard(num_shards=self.world_size, index=self.rank).batch(batch_size=1)

        self.model = GPT(config.model)

        if self.world_size > 1:
            dist_utils.init_distributed(config.comm)
            dp_replicate, dp_shard = config.dist.dp_replicate, config.dist.dp_shard

            self.mesh = init_device_mesh("cuda", [dp_replicate, dp_shard], mesh_dim_names=["dp_replicate", "dp_shard"])
            self.mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")

            fsdp_kwargs = {
                "mesh": self.mesh,
                "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
            }

            for block in self.model.layers:
                fully_shard(block, **fsdp_kwargs)

            fully_shard(self.model, reshard_after_forward=False, **fsdp_kwargs)
        else:
            self.mesh = None

        self.optimizer = Muon(
            self.model.parameters(),
            distributed_mesh=self.mesh,
            lr=config.optim.lr,
        )

        if config.ckpt.resume_from:
            load_checkpoint(Path(config.ckpt.resume_from), self.model, self.optimizer)
            logger.info(f"Resumed from {config.ckpt.resume_from}")

    def forward_backward(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        loss = self.model(**batch)
        loss.backward()
        return loss

    def optimizer_step(self):
        grad_norm = clip(self.model.parameters(), self.config.optim.max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return grad_norm

    def train_step(self, batch):
        loss = self.forward_backward(batch)
        grad_norm = self.optimizer_step()
        self.step += 1

        if self.step % self.config.trainer.log_every == 0:
            logger.info(f"step={self.step} loss={loss.item():.4f} grad_norm={grad_norm.item():.4f}")

        if self.step % self.config.ckpt.save_every == 0:
            save_checkpoint(self.step, self.config, self.model, self.optimizer)

        return loss

    def train(self):
        for batch in self.dataset:
            if self.step >= self.config.trainer.steps:
                break
            self.train_step(batch)
        return self.model
