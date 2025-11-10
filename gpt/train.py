import os
from contextlib import nullcontext
from pathlib import Path

import torch
from loguru import logger
from tokenizers import Tokenizer
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.distributed.utils import clip_grad_norm_ as clip
from torchtitan.distributed.utils import init_distributed, set_determinism
from torchtitan.tools.utils import GarbageCollection, device_module, device_type, get_peak_flops, set_default_dtype

from gpt.ckpt import load_checkpoint, save_checkpoint
from gpt.config import Config
from gpt.data import build_dataset
from gpt.loss import build_loss
from gpt.models.gpt import GPT
from gpt.optim import build_optimizer


class Trainer:
    def __init__(self, config: Config):
        self.step = 0
        self.config = config

        self.gc_handler = GarbageCollection()
        self.device_monitor = build_device_memory_monitor()
        self.peak_flops = get_peak_flops(self.device_monitor.device_name)

        self.rank = int(os.getenv('LOCAL_RANK', '0'))
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        device_module.set_device(torch.device(f"{device_type}:{self.rank}"))
        self.device = device_module.current_device()

        if self.world_size > 1:
            init_distributed(config.comm)
            mesh_shape = [config.dist.dp_replicate, config.dist.dp_shard]
            self.mesh = init_device_mesh("cuda", mesh_shape, mesh_dim_names=["dp_replicate", "dp_shard"])
            self.mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")
        else:
            self.mesh = None

        set_determinism(self.mesh, self.device, config.debug, distinct_seed_mesh_dims=[])

        tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
        eos_token_id = tokenizer.token_to_id("<|end_of_text|>")
        ds = build_dataset(config.data, tokenizer, eos_token_id).to_iterable_dataset(self.world_size)
        self.dataset = ds.shard(self.world_size, self.rank).batch(1, drop_last_batch=True)

        param_dtype = getattr(torch, config.trainer.param_dtype)
        reduce_dtype = getattr(torch, config.trainer.reduce_dtype)

        with (torch.device("meta"), set_default_dtype(param_dtype)):
            self.model = GPT(config.model)

        self.loss_fn = build_loss(config.loss)

        if self.mesh is not None:
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
            for block in self.model.layers:
                fully_shard(block, mesh=self.mesh, mp_policy=mp_policy)
            fully_shard(self.model, mesh=self.mesh, mp_policy=mp_policy, reshard_after_forward=False)
            self.maybe_amp = nullcontext()
        else:
            self.maybe_amp = torch.autocast(device_type, dtype=param_dtype)

        self.model.to_empty(device=self.device)
        with torch.no_grad():
            self.model.init_weights()
        self.model.train()

        self.optimizer = build_optimizer(self.model, self.config, self.mesh)
        self.scheduler = build_lr_schedulers([self.optimizer], self.config.sched, self.config.trainer.steps)

        if config.ckpt.resume_from:
            load_checkpoint(Path(config.ckpt.resume_from), self.model, self.optimizer)
            logger.info(f"Resumed from {config.ckpt.resume_from}")

    def forward_backward(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        with self.maybe_amp:
            hidden_states = self.model(**batch)
            loss = self.loss_fn(hidden_states, batch['labels'], self.model.lm_head)

        loss.backward()
        return loss

    def optimizer_step(self):
        grad_norm = clip(self.model.parameters(), self.config.optim.max_norm)
        self.optimizer.step()
        self.scheduler.step()
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
