import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from tokenizers import Tokenizer
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_metrics_processor
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.utils import clip_grad_norm_ as clip
from torchtitan.distributed.utils import init_distributed, set_determinism
from torchtitan.tools import utils
from torchtitan.tools.logging import logger

from gpt.ckpt import load_checkpoint, save_checkpoint
from gpt.config import Config
from gpt.data import build_dataset
from gpt.loss import build_loss
from gpt.models import MODEL_REGISTRY
from gpt.optim import build_optimizer


class Trainer:
    def __init__(self, config: Config):
        self.step = 0
        self.config = config

        self.rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))

        utils.device_module.set_device(torch.device(f"{utils.device_type}:{self.rank}"))
        self.device = utils.device_module.current_device()
        self.gc = utils.GarbageCollection()

        init_distributed(config.comm, enable_cpu_backend=config.trainer.enable_cpu_offload)

        self.parallel_dims = ParallelDims(
            **vars(config.dist), pp=1, cp=1, ep=1, etp=1, world_size=self.world_size
        )
        self.mesh = self.parallel_dims.world_mesh if self.world_size > 1 else None
        self.metrics = build_metrics_processor(config, self.parallel_dims)

        set_determinism(self.mesh, self.device, config.debug, distinct_seed_mesh_dims=[])

        tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
        eos_token_id = tokenizer.token_to_id(config.data.eos_token)
        ds = build_dataset(config.data, tokenizer, eos_token_id).to_iterable_dataset(self.world_size)
        self.dataset = iter(ds.shard(self.world_size, self.rank).batch(1, drop_last_batch=True))

        param_dtype = getattr(torch, config.trainer.param_dtype)
        reduce_dtype = getattr(torch, config.trainer.reduce_dtype)

        with torch.device("meta"), utils.set_default_dtype(param_dtype):
            self.model = MODEL_REGISTRY[config.model.type](config.model)

        self.loss_fn = build_loss(config.loss)

        if self.mesh is not None:
            mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
            for block in self.model.layers:
                fully_shard(block, mesh=self.mesh, mp_policy=mp_policy)
            fully_shard(self.model, mesh=self.mesh, mp_policy=mp_policy, reshard_after_forward=False)
            self.maybe_amp = nullcontext()
        else:
            self.maybe_amp = torch.autocast(utils.device_type, dtype=param_dtype)

        self.model.to_empty(device=self.device)
        with torch.no_grad():
            self.model.init_weights()
        self.model.train()

        self.optimizer = build_optimizer(self.model, self.config, self.mesh)
        self.scheduler = build_lr_schedulers([self.optimizer], self.config.sched, self.config.trainer.steps)

        if config.ckpt.resume_from:
            load_checkpoint(Path(config.ckpt.resume_from), self.model, self.optimizer)
            self.gc.collect("GC after checkpoint load")
            logger.info(f"Resumed from {config.ckpt.resume_from}")

        self.metrics.num_flops_per_token = getattr(self.model, "flops_per_token", 1)
        self.total_tokens = 0

    def forward_backward(self, batch):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

        with self.maybe_amp:
            hidden_states = self.model(**batch)
            loss = self.loss_fn(hidden_states, batch["labels"], self.model.lm_head)

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

        ntokens = batch["input_ids"].numel()
        self.metrics.ntokens_since_last_log += ntokens
        self.total_tokens += ntokens

        if self.metrics.should_log(self.step):
            self.metrics.log(
                step=self.step,
                global_avg_loss=loss.item(),
                global_max_loss=loss.item(),
                grad_norm=grad_norm.item(),
                extra_metrics={"total_tokens": self.total_tokens},
            )

        if self.step % self.config.ckpt.save_every == 0:
            save_checkpoint(self.step, self.config, self.model, self.optimizer)
            self.gc.collect("GC after checkpoint save")

        return loss

    def train(self):
        while self.step < self.config.trainer.steps:
            data_load_start = time.perf_counter()
            batch = next(self.dataset)
            self.metrics.data_loading_times.append(time.perf_counter() - data_load_start)

            self.train_step(batch)
            self.gc.run(self.step)
        return self.model

    def close(self) -> None:
        self.metrics.close()
