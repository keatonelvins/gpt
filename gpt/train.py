import os

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
from torchtitan.tools import utils

from gpt.config import Config
from gpt.data import build_dataset
from gpt.models.gpt import GPT
from gpt.optimizer import Muon
from gpt.utils import setup_logger


@logger.catch(reraise=True)
def train(config: Config):
    setup_logger()

    _ = utils.GarbageCollection()
    device_memory_monitor = build_device_memory_monitor()
    _ = utils.get_peak_flops(device_memory_monitor.device_name)
    device_module, device_type = utils.device_module, utils.device_type
    rank, world_size = int(os.getenv('LOCAL_RANK', '0')), int(os.getenv('WORLD_SIZE', '1'))
    device_module.set_device(torch.device(f"{device_type}:{rank}"))

    tokenizer: Tokenizer = Tokenizer.from_file(config.data.tokenizer_path)
    eos_token_id = tokenizer.token_to_id("<|end_of_text|>")
    config.model.vocab_size = tokenizer.get_vocab_size()

    dataset = build_dataset(config.data, tokenizer, eos_token_id)
    logger.info(f"Built dataset with {len(dataset)} samples")
    dataset = dataset.to_iterable_dataset(num_shards=world_size)
    dataset = dataset.shard(num_shards=world_size, index=rank).batch(batch_size=1)

    model = GPT(config.model)

    if world_size > 1:
        dist_utils.init_distributed(config.comm)
        dp_replicate, dp_shard = config.dist.dp_replicate, config.dist.dp_shard

        mesh = init_device_mesh("cuda", [dp_replicate, dp_shard], mesh_dim_names=["dp_replicate", "dp_shard"])
        mesh["dp_replicate", "dp_shard"]._flatten(mesh_dim_name="dp")

        fsdp_kwargs = {
            "mesh": mesh,
            "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
        }

        for block in model.layers:
            fully_shard(block, **fsdp_kwargs)

        fully_shard(model, reshard_after_forward=False, **fsdp_kwargs)
    else:
        mesh = None

    optimizer = Muon(
        model.parameters(),
        distributed_mesh=mesh,
        lr=config.optim.lr,
    )

    for step, batch in enumerate(dataset):
        if step >= config.trainer.steps:
            break

        batch = {k: v.to(device_module.current_device(), non_blocking=True) for k, v in batch.items()}
        loss = model(**batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return model
