import os
import sys
import warnings

import torch
import tyro
from kernels import get_kernel
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.config.manager import ConfigManager, custom_registry
from torchtitan.tools.logging import init_logger

fa_version = "3" if "H100" in torch.cuda.get_device_name() else "2"
sys.modules["flash_attn"] = get_kernel("kernels-community/flash-attn" + fa_version)

from gpt.config import Config  # noqa: E402
from gpt.train import Trainer  # noqa: E402

warnings.filterwarnings("ignore", message="To copy construct from a tensor.*", category=UserWarning)

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "gpt"


def train():
    cmd = ["torchrun", "gpt/__init__.py"] + sys.argv[1:]
    os.execvp("torchrun", cmd)


@record
def main():
    init_logger()
    config, manager = Config(), ConfigManager(config_cls=Config)

    if len(sys.argv) > 1:
        if sys.argv[1].startswith("--"):
            config = tyro.cli(Config, args=sys.argv[1:], registry=custom_registry)
        else:
            config_path = sys.argv[1]
            cli_args = [f"--job.config-file={config_path}"] + sys.argv[2:]
            toml_values = manager._maybe_load_toml(cli_args)
            base_config = manager._dict_to_dataclass(Config, toml_values)
            config = tyro.cli(Config, args=cli_args, default=base_config, registry=custom_registry)

    trainer = Trainer(config)
    try:
        trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
