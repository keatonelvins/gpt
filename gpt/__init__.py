import os
import sys
import warnings

import tyro
from kernels import get_kernel
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.config.manager import ConfigManager, custom_registry
from torchtitan.tools.logging import init_logger

sys.modules["flash_attn"] = get_kernel("kernels-community/flash-attn2")

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
        config_path = sys.argv[1]
        cli_args = [f"--job.config-file={config_path}"] + sys.argv[2:]
        toml_values = manager._maybe_load_toml(cli_args)
        base_config = manager._dict_to_dataclass(Config, toml_values)
        config = tyro.cli(Config, args=cli_args, default=base_config, registry=custom_registry)

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
