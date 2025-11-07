import sys
import tomllib
from pathlib import Path

from kernels import get_kernel
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.config.manager import ConfigManager

flash_attn = get_kernel("kernels-community/flash-attn2")
sys.modules["flash_attn"] = flash_attn

from gpt.config import Config  # noqa: E402
from gpt.train import train  # noqa: E402


@record
def main():
    config_path = sys.argv[1]
    with Path(config_path).open("rb") as f:
        config_dict = tomllib.load(f)

    manager = ConfigManager()
    config = manager._dict_to_dataclass(Config, config_dict)
    train(config)
