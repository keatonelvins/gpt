import sys
import tomllib
from pathlib import Path

from kernels import get_kernel
from loguru import logger
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.config.manager import ConfigManager

sys.modules["flash_attn"] = get_kernel("kernels-community/flash-attn2")

from gpt.config import Config  # noqa: E402
from gpt.train import Trainer  # noqa: E402
from gpt.utils import setup_logger  # noqa: E402


@record
@logger.catch(reraise=True)
def main():
    setup_logger()

    config_path = sys.argv[1]
    with Path(config_path).open("rb") as f:
        config_dict = tomllib.load(f)

    manager = ConfigManager()
    config = manager._dict_to_dataclass(Config, config_dict)
    trainer = Trainer(config)
    trainer.train()
