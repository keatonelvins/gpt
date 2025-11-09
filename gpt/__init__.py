import sys
import warnings

import tyro
from kernels import get_kernel
from loguru import logger
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.config.manager import ConfigManager, custom_registry

sys.modules["flash_attn"] = get_kernel("kernels-community/flash-attn2")

from gpt.config import Config  # noqa: E402
from gpt.train import Trainer  # noqa: E402
from gpt.utils import setup_logger  # noqa: E402

warnings.filterwarnings("ignore", message="To copy construct from a tensor.*", category=UserWarning)


@record
@logger.catch(reraise=True)
def main():
    setup_logger()

    config_path = sys.argv[1]
    cli_args = [f"--job.config-file={config_path}"] + sys.argv[2:]

    manager = ConfigManager(config_cls=Config)
    toml_values = manager._maybe_load_toml(cli_args)
    base_config = manager._dict_to_dataclass(Config, toml_values) if toml_values else Config()
    config = tyro.cli(Config, args=cli_args, default=base_config, registry=custom_registry)

    trainer = Trainer(config)
    trainer.train()
