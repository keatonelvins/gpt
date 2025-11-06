from loguru import logger
from torchtitan.components.metrics import build_device_memory_monitor
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools import utils

from gpt.config import Config
from gpt.data import build_dataset
from gpt.utils import setup_logger


@logger.catch(reraise=True)
def train(config: Config):
    setup_logger()
    dataset = build_dataset(config.data)
    logger.info(f"Built dataset with {len(dataset)} samples")

    _ = utils.GarbageCollection()
    dist_utils.init_distributed(config.comm)
    device_memory_monitor = build_device_memory_monitor()
    _ = utils.get_peak_flops(device_memory_monitor.device_name)
