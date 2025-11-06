from loguru import logger

from gpt.config import Config
from gpt.data import build_dataset
from gpt.utils import setup_logger


def train(config: Config):
    setup_logger()
    dataset = build_dataset(config.data)
    logger.info(f"Built dataset with {len(dataset)} samples")
