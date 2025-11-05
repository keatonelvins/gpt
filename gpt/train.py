from gpt.config import Config
from gpt.data import build_dataset


def train(config: Config):
    print(config)
    dataset = build_dataset(config.data)