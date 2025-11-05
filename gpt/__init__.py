import sys
import tomllib
from pathlib import Path

from torchtitan.config.manager import ConfigManager

from gpt.config import Config
from gpt.train import train


def main():
    config_path = sys.argv[1]
    with Path(config_path).open("rb") as f:
        config_dict = tomllib.load(f)

    manager = ConfigManager()
    config = manager._dict_to_dataclass(Config, config_dict)
    train(config)
