from dataclasses import dataclass


@dataclass
class Data:
    dataset: str = "keatone/TinierStories"
    tokenizer_path: str = "gpt/tokenizer"

@dataclass
class Optim:
    lr: float = 3e-4


@dataclass
class Model:
    d: int = 1024


@dataclass
class Trainer:
    steps: int = 100000


@dataclass
class Config:
    data: Data
    model: Model
    optim: Optim
    trainer: Trainer
    project: str = "gpt"


default_config = Config(Data(), Model(), Optim(), Trainer())
