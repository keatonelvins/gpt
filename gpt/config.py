from dataclasses import dataclass, field


@dataclass
class Dataset:
    path: str = "keatone/TinierStories"
    split: str = "train"
    name: str | None = None


@dataclass
class Data:
    pad_to: int = 2048
    seq_len: int = 2048
    column: str = "text"
    skip_cache: bool = False
    process_batch_size: int = 100000
    tokenizer_path: str = "gpt/tokenizer/llama.json"
    dataset: Dataset = field(default_factory=Dataset)


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
