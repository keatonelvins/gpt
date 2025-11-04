from dataclasses import dataclass, field


@dataclass
class Dataset:
    path: str = "keatone/TinierStories"
    split: str = "train"
    name: str | None = None


@dataclass
class Data:
    dataset: Dataset = field(default_factory=Dataset)
    tokenizer_path: str = "gpt/tokenizer/llama.json"
    seq_len: int = 2048
    pad_to: int = 2048
    column: str = "text"
    skip_cache: bool = False
    process_batch_size: int = 100000


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
    data: Data = field(default_factory=Data)
    model: Model = field(default_factory=Model)
    optim: Optim = field(default_factory=Optim)
    trainer: Trainer = field(default_factory=Trainer)
    project: str = "gpt"
