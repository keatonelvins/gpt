from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    path: str = "keatone/TinierStories"
    split: str = "train"
    name: str | None = None


@dataclass
class DataConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tokenizer_path: str = "gpt/tokenizer/llama.json"
    seq_len: int = 2048
    pad_to: int = 2048
    column: str = "text"
    skip_cache: bool = False
    process_batch_size: int = 100000


@dataclass
class OptimConfig:
    lr: float = 3e-4


@dataclass
class AttnConfig:
    num_heads: int = 16
    rope_theta: float = 10000.0
    layers: list[int] = field(default_factory=list)


@dataclass
class ModelConfig:
    hidden_size: int = 1024
    head_dim: int = 128
    num_heads: int = 16
    norm_eps: float = 1e-6
    attn: AttnConfig = field(default_factory=AttnConfig)


@dataclass
class TrainerConfig:
    steps: int = 100000


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    project: str = "gpt"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
