from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    path: str = "karpathy/fineweb-edu-100b-shuffle"
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
    process_batch_size: int = 1000000
    streaming: bool = True


@dataclass
class DistributedConfig:
    dp_replicate: int = 1
    dp_shard: int = -1


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
class Comm:
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100
    trace_buf_size: int = 20000
    save_traces_folder: str = "comm_traces"
    save_traces_file_prefix: str = "rank_"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    dist: DistributedConfig = field(default_factory=DistributedConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    comm: Comm = field(default_factory=Comm)

    project: str = "gpt"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
