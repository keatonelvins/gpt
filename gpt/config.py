from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from tokenizers import Tokenizer


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
    process_batch_size: int = 1000000
    streaming: bool = True


@dataclass
class DistributedConfig:
    dp_replicate: int = 1
    dp_shard: int = -1


@dataclass
class OptimConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    cautious_wd: bool = False
    max_norm: float = 1.0


@dataclass
class SchedulerConfig:
    warmup_steps: int = 20
    decay_ratio: float | None = None
    decay_type: Literal["linear", "sqrt", "cosine"] = "linear"
    min_lr_factor: float = 0.0


@dataclass
class AttnConfig:
    num_heads: int = 16
    rope_theta: float = 10000.0
    layers: list[int] = field(default_factory=list)


@dataclass
class ModelConfig:
    type: Literal["transformer", "kda"] = "transformer"
    hidden_size: int = 1024
    num_layers: int = 24
    head_dim: int = 128
    num_heads: int = 16
    norm_eps: float = 1e-6
    vocab_size: int | None = None
    attn: AttnConfig = field(default_factory=AttnConfig)


@dataclass
class LossConfig:
    type: Literal["fused_linear", "fused", "torch"] = "fused_linear"
    use_l2warp: bool = False
    num_chunks: int = 8


@dataclass
class TrainerConfig:
    steps: int = 100
    param_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    reduce_dtype: Literal["float32"] = "float32"
    enable_cpu_offload: bool = False


@dataclass
class MetricsConfig:
    log_freq: int = 1
    enable_wandb: bool = True
    enable_tensorboard: bool = False
    save_tb_folder: str = "metrics"
    save_for_all_ranks: bool = False
    disable_color_printing: bool = False


@dataclass
class Debug:
    seed: int = 42
    deterministic: bool = False
    deterministic_warn_only: bool = False


@dataclass
class CheckpointConfig:
    save_dir: str = "weights"
    save_every: int = 1000
    resume_from: str | None = None


@dataclass
class Job:
    config_file: str | None = None
    dump_folder: str = "./outputs"


@dataclass
class Comm:
    init_timeout_seconds: int = 300
    train_timeout_seconds: int = 100
    trace_buf_size: int = 20000
    save_traces_folder: str = "comm_traces"
    save_traces_file_prefix: str = "rank_"


@dataclass
class FaultToleranceConfig:
    enable: bool = False


@dataclass
class Config:
    job: Job = field(default_factory=Job)
    comm: Comm = field(default_factory=Comm)
    ckpt: CheckpointConfig = field(default_factory=CheckpointConfig)
    data: DataConfig = field(default_factory=DataConfig)
    debug: Debug = field(default_factory=Debug)
    dist: DistributedConfig = field(default_factory=DistributedConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    sched: SchedulerConfig = field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    fault_tolerance: FaultToleranceConfig = field(default_factory=FaultToleranceConfig)

    project: str = "gpt"

    def __post_init__(self):
        if self.model.vocab_size is None:
            tokenizer = Tokenizer.from_file(self.data.tokenizer_path)
            self.model.vocab_size = tokenizer.get_vocab_size()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
