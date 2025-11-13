from gpt.models.kda.model import KDA
from gpt.models.transformer.model import Transformer

MODEL_REGISTRY = {
    "transformer": Transformer,
    "kda": KDA,
}

__all__ = ["MODEL_REGISTRY", "KDA", "Transformer"]
