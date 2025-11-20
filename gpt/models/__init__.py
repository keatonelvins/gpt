from gpt.models.olmo.model import Olmo

MODEL_REGISTRY = {
    "olmo": Olmo,
}

__all__ = ["MODEL_REGISTRY"]
