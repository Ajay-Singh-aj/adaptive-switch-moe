"""Switch Transformer Mixture-of-Experts with adaptive capacity routing."""

from .model import (
    AdaptiveSwitchFFN,
    ModelConfig,
    SwitchFFN,
    SwitchTransformerLM,
)

__all__ = ["AdaptiveSwitchFFN", "ModelConfig", "SwitchFFN", "SwitchTransformerLM"]
