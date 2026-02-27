"""Model configurations for LLM inference routing."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for a single LLM model endpoint."""

    name: str
    cost_per_call: float  # USD
    latency_mean: float   # seconds
    latency_std: float    # seconds
    quality_score: float  # 0-1

    def sample_latency(self, rng: np.random.Generator) -> float:
        """Sample latency from a truncated normal distribution (min 0.01s)."""
        raw = rng.normal(self.latency_mean, self.latency_std)
        return max(0.01, raw)

    def sample_quality(self, prompt_complexity: float, rng: np.random.Generator) -> float:
        """
        Sample effective quality given prompt complexity.

        High-complexity prompts expose quality gaps between models more than
        low-complexity ones. Adds small Gaussian noise for realism.
        """
        # Complexity amplifies the gap from 1.0
        gap = 1.0 - self.quality_score
        effective = 1.0 - gap * (0.5 + 0.5 * prompt_complexity)
        noise = rng.normal(0, 0.02)
        return float(np.clip(effective + noise, 0.0, 1.0))


# Five preset model tiers shipped with the environment
DEFAULT_MODELS: list[ModelConfig] = [
    ModelConfig(
        name="tier1_large",
        cost_per_call=0.030,
        latency_mean=2.0,
        latency_std=0.5,
        quality_score=0.95,
    ),
    ModelConfig(
        name="tier1_small",
        cost_per_call=0.003,
        latency_mean=0.5,
        latency_std=0.1,
        quality_score=0.82,
    ),
    ModelConfig(
        name="tier2_large",
        cost_per_call=0.015,
        latency_mean=1.5,
        latency_std=0.4,
        quality_score=0.90,
    ),
    ModelConfig(
        name="tier2_small",
        cost_per_call=0.001,
        latency_mean=0.3,
        latency_std=0.08,
        quality_score=0.75,
    ),
    ModelConfig(
        name="open_source",
        cost_per_call=0.0005,
        latency_mean=0.8,
        latency_std=0.3,
        quality_score=0.70,
    ),
]
