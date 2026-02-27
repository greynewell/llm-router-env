"""Prompt traffic generator with realistic load patterns."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PromptRequest:
    """A single incoming prompt request."""

    length: float          # normalized 0-1
    complexity: float      # 0-1 (beta distributed)
    quality_required: float  # minimum acceptable quality for this request


class TrafficGenerator:
    """
    Generates synthetic prompt traffic with:
    - Varying complexity via beta distribution
    - Sinusoidal time-of-day load patterns with noise
    - Per-request quality requirements
    """

    def __init__(
        self,
        rng: np.random.Generator,
        complexity_alpha: float = 2.0,
        complexity_beta: float = 5.0,
        length_alpha: float = 2.0,
        length_beta: float = 3.0,
    ):
        self.rng = rng
        self.complexity_alpha = complexity_alpha
        self.complexity_beta = complexity_beta
        self.length_alpha = length_alpha
        self.length_beta = length_beta

    def sample(self, time_of_day: float) -> PromptRequest:
        """
        Sample a prompt request.

        Args:
            time_of_day: Normalized time 0-1 (0=midnight, 0.5=noon, 1=midnight).

        Returns:
            PromptRequest with sampled features.
        """
        complexity = float(self.rng.beta(self.complexity_alpha, self.complexity_beta))
        length = float(self.rng.beta(self.length_alpha, self.length_beta))

        # High-complexity prompts more likely to need high quality
        quality_base = 0.5 + 0.4 * complexity
        quality_noise = self.rng.normal(0, 0.05)
        quality_required = float(np.clip(quality_base + quality_noise, 0.0, 1.0))

        return PromptRequest(
            length=length,
            complexity=complexity,
            quality_required=quality_required,
        )

    def load_factor(self, time_of_day: float) -> float:
        """
        Compute load factor at the given time of day (0-1).

        Uses a sinusoidal pattern peaking at business hours (~0.375 = 9am)
        with added noise.
        """
        # Peak around 9am (0.375 of day), trough around 3am (0.125)
        base = 0.5 + 0.4 * np.sin(2 * np.pi * (time_of_day - 0.125))
        noise = self.rng.normal(0, 0.05)
        return float(np.clip(base + noise, 0.1, 1.0))
