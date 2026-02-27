"""Reward function for LLM inference routing."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Weights and thresholds for the routing reward function."""

    cost_weight: float = 1.0
    quality_weight: float = 0.5
    latency_penalty: float = 2.0
    sla_threshold: float = 1.0  # seconds


def compute_reward(
    cost: float,
    quality: float,
    latency: float,
    config: RewardConfig,
) -> float:
    """
    Compute the per-step reward.

    r = -cost_weight * cost + quality_weight * quality
        - latency_penalty * max(0, latency - sla_threshold)

    Args:
        cost: Cost incurred for this call (USD).
        quality: Effective quality score (0-1).
        latency: Response latency (seconds).
        config: Reward weights and SLA threshold.

    Returns:
        Scalar reward value.
    """
    latency_violation = max(0.0, latency - config.sla_threshold)
    return (
        -config.cost_weight * cost
        + config.quality_weight * quality
        - config.latency_penalty * latency_violation
    )
