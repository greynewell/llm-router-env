"""Reward function for LLM inference routing."""

from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Weights and thresholds for the routing reward function."""

    cost_weight: float = 1.0
    quality_weight: float = 0.5
    latency_penalty: float = 2.0
    sla_threshold: float = 1.0  # seconds
    quality_miss_penalty: float = 1.0  # penalty per unit of quality shortfall


def compute_reward(
    cost: float,
    quality: float,
    latency: float,
    config: RewardConfig,
    quality_required: float = 0.0,
) -> float:
    """
    Compute the per-step reward.

    r = -cost_weight * cost + quality_weight * quality
        - latency_penalty * max(0, latency - sla_threshold)
        - quality_miss_penalty * max(0, quality_required - quality)

    Args:
        cost: Cost incurred for this call (USD).
        quality: Effective quality score (0-1).
        latency: Response latency (seconds).
        config: Reward weights and SLA threshold.
        quality_required: Minimum quality required for this request (0-1).

    Returns:
        Scalar reward value.
    """
    latency_violation = max(0.0, latency - config.sla_threshold)
    quality_shortfall = max(0.0, quality_required - quality)
    return (
        -config.cost_weight * cost
        + config.quality_weight * quality
        - config.latency_penalty * latency_violation
        - config.quality_miss_penalty * quality_shortfall
    )
