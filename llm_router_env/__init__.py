"""LLM Router Env — Gymnasium environment for LLM inference routing optimization."""

from gymnasium.envs.registration import register

from .env import LLMRouterEnv
from .models import DEFAULT_MODELS, ModelConfig
from .reward import RewardConfig, compute_reward
from .traffic import PromptRequest, TrafficGenerator

register(
    id="LLMRouter-v0",
    entry_point="llm_router_env.env:LLMRouterEnv",
    max_episode_steps=1000,
)

__all__ = [
    "LLMRouterEnv",
    "ModelConfig",
    "DEFAULT_MODELS",
    "RewardConfig",
    "compute_reward",
    "TrafficGenerator",
    "PromptRequest",
]
