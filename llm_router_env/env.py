"""Core Gymnasium environment for LLM inference routing optimization."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .models import DEFAULT_MODELS, ModelConfig
from .reward import RewardConfig, compute_reward
from .traffic import TrafficGenerator


class LLMRouterEnv(gym.Env):
    """
    Gymnasium environment for training an RL agent to route LLM inference requests.

    The agent observes prompt features and system state, then selects a model
    to handle each incoming request. The objective is to minimize cost while
    meeting latency SLAs and quality thresholds.

    Observation space (Box, float32):
        [0] prompt_length       — normalized 0-1
        [1] prompt_complexity   — 0-1 (beta distributed)
        [2..N+1] queue_depths   — normalized queue depth per model (0-1)
        [N+2] time_of_day       — normalized 0-1
        [N+3] budget_remaining  — normalized 0-1

    Action space (Discrete):
        Index into the list of available models.

    Reward:
        r = -cost_weight * cost + quality_weight * quality
            - latency_penalty * max(0, latency - sla_threshold)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        models: list[ModelConfig] | None = None,
        reward_config: RewardConfig | None = None,
        episode_length: int = 1000,
        budget: float = 10.0,
        max_queue_depth: float = 50.0,
        seed: int | None = None,
    ):
        """
        Args:
            models: List of ModelConfig presets. Defaults to DEFAULT_MODELS.
            reward_config: Reward weights and SLA. Defaults to RewardConfig().
            episode_length: Number of prompts per episode.
            budget: Total cost budget for the episode (USD).
            max_queue_depth: Max queue depth for normalization.
            seed: Optional RNG seed.
        """
        super().__init__()

        self.models = models if models is not None else DEFAULT_MODELS
        self.reward_config = reward_config if reward_config is not None else RewardConfig()
        self.episode_length = episode_length
        self.initial_budget = budget
        self.max_queue_depth = max_queue_depth
        self._seed = seed

        n_models = len(self.models)
        # obs: [prompt_length, prompt_complexity, *queue_depths, time_of_day, budget_remaining]
        obs_dim = 2 + n_models + 2
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_models)

        # State (initialized in reset)
        self._rng: np.random.Generator | None = None
        self._traffic: TrafficGenerator | None = None
        self._queue_depths: np.ndarray | None = None
        self._step_count: int = 0
        self._budget_remaining: float = 0.0
        self._time_of_day: float = 0.0
        self._current_prompt_length: float = 0.0
        self._current_prompt_complexity: float = 0.0
        self._current_quality_required: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng_seed = seed if seed is not None else self._seed
        self._rng = np.random.default_rng(rng_seed)
        self._traffic = TrafficGenerator(rng=self._rng)

        self._step_count = 0
        self._budget_remaining = self.initial_budget
        # Randomize start time of day each episode
        self._time_of_day = float(self._rng.uniform(0.0, 1.0))
        # Randomize initial queue depths (lightly loaded)
        self._queue_depths = self._rng.uniform(0.0, 0.2, size=len(self.models)).astype(
            np.float32
        )

        obs = self._get_obs()
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._rng is not None, "Call reset() before step()"
        assert self.action_space.contains(action), f"Invalid action: {action}"

        model = self.models[action]
        prompt_complexity = self._current_prompt_complexity

        # Sample outcomes for the selected model
        latency = model.sample_latency(self._rng)
        quality = model.sample_quality(prompt_complexity, self._rng)
        cost = model.cost_per_call

        # Update state
        self._budget_remaining = max(0.0, self._budget_remaining - cost)
        self._step_count += 1

        # Advance time (1 step = ~1 minute of simulated time, 1440 steps = 1 day)
        self._time_of_day = (self._time_of_day + 1.0 / 1440.0) % 1.0

        # Update queue depths: selected model gets a small load spike, others decay
        self._queue_depths[action] = min(
            self.max_queue_depth,
            self._queue_depths[action] + self._rng.exponential(2.0),
        )
        decay = self._rng.uniform(0.9, 0.98, size=len(self.models))
        self._queue_depths = (self._queue_depths * decay).astype(np.float32)

        reward = compute_reward(cost, quality, latency, self.reward_config)

        # Sample next prompt
        prompt = self._traffic.sample(self._time_of_day)
        self._current_prompt_length = prompt.length
        self._current_prompt_complexity = prompt.complexity
        self._current_quality_required = prompt.quality_required

        terminated = self._step_count >= self.episode_length or self._budget_remaining <= 0.0
        truncated = False

        info = {
            "cost": cost,
            "quality": quality,
            "latency": latency,
            "model_name": model.name,
            "budget_remaining": self._budget_remaining,
            "quality_required": self._current_quality_required,
            "sla_violated": latency > self.reward_config.sla_threshold,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build normalized observation vector."""
        # Sample initial prompt if at step 0
        if self._step_count == 0:
            assert self._traffic is not None
            prompt = self._traffic.sample(self._time_of_day)
            self._current_prompt_length = prompt.length
            self._current_prompt_complexity = prompt.complexity
            self._current_quality_required = prompt.quality_required

        queue_norm = np.clip(
            self._queue_depths / self.max_queue_depth, 0.0, 1.0
        ).astype(np.float32)
        budget_norm = float(np.clip(self._budget_remaining / self.initial_budget, 0.0, 1.0))

        obs = np.concatenate([
            [self._current_prompt_length, self._current_prompt_complexity],
            queue_norm,
            [self._time_of_day, budget_norm],
        ]).astype(np.float32)

        return obs
