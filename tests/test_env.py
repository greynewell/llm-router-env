"""Tests for LLMRouterEnv — env checker and basic step/reset."""

import numpy as np
from gymnasium.utils.env_checker import check_env

from llm_router_env import DEFAULT_MODELS, LLMRouterEnv, RewardConfig


def make_env(**kwargs) -> LLMRouterEnv:
    env = LLMRouterEnv(seed=42, **kwargs)
    env.reset(seed=42)
    return env


class TestEnvChecker:
    def test_gymnasium_check_env(self):
        """gymnasium.utils.env_checker.check_env must pass without warnings."""
        env = LLMRouterEnv(episode_length=10, seed=0)
        check_env(env, warn=True, skip_render_check=True)
        env.close()


class TestObservationSpace:
    def test_obs_shape(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        n_models = len(DEFAULT_MODELS)
        expected_dim = 2 + n_models + 3
        assert obs.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {obs.shape}"

    def test_obs_includes_quality_required(self):
        """Last obs dimension (quality_required) should be in [0, 1]."""
        env = make_env()
        obs, _ = env.reset(seed=0)
        quality_required = obs[-1]
        assert 0.0 <= quality_required <= 1.0, f"quality_required out of range: {quality_required}"

    def test_obs_dtype(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_obs_bounds(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        assert np.all(obs >= 0.0), "Observation contains values below 0"
        assert np.all(obs <= 1.0), "Observation contains values above 1"

    def test_obs_in_space(self):
        env = make_env()
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)


class TestActionSpace:
    def test_action_space_size(self):
        env = make_env()
        assert env.action_space.n == len(DEFAULT_MODELS)

    def test_all_actions_valid(self):
        env = make_env(episode_length=100)
        env.reset(seed=0)
        for action in range(env.action_space.n):
            obs, reward, terminated, truncated, info = env.step(action)
            assert env.observation_space.contains(obs)
            assert isinstance(reward, float)
            if terminated:
                env.reset(seed=0)


class TestStepReset:
    def test_reset_returns_valid_obs(self):
        env = LLMRouterEnv(seed=0)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = LLMRouterEnv(episode_length=10, seed=0)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.dtype == np.float32
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_terminates_at_length(self):
        n = 20
        env = LLMRouterEnv(episode_length=n, seed=0)
        env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if steps > n + 5:
                break
        assert steps <= n, f"Episode ran {steps} steps, expected <= {n}"

    def test_budget_depletion_terminates(self):
        """With a tiny budget and expensive model, episode ends early."""
        env = LLMRouterEnv(
            episode_length=10000,
            budget=0.003,  # just barely enough for 1 tier1_large call
            seed=0,
        )
        env.reset(seed=0)
        terminated = False
        steps = 0
        while not terminated and steps < 10000:
            _, _, terminated, _, info = env.step(0)  # always pick tier1_large (most expensive)
            steps += 1
        assert terminated, "Episode should have terminated when budget ran out"

    def test_info_keys(self):
        env = LLMRouterEnv(seed=0)
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        for key in ("cost", "quality", "latency", "model_name", "budget_remaining", "sla_violated", "quality_required"):
            assert key in info, f"Missing info key: {key}"

    def test_multiple_resets(self):
        env = LLMRouterEnv(seed=0)
        for i in range(5):
            obs, _ = env.reset(seed=i)
            assert env.observation_space.contains(obs)

    def test_reward_is_finite(self):
        env = LLMRouterEnv(episode_length=50, seed=0)
        env.reset(seed=0)
        for _ in range(50):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert np.isfinite(reward), f"Reward is not finite: {reward}"
            if terminated:
                break


class TestRewardConfig:
    def test_custom_reward_config(self):
        config = RewardConfig(cost_weight=2.0, quality_weight=1.0, latency_penalty=0.0, quality_miss_penalty=0.0)
        env = LLMRouterEnv(reward_config=config, episode_length=10, seed=0)
        env.reset(seed=0)
        obs, reward, _, _, info = env.step(0)
        # With zero latency and quality miss penalty, reward = -2*cost + 1*quality
        expected = -2.0 * info["cost"] + 1.0 * info["quality"]
        assert abs(reward - expected) < 1e-6

    def test_quality_miss_penalty(self):
        """quality_miss_penalty should lower reward when quality < quality_required."""
        from llm_router_env.reward import compute_reward

        config = RewardConfig(cost_weight=0.0, quality_weight=0.0, latency_penalty=0.0, quality_miss_penalty=2.0)
        # quality below required: shortfall = 0.3, penalty = 2.0 * 0.3 = 0.6
        reward = compute_reward(cost=0.0, quality=0.5, latency=0.0, config=config, quality_required=0.8)
        assert abs(reward - (-2.0 * 0.3)) < 1e-6

        # quality meets required: no penalty
        reward_no_miss = compute_reward(cost=0.0, quality=0.9, latency=0.0, config=config, quality_required=0.8)
        assert abs(reward_no_miss - 0.0) < 1e-6

    def test_quality_miss_penalty_in_env(self):
        """Environment applies quality_miss_penalty when quality falls short of quality_required."""
        config = RewardConfig(cost_weight=0.0, quality_weight=0.0, latency_penalty=0.0, quality_miss_penalty=5.0)
        env = LLMRouterEnv(reward_config=config, episode_length=10, seed=0)
        obs, _ = env.reset(seed=0)
        # obs[-1] is quality_required for the first prompt (the one step() will serve)
        quality_required = float(obs[-1])
        _, reward, _, _, info = env.step(0)
        expected = -5.0 * max(0.0, quality_required - info["quality"])
        assert abs(reward - expected) < 1e-6
