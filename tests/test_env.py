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
        expected_dim = 2 + n_models + 2
        assert obs.shape == (expected_dim,), f"Expected shape ({expected_dim},), got {obs.shape}"

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
        for key in ("cost", "quality", "latency", "model_name", "budget_remaining", "sla_violated"):
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
        config = RewardConfig(cost_weight=2.0, quality_weight=1.0, latency_penalty=0.0)
        env = LLMRouterEnv(reward_config=config, episode_length=10, seed=0)
        env.reset(seed=0)
        obs, reward, _, _, info = env.step(0)
        # With zero latency penalty, reward = -2*cost + 1*quality
        expected = -2.0 * info["cost"] + 1.0 * info["quality"]
        assert abs(reward - expected) < 1e-6
