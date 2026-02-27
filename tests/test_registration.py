"""Tests that gymnasium.make() works for registered environments."""

import gymnasium as gym
import numpy as np


class TestRegistration:
    def test_make_llm_router(self):
        """gymnasium.make('LLMRouter-v0') should succeed."""
        env = gym.make("LLMRouter-v0")
        obs, info = env.reset()
        assert obs is not None
        assert env.observation_space.contains(obs)
        env.close()

    def test_make_with_kwargs(self):
        """gymnasium.make() should forward kwargs to the environment."""
        env = gym.make("LLMRouter-v0", episode_length=50)
        assert env.unwrapped.episode_length == 50
        env.close()

    def test_make_step(self):
        """A full step cycle via gymnasium.make() should work."""
        env = gym.make("LLMRouter-v0")
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert isinstance(reward, float)
        assert np.isfinite(reward)
        env.close()
