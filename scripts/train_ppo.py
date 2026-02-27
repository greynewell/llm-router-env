#!/usr/bin/env python3
"""Train a PPO agent to optimize LLM inference routing."""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import llm_router_env  # noqa: F401 — registers LLMRouter-v0


def make_env(seed: int = 0):
    env = gym.make("LLMRouter-v0", episode_length=1000)
    env = Monitor(env)
    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on LLMRouter-v0")
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="ppo_llm_router")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    args = parser.parse_args()

    env = make_env(args.seed)
    eval_env = make_env(args.seed + 1)

    eval_callback = EvalCallback(
        eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        verbose=1,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print(f"Training PPO for {args.total_timesteps:,} timesteps...")
    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(args.output)
    print(f"Model saved to {args.output}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
