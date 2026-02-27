#!/usr/bin/env python3
"""Compare trained PPO agent vs random, round-robin, and cheapest-first baselines."""

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import llm_router_env  # noqa: F401
from llm_router_env import DEFAULT_MODELS


def rollout(policy_fn, n_episodes: int = 20, episode_length: int = 1000, seed: int = 0):
    """Run n_episodes and return mean episode reward and cost."""
    env = gym.make("LLMRouter-v0", episode_length=episode_length)
    rewards = []
    costs = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        ep_cost = 0.0
        done = False
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            ep_cost += info["cost"]
            done = terminated or truncated
        rewards.append(ep_reward)
        costs.append(ep_cost)

    env.close()
    return np.mean(rewards), np.std(rewards), np.mean(costs)


def main():
    parser = argparse.ArgumentParser(description="Evaluate routing strategies")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained PPO model (.zip). Skipped if not provided.")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    n_models = len(DEFAULT_MODELS)
    cheapest_idx = int(np.argmin([m.cost_per_call for m in DEFAULT_MODELS]))

    strategies = {
        "Random": lambda obs: np.random.randint(n_models),
        "Round-Robin": None,  # handled specially below
        "Cheapest-First": lambda obs: cheapest_idx,
    }

    # Round-robin counter
    rr_counter = [0]

    def round_robin(obs):
        idx = rr_counter[0] % n_models
        rr_counter[0] += 1
        return idx

    strategies["Round-Robin"] = round_robin

    if args.model_path and Path(args.model_path).exists():
        ppo_model = PPO.load(args.model_path)
        strategies["PPO Agent"] = lambda obs: int(ppo_model.predict(obs, deterministic=True)[0])

    print(f"\n{'Strategy':<20} {'Mean Reward':>14} {'Std':>10} {'Mean Cost (USD)':>16}")
    print("-" * 65)

    for name, policy_fn in strategies.items():
        if name == "Round-Robin":
            rr_counter[0] = 0
        mean_r, std_r, mean_c = rollout(policy_fn, args.n_episodes, seed=args.seed)
        print(f"{name:<20} {mean_r:>14.2f} {std_r:>10.2f} {mean_c:>16.4f}")

    print()


if __name__ == "__main__":
    main()
