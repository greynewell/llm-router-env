#!/usr/bin/env python3
"""
Demo: Train a PPO agent and plot training curves vs baselines.

Run with: python scripts/demo_notebook.py
Outputs: training_curves.png
"""

import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

import llm_router_env  # noqa: F401
from llm_router_env import DEFAULT_MODELS


def train_ppo(total_timesteps: int = 100_000, seed: int = 0, log_dir: str = "/tmp/ppo_demo"):
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(gym.make("LLMRouter-v0", episode_length=200), log_dir)
    model = PPO(
        "MlpPolicy", env, verbose=0, seed=seed,
        n_steps=512, batch_size=64, n_epochs=5, ent_coef=0.01,
    )
    model.learn(total_timesteps=total_timesteps)
    env.close()
    return model, log_dir


def baseline_rewards(strategy: str, n_episodes: int = 50, episode_length: int = 200, seed: int = 0):
    cheapest = int(np.argmin([m.cost_per_call for m in DEFAULT_MODELS]))
    n_models = len(DEFAULT_MODELS)
    rr = [0]

    policies = {
        "random": lambda _: np.random.randint(n_models),
        "round_robin": lambda _: (rr.__setitem__(0, rr[0] + 1), (rr[0] - 1) % n_models)[1],
        "cheapest": lambda _: cheapest,
    }
    policy = policies[strategy]
    env = gym.make("LLMRouter-v0", episode_length=episode_length)
    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_r = 0.0
        done = False
        while not done:
            obs, r, term, trunc, _ = env.step(policy(obs))
            ep_r += r
            done = term or trunc
        rewards.append(ep_r)
    env.close()
    return np.mean(rewards)


def main():
    print("Training PPO for 100k steps (episode_length=200)...")
    model, log_dir = train_ppo(total_timesteps=100_000)

    print("\nComputing baseline rewards...")
    random_r = baseline_rewards("random")
    rr_r = baseline_rewards("round_robin")
    cheap_r = baseline_rewards("cheapest")

    # Evaluate trained PPO
    env = gym.make("LLMRouter-v0", episode_length=200)
    ppo_rewards = []
    for ep in range(50):
        obs, _ = env.reset(seed=100 + ep)
        ep_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(action))
            ep_r += r
            done = term or trunc
        ppo_rewards.append(ep_r)
    env.close()
    ppo_r = np.mean(ppo_rewards)

    print("\n=== Results (mean episode reward over 50 episodes) ===")
    print(f"  Random:      {random_r:.2f}")
    print(f"  Round-robin: {rr_r:.2f}")
    print(f"  Cheapest:    {cheap_r:.2f}")
    print(f"  PPO Agent:   {ppo_r:.2f}")

    if ppo_r > random_r:
        improvement = (ppo_r - random_r) / abs(random_r) * 100
        print(f"\nPPO beats random by {improvement:.1f}%")

    try:
        import matplotlib.pyplot as plt
        x, y = ts2xy(load_results(log_dir), "timesteps")
        # Smooth with rolling mean
        window = max(1, len(y) // 20)
        y_smooth = np.convolve(y, np.ones(window) / window, mode="valid")
        x_smooth = x[window - 1:]

        plt.figure(figsize=(10, 5))
        plt.plot(x_smooth, y_smooth, label="PPO (smoothed)", color="blue")
        plt.axhline(random_r, linestyle="--", color="gray", label=f"Random ({random_r:.1f})")
        plt.axhline(rr_r, linestyle="--", color="orange", label=f"Round-Robin ({rr_r:.1f})")
        plt.axhline(cheap_r, linestyle="--", color="green", label=f"Cheapest ({cheap_r:.1f})")
        plt.xlabel("Timesteps")
        plt.ylabel("Episode Reward")
        plt.title("PPO vs Baselines — LLM Inference Routing")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curves.png", dpi=150)
        print("\nSaved training_curves.png")
    except ImportError:
        print("\n(matplotlib not installed — skipping plot)")


if __name__ == "__main__":
    main()
