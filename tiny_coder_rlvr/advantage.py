from __future__ import annotations

import torch

DEFAULT_EPS = 1e-8


def grpo_advantages(rewards: list[float], *, eps: float = DEFAULT_EPS) -> list[float]:
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean = rewards_t.mean()
    std = rewards_t.std()
    return ((rewards_t - mean) / (std + eps)).tolist()


def grpo_advantages_batch(reward_groups: list[list[float]], *, eps: float = DEFAULT_EPS) -> list[list[float]]:
    return [grpo_advantages(rewards, eps=eps) for rewards in reward_groups]
