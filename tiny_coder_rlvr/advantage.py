from __future__ import annotations

DEFAULT_EPS = 1e-8


def grpo_advantages(rewards: list[float], *, eps: float = DEFAULT_EPS) -> list[float]:
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    std = variance ** 0.5
    
    return [(reward - mean) / (std + eps) for reward in rewards]


def grpo_advantages_batch(reward_groups: list[list[float]], *, eps: float = DEFAULT_EPS) -> list[list[float]]:
    return [grpo_advantages(rewards, eps=eps) for rewards in reward_groups]
