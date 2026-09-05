from __future__ import annotations

import torch

from tiny_coder_rlvr import settings


def grpo_advantages(rewards: list[float], *, eps: float | None = None) -> list[float]:
    if eps is None:
        eps = float(settings.advantage_eps)
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean = rewards_t.mean()
    std = rewards_t.std()
    return ((rewards_t - mean) / (std + eps)).tolist()


def grpo_advantages_batch(reward_groups: list[list[float]], *, eps: float | None = None) -> list[list[float]]:
    if eps is None:
        eps = float(settings.advantage_eps)
    return [grpo_advantages(rewards, eps=eps) for rewards in reward_groups]


def __getattr__(name: str):
    if name == "DEFAULT_EPS":
        return float(settings.advantage_eps)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
