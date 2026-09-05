from __future__ import annotations

import torch

from tiny_coder_rlvr import settings


def grpo_loss(
    mask: torch.Tensor,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    epsilon_low: float | None = None,
    epsilon_high: float | None = None,
    normalize_tokens: float | torch.Tensor | None = None,
) -> torch.Tensor:
    """
    DAPO Style GRPO Loss, make sure mask is 1 for only reponse tokens.

    If normalize_tokens is set, divide by that (for microbatch grad accumulation
    that matches a global token-mean over the full group).
    """
    if epsilon_low is None:
        epsilon_low = float(settings.epsilon_low)
    if epsilon_high is None:
        epsilon_high = float(settings.epsilon_high)

    a = advantages.unsqueeze(-1)
    r = (log_probs - old_log_probs).exp()
    per_token = torch.min(r * a, torch.clamp(r, 1.0 - epsilon_low, 1 + epsilon_high) * a)
    denom = mask.sum().clamp_min(1) if normalize_tokens is None else normalize_tokens
    loss = (per_token * mask).sum() / denom

    return -loss


def __getattr__(name: str):
    if name == "EPSILON_LOW":
        return float(settings.epsilon_low)
    if name == "EPSILON_HIGH":
        return float(settings.epsilon_high)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
