from __future__ import annotations

import torch

EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28


def grpo_loss(mask: torch.Tensor, log_probs: torch.Tensor, old_log_probs: torch.Tensor, advantages: torch.Tensor, *, epsilon_low: float = EPSILON_LOW, epsilon_high: float = EPSILON_HIGH) -> torch.Tensor:
    """
    DAPO Style GRPO Loss, make sure mask is 1 for only reponse tokens
    """
    a = advantages.unsqueeze(-1)
    r = (log_probs - old_log_probs).exp()
    per_token = torch.min(r * a, torch.clamp(r, 1.0-epsilon_low, 1+epsilon_high) * a)
    loss = (per_token * mask).sum() / mask.sum().clamp_min(1)

    return -loss
