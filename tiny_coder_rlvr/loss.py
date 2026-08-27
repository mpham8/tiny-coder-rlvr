from __future__ import annotations

import torch
from torch import Tensor

EPSILON_LOW = 0.2
EPSILON_HIGH = 0.28


def grpo_loss(log_probs: Tensor, old_log_probs: Tensor, ref_log_probs: Tensor, advantages: Tensor, *, epsilon_low: float = EPSILON_LOW, epsilon_high: float = EPSILON_HIGH) -> Tensor:
    """
    DAPO Style GRPO Loss
    """

    return
