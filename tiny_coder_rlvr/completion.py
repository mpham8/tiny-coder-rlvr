from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Completion:
    """Unified rollout object shared by generation → reward → loss."""

    text: str
    token_ids: list[int]
    old_logprobs: list[float] | None = None 
    thinking: str | None = None
    solution: str | None = None
    finish_reason: str | None = None
    reward: float | None = None
    base_reward: float | None = None
    overlong_penalty: float | None = None

    @property
    def response_tokens(self) -> int:
        return len(self.token_ids)
