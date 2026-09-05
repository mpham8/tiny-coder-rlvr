from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from data.prepare_data import LeetCodeSample
from tiny_coder_rlvr import settings
from tiny_coder_rlvr.completion import Completion
from tiny_coder_rlvr.sandbox.sandbox import Candidate

THINK_END = "</think>"
PYTHON_FENCE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _format_fail_reward() -> float:
    return float(settings.format_fail_reward)


def _pass_reward() -> float:
    return float(settings.pass_reward)


def _fail_reward() -> float:
    return float(settings.fail_reward)


# Module constants (PASS_REWARD, L_MAX, ...) are resolved via __getattr__ from settings.


def __getattr__(name: str):
    mapping = {
        "FORMAT_FAIL_REWARD": "format_fail_reward",
        "PASS_REWARD": "pass_reward",
        "FAIL_REWARD": "fail_reward",
        "L_MAX": "l_max",
        "L_CACHE": "l_cache",
        "DEFAULT_TOKENIZER_PATH": "tokenizer_path",
    }
    if name in mapping:
        value = settings.get(mapping[name])
        if name == "DEFAULT_TOKENIZER_PATH":
            return settings.REPO_ROOT / str(value)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class SandboxRunner(Protocol):
    def submit(self, candidate: Candidate) -> None: ...

    def poll_results(self) -> list[tuple[str, int]]: ...


@dataclass
class PendingRollout:
    completion: Completion
    reward: float | None = None


@lru_cache(maxsize=1)
def _default_tokenizer():
    from transformers import AutoTokenizer

    default_path = settings.REPO_ROOT / str(settings.tokenizer_path)
    path = Path(os.environ.get("TINY_CODER_TOKENIZER", default_path))
    if not path.exists():
        raise RuntimeError(
            f"tokenizer not found at {path}; pass Completion.token_ids from vLLM or a tokenizer"
        )
    return AutoTokenizer.from_pretrained(path)


def response_token_count(completion: str, *, token_ids: list[int] | None = None, tokenizer: Any | None = None) -> int:
    """Token length of a completion string."""
    if token_ids is not None:
        return len(token_ids)
    tok = tokenizer if tokenizer is not None else _default_tokenizer()
    return len(tok.encode(completion, add_special_tokens=False))


def extract_code(completion: str) -> str | None:
    """Extract python code from a model response via fences / Solution class."""
    text = completion.strip()
    if THINK_END in text:
        text = text.split(THINK_END, 1)[1].strip()
    match = PYTHON_FENCE.search(text)
    if match:
        return match.group(1).strip()
    if "class Solution" in text:
        return text
    return None


def make_candidate(completion: str, sample: LeetCodeSample, *, rollout_id: str) -> Candidate | None:
    """Build a sandbox Candidate from completion text + dataset sample."""
    code = extract_code(completion)
    if not code:
        return None
    if not code.endswith("\n"):
        code += "\n"
    return Candidate(id=rollout_id, imports=sample.prompt, code=code, tests=sample.test, entry_point=sample.entry_point)


def reward_from_status(status: int) -> float:
    """Map sandbox wait status to pass/fail reward."""
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return _pass_reward()
    return _fail_reward()


def overlong_penalty(
    response_tokens: int,
    *,
    l_max: int | None = None,
    l_cache: int | None = None,
) -> float:
    """DAPO-style soft overlong penalty."""
    if l_max is None:
        l_max = int(settings.l_max)
    if l_cache is None:
        l_cache = int(settings.l_cache)
    if l_cache <= 0:
        raise ValueError("l_cache must be positive")
    safe_length = l_max - l_cache
    if response_tokens <= safe_length:
        return 0.0
    if response_tokens <= l_max:
        return (safe_length - response_tokens) / l_cache
    return -1.0


def compute_reward_batch(
    runner: SandboxRunner,
    completions: list[Completion],
    sample: LeetCodeSample,
    *,
    tokenizer: Any | None = None,
    timeout: float = 30.0,
) -> list[Completion]:
    """Grade completions in-place and return the same list with reward fields filled."""
    pending: dict[str, PendingRollout] = {}

    for i, completion in enumerate(completions):
        rollout_id = f"{sample.task_id}:{i}"
        if not completion.token_ids:
            tok = tokenizer if tokenizer is not None else _default_tokenizer()
            completion.token_ids = tok.encode(completion.text, add_special_tokens=False)

        pending[rollout_id] = PendingRollout(completion=completion)

        candidate = make_candidate(completion.text, sample, rollout_id=rollout_id)
        if candidate is None:
            pending[rollout_id].reward = _format_fail_reward()
            continue

        runner.submit(candidate)

    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(entry.reward is not None for entry in pending.values()):
            break

        for rollout_id, status in runner.poll_results():
            entry = pending.get(rollout_id)
            if entry is not None and entry.reward is None:
                entry.reward = reward_from_status(status)

        time.sleep(0.05)

    for i, completion in enumerate(completions):
        rollout_id = f"{sample.task_id}:{i}"
        entry = pending[rollout_id]
        base_reward = entry.reward if entry.reward is not None else _fail_reward()
        penalty = overlong_penalty(completion.response_tokens)
        completion.base_reward = base_reward
        completion.overlong_penalty = penalty
        completion.reward = base_reward + penalty

    return completions
