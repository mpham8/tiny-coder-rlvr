from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

from data.prepare_data import LeetCodeSample
from tiny_coder_rlvr.sandbox.runner import Runner
from tiny_coder_rlvr.sandbox.sandbox import Candidate

THINK_END = "</think>"
PYTHON_FENCE = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)

FORMAT_FAIL_REWARD = -1.0
PASS_REWARD = 1.0
FAIL_REWARD = -1.0
L_MAX = 7168
L_CACHE = 4096

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOKENIZER_PATH = _REPO_ROOT / "checkpoints" / "hf-vllm-handoff"


class SandboxRunner(Protocol):
    def submit(self, candidate: Candidate) -> None: ...

    def poll_results(self) -> list[tuple[str, int]]: ...


@dataclass
class GradedRollout:
    reward: float
    response_tokens: int
    base_reward: float
    overlong_penalty: float


@dataclass
class PendingRollout:
    response_tokens: int
    reward: float | None = None


@lru_cache(maxsize=1)
def _default_tokenizer():
    from transformers import AutoTokenizer

    path = Path(os.environ.get("TINY_CODER_TOKENIZER", DEFAULT_TOKENIZER_PATH))
    if not path.exists():
        raise RuntimeError(
            f"tokenizer not found at {path}; pass completion_token_counts from vLLM or a tokenizer"
        )
    return AutoTokenizer.from_pretrained(path)


def response_token_count(completion: str, *, token_ids: list[int] | None = None, tokenizer: Any | None = None) -> int:
    if token_ids is not None:
        return len(token_ids)
    tok = tokenizer if tokenizer is not None else _default_tokenizer()
    return len(tok.encode(completion, add_special_tokens=False))


def extract_code(completion: str) -> str | None:
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
    code = extract_code(completion)
    if not code:
        return None
    if not code.endswith("\n"):
        code += "\n"
    return Candidate(id=rollout_id, imports=sample.prompt, code=code, tests=sample.test, entry_point=sample.entry_point)


def reward_from_status(status: int) -> float:
    if os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0:
        return PASS_REWARD
    return FAIL_REWARD


def overlong_penalty(response_tokens: int, *, l_max: int = L_MAX, l_cache: int = L_CACHE) -> float:
    if l_cache <= 0:
        raise ValueError("l_cache must be positive")
    safe_length = l_max - l_cache
    if response_tokens <= safe_length:
        return 0.0
    if response_tokens <= l_max:
        return (safe_length - response_tokens) / l_cache
    return -1.0


def final_reward(base_reward: float, response_tokens: int, *, l_max: int = L_MAX, l_cache: int = L_CACHE) -> GradedRollout:
    penalty = overlong_penalty(response_tokens, l_max=l_max, l_cache=l_cache)
    return GradedRollout(
        reward=base_reward + penalty,
        response_tokens=response_tokens,
        base_reward=base_reward,
        overlong_penalty=penalty,
    )


def compute_reward(completion: str, sample: LeetCodeSample, *, token_ids: list[int] | None = None, tokenizer: Any | None = None) -> GradedRollout:
    tokens = response_token_count(completion, token_ids=token_ids, tokenizer=tokenizer)
    if make_candidate(completion, sample, rollout_id=sample.task_id) is None:
        return final_reward(FORMAT_FAIL_REWARD, tokens)

    runner = Runner()
    runner.start()
    try:
        graded = grade_completions(runner, [completion], sample, completion_token_counts=[tokens])
        return graded[0]
    finally:
        runner.stop()


def grade_completions(runner: SandboxRunner, completions: list[str], sample: LeetCodeSample, *, completion_token_counts: list[int] | None = None, tokenizer: Any | None = None, timeout: float = 30.0) -> list[GradedRollout]:
    if completion_token_counts is not None and len(completion_token_counts) != len(completions):
        raise ValueError("completion_token_counts must match completions length")

    pending: dict[str, PendingRollout] = {}

    for i, completion in enumerate(completions):
        rollout_id = f"{sample.task_id}:{i}"
        if completion_token_counts is not None:
            response_tokens = completion_token_counts[i]
        else:
            response_tokens = response_token_count(completion, tokenizer=tokenizer)
        pending[rollout_id] = PendingRollout(response_tokens=response_tokens)

        candidate = make_candidate(completion, sample, rollout_id=rollout_id)
        if candidate is None:
            pending[rollout_id].reward = FORMAT_FAIL_REWARD
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

    graded: list[GradedRollout] = []
    for i in range(len(completions)):
        rollout_id = f"{sample.task_id}:{i}"
        entry = pending[rollout_id]
        base_reward = entry.reward if entry.reward is not None else FAIL_REWARD
        graded.append(final_reward(base_reward, entry.response_tokens))
    return graded
