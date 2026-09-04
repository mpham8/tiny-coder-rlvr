import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare_data import LeetCodeRLVRDataset, LeetCodeSample, load_leetcode_dataset
from tests.sandbox_helpers import docker_image_exists
from tiny_coder_rlvr.completion import Completion
from tiny_coder_rlvr.reward import (
    FAIL_REWARD,
    FORMAT_FAIL_REWARD,
    L_CACHE,
    L_MAX,
    PASS_REWARD,
    compute_reward_batch,
    extract_code,
    make_candidate,
    overlong_penalty,
    response_token_count,
    reward_from_status,
)
from tiny_coder_rlvr.sandbox.runner import create_sandbox_runner_pool


def _completion(text: str, token_ids: list[int]) -> Completion:
    return Completion(text=text, token_ids=token_ids)


def compute_reward_single(completion: str, sample: LeetCodeSample, *, token_ids: list[int] | None = None) -> Completion:
    rollout = _completion(completion, list(token_ids) if token_ids is not None else [])
    if make_candidate(completion, sample, rollout_id=sample.task_id) is None:
        penalty = overlong_penalty(rollout.response_tokens)
        rollout.base_reward = FORMAT_FAIL_REWARD
        rollout.overlong_penalty = penalty
        rollout.reward = FORMAT_FAIL_REWARD + penalty
        return rollout

    runner = create_sandbox_runner_pool(1)
    try:
        return compute_reward_batch(runner, [rollout], sample)[0]
    finally:
        runner.stop()


class RewardParsingTest(unittest.TestCase):
    def test_extract_code_from_python_fence(self):
        completion = "Here is the answer:\n```python\nclass Solution:\n    pass\n```"
        self.assertEqual(extract_code(completion), "class Solution:\n    pass")

    def test_extract_code_after_thinking_block(self):
        completion = (
            "plan the solution\n"
            "</think>\n"
            "```python\n"
            "class Solution:\n"
            "    def solve(self):\n"
            "        return 1\n"
            "```"
        )
        self.assertIn("class Solution", extract_code(completion) or "")

    def test_extract_code_returns_none_for_garbage(self):
        self.assertIsNone(extract_code("still thinking with no code"))


class RewardStatusTest(unittest.TestCase):
    def test_reward_from_success_status(self):
        self.assertEqual(reward_from_status(0), PASS_REWARD)

    def test_reward_from_failure_status(self):
        self.assertEqual(reward_from_status(256), FAIL_REWARD)

    def test_response_token_count_uses_tokenizer(self):
        self.assertEqual(response_token_count("abc", token_ids=[1, 2, 3]), 3)


class OverlongPenaltyTest(unittest.TestCase):
    def test_no_penalty_in_safe_zone(self):
        self.assertEqual(overlong_penalty(L_MAX - L_CACHE), 0.0)
        self.assertEqual(overlong_penalty(100), 0.0)

    def test_linear_penalty_in_soft_zone(self):
        safe_length = L_MAX - L_CACHE
        mid = safe_length + L_CACHE // 2
        self.assertEqual(overlong_penalty(mid), -0.5)

    def test_max_penalty_at_l_max(self):
        self.assertEqual(overlong_penalty(L_MAX), -1.0)

    def test_max_penalty_beyond_l_max(self):
        self.assertEqual(overlong_penalty(L_MAX + 1), -1.0)


@unittest.skipUnless(docker_image_exists(), "sandbox image not built")
class RewardSyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = load_leetcode_dataset(split="train", decontaminate=False)
        cls.sample = LeetCodeRLVRDataset(dataset)[0]

    def test_compute_reward_passes_gold_completion(self):
        graded = compute_reward_single(self.sample.completion, self.sample, token_ids=[1, 2, 3])
        self.assertEqual(graded.base_reward, PASS_REWARD)
        self.assertEqual(graded.overlong_penalty, 0.0)
        self.assertEqual(graded.reward, PASS_REWARD)
        self.assertEqual(graded.response_tokens, 3)

    def test_compute_reward_rejects_unparseable_completion(self):
        graded = compute_reward_single("no code here", self.sample, token_ids=[9, 8])
        self.assertEqual(graded.base_reward, FORMAT_FAIL_REWARD)
        self.assertEqual(graded.overlong_penalty, 0.0)
        self.assertEqual(graded.reward, FORMAT_FAIL_REWARD)
        self.assertEqual(graded.response_tokens, 2)

    def test_compute_reward_applies_overlong_penalty(self):
        graded = compute_reward_single("no code here", self.sample, token_ids=list(range(L_MAX)))
        self.assertEqual(graded.base_reward, FORMAT_FAIL_REWARD)
        self.assertEqual(graded.overlong_penalty, -1.0)
        self.assertEqual(graded.reward, FORMAT_FAIL_REWARD - 1.0)


@unittest.skipUnless(docker_image_exists(), "sandbox image not built")
class RewardAsyncTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = load_leetcode_dataset(split="train", decontaminate=False)
        cls.sample = LeetCodeRLVRDataset(dataset)[0]

    def test_grade_completions_batch(self):
        bad_code = (
            "```python\n"
            "class Solution:\n"
            "    def twoSum(self, nums, target):\n"
            "        return []\n"
            "```"
        )
        completions = [
            _completion(self.sample.completion, list(range(100))),
            _completion("not valid output", list(range(5))),
            _completion(bad_code, list(range(42))),
        ]

        runner = create_sandbox_runner_pool(1)
        try:
            graded = compute_reward_batch(runner, completions, self.sample)
        finally:
            runner.stop()

        self.assertEqual(len(graded), 3)
        self.assertEqual(graded[0].base_reward, PASS_REWARD)
        self.assertEqual(graded[0].overlong_penalty, 0.0)
        self.assertEqual(graded[0].reward, PASS_REWARD)
        self.assertEqual(graded[0].response_tokens, 100)
        self.assertEqual(graded[1].base_reward, FORMAT_FAIL_REWARD)
        self.assertEqual(graded[1].overlong_penalty, 0.0)
        self.assertEqual(graded[1].reward, FORMAT_FAIL_REWARD)
        self.assertEqual(graded[1].response_tokens, 5)
        self.assertEqual(graded[2].base_reward, FAIL_REWARD)
        self.assertEqual(graded[2].overlong_penalty, 0.0)
        self.assertEqual(graded[2].reward, FAIL_REWARD)
        self.assertEqual(graded[2].response_tokens, 42)

    def test_make_candidate_matches_sandbox_shape(self):
        candidate = make_candidate(f"```python\n{self.sample.completion}\n```", self.sample, rollout_id="test-0")
        self.assertIsNotNone(candidate)
        assert candidate is not None
        self.assertEqual(candidate.imports, self.sample.prompt)
        self.assertEqual(candidate.tests, self.sample.test)
        self.assertEqual(candidate.entry_point, self.sample.entry_point)


if __name__ == "__main__":
    unittest.main()
