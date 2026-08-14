import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_coder_rlvr.sandbox.runner import Runner
from tiny_coder_rlvr.sandbox.sandbox import Candidate


def exited_successfully(status: int) -> bool:
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def grade(candidate: Candidate, timeout: float = 10.0) -> int:
    runner = Runner()
    runner.start()
    try:
        runner.submit(candidate)
        deadline = time.time() + timeout
        while time.time() < deadline:
            for _, status in runner.poll_results():
                return status
            time.sleep(0.05)
        raise TimeoutError(f"timed out waiting for {candidate.id}")
    finally:
        runner.stop()


class SandboxTest(unittest.TestCase):
    def test_add(self):
        candidate = Candidate(
            id="add",
            imports="",
            code="def add(a, b):\n    return a + b\n",
            tests="def check(entry_point):\n    assert entry_point(1, 2) == 3\n",
            entry_point="add",
        )

        status = grade(candidate)
        self.assertTrue(exited_successfully(status))

    def test_leetcode_sample(self):
        from datasets import load_dataset

        sample = load_dataset("newfacade/LeetCodeDataset", split="train")[0]
        candidate = Candidate(
            id="leetcode-0",
            imports=sample["prompt"],
            code=sample["completion"],
            tests=sample["test"],
            entry_point=sample["entry_point"],
        )

        status = grade(candidate, timeout=30)
        self.assertTrue(exited_successfully(status))


if __name__ == "__main__":
    unittest.main()
