import os
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from tiny_coder_rlvr.sandbox.runner import Runner
from tiny_coder_rlvr.sandbox.sandbox import Candidate


def exited_successfully(status: int) -> bool:
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def wait_for_result(runner: Runner, candidate_id: str, timeout: float = 30.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for result_id, status in runner.poll_results():
            if result_id == candidate_id:
                return status
        time.sleep(0.05)
    raise TimeoutError(f"no result for {candidate_id!r}")


class SandboxTest(unittest.TestCase):
    def test_add(self):
        candidate = Candidate(
            id="add",
            imports="",
            code=(
                "def add(a, b):\n"
                "    return a + b\n"
            ),
            tests=(
                "def check(entry_point):\n"
                "    assert entry_point(1, 2) == 3\n"
            ),
            entry_point="add",
        )

        runner = Runner()
        runner.start()
        try:
            runner.submit(candidate)
            status = wait_for_result(runner, candidate.id)
        finally:
            runner.stop()

        self.assertTrue(exited_successfully(status))

    def test_leetcode_sample(self):
        sample = load_dataset("newfacade/LeetCodeDataset", split="train")[0]
        candidate = Candidate(
            id="leetcode-0",
            imports=sample["prompt"],
            code=sample["completion"],
            tests=sample["test"],
            entry_point=sample["entry_point"],
        )

        runner = Runner()
        runner.start()
        try:
            runner.submit(candidate)
            status = wait_for_result(runner, candidate.id, timeout=30.0)
        finally:
            runner.stop()

        self.assertTrue(exited_successfully(status))

    def test_mem_limit(self):
        candidate = Candidate(
            id="memory-hog",
            imports="",
            code=(
                "def eat_memory():\n"
                "    bytearray(1024 ** 3)\n"
            ),
            tests=(
                "def check(entry_point):\n"
                "    entry_point()\n"
            ),
            entry_point="eat_memory",
        )

        runner = Runner()
        runner.start()
        try:
            runner.submit(candidate)
            status = wait_for_result(runner, candidate.id, timeout=10.0)
        finally:
            runner.stop()

        self.assertFalse(exited_successfully(status))


if __name__ == "__main__":
    unittest.main()
