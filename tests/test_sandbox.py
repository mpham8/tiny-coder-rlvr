import os
import signal
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import load_dataset

from tiny_coder_rlvr.sandbox.supervisor import Candidate, running, start


#wait until child finished
def wait_child(pid: int, timeout: float = 10.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        child_pid, status = os.waitpid(pid, os.WNOHANG)
        if child_pid == pid:
            return status
        time.sleep(0.05)

    os.kill(pid, signal.SIGKILL)
    _, status = os.waitpid(pid, 0)
    return status


def exited_successfully(status: int) -> bool:
    #check child exit normal, check exit code sucess
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


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

        status = wait_child(start(candidate))
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

        status = wait_child(start(candidate), timeout=30)
        self.assertTrue(exited_successfully(status))


if __name__ == "__main__":
    unittest.main()
