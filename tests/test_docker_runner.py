import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_coder_rlvr.sandbox.runner import DEFAULT_DOCKER_IMAGE, DockerRunner
from tiny_coder_rlvr.sandbox.sandbox import Candidate


def docker_image_exists(name: str) -> bool:
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(
        ["docker", "image", "inspect", name],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def exited_successfully(status: int) -> bool:
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def wait_for_result(runner: DockerRunner, candidate_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        for result_id, status in runner.poll_results():
            if result_id == candidate_id:
                return status
        time.sleep(0.05)
    raise TimeoutError(f"no result for {candidate_id!r}")


@unittest.skipUnless(docker_image_exists(DEFAULT_DOCKER_IMAGE), "sandbox image not built")
class DockerRunnerTest(unittest.TestCase):
    def test_add(self):
        runner = DockerRunner()
        runner.start()
        try:
            runner.submit(
                Candidate(
                    id="add",
                    imports="",
                    code="def add(a, b):\n    return a + b\n",
                    tests="def check(entry_point):\n    assert entry_point(1, 2) == 3\n",
                    entry_point="add",
                )
            )
            status = wait_for_result(runner, "add")
            self.assertTrue(exited_successfully(status))
        finally:
            runner.stop()


if __name__ == "__main__":
    unittest.main()
