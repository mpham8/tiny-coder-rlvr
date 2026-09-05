import os
import shutil
import subprocess
import time

from typing import Protocol

from tiny_coder_rlvr import settings


class _PollableRunner(Protocol):
    def poll_results(self) -> list[tuple[str, int]]: ...


def docker_image_exists(name: str | None = None) -> bool:
    if name is None:
        name = str(settings.docker_image)
    if shutil.which("docker") is None:
        return False
    result = subprocess.run(["docker", "image", "inspect", name], capture_output=True, check=False)
    return result.returncode == 0


def exited_successfully(status: int) -> bool:
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


def wait_for_result(runner: _PollableRunner, candidate_id: str, timeout: float = 30.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        for result_id, status in runner.poll_results():
            if result_id == candidate_id:
                return status
        time.sleep(0.05)
    raise TimeoutError(f"no result for {candidate_id!r}")
