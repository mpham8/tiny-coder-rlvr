from dataclasses import dataclass
import collections
import os
import signal
import subprocess
import sys
import tempfile
import time

from tiny_coder_rlvr.sandbox.process_limits import RECURSION_LIMIT, apply_limits


MAX_CONCURRENT = 16

candidates_queue = collections.deque()
running = {}


@dataclass
class Candidate:
    id: str
    imports: str
    code: str
    tests: str
    entry_point: str


@dataclass
class RunningJob:
    candidate: Candidate
    wallclock_limit: float
    proc: subprocess.Popen[bytes]


def returncode_to_wait_status(returncode: int) -> int:
    if returncode < 0:
        return -returncode
    if returncode == 0:
        return 0
    return returncode << 8


def _child_setup() -> None:
    apply_limits()


def _assert_worker_process() -> None:
    if os.environ.get("TINY_CODER_SANDBOX_WORKER") != "1":
        raise RuntimeError("sandbox.start() must run inside the sandbox worker process")


def start(candidate):
    _assert_worker_process()
    max_wallclock_time = 10

    full_source = (
        f"import sys; sys.setrecursionlimit({RECURSION_LIMIT})\n"
        f"{candidate.imports}\n"
        f"{candidate.code}\n"
        f"{candidate.tests}\n"
        f"check({candidate.entry_point})\n"
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(full_source)
    tmp.close()
    candidate_path = tmp.name

    safe_env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "LANG": "C.UTF-8",
    }

    proc = subprocess.Popen(
        [sys.executable, candidate_path],
        env=safe_env,
        start_new_session=True,
        preexec_fn=_child_setup,
        stdout=subprocess.DEVNULL,
        stderr=None,
    )

    running[proc.pid] = RunningJob(candidate=candidate, wallclock_limit=time.time() + max_wallclock_time, proc=proc)
    return proc.pid


def sweep_once():
    finished = []

    while len(running) < MAX_CONCURRENT and candidates_queue:
        start(candidates_queue.popleft())

    for pid in list(running.keys()):
        job = running[pid]
        returncode = job.proc.poll()

        if returncode is not None:
            candidate_id = job.candidate.id
            running.pop(pid, None)
            status = returncode_to_wait_status(returncode)
            finished.append((candidate_id, status))
            continue

        if time.time() > job.wallclock_limit:
            os.killpg(os.getpgid(job.proc.pid), signal.SIGKILL)
            job.proc.wait()
            job = running.pop(pid, None)
            status = returncode_to_wait_status(job.proc.returncode)
            finished.append((job.candidate.id, status))
            continue

    return finished

