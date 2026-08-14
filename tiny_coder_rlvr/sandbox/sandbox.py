from dataclasses import dataclass
import os
import signal
import subprocess
import sys
import time
import collections
import tempfile

MAX_CONCURRENT = 16

candidates_queue = collections.deque()
running = {}
results = {}


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


def start(candidate):
    max_wallclock_time = 10

    full_source = (
        candidate.imports
        + candidate.code
        + candidate.tests
        + f"\ncheck({candidate.entry_point})\n"
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(full_source)
    tmp.close()
    candidate_path = tmp.name

    proc = subprocess.Popen(
        [sys.executable, candidate_path],
        env=os.environ.copy(),  # TODO: change later for true isolation
    )
    running[proc.pid] = RunningJob(
        candidate=candidate,
        wallclock_limit=time.time() + max_wallclock_time,
    )
    return proc.pid


def sweep_once():
    finished = []

    while len(running) < MAX_CONCURRENT and candidates_queue:
        start(candidates_queue.popleft())

    for pid in list(running.keys()):
        child_pid, status = os.waitpid(pid, os.WNOHANG)

        if child_pid == pid:
            candidate_id = running[pid].candidate.id
            running.pop(pid, None)
            results[candidate_id] = status
            finished.append((candidate_id, status))
            break

        if time.time() > running[pid].wallclock_limit:
            os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
            job = running.pop(pid, None)
            results[job.candidate.id] = status
            finished.append((job.candidate.id, status))
            break

    return finished


def sweep():
    while True:
        sweep_once()
        time.sleep(0.5)
