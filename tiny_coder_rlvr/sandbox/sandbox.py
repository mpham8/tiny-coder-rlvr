from dataclasses import dataclass
import os
import signal
import sys
import time
import collections
import tempfile

from tiny_coder_rlvr.sandbox.process_limits import RECURSION_LIMIT, apply_limits


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

    #write source to a temp file
    full_source = (
        f"import sys; sys.setrecursionlimit({RECURSION_LIMIT})\n"
        + candidate.imports
        + candidate.code
        + candidate.tests
        + f"\ncheck({candidate.entry_point})\n"
    )
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(full_source)
    tmp.close()
    candidate_path = tmp.name

    pid = os.fork()

    #in child process
    if pid == 0:
        os.setsid() #make candidate its own session leader
        
        apply_limits()
        
        safe_env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1", 
            "LANG": "C.UTF-8",
        }

        #execute the python
        os.execve(
            sys.executable,
            [sys.executable, candidate_path],
            safe_env,  
        )
        os._exit(127)

    #in parent process
    if pid > 0:
        running[pid] = RunningJob(candidate=candidate, wallclock_limit=time.time() + max_wallclock_time)
        return pid

    raise OSError("fork failed")


def sweep_once():
    finished = []

    while len(running) < MAX_CONCURRENT and candidates_queue:
        start(candidates_queue.popleft())

    for pid in list(running.keys()):
        child_pid, status = os.waitpid(pid, os.WNOHANG)

        if child_pid == pid:
            #add pid to results
            candidate_id = running[pid].candidate.id
            
            #remove pid from running
            running.pop(pid, None)
            results[candidate_id] = status
            finished.append((candidate_id, status))
            
            continue

        if time.time() > running[pid].wallclock_limit:
            #kill process
            os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
            
            #remove pid from running
            job = running.pop(pid, None)
            results[job.candidate.id] = status
            finished.append((job.candidate.id, status))
            
            continue

    return finished


def sweep():
    while True:
        sweep_once()
        time.sleep(0.5)
