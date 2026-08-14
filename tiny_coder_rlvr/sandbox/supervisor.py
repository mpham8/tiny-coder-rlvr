from dataclasses import dataclass
import os
import signal
import time
import collections
import tempfile


candidates_queue = collections.deque() #TODO move out of global scope
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
    max_wallclock_time = 9
    
    
    pid = os.fork()
    
    #in child process
    if pid == 0:
        #write to a temp file
        full_source = candidate.imports + candidate.code + candidate.tests + f"\ncheck({candidate.entry_point})\n"
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
        tmp.write(full_source)
        tmp.close()
        candidate_path = tmp.name

        #execute the python
        os.execve(
            "/usr/bin/python3",
            ["/usr/bin/python3", candidate_path],
            os.environ.copy(),  # TODO: change later for true isolation
        )
    #in parent process
    elif pid > 0:
        #keep track of running
        job = RunningJob(candidate=candidate, wallclock_limit=time.time() + max_wallclock_time)
        running[pid] = job
        return pid

    raise OSError("fork failed")


def sweep():
    """
    parent process that checks while jobs done, timeout and start new processes
    """

    while True:
        for pid in list(running.keys()):
            child_pid, status = os.waitpid(pid, os.WNOHANG)
            
            if child_pid != 0:
                #add pid to results
                results[running[pid].candidate.id] = status #first index of candidate is its id
                
                #remove pid from running
                running.pop(pid, None)

                #start next candidate if queue not empty
                if candidates_queue:
                    candidate = candidates_queue.popleft()
                    start(candidate)

                break

            if time.time() > running[pid].wallclock_limit:
                #kill process
                os.kill(pid, signal.SIGKILL)
                child_pid, status = os.waitpid(pid, 0)
                print(child_pid, " exited ", os.WIFEXITED(status))
                
                #remove pid from running
                running.pop(pid, None)

                #start next candidate if queue not empty
                if candidates_queue:
                    candidate = candidates_queue.popleft()
                    start(candidate)
        
        time.sleep(0.5)


    return