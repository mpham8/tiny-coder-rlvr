import json
import os
import queue
import subprocess
import sys
import threading
import time

from tiny_coder_rlvr.sandbox.sandbox import Candidate, candidates_queue, running, sweep_once

DEFAULT_DOCKER_IMAGE = "tiny-coder-sandbox"


def _candidate_from_message(message: dict) -> Candidate:
    return Candidate(id=message["id"], imports=message.get("imports", ""), code=message["code"], tests=message["tests"], entry_point=message["entry_point"])


def _candidate_to_message(candidate: Candidate) -> dict:
    return {"type": "submit", "id": candidate.id, "imports": candidate.imports, "code": candidate.code, "tests": candidate.tests, "entry_point": candidate.entry_point}


class DockerRunner:
    """Production sandbox client: long-lived worker in an isolated Docker container."""

    def __init__(self, image: str = DEFAULT_DOCKER_IMAGE, docker_bin: str = "docker"):
        self._image = image
        self._docker_bin = docker_bin
        self._proc = None
        self._results: queue.Queue = queue.Queue()
        self._reader_thread = None

    def start(self):
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [self._docker_bin, "run", "-i", "--rm", "--network", "none", self._image],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader_thread.start()

    def _read_stdout(self):
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            message = json.loads(line)
            if message.get("type") == "result":
                self._results.put((message["id"], message["status"]))

    def submit(self, candidate: Candidate):
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("DockerRunner is not started")
        self._proc.stdin.write(json.dumps(_candidate_to_message(candidate)) + "\n")
        self._proc.stdin.flush()

    def poll_results(self):
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self, timeout: float = 30.0):
        if self._proc is None:
            return
        if self._proc.stdin is not None:
            self._proc.stdin.write('{"type":"shutdown"}\n')
            self._proc.stdin.flush()
            self._proc.stdin.close()
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None


def _create_sandbox_runner(image: str = DEFAULT_DOCKER_IMAGE, docker_bin: str = "docker") -> DockerRunner:
    runner = DockerRunner(image=image, docker_bin=docker_bin)
    runner.start()
    return runner


class DockerRunnerPool:
    """Spread candidates across N Docker containers using least in-flight load."""

    def __init__(self, runners: list[DockerRunner]):
        if not runners:
            raise ValueError("runners must not be empty")
        self._runners = runners
        self._in_flight = [0] * len(runners)

    def submit(self, candidate: Candidate):
        index = min(range(len(self._runners)), key=lambda i: self._in_flight[i])
        self._runners[index].submit(candidate)
        self._in_flight[index] += 1

    def poll_results(self):
        out = []
        for index, runner in enumerate(self._runners):
            for candidate_id, status in runner.poll_results():
                self._in_flight[index] -= 1
                out.append((candidate_id, status))
        return out

    def stop(self, timeout: float = 30.0):
        for runner in self._runners:
            runner.stop(timeout=timeout)


def create_sandbox_runner_pool(n: int, *, image: str = DEFAULT_DOCKER_IMAGE, docker_bin: str = "docker") -> DockerRunnerPool:
    if n < 1:
        raise ValueError("n must be at least 1")
    return DockerRunnerPool([_create_sandbox_runner(image=image, docker_bin=docker_bin) for _ in range(n)])


def _stdin_reader(job_queue: queue.Queue):
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if message.get("type") == "shutdown":
            job_queue.put(None)
            return
        if message.get("type") == "submit":
            job_queue.put(_candidate_from_message(message))
            continue
        raise ValueError(f"unknown message type: {message.get('type')!r}")
    job_queue.put(None)


def run_runner_stdio():
    """Container entrypoint: JSON lines on stdin, results on stdout."""
    os.environ["TINY_CODER_SANDBOX_WORKER"] = "1"
    job_queue = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(job_queue,), daemon=True).start()

    shutting_down = False
    while True:
        try:
            while True:
                candidate = job_queue.get_nowait()
                if candidate is None:
                    shutting_down = True
                    break
                candidates_queue.append(candidate)
        except queue.Empty:
            pass

        for candidate_id, status in sweep_once():
            print(json.dumps({"type": "result", "id": candidate_id, "status": status}), flush=True)

        if shutting_down and not candidates_queue and not running:
            return

        time.sleep(0.05)


def main():
    run_runner_stdio()


if __name__ == "__main__":
    main()
