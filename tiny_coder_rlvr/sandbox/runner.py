import json
import multiprocessing
import os
import queue
import subprocess
import sys
import threading
import time

from tiny_coder_rlvr.sandbox.sandbox import Candidate, candidates_queue, running, sweep_once

DEFAULT_DOCKER_IMAGE = "tiny-coder-sandbox"


def run_runner(job_queue, result_queue):
    """Lean runner process: no vLLM/datasets, drives sandbox.sweep_once()."""
    os.environ["TINY_CODER_SANDBOX_WORKER"] = "1"
    while True:
        try:
            while True:
                candidate = job_queue.get_nowait()
                if candidate is None:
                    return
                candidates_queue.append(candidate)
        except queue.Empty:
            pass

        for candidate_id, status in sweep_once():
            result_queue.put((candidate_id, status))

        time.sleep(0.05)


def _candidate_from_message(message: dict) -> Candidate:
    return Candidate(id=message["id"], imports=message.get("imports", ""), code=message["code"], tests=message["tests"], entry_point=message["entry_point"])


def _candidate_to_message(candidate: Candidate) -> dict:
    return {
        "type": "submit",
        "id": candidate.id,
        "imports": candidate.imports,
        "code": candidate.code,
        "tests": candidate.tests,
        "entry_point": candidate.entry_point,
    }


class Runner:
    def __init__(self):
        self._ctx = multiprocessing.get_context("spawn")
        self._jobs = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._proc = None

    def start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(target=run_runner, args=(self._jobs, self._results), daemon=True)
        self._proc.start()

    def submit(self, candidate: Candidate):
        self._jobs.put(candidate)

    def poll_results(self):
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                break
        return out

    def stop(self, timeout=10.0):
        if self._proc is None:
            return
        self._jobs.put(None)
        self._proc.join(timeout=timeout)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()
        self._proc = None


class DockerRunner:
    """Host client for model 2: long-lived sandbox worker in Docker."""

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
            [
                self._docker_bin,
                "run",
                "-i",
                "--rm",
                "--network",
                "none",
                self._image,
            ],
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
    """Long-lived worker for Docker: JSON lines on stdin, results on stdout."""
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
