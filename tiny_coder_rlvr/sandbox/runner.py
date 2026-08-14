import multiprocessing
import queue
import time

from tiny_coder_rlvr.sandbox.sandbox import Candidate, candidates_queue, sweep_once


def run_runner(job_queue, result_queue):
    """Lean runner process: no vLLM/datasets, drives sandbox.sweep_once()."""
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


class Runner:
    def __init__(self):
        self._ctx = multiprocessing.get_context("spawn")
        self._jobs = self._ctx.Queue()
        self._results = self._ctx.Queue()
        self._proc = None

    def start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        self._proc = self._ctx.Process(
            target=run_runner,
            args=(self._jobs, self._results),
            daemon=True,
        )
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
