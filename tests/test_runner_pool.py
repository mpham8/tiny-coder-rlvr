import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.sandbox_helpers import docker_image_exists, exited_successfully, wait_for_result
from tiny_coder_rlvr.sandbox.runner import DockerRunnerPool, create_sandbox_runner_pool
from tiny_coder_rlvr.sandbox.sandbox import Candidate


class FakeRunner:
    def __init__(self):
        self.submitted: list[Candidate] = []

    def submit(self, candidate: Candidate):
        self.submitted.append(candidate)

    def poll_results(self):
        return []

    def stop(self, timeout: float = 30.0):
        return None


class DockerRunnerPoolTest(unittest.TestCase):
    def test_submit_picks_least_loaded_runner(self):
        runners = [FakeRunner(), FakeRunner(), FakeRunner()]
        pool = DockerRunnerPool(runners)

        for i in range(7):
            pool.submit(Candidate(id=f"c{i}", imports="", code="", tests="", entry_point="f"))

        self.assertEqual(len(runners[0].submitted), 3)
        self.assertEqual(len(runners[1].submitted), 2)
        self.assertEqual(len(runners[2].submitted), 2)

    def test_poll_results_decrements_in_flight(self):
        runners = [FakeRunner(), FakeRunner()]
        pool = DockerRunnerPool(runners)
        pool.submit(Candidate(id="a", imports="", code="", tests="", entry_point="f"))
        pool.submit(Candidate(id="b", imports="", code="", tests="", entry_point="f"))
        self.assertEqual(pool._in_flight, [1, 1])

        runners[0].poll_results = lambda: [("a", 0)]
        results = pool.poll_results()
        self.assertEqual(results, [("a", 0)])
        self.assertEqual(pool._in_flight, [0, 1])


@unittest.skipUnless(docker_image_exists(), "sandbox image not built")
class DockerRunnerPoolIntegrationTest(unittest.TestCase):
    def test_pool_runs_candidates(self):
        pool = create_sandbox_runner_pool(2)
        try:
            for i in range(4):
                pool.submit(
                    Candidate(
                        id=f"add-{i}",
                        imports="",
                        code="def add(a, b):\n    return a + b\n",
                        tests="def check(entry_point):\n    assert entry_point(1, 2) == 3\n",
                        entry_point="add",
                    )
                )

            seen = {}
            deadline = time.time() + 30.0
            while len(seen) < 4 and time.time() < deadline:
                for candidate_id, status in pool.poll_results():
                    seen[candidate_id] = status
                time.sleep(0.05)
        finally:
            pool.stop()

        self.assertEqual(len(seen), 4)
        self.assertTrue(all(exited_successfully(status) for status in seen.values()))


if __name__ == "__main__":
    unittest.main()
