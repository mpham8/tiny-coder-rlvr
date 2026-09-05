import sys
import time
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_coder_rlvr.completion import Completion
from tiny_coder_rlvr.generation import VllmGenerator
from tiny_coder_rlvr.model import Policy

SOURCE_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "hf-vllm-handoff"
SAMPLE = {"query": "Write a Python function `answer()` that returns 42."}


def _assert_valid_rollouts(test: unittest.TestCase, rollouts: list[Completion], *, n: int) -> None:
    test.assertEqual(len(rollouts), n)
    for rollout in rollouts:
        test.assertIsInstance(rollout, Completion)
        test.assertIsInstance(rollout.text, str)
        test.assertGreater(len(rollout.text), 0)
        test.assertIsInstance(rollout.token_ids, list)
        test.assertGreater(len(rollout.token_ids), 0)
        test.assertIsNotNone(rollout.old_logprobs)
        test.assertEqual(len(rollout.old_logprobs), len(rollout.token_ids))
        test.assertTrue(all(lp <= 0.0 for lp in rollout.old_logprobs))
        test.assertIsNotNone(rollout.finish_reason)


@unittest.skipUnless(SOURCE_CHECKPOINT.is_dir(), "local policy checkpoint missing")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for vLLM generation")
class IpcWeightSyncTest(unittest.TestCase):
    def test_ipc_sleep_sync_generate_rollouts(self):
        """Colocated RL handoff: generate → sleep(2) → train → wake weights → IPC → wake KV."""
        t0 = time.perf_counter()
        policy = Policy(str(SOURCE_CHECKPOINT), dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-6)
        hf_load_s = time.perf_counter() - t0

        generator = VllmGenerator(
            model_name=str(SOURCE_CHECKPOINT),
            model_path=str(SOURCE_CHECKPOINT),
            dtype="bfloat16",
            max_model_len=2048,
            gpu_memory_utilization=0.85,
        )
        try:
            policy.to_cpu(optimizer)

            t0 = time.perf_counter()
            generator.load()
            vllm_load_s = time.perf_counter() - t0
            self.assertIsNotNone(generator.llm)
            engine_id = id(generator.llm)

            t0 = time.perf_counter()
            generator.init_weight_sync(policy.model)
            init_sync_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            rollouts_v1 = generator.generate(SAMPLE, n=2, enable_thinking=True)
            generate_v1_s = time.perf_counter() - t0
            _assert_valid_rollouts(self, rollouts_v1, n=2)

            t0 = time.perf_counter()
            generator.sleep(level=2)
            sleep_s = time.perf_counter() - t0
            free_after_sleep = torch.cuda.mem_get_info()[0]

            t0 = time.perf_counter()
            policy.to_gpu(optimizer)
            with torch.no_grad():
                first_param = next(policy.model.parameters())
                first_param.add_(0.01)
            loss = first_param.float().sum() * 0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            generator.wake_up(tags=["weights"])
            generator.sync_weights()
            policy.to_cpu(optimizer)
            generator.wake_up(tags=["kv_cache"])
            wake_sync_s = time.perf_counter() - t0

            self.assertIsNotNone(generator.llm)
            self.assertEqual(id(generator.llm), engine_id)
            self.assertFalse(generator._sleeping)

            t0 = time.perf_counter()
            rollouts_v2 = generator.generate(SAMPLE, n=2, enable_thinking=True)
            generate_v2_s = time.perf_counter() - t0
            _assert_valid_rollouts(self, rollouts_v2, n=2)

            print(
                "\n"
                f"HF load:       {hf_load_s:6.2f}s\n"
                f"vLLM load:     {vllm_load_s:6.2f}s\n"
                f"init IPC:      {init_sync_s:6.2f}s\n"
                f"generate v1:   {generate_v1_s:6.2f}s\n"
                f"sleep(2):      {sleep_s:6.2f}s\n"
                f"train step:    {train_s:6.2f}s\n"
                f"wake+IPC+KV:   {wake_sync_s:6.2f}s\n"
                f"generate v2:   {generate_v2_s:6.2f}s\n"
                f"free after sleep: {free_after_sleep / 1e9:.2f} GB\n"
                f"total:         {hf_load_s + vllm_load_s + init_sync_s + generate_v1_s + sleep_s + train_s + wake_sync_s + generate_v2_s:6.2f}s"
            )
        finally:
            generator.shutdown()
            policy.to_cpu(optimizer)
            del policy
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
