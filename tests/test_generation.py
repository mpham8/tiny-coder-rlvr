import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_coder_rlvr.completion import Completion
from tiny_coder_rlvr.generation import VllmGenerator
from tiny_coder_rlvr.model import Policy

SOURCE_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "hf-vllm-handoff"


@unittest.skipUnless(SOURCE_CHECKPOINT.is_dir(), "local policy checkpoint missing")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for vLLM generation")
class HfToVllmHandoffTest(unittest.TestCase):
    def test_save_load_generate_rollouts(self):
        # HF weights v1
        t0 = time.perf_counter()
        policy = Policy(str(SOURCE_CHECKPOINT), dtype=torch.bfloat16)
        hf_load_s = time.perf_counter() - t0

        with tempfile.TemporaryDirectory(prefix="hf-vllm-handoff-") as tmp:
            checkpoint = Path(tmp)

            # ↓ save → disk checkpoint v1
            t0 = time.perf_counter()
            policy.save(checkpoint)
            save_s = time.perf_counter() - t0
            self.assertTrue((checkpoint / "config.json").is_file())
            self.assertTrue(any(checkpoint.glob("*.safetensors")) or (checkpoint / "pytorch_model.bin").is_file())

            del policy
            torch.cuda.empty_cache()

            # ↓ load() → vLLM weights v1
            generator = VllmGenerator(
                model_name=str(checkpoint),
                checkpoint_path=str(checkpoint),
                dtype="bfloat16",
                max_model_len=2048,
                gpu_memory_utilization=0.85,
            )
            try:
                t0 = time.perf_counter()
                generator.load()
                vllm_load_s = time.perf_counter() - t0
                self.assertIsNotNone(generator.llm)

                # ↓ generate rollouts
                sample = {"query": "Write a Python function `answer()` that returns 42."}
                t0 = time.perf_counter()
                rollouts = generator.generate(sample, n=2, enable_thinking=True)
                generate_s = time.perf_counter() - t0

                self.assertEqual(len(rollouts), 2)
                for rollout in rollouts:
                    self.assertIsInstance(rollout, Completion)
                    self.assertIsInstance(rollout.text, str)
                    self.assertGreater(len(rollout.text), 0)
                    self.assertIsInstance(rollout.token_ids, list)
                    self.assertGreater(len(rollout.token_ids), 0)
                    self.assertIsNotNone(rollout.old_logprobs)
                    self.assertEqual(len(rollout.old_logprobs), len(rollout.token_ids))
                    self.assertTrue(all(lp <= 0.0 for lp in rollout.old_logprobs))
                    self.assertIsNotNone(rollout.finish_reason)

                print(
                    "\n"
                    f"HF load:      {hf_load_s:6.2f}s\n"
                    f"save:         {save_s:6.2f}s\n"
                    f"vLLM load:    {vllm_load_s:6.2f}s\n"
                    f"generate:     {generate_s:6.2f}s\n"
                    f"total:        {hf_load_s + save_s + vllm_load_s + generate_s:6.2f}s"
                )
            finally:
                generator.shutdown()


if __name__ == "__main__":
    unittest.main()
