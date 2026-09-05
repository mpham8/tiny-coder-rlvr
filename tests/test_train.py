import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare_data import LeetCodeSample, LeetCodeRLVRDataset, collate_leetcode_samples, load_leetcode_dataset
from tests.sandbox_helpers import docker_image_exists
from tiny_coder_rlvr.completion import Completion
from tiny_coder_rlvr.generation import VllmGenerator
from tiny_coder_rlvr.model import Policy
from tiny_coder_rlvr.sandbox.runner import create_sandbox_runner_pool
from tiny_coder_rlvr.train import TRAIN_STATE_NAME, Trainer

SOURCE_CHECKPOINT = Path(__file__).resolve().parents[1] / "checkpoints" / "hf-vllm-handoff"
NUM_RUNNERS = 2
GROUP_SIZE = 2


def _sample(task_id: str = "two-sum") -> LeetCodeSample:
    return LeetCodeSample(
        task_id=task_id,
        question_id=1,
        difficulty="Easy",
        problem_description="add two numbers",
        starter_code="class Solution:\n    pass\n",
        query="Write a function that adds two numbers.",
        response="",
        prompt="",
        completion="class Solution:\n    pass\n",
        entry_point="Solution",
        test="def check():\n    assert True\n",
    )


class FakeTokenizer:
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


class FakeParam:
    device = torch.device("cpu")


class FakeModel:
    def parameters(self):
        yield FakeParam()


class FakePolicy:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.model = FakeModel()
        self.saved_paths: list[str] = []

    def token_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return torch.full(
            (input_ids.shape[0], input_ids.shape[1] - 1),
            -0.5,
            dtype=torch.float32,
            requires_grad=True,
        )

    def to_cpu(self, optimizer=None):
        return None

    def to_gpu(self, optimizer=None):
        return None

    def save(self, path):
        self.saved_paths.append(str(path))


class TrainerHelpersTest(unittest.TestCase):
    def test_build_training_tensors_shapes_and_mask(self):
        policy = FakePolicy()
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                policy=policy,
                generator=MagicMock(),
                optimizer=MagicMock(),
                dataloader=[],
                group_size=2,
                num_epochs=1,
                intermediate_path=tmp,
                checkpoint_path=tmp,
                runner=MagicMock(),
                resume=False,
            )

            completions = [
                Completion(text="hi", token_ids=[10, 11], old_logprobs=[-0.1, -0.2], reward=1.0),
                Completion(text="hey", token_ids=[10], old_logprobs=[-0.3], reward=-1.0),
            ]
            prompt_token_ids = [[1, 2, 3], [1, 2]]
            advantages = [0.5, -0.5]

            mask, log_probs, old_log_probs, advantages_t = trainer.build_training_tensors(
                completions,
                advantages,
                prompt_token_ids,
            )

        self.assertEqual(mask.shape, (2, 2))
        self.assertEqual(log_probs.shape, (2, 2))
        self.assertEqual(old_log_probs.shape, (2, 2))
        self.assertEqual(advantages_t.shape, (2,))

        self.assertTrue(torch.equal(mask[0], torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.equal(mask[1], torch.tensor([1.0, 0.0])))
        self.assertAlmostEqual(old_log_probs[0, 0].item(), -0.1, places=5)
        self.assertAlmostEqual(old_log_probs[1, 0].item(), -0.3, places=5)
        self.assertEqual(old_log_probs[1, 1].item(), 0.0)
        self.assertTrue(log_probs.requires_grad)

    def test_build_training_tensors_requires_old_logprobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = Trainer(
                policy=FakePolicy(),
                generator=MagicMock(),
                optimizer=MagicMock(),
                dataloader=[],
                group_size=1,
                num_epochs=1,
                intermediate_path=tmp,
                checkpoint_path=tmp,
                runner=MagicMock(),
                resume=False,
            )
            completions = [Completion(text="x", token_ids=[1], old_logprobs=None)]

            with self.assertRaises(ValueError):
                trainer.build_training_tensors(completions, [0.0], [[1]])

    def test_save_and_resume_train_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = FakePolicy()
            param = torch.nn.Parameter(torch.tensor([1.0]))
            optimizer = torch.optim.SGD([param], lr=0.1)
            optimizer.step()

            trainer = Trainer(
                policy=policy,
                generator=MagicMock(),
                optimizer=optimizer,
                dataloader=[],
                group_size=1,
                num_epochs=3,
                intermediate_path=tmp,
                checkpoint_path=tmp,
                runner=MagicMock(),
                resume=False,
            )
            trainer.save_train_checkpoint(epoch=1, batch_idx=4, global_step=12)

            self.assertTrue((Path(tmp) / TRAIN_STATE_NAME).is_file())

            param2 = torch.nn.Parameter(torch.tensor([0.0]))
            optimizer2 = torch.optim.SGD([param2], lr=0.1)
            resumed = Trainer(
                policy=FakePolicy(),
                generator=MagicMock(),
                optimizer=optimizer2,
                dataloader=[],
                group_size=1,
                num_epochs=3,
                intermediate_path=tmp,
                checkpoint_path=tmp,
                runner=MagicMock(),
                resume=True,
            )

            self.assertEqual(resumed.start_epoch, 1)
            self.assertEqual(resumed.start_batch_idx, 4)
            self.assertEqual(resumed.global_step, 12)
            self.assertIn(str(Path(tmp)), policy.saved_paths)


@unittest.skipUnless(SOURCE_CHECKPOINT.is_dir(), "local Qwen checkpoint missing")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA required for vLLM + policy train step")
@unittest.skipUnless(docker_image_exists(), "sandbox image not built")
class TrainerRealIntegrationTest(unittest.TestCase):
    def test_one_train_step_real_qwen_and_runner_pool(self):
        dataset = LeetCodeRLVRDataset(load_leetcode_dataset(split="train", decontaminate=False))
        sample = dataset[0]
        batch = collate_leetcode_samples([sample])

        policy = Policy(str(SOURCE_CHECKPOINT), dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(policy.model.parameters(), lr=1e-6)

        with tempfile.TemporaryDirectory(prefix="train-step-") as tmp:
            intermediate = Path(tmp) / "intermediate"
            checkpoint = Path(tmp) / "checkpoint"
            intermediate.mkdir()
            checkpoint.mkdir()

            generator = VllmGenerator(
                model_name=str(SOURCE_CHECKPOINT),
                checkpoint_path=str(intermediate),
                dtype="bfloat16",
                max_model_len=2048,
                gpu_memory_utilization=0.85,
            )
            runner = create_sandbox_runner_pool(NUM_RUNNERS)
            try:
                trainer = Trainer(
                    policy=policy,
                    generator=generator,
                    optimizer=optimizer,
                    dataloader=[batch],
                    group_size=GROUP_SIZE,
                    num_epochs=1,
                    intermediate_path=str(intermediate),
                    checkpoint_path=str(checkpoint),
                    runner=runner,
                    cfg={"WANDB_ENABLED": False},
                    resume=False,
                )
                trainer.train()
            finally:
                generator.shutdown()
                runner.stop()
                policy.to_cpu()
                del policy
                torch.cuda.empty_cache()

            self.assertTrue((checkpoint / "config.json").is_file())
            self.assertTrue(
                any(checkpoint.glob("*.safetensors")) or (checkpoint / "pytorch_model.bin").is_file()
            )
            self.assertTrue((checkpoint / TRAIN_STATE_NAME).is_file())


if __name__ == "__main__":
    unittest.main()
