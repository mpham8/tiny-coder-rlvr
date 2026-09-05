import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_coder_rlvr.loss import EPSILON_HIGH, grpo_loss


def _ones_mask(*shape: int) -> torch.Tensor:
    return torch.ones(shape)


class GrpoLossTest(unittest.TestCase):
    def test_returns_scalar(self):
        log_probs = torch.zeros(2, 3, requires_grad=True)
        old_log_probs = torch.zeros(2, 3)
        advantages = torch.tensor([1.0, -1.0])
        mask = _ones_mask(2, 3)

        loss = grpo_loss(mask, log_probs, old_log_probs, advantages)

        self.assertEqual(loss.shape, torch.Size([]))

    def test_zero_advantages_zero_loss(self):
        log_probs = torch.tensor([[0.0, 0.0], [0.5, 0.5]], requires_grad=True)
        old_log_probs = torch.zeros(2, 2)
        advantages = torch.zeros(2)
        mask = _ones_mask(2, 2)

        loss = grpo_loss(mask, log_probs, old_log_probs, advantages)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_symmetric_advantages_cancel_when_ratio_is_one(self):
        log_probs = torch.zeros(2, 1, requires_grad=True)
        old_log_probs = torch.zeros(2, 1)
        advantages = torch.tensor([1.0, -1.0])
        mask = _ones_mask(2, 1)

        loss = grpo_loss(mask, log_probs, old_log_probs, advantages)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_positive_advantage_negative_loss_when_policy_improves(self):
        # ratio > 1 with A > 0 → objective > 0 → loss < 0
        log_probs = torch.tensor([[0.5]], requires_grad=True)
        old_log_probs = torch.tensor([[0.0]])
        advantages = torch.tensor([1.0])
        mask = _ones_mask(1, 1)

        loss = grpo_loss(mask, log_probs, old_log_probs, advantages)

        self.assertLess(loss.item(), 0.0)

    def test_clip_high_limits_large_ratio(self):
        log_probs = torch.tensor([[5.0]], requires_grad=True)
        old_log_probs = torch.tensor([[0.0]])
        advantages = torch.tensor([1.0])
        mask = _ones_mask(1, 1)

        loss = grpo_loss(mask, log_probs, old_log_probs, advantages)

        self.assertAlmostEqual(loss.item(), -(1.0 + EPSILON_HIGH), places=5)

    def test_mask_excludes_padding(self):
        log_probs = torch.tensor([[0.5, 99.0]], requires_grad=True)
        old_log_probs = torch.zeros(1, 2)
        advantages = torch.tensor([1.0])
        mask = torch.tensor([[1.0, 0.0]])

        loss = grpo_loss(mask, log_probs, old_log_probs, advantages)
        expected = grpo_loss(
            _ones_mask(1, 1),
            log_probs[:, :1],
            old_log_probs[:, :1],
            advantages,
        )

        self.assertAlmostEqual(loss.item(), expected.item(), places=5)
    def test_gradient_flows_to_log_probs(self):
        log_probs = torch.tensor([[0.2, -0.1]], requires_grad=True)
        old_log_probs = torch.zeros(1, 2)
        advantages = torch.tensor([1.0])
        mask = _ones_mask(1, 2)

        loss = grpo_loss(mask, log_probs, old_log_probs, advantages)
        loss.backward()

        self.assertIsNotNone(log_probs.grad)
        self.assertTrue(torch.isfinite(log_probs.grad).all())


if __name__ == "__main__":
    unittest.main()
