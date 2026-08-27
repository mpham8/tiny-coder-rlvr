import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tiny_coder_rlvr.loss import DEFAULT_EPSILON_HIGH, DEFAULT_KL_BETA, grpo_loss


class GrpoLossTest(unittest.TestCase):
    def test_returns_scalar(self):
        log_probs = torch.zeros(2, 3, requires_grad=True)
        old_log_probs = torch.zeros(2, 3)
        ref_log_probs = torch.zeros(2, 3)
        advantages = torch.tensor([1.0, -1.0])

        loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)

        self.assertEqual(loss.shape, torch.Size([]))

    def test_zero_advantages_zero_policy_term(self):
        log_probs = torch.tensor([[0.0, 0.0], [0.5, 0.5]], requires_grad=True)
        old_log_probs = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        ref_log_probs = old_log_probs.clone()
        advantages = torch.zeros(2)

        loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_symmetric_advantages_cancel_when_ratio_is_one(self):
        log_probs = torch.zeros(2, 1, requires_grad=True)
        old_log_probs = torch.zeros(2, 1)
        ref_log_probs = torch.zeros(2, 1)
        advantages = torch.tensor([1.0, -1.0])

        loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)

        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_positive_advantage_increases_loss_when_policy_improves(self):
        log_probs = torch.tensor([[0.5]], requires_grad=True)
        old_log_probs = torch.tensor([[0.0]])
        ref_log_probs = torch.tensor([[0.0]])
        advantages = torch.tensor([1.0])

        loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)

        self.assertLess(loss.item(), 0.0)

    def test_clip_high_limits_large_ratio(self):
        log_probs = torch.tensor([[5.0]], requires_grad=True)
        old_log_probs = torch.tensor([[0.0]])
        ref_log_probs = torch.tensor([[0.0]])
        advantages = torch.tensor([1.0])

        loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)

        expected_policy = -(1.0 + DEFAULT_EPSILON_HIGH)
        self.assertAlmostEqual(loss.item(), expected_policy, places=5)

    def test_kl_term_when_reference_differs(self):
        log_probs = torch.zeros(1, 2, requires_grad=True)
        old_log_probs = torch.zeros(1, 2)
        ref_log_probs = torch.full((1, 2), -1.0)
        advantages = torch.zeros(1)

        loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)

        self.assertGreater(loss.item(), 0.0)
        self.assertAlmostEqual(loss.item(), DEFAULT_KL_BETA, places=5)

    def test_gradient_flows_to_log_probs(self):
        log_probs = torch.tensor([[0.2, -0.1]], requires_grad=True)
        old_log_probs = torch.zeros(1, 2)
        ref_log_probs = torch.zeros(1, 2)
        advantages = torch.tensor([1.0])

        loss = grpo_loss(log_probs, old_log_probs, ref_log_probs, advantages)
        loss.backward()

        self.assertIsNotNone(log_probs.grad)
        self.assertTrue(torch.isfinite(log_probs.grad).all())


if __name__ == "__main__":
    unittest.main()
