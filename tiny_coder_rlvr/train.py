from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm
import wandb

from data.prepare_data import LeetCodeSample, sample_to_dict
from tiny_coder_rlvr.advantage import grpo_advantages_batch
from tiny_coder_rlvr.completion import Completion
from tiny_coder_rlvr.loss import grpo_loss
from tiny_coder_rlvr.reward import PASS_REWARD, compute_reward_batch

TRAIN_STATE_NAME = "train_state.pt"


class Trainer:
    def __init__(
        self,
        policy,
        generator,
        optimizer,
        dataloader,
        group_size,
        num_epochs,
        intermediate_path,
        checkpoint_path,
        runner,
        cfg: dict | None = None,
        resume: bool = True,
        eval_dataloader=None,
        eval_every: int = 50,
    ):
        self.policy = policy
        self.generator = generator
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.group_size = group_size
        self.num_epochs = num_epochs
        self.intermediate_path = Path(intermediate_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.runner = runner
        self.cfg = cfg or {}
        self.eval_dataloader = eval_dataloader
        self.eval_every = eval_every

        self.start_epoch = 0
        self.start_batch_idx = 0
        self.global_step = 0

        state_path = self.checkpoint_path / TRAIN_STATE_NAME
        if resume and state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.start_epoch = int(state["epoch"])
            self.start_batch_idx = int(state["batch_idx"])
            self.global_step = int(state.get("global_step", 0))
            self.optimizer.load_state_dict(state["optimizer"])

    def build_training_tensors(
        self,
        completions: list[Completion],
        advantages: list[float] | torch.Tensor,
        prompt_token_ids: list[list[int]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build padded tensors for grpo_loss.

        Returns:
            mask, log_probs, old_log_probs, advantages
            all on the policy device; response-aligned shapes (N, T) / (N,).
        """
        if len(completions) != len(prompt_token_ids):
            raise ValueError("completions and prompt_token_ids must have the same length")
        if any(c.old_logprobs is None for c in completions):
            raise ValueError("every Completion needs old_logprobs from vLLM")

        pad_id = self.policy.tokenizer.pad_token_id
        device = next(self.policy.model.parameters()).device

        prompt_lens = [len(p) for p in prompt_token_ids]
        resp_lens = [len(c.token_ids) for c in completions]
        max_prompt = max(prompt_lens)
        max_resp = max(resp_lens)
        n = len(completions)
        seq_len = max_prompt + max_resp

        input_ids = torch.full((n, seq_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(n, seq_len, dtype=torch.long)
        old_log_probs = torch.zeros(n, max_resp, dtype=torch.float32)
        mask = torch.zeros(n, max_resp, dtype=torch.float32)

        for i, completion in enumerate(completions):
            prompt = prompt_token_ids[i]
            response = completion.token_ids
            pl, rl = prompt_lens[i], resp_lens[i]

            input_ids[i, :pl] = torch.tensor(prompt, dtype=torch.long)
            input_ids[i, pl : pl + rl] = torch.tensor(response, dtype=torch.long)
            attention_mask[i, : pl + rl] = 1

            old_log_probs[i, :rl] = torch.tensor(completion.old_logprobs, dtype=torch.float32)
            mask[i, :rl] = 1.0

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        old_log_probs = old_log_probs.to(device)
        mask = mask.to(device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)

        full_log_probs = self.policy.token_log_probs(input_ids, attention_mask)
        log_probs = torch.zeros(n, max_resp, dtype=full_log_probs.dtype, device=device)
        for i, (pl, rl) in enumerate(zip(prompt_lens, resp_lens)):
            log_probs[i, :rl] = full_log_probs[i, pl - 1 : pl - 1 + rl]

        return mask, log_probs, old_log_probs, advantages_t

    def save_train_checkpoint(self, *, epoch: int, batch_idx: int, global_step: int) -> None:
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.policy.save(self.checkpoint_path)
        torch.save(
            {
                "epoch": epoch,
                "batch_idx": batch_idx,
                "global_step": global_step,
                "optimizer": self.optimizer.state_dict(),
            },
            self.checkpoint_path / TRAIN_STATE_NAME,
        )

    def evaluate(self) -> dict[str, float]:
        """Run test set with vLLM (must already be loaded). Returns mean reward / pass@1."""
        if self.eval_dataloader is None:
            raise RuntimeError("eval_dataloader was not provided")
        if self.generator.llm is None:
            raise RuntimeError("generator.load() must be called before evaluate()")

        rewards: list[float] = []
        passes = 0
        total = 0
        for batch in tqdm(self.eval_dataloader, desc="eval", leave=False):
            batch_size = len(batch["task_id"])
            samples = [
                LeetCodeSample(**{key: batch[key][i] for key in batch})
                for i in range(batch_size)
            ]
            for sample in samples:
                sample_dict = sample_to_dict(sample)
                completion = self.generator.generate(sample=sample_dict, n=1)[0]
                compute_reward_batch(
                    self.runner,
                    [completion],
                    sample,
                    tokenizer=self.policy.tokenizer,
                )
                rewards.append(float(completion.reward))
                passes += int(completion.base_reward == PASS_REWARD)
                total += 1

        return {
            "eval/reward": sum(rewards) / max(len(rewards), 1),
            "eval/pass_rate": passes / max(total, 1),
            "eval/n": float(total),
        }

    def train(self):
        #wb setup
        wandb_run = None
        cfg = self.cfg
        if cfg.get("WANDB_ENABLED", False):

            init_kwargs = {
                "project": cfg.get("WANDB_PROJECT", "rlhf-summarize"),
                "config": cfg,
            }
            if cfg.get("WANDB_ENTITY"):
                init_kwargs["entity"] = cfg["WANDB_ENTITY"]
            if cfg.get("WANDB_RUN_NAME"):
                init_kwargs["name"] = cfg["WANDB_RUN_NAME"]
            wandb_run = wandb.init(**init_kwargs)

        global_step = self.global_step

        for epoch in range(self.start_epoch, self.num_epochs):
            pbar = tqdm(self.dataloader, desc=f"epoch {epoch + 1}/{self.num_epochs}", leave=True)
            for batch_idx, batch in enumerate(pbar):
                if epoch == self.start_epoch and batch_idx < self.start_batch_idx:
                    continue

                #move policy to cpu, save policy, load vllm
                self.policy.to_cpu()
                self.policy.save(self.intermediate_path)
                self.generator.checkpoint_path = str(self.intermediate_path)
                self.generator.load()

                # periodic test eval (reuse loaded vLLM)
                if (
                    self.eval_dataloader is not None
                    and self.eval_every > 0
                    and global_step % self.eval_every == 0
                ):
                    eval_metrics = self.evaluate()
                    if wandb_run is not None:
                        wandb_run.log({**eval_metrics, "global_step": global_step, "epoch": epoch})

                #batch x G rollouts
                batch_size = len(batch["task_id"])
                samples = [
                    LeetCodeSample(**{key: batch[key][i] for key in batch})
                    for i in range(batch_size)
                ]
                grouped_completions: list[list[Completion]] = []
                prompt_token_ids: list[list[int]] = []
                for sample in samples:
                    sample_dict = sample_to_dict(sample)
                    prompt = self.generator.build_prompt(sample_dict)
                    prompt_ids = self.generator.tokenizer.encode(prompt, add_special_tokens=False)
                    group = self.generator.generate(sample=sample_dict, n=self.group_size)
                    grouped_completions.append(group)
                    prompt_token_ids.extend([prompt_ids] * len(group))

                #reward rollouts
                for sample, group in zip(samples, grouped_completions):
                    compute_reward_batch(
                        self.runner,
                        group,
                        sample,
                        tokenizer=self.policy.tokenizer,
                    )

                #compute grpo advantages
                reward_groups = [[c.reward for c in group] for group in grouped_completions]
                advantage_groups = grpo_advantages_batch(reward_groups)
                advantages = [a for group in advantage_groups for a in group]
                completions = [c for group in grouped_completions for c in group]

                #free vllm, move policy to gpu
                self.generator.shutdown()
                self.policy.to_gpu()

                #build training tensors
                mask, log_probs, old_log_probs, advantages_t = self.build_training_tensors(
                    completions,
                    advantages,
                    prompt_token_ids,
                )

                #grpo loss, optimizer step
                loss = grpo_loss(mask, log_probs, old_log_probs, advantages_t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                global_step += 1

                next_batch_idx = batch_idx + 1
                next_epoch = epoch
                if next_batch_idx >= len(self.dataloader):
                    next_epoch = epoch + 1
                    next_batch_idx = 0
                self.save_train_checkpoint(
                    epoch=next_epoch,
                    batch_idx=next_batch_idx,
                    global_step=global_step,
                )

                #log
                mean_reward = sum(c.reward for c in completions) / max(len(completions), 1)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    reward=f"{mean_reward:.3f}",
                    step=global_step,
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/loss": loss.item(),
                            "train/reward": mean_reward,
                            "epoch": epoch,
                            "batch_idx": batch_idx,
                            "global_step": global_step,
                        }
                    )

            self.start_batch_idx = 0

        if wandb_run is not None:
            wandb_run.finish()
