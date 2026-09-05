from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python tiny_coder_rlvr/run.py` (or cwd inside the package) to find repo packages.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from data.prepare_data import get_dataloader
from tiny_coder_rlvr import settings
from tiny_coder_rlvr.generation import VllmGenerator
from tiny_coder_rlvr.model import Policy
from tiny_coder_rlvr.sandbox.runner import create_sandbox_runner_pool
from tiny_coder_rlvr.train import TRAIN_STATE_NAME, Trainer

REPO_ROOT = settings.REPO_ROOT
DEFAULT_CONFIG = settings.DEFAULT_CONFIG_PATH


def load_config(path: Path) -> dict:
    return settings.load_config(path)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_policy_path(cfg: dict) -> Path:
    checkpoint_path = resolve_path(cfg["checkpoint_path"])
    init_checkpoint = resolve_path(cfg.get("init_checkpoint", cfg["model"]))
    # Resume from last train checkpoint when weights + train_state exist.
    if (checkpoint_path / "config.json").is_file() and (checkpoint_path / TRAIN_STATE_NAME).is_file():
        return checkpoint_path
    if init_checkpoint.is_dir():
        return init_checkpoint
    return Path(cfg["model"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run GRPO training for tiny-coder-rlvr")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    resume = cfg.get("resume", True) if args.resume is None else args.resume

    policy_path = resolve_policy_path(cfg)
    checkpoint_path = resolve_path(cfg["checkpoint_path"])
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    dtype_name = str(cfg.get("dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name)

    print(f"loading policy from {policy_path}")
    policy = Policy(str(policy_path), dtype=torch_dtype)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=float(cfg.get("lr", 1e-6)))

    train_loader = get_dataloader(
        split="train",
        batch_size=int(cfg.get("batch_size", 1)),
        seed=int(cfg["seed"]) if cfg.get("seed") is not None else None,
        decontaminate=bool(cfg.get("decontaminate", True)),
        ngram_size=int(cfg.get("ngram_size", 12)),
    )
    eval_loader = get_dataloader(
        split="test",
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=False,
        decontaminate=bool(cfg.get("decontaminate", True)),
        ngram_size=int(cfg.get("ngram_size", 12)),
    )

    generator = VllmGenerator(
        model_name=str(policy_path),
        model_path=str(policy_path),
        dtype=dtype_name,
        max_model_len=int(cfg.get("max_model_len", 8192)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)),
        reasoning_parser_name=str(cfg.get("reasoning_parser_name", "qwen3")),
        max_new_tokens=int(cfg.get("max_new_tokens", 7168)),
        temperature=float(cfg.get("temperature", 0.6)),
        top_p=float(cfg.get("top_p", 0.95)),
        top_k=int(cfg.get("top_k", 20)),
    )
    runner = create_sandbox_runner_pool(
        int(cfg.get("num_runners", 2)),
        image=str(cfg.get("docker_image", "tiny-coder-sandbox")),
    )

    trainer = Trainer(
        policy=policy,
        generator=generator,
        optimizer=optimizer,
        dataloader=train_loader,
        group_size=int(cfg.get("group_size", 8)),
        num_epochs=int(cfg.get("num_epochs", 1)),
        checkpoint_path=str(checkpoint_path),
        runner=runner,
        cfg=cfg,
        resume=resume,
        eval_dataloader=eval_loader,
        eval_every=int(cfg.get("eval_every", 50)),
        eval_samples=int(cfg.get("eval_samples", 4)),
        save_every=int(cfg.get("save_every", 30)),
    )

    try:
        trainer.train()
    finally:
        generator.shutdown()
        runner.stop()
        policy.to_cpu(optimizer)


if __name__ == "__main__":
    main()
