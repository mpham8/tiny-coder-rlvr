from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from data.prepare_data import get_dataloader
from tiny_coder_rlvr.generation import VllmGenerator
from tiny_coder_rlvr.model import Policy
from tiny_coder_rlvr.sandbox.runner import create_sandbox_runner_pool
from tiny_coder_rlvr.train import TRAIN_STATE_NAME, Trainer

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


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
    intermediate_path = resolve_path(cfg["intermediate_path"])
    intermediate_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    dtype_name = str(cfg.get("dtype", "bfloat16"))
    torch_dtype = getattr(torch, dtype_name)

    print(f"loading policy from {policy_path}")
    policy = Policy(str(policy_path), dtype=torch_dtype)
    optimizer = torch.optim.AdamW(policy.model.parameters(), lr=float(cfg.get("lr", 1e-6)))

    train_loader = get_dataloader(
        split="train",
        batch_size=int(cfg.get("batch_size", 1)),
        decontaminate=bool(cfg.get("decontaminate", True)),
    )
    eval_loader = get_dataloader(
        split="test",
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=False,
        decontaminate=bool(cfg.get("decontaminate", True)),
    )

    generator = VllmGenerator(
        model_name=str(policy_path),
        checkpoint_path=str(intermediate_path),
        dtype=dtype_name,
        max_model_len=int(cfg.get("max_model_len", 8192)),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)),
        reasoning_parser_name=str(cfg.get("reasoning_parser_name", "qwen3")),
    )
    runner = create_sandbox_runner_pool(int(cfg.get("num_runners", 2)))

    trainer = Trainer(
        policy=policy,
        generator=generator,
        optimizer=optimizer,
        dataloader=train_loader,
        group_size=int(cfg.get("group_size", 8)),
        num_epochs=int(cfg.get("num_epochs", 1)),
        intermediate_path=str(intermediate_path),
        checkpoint_path=str(checkpoint_path),
        runner=runner,
        cfg=cfg,
        resume=resume,
        eval_dataloader=eval_loader,
        eval_every=int(cfg.get("eval_every", 50)),
    )

    try:
        trainer.train()
    finally:
        generator.shutdown()
        runner.stop()
        policy.to_cpu()


if __name__ == "__main__":
    main()
