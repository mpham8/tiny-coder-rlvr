from __future__ import annotations

import base64
import os
import pickle
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import StructuredOutputsConfig, WeightTransferConfig
from vllm.distributed.weight_transfer import ModuleSource
from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo
from vllm.reasoning import ReasoningParserManager

from tiny_coder_rlvr.completion import Completion
from tiny_coder_rlvr import settings


def _rpc_safe_update_info(update_info: dict[str, Any]) -> dict[str, Any]:
    """Pickle CUDA IPC handles so multiprocess LLM RPC can carry them.

    Same encoding as vLLM's HTTPVLLMWeightSyncClient; the engine worker
    unpickles when VLLM_ALLOW_INSECURE_SERIALIZATION=1.
    """
    ipc_handles = update_info.get("ipc_handles")
    if ipc_handles is None:
        return update_info
    out = {k: v for k, v in update_info.items() if k != "ipc_handles"}
    out["ipc_handles_pickled"] = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
    return out


class _LLMWeightSyncClient:
    """Adapt vLLM's LLM API to the trainer WeightSyncClient protocol."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        self.llm.init_weight_transfer_engine({"init_info": init_info})

    def start_weight_update(self) -> None:
        self.llm.start_weight_update()

    def update_weights(self, update_info: dict) -> None:
        self.llm.update_weights({"update_info": _rpc_safe_update_info(update_info)})

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self.llm.finish_weight_update(weight_version)


class VllmGenerator:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        dtype: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        reasoning_parser_name: str = "qwen3",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reasoning_parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)(
            self.tokenizer
        )

        self.model_name = model_name
        # Initial HF weights used only to construct the engine once.
        self.model_path = model_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.reasoning_parser_name = reasoning_parser_name
        self.max_new_tokens = int(settings.max_new_tokens if max_new_tokens is None else max_new_tokens)
        self.temperature = float(settings.temperature if temperature is None else temperature)
        self.top_p = float(settings.top_p if top_p is None else top_p)
        self.top_k = int(settings.top_k if top_k is None else top_k)
        self.llm: LLM | None = None
        self._transfer_engine = None
        self._sleeping = False
        self._awake_tags: set[str] = {"weights", "kv_cache"}

    def load(self):
        """Create the vLLM engine once (subsequent weight updates use IPC)."""
        if self.llm is not None:
            return

        # Required so EngineCore can unpickle CUDA IPC handles from the trainer.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        self.llm = LLM(
            model=self.model_path,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            structured_outputs_config=StructuredOutputsConfig(
                reasoning_parser=self.reasoning_parser_name
            ),
            weight_transfer_config=WeightTransferConfig(backend="ipc"),
            # Level-2 sleep frees weights+KV while keeping the engine process alive.
            enable_sleep_mode=True,
        )
        self._transfer_engine = None
        self._sleeping = False
        self._awake_tags = {"weights", "kv_cache"}

    def init_weight_sync(self, policy_module) -> None:
        """Bind HF policy module as the IPC weight source."""
        if self.llm is None:
            raise RuntimeError("VllmGenerator.load() must be called before init_weight_sync()")

        self._transfer_engine = WeightTransferTrainerFactory.trainer_init(
            IPCTrainerInitInfo(rank=0, packed=True),
            client=_LLMWeightSyncClient(self.llm),
            source=ModuleSource(policy_module),
        )

    def sleep(self, level: int = 2) -> None:
        """Discard vLLM GPU allocations (level 2: weights + KV) without killing the engine."""
        if self.llm is None:
            raise RuntimeError("VllmGenerator.load() must be called before sleep()")
        if self._sleeping and not self._awake_tags:
            return
        self.llm.sleep(level=level)
        self._sleeping = True
        self._awake_tags = set()

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Restore slept GPU allocations. tags: weights / kv_cache / scheduling / None=all."""
        if self.llm is None:
            raise RuntimeError("VllmGenerator.load() must be called before wake_up()")
        self.llm.wake_up(tags=tags)
        if tags is None:
            self._awake_tags = {"weights", "kv_cache"}
        else:
            self._awake_tags.update(tags)
        self._sleeping = not {"weights", "kv_cache"}.issubset(self._awake_tags)

    def ensure_awake(self) -> None:
        """Fully wake the engine before generate/eval if it was put to sleep."""
        if self.llm is None or not self._sleeping:
            return
        self.wake_up(tags=None)

    def sync_weights(self) -> None:
        """Push current HF policy weights into the live vLLM engine via CUDA IPC.

        Caller should wake_up(tags=['weights']) first after level-2 sleep so
        weight storage exists; KV can stay asleep until after HF moves off GPU.
        """
        if self._transfer_engine is None:
            raise RuntimeError("init_weight_sync() must be called before sync_weights()")
        self._transfer_engine.send_weights()

    def build_prompt(self, sample: dict, *, enable_thinking: bool = True) -> str:
        messages = [{"role": "user", "content": sample["query"]}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def make_sampling_params(self, prompt: str, *, n: int = 16) -> SamplingParams:
        prompt_len = len(self.tokenizer.encode(prompt))
        max_tokens = min(self.max_new_tokens, self.max_model_len - prompt_len - 64)
        if max_tokens < 1:
            raise ValueError(f"prompt too long for max_model_len={self.max_model_len}: {prompt_len} tokens")
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=max_tokens,
            n=n,
            logprobs=1,
        )

    def sampled_logprobs(self, token_ids: list[int], logprobs) -> list[float]:
        """Extract π_old logprob of each sampled token from vLLM's per-position maps."""
        if logprobs is None:
            raise RuntimeError("vLLM returned no logprobs; SamplingParams(logprobs=...) is required")
        if len(logprobs) != len(token_ids):
            raise RuntimeError(f"logprobs length {len(logprobs)} != token_ids length {len(token_ids)}")
        return [pos[tid].logprob for tid, pos in zip(token_ids, logprobs)]

    def generate(
        self,
        sample: dict,
        *,
        n: int = 16,
        enable_thinking: bool = True,
    ) -> list[Completion]:
        if self.llm is None:
            raise RuntimeError("VllmGenerator.load() must be called before generate()")
        self.ensure_awake()

        prompt = self.build_prompt(sample, enable_thinking=enable_thinking)
        sampling_params = self.make_sampling_params(prompt, n=n)
        outputs = self.llm.generate([prompt], sampling_params=sampling_params)

        results: list[Completion] = []
        for completion in outputs[0].outputs:
            thinking, solution = self.reasoning_parser.extract_reasoning(
                completion.text.strip(),
                request=None,
            )
            token_ids = list(completion.token_ids)
            results.append(
                Completion(
                    text=completion.text,
                    token_ids=token_ids,
                    old_logprobs=self.sampled_logprobs(token_ids, completion.logprobs),
                    thinking=thinking,
                    solution=solution,
                    finish_reason=completion.finish_reason,
                )
            )
        return results

    def shutdown(self):
        if self.llm is None:
            return
        self.llm.llm_engine.engine_core.shutdown()
        self.llm = None
        self._transfer_engine = None
        self._sleeping = False
        self._awake_tags = set()
