from __future__ import annotations

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import StructuredOutputsConfig
from vllm.reasoning import ReasoningParserManager

from tiny_coder_rlvr.completion import Completion

MAX_NEW_TOKENS = 7168


class VllmGenerator:
    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        dtype: str,
        max_model_len: int,
        gpu_memory_utilization: float,
        reasoning_parser_name: str = "qwen3",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reasoning_parser = ReasoningParserManager.get_reasoning_parser(reasoning_parser_name)(
            self.tokenizer
        )

        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.reasoning_parser_name = reasoning_parser_name
        self.llm: LLM | None = None

    def load(self):
        """Create the vLLM engine and load the latest checkpoint."""
        self.llm = LLM(
            model=self.checkpoint_path,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            structured_outputs_config=StructuredOutputsConfig(
                reasoning_parser=self.reasoning_parser_name
            ),
        )

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
        max_tokens = min(MAX_NEW_TOKENS, self.max_model_len - prompt_len - 64)
        if max_tokens < 1:
            raise ValueError(f"prompt too long for max_model_len={self.max_model_len}: {prompt_len} tokens")
        return SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
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
