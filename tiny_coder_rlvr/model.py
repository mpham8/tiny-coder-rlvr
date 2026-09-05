import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class Policy:
    def __init__(self, model_name: str, dtype: torch.dtype = torch.bfloat16) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

        self.model.train()

    def token_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        returns individual token log-probailities
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        log_probs = torch.log_softmax(outputs.logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs[:, :-1, :], dim=-1, index=input_ids[:, 1:].unsqueeze(-1)
        )

        return token_log_probs.squeeze(-1)

    def to_gpu(self, optimizer: torch.optim.Optimizer | None = None):
        self.model.to("cuda")
        if optimizer is not None:
            _move_optimizer_state(optimizer, "cuda")

    def to_cpu(self, optimizer: torch.optim.Optimizer | None = None):
        self.model.to("cpu")
        if optimizer is not None:
            _move_optimizer_state(optimizer, "cpu")
        torch.cuda.empty_cache()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Adam(W) moment buffers live outside the module; move them with the policy."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=False)
