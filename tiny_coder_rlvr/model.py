import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


class Policy:
    def __init__(self, model_name: str, dtype: torch.dtype = torch.bfloat16) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
        self.model.gradient_checkpointing_enable()
        self.model.train()

    def token_log_probs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Per-token log π(x_t | x_<t). Avoids materializing a full log_softmax over V."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = input_ids[:, 1:].unsqueeze(-1)
        # log π(target) = logit[target] - logsumexp(logits)
        selected = logits.gather(-1, targets).squeeze(-1)
        return selected - torch.logsumexp(logits, dim=-1)

    def to_gpu(self, optimizer: torch.optim.Optimizer | None = None):
        self.model.to("cuda")
        if optimizer is not None:
            move_optimizer_state(optimizer, "cuda")

    def to_cpu(self, optimizer: torch.optim.Optimizer | None = None):
        self.model.to("cpu")
        if optimizer is not None:
            move_optimizer_state(optimizer, "cpu")
        torch.cuda.empty_cache()

    def save(self, path):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: str) -> None:
    """Adam(W) moment buffers live outside the module; move them independently of weights."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device, non_blocking=False)
