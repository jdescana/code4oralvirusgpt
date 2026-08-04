from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from torch import Tensor, nn


@dataclass(slots=True)
class LoraSpec:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: tuple[str, ...] = ("wqkv", "wo", "w1", "w2", "w3")
    bias: str = "none"
    modules_to_save: tuple[str, ...] = field(default_factory=tuple)


class TextEncoder(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        tokenizer: object,
        lora_spec: LoraSpec | None = None,
        max_length: int = 512,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.lora_spec = lora_spec or LoraSpec()
        self.max_length = max_length

    def encode_text(self, prompts: Sequence[str]) -> dict[str, Tensor]:
        if not callable(self.tokenizer):
            raise RuntimeError("tokenizer is not callable")
        return self.tokenizer(
            list(prompts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        inputs_embeds: Tensor | None = None,
    ) -> Tensor:
        kwargs: dict[str, Tensor] = {}
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        else:
            kwargs["input_ids"] = input_ids
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        out = self.language_model(**kwargs, output_hidden_states=True, return_dict=True)
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return out.hidden_states[-1]
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        raise RuntimeError("language model did not return hidden states")
