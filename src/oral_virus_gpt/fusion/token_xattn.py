from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def scaled_dot_product(q: Tensor, k: Tensor, v: Tensor, num_heads: int) -> Tensor:
    if q.shape[-1] % num_heads != 0:
        raise ValueError(f"hidden dim {q.shape[-1]} not divisible by num_heads {num_heads}")
    b, n_q, d = q.shape
    n_k = k.shape[1]
    d_k = d // num_heads
    q_h = q.view(b, n_q, num_heads, d_k).transpose(1, 2)
    k_h = k.view(b, n_k, num_heads, d_k).transpose(1, 2)
    v_h = v.view(b, n_k, num_heads, d_k).transpose(1, 2)
    scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(d_k)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v_h)
    return out.transpose(1, 2).contiguous().view(b, n_q, d)


class KVProjection(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, text: Tensor) -> tuple[Tensor, Tensor]:
        return self.k(text), self.v(text)


class TokenCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int = 16,
        kv_proj: KVProjection | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by num_heads {num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv = kv_proj if kv_proj is not None else KVProjection(d_model)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query_tokens: Tensor, key_value_tokens: Tensor) -> Tensor:
        q = self.q_proj(query_tokens)
        k, v = self.kv(key_value_tokens)
        out = scaled_dot_product(q, k, v, self.num_heads)
        return self.out_proj(out)


class RadiographCrossAttention(TokenCrossAttention):
    pass
