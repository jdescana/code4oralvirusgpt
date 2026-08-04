from __future__ import annotations

import torch

from oral_virus_gpt.fusion.token_xattn import (
    KVProjection,
    RadiographCrossAttention,
    TokenCrossAttention,
)


def test_token_xattn_output_shape_matches_query() -> None:
    block = TokenCrossAttention(d_model=16, num_heads=4)
    q = torch.rand(2, 5, 16)
    k = torch.rand(2, 7, 16)
    out = block(q, k)
    assert out.shape == q.shape


def test_shared_kv_means_shared_parameters() -> None:
    kv = KVProjection(d_model=16)
    photo = TokenCrossAttention(d_model=16, num_heads=4, kv_proj=kv)
    radio = RadiographCrossAttention(d_model=16, num_heads=4, kv_proj=kv)
    assert photo.kv is radio.kv
    assert photo.q_proj is not radio.q_proj


def test_attention_is_deterministic_under_seed() -> None:
    torch.manual_seed(0)
    block = TokenCrossAttention(d_model=16, num_heads=4)
    q = torch.rand(1, 4, 16)
    k = torch.rand(1, 6, 16)
    a = block(q, k)
    b = block(q, k)
    assert torch.allclose(a, b)
