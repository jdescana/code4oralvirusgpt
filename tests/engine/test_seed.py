from __future__ import annotations

import torch

from oral_virus_gpt.engine.seed import seed_everything


def test_seed_everything_makes_torch_deterministic() -> None:
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.allclose(a, b)


def test_seed_must_be_non_negative() -> None:
    import pytest

    with pytest.raises(ValueError):
        seed_everything(-1)
