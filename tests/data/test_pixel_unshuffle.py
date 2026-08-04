from __future__ import annotations

import torch

from oral_virus_gpt.data.pixel_unshuffle import unshuffle


def test_unshuffle_quadruples_channels_halves_spatial() -> None:
    x = torch.rand(2, 3, 8, 12)
    y = unshuffle(x, factor=2)
    assert y.shape == (2, 12, 4, 6)


def test_unshuffle_factor_four() -> None:
    x = torch.rand(1, 2, 16, 16)
    y = unshuffle(x, factor=4)
    assert y.shape == (1, 32, 4, 4)
