from __future__ import annotations

import math

import pytest
import torch

from oral_virus_gpt.data.tile_window import dynamic_tile, expected_token_count


@pytest.mark.parametrize("h,w", [(448, 448), (640, 480), (513, 1024), (449, 449)])
def test_tile_count_matches_formula(h: int, w: int) -> None:
    image = torch.rand(3, h, w)
    tiles, grid = dynamic_tile(image, tile_size=448)
    assert grid.rows == math.ceil(h / 448)
    assert grid.cols == math.ceil(w / 448)
    assert tiles.shape[0] == grid.num_tiles
    assert tiles.shape[2] == 448 and tiles.shape[3] == 448


def test_token_count_formula() -> None:
    assert expected_token_count(448, 448) == 64
    assert expected_token_count(896, 896) == 256
    assert expected_token_count(449, 449) == 256


def test_tile_rejects_non_chw() -> None:
    with pytest.raises(ValueError):
        dynamic_tile(torch.rand(448, 448), tile_size=448)
