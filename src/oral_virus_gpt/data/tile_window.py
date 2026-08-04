from __future__ import annotations

import math
from dataclasses import dataclass

from torch import Tensor
from torch.nn import functional as F


@dataclass(slots=True)
class TileGrid:
    rows: int
    cols: int
    tile_size: int

    @property
    def num_tiles(self) -> int:
        return self.rows * self.cols


def dynamic_tile(image: Tensor, tile_size: int = 448) -> tuple[Tensor, TileGrid]:
    if image.dim() != 3:
        raise ValueError("image must be a 3D CHW tensor")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    c, h, w = image.shape
    rows = math.ceil(h / tile_size)
    cols = math.ceil(w / tile_size)
    target_h = rows * tile_size
    target_w = cols * tile_size
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    tiles = image.unfold(1, tile_size, tile_size).unfold(2, tile_size, tile_size)
    tiles = tiles.contiguous().view(c, rows * cols, tile_size, tile_size).permute(1, 0, 2, 3)
    return tiles, TileGrid(rows=rows, cols=cols, tile_size=tile_size)


def expected_token_count(
    image_height: int,
    image_width: int,
    tile_size: int = 448,
    tokens_per_tile: int = 256,
    unshuffle_factor: int = 4,
) -> int:
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size <= 0 or tokens_per_tile <= 0 or unshuffle_factor <= 0:
        raise ValueError("token-count parameters must be positive")
    rows = math.ceil(image_height / tile_size)
    cols = math.ceil(image_width / tile_size)
    return rows * cols * tokens_per_tile // unshuffle_factor
