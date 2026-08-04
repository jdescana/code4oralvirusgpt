from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from oral_virus_gpt.engine.seed import seed_everything


@pytest.fixture(autouse=True)
def _deterministic() -> Iterator[None]:
    seed_everything(0)
    torch.set_num_threads(1)
    yield


@pytest.fixture
def tmp_ckpt_dir(tmp_path: Path) -> Path:
    out = tmp_path / "ckpt"
    out.mkdir(parents=True, exist_ok=True)
    return out
