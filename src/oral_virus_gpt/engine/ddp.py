from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch import nn


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main_process() -> bool:
    return rank() == 0


@contextmanager
def maybe_no_sync(model: nn.Module) -> Iterator[None]:
    if hasattr(model, "no_sync") and is_distributed():
        with model.no_sync():
            yield
    else:
        yield


def init_process_group_from_env() -> None:
    if not dist.is_available():
        return
    if dist.is_initialized():
        return
    if "WORLD_SIZE" not in os.environ:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
