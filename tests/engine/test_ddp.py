from __future__ import annotations

from oral_virus_gpt.engine.ddp import is_distributed, is_main_process, rank, world_size


def test_singleton_world_size_when_no_ddp() -> None:
    assert not is_distributed()
    assert world_size() == 1
    assert rank() == 0
    assert is_main_process()
