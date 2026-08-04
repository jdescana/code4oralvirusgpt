from __future__ import annotations

from oral_virus_gpt.data.splits import patient_stratified_split


def test_no_patient_leak_across_splits() -> None:
    pids = [f"p{i // 3:03d}" for i in range(300)]
    split = patient_stratified_split(pids, seed=42)
    train_pids = {pids[i] for i in split.train_indices.tolist()}
    val_pids = {pids[i] for i in split.val_indices.tolist()}
    test_pids = {pids[i] for i in split.test_indices.tolist()}
    assert train_pids.isdisjoint(val_pids)
    assert train_pids.isdisjoint(test_pids)
    assert val_pids.isdisjoint(test_pids)


def test_split_ratios_within_tolerance() -> None:
    pids = [f"p{i:04d}" for i in range(1000)]
    split = patient_stratified_split(pids, seed=0)
    n = len(pids)
    assert abs(len(split.train_indices) / n - 0.7) < 0.02
    assert abs(len(split.val_indices) / n - 0.1) < 0.02
    assert abs(len(split.test_indices) / n - 0.2) < 0.02
