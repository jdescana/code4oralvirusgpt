from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset


@dataclass(slots=True)
class DatasetIndex:
    name: str
    split: str
    records: list[dict[str, Any]]


class BaseDentalDataset(Dataset[dict[str, Any]], ABC):
    def __init__(self, root: Path | str, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.index = self._load_index()

    @property
    @abstractmethod
    def manifest_filename(self) -> str: ...

    def _load_index(self) -> DatasetIndex:
        manifest_path = self.root / self.manifest_filename
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Dataset manifest not found at {manifest_path}. "
                "Run scripts/prepare_<dataset>.sh first."
            )
        payload = json.loads(manifest_path.read_text())
        records = [r for r in payload.get("records", []) if r.get("split") == self.split]
        return DatasetIndex(name=payload.get("name", "?"), split=self.split, records=records)

    def __len__(self) -> int:
        return len(self.index.records)

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...
