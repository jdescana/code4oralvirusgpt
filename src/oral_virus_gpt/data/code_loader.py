from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from oral_virus_gpt.data._base import BaseDentalDataset
from oral_virus_gpt.data.transforms import (
    PhotoStats,
    RadiographStats,
    photo_normalize,
    radiograph_normalize,
)
from oral_virus_gpt.utils.text_template import ClinicalRecord


class CODeDataset(BaseDentalDataset):
    manifest_filename = "code_manifest.json"

    def __init__(self, root: Path | str, split: str = "train", tile_size: int = 448) -> None:
        super().__init__(root=root, split=split)
        self.tile_size = tile_size
        self.photo_stats = PhotoStats()
        self.radio_stats = RadiographStats()

    def _load_image(self, relative_path: str) -> Tensor:
        path = self.root / relative_path
        with Image.open(path) as img:
            arr = to_tensor(img.convert("RGB"))
        return arr

    def _load_radiograph(self, relative_path: str) -> Tensor:
        path = self.root / relative_path
        with Image.open(path) as img:
            arr = to_tensor(img.convert("L"))
        return arr.expand(3, -1, -1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.index.records[index]
        photo = self._load_image(rec["photo"])
        radio = (
            self._load_radiograph(rec["radiograph"])
            if rec.get("radiograph")
            else torch.zeros(3, self.tile_size, self.tile_size)
        )
        clinical = ClinicalRecord(
            demographics=rec.get("demographics", "n/a"),
            chief_complaint=rec.get("chief_complaint", "n/a"),
            history=rec.get("history", "n/a"),
            findings=rec.get("findings", "n/a"),
        )
        return {
            "photo": photo_normalize(photo, self.photo_stats),
            "radiograph": radiograph_normalize(radio[0:1], self.radio_stats).expand(3, -1, -1),
            "text": clinical.render(),
            "label": int(rec["label"]),
            "patient_id": str(rec["patient_id"]),
            "photo_present": rec.get("photo_present", True),
            "radiograph_present": rec.get("radiograph_present", True),
        }
