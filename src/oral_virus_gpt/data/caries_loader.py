from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from oral_virus_gpt.data._base import BaseDentalDataset
from oral_virus_gpt.data.transforms import PhotoStats, photo_normalize


class CariesDataset(BaseDentalDataset):
    manifest_filename = "caries_manifest.json"

    def __init__(self, root: Any, split: str = "test") -> None:
        super().__init__(root=root, split=split)
        self.stats = PhotoStats()

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.index.records[index]
        with Image.open(self.root / rec["photo"]) as img:
            arr = to_tensor(img.convert("RGB"))
        return {
            "photo": photo_normalize(arr, self.stats),
            "radiograph": torch.zeros_like(arr),
            "text": rec.get("text", "Intraoral photograph annotated for caries detection."),
            "label": int(rec["label"]),
            "patient_id": str(rec.get("patient_id", index)),
            "photo_present": True,
            "radiograph_present": False,
            "bbox": rec.get("bbox", []),
        }
