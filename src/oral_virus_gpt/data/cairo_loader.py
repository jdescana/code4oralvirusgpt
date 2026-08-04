from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from oral_virus_gpt.data._base import BaseDentalDataset
from oral_virus_gpt.data.transforms import PhotoStats, photo_normalize


class CairoIntraoralDataset(BaseDentalDataset):
    manifest_filename = "cairo_manifest.json"

    def __init__(self, root: Any, split: str = "train") -> None:
        super().__init__(root=root, split=split)
        self.stats = PhotoStats()

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.index.records[index]
        path = self.root / rec["photo"]
        with Image.open(path) as img:
            arr = to_tensor(img.convert("RGB"))
        return {
            "photo": photo_normalize(arr, self.stats),
            "radiograph": torch.zeros_like(arr),
            "text": rec.get(
                "text", "Intraoral photograph for malignant transformation risk classification."
            ),
            "label": int(rec["label"]),
            "patient_id": str(rec["patient_id"]),
            "photo_present": True,
            "radiograph_present": False,
        }
