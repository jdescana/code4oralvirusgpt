from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from oral_virus_gpt.data._base import BaseDentalDataset
from oral_virus_gpt.data.transforms import RadiographStats, radiograph_normalize


class DentexDataset(BaseDentalDataset):
    manifest_filename = "dentex_manifest.json"

    def __init__(self, root: Any, split: str = "train") -> None:
        super().__init__(root=root, split=split)
        self.stats = RadiographStats()

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.index.records[index]
        path = self.root / rec["radiograph"]
        with Image.open(path) as img:
            arr = to_tensor(img.convert("L"))
        radio = arr.expand(3, -1, -1)
        return {
            "photo": torch.zeros_like(radio),
            "radiograph": radiograph_normalize(radio[0:1], self.stats).expand(3, -1, -1),
            "text": rec.get("text", "Panoramic dental radiograph for pathology classification."),
            "label": int(rec["label"]),
            "patient_id": str(rec.get("patient_id", index)),
            "photo_present": False,
            "radiograph_present": True,
        }
