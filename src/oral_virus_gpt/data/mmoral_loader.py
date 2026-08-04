from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from oral_virus_gpt.data._base import BaseDentalDataset
from oral_virus_gpt.data.transforms import RadiographStats, radiograph_normalize


class MMOralDataset(BaseDentalDataset):
    manifest_filename = "mmoral_manifest.json"

    def __init__(self, root: Any, split: str = "test", task: str = "vqa") -> None:
        super().__init__(root=root, split=split)
        self.task = task
        self.stats = RadiographStats()

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.index.records[index]
        with Image.open(self.root / rec["radiograph"]) as img:
            arr = to_tensor(img.convert("L"))
        radio = arr.expand(3, -1, -1)
        return {
            "photo": torch.zeros_like(radio),
            "radiograph": radiograph_normalize(radio[0:1], self.stats).expand(3, -1, -1),
            "text": rec.get("question", "Describe the panoramic radiograph."),
            "answer": rec.get("answer", ""),
            "label": int(rec.get("label", -1)),
            "patient_id": str(rec.get("image_id", index)),
            "photo_present": False,
            "radiograph_present": True,
            "task": self.task,
        }
