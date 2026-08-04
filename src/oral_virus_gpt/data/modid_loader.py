from __future__ import annotations

from typing import Any

import numpy as np
import tifffile
import torch

from oral_virus_gpt.data._base import BaseDentalDataset


class MODIDDataset(BaseDentalDataset):
    manifest_filename = "modid_manifest.json"

    def __init__(self, root: Any, split: str = "test") -> None:
        super().__init__(root=root, split=split)

    def __getitem__(self, index: int) -> dict[str, Any]:
        rec = self.index.records[index]
        cube = tifffile.imread(str(self.root / rec["multispectral"]))
        if cube.ndim != 3:
            raise ValueError(f"expected 3D multispectral cube, got shape {cube.shape}")
        cube = np.transpose(cube, (2, 0, 1)) if cube.shape[0] != 16 else cube
        cube_t = torch.from_numpy(cube.astype(np.float32))
        per_band_mean = cube_t.mean(dim=(1, 2), keepdim=True)
        per_band_std = cube_t.std(dim=(1, 2), keepdim=True).clamp_min(1.0e-6)
        normed = (cube_t - per_band_mean) / per_band_std
        return {
            "multispectral": normed,
            "label": int(rec["label"]),
            "subject_id": str(rec.get("subject_id", index)),
            "spectral_range_nm": rec.get("spectral_range_nm", [460, 600]),
        }
