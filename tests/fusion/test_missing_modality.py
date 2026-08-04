from __future__ import annotations

import torch

from oral_virus_gpt.fusion.missing_modality import NullEmbedding, mask_modality


def test_mask_replaces_absent_with_null() -> None:
    semantic = torch.rand(3, 4, 8)
    null = NullEmbedding(num_concept_slots=4, d_model=8)
    presence = torch.tensor([1.0, 0.0, 1.0])
    masked, mask = mask_modality(semantic, presence, null)
    assert masked.shape == semantic.shape
    assert torch.allclose(masked[0], semantic[0])
    assert torch.allclose(masked[2], semantic[2])
    assert not torch.allclose(masked[1], semantic[1])
    assert mask.tolist() == [1.0, 0.0, 1.0]
