from __future__ import annotations

import torch

from oral_virus_gpt.uq.severity import SeverityHead


def test_severity_zero_init_yields_zero() -> None:
    head = SeverityHead(num_classes=4)
    probs = torch.softmax(torch.randn(3, 4), dim=-1)
    entropy = torch.rand(3)
    set_size = torch.tensor([1, 2, 3])
    out = head(probs, entropy, set_size)
    assert torch.allclose(out, torch.zeros(3))
