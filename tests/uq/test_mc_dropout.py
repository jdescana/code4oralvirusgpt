from __future__ import annotations

import torch
from torch import nn

from oral_virus_gpt.uq.mc_dropout import MCDropoutEnsemble, predictive_entropy


class _Stub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(8, 4)
        self.drop = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(self.drop(x))


def test_mc_dropout_returns_distinct_samples() -> None:
    model = _Stub()
    ensemble = MCDropoutEnsemble(model, num_samples=5)
    inputs = torch.rand(3, 8)
    samples = ensemble.collect(lambda: model(inputs))
    assert samples.shape == (5, 3, 4)
    pairwise_max = max(
        (samples[i] - samples[j]).abs().max().item() for i in range(5) for j in range(i + 1, 5)
    )
    assert pairwise_max > 1.0e-4


def test_predictive_entropy_zero_on_certain_distribution() -> None:
    probs = torch.zeros(3, 5)
    probs[:, 0] = 1.0
    h = predictive_entropy(probs)
    assert h.shape == (3,)
    assert torch.all(h < 1.0e-5)


def test_predictive_entropy_max_on_uniform() -> None:
    probs = torch.full((2, 4), 0.25)
    h = predictive_entropy(probs)
    expected = torch.log(torch.tensor(4.0))
    assert torch.allclose(h, expected.expand(2), atol=1.0e-6)
