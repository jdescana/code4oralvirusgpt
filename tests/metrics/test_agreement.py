from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score

from oral_virus_gpt.metrics.agreement import cohens_kappa


def test_cohens_kappa_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=100)
    b = rng.integers(0, 4, size=100)
    assert cohens_kappa(a, b) == cohen_kappa_score(a, b)
