from __future__ import annotations

import numpy as np

from oral_virus_gpt.metrics.significance import bh_fdr, holm_bonferroni, paired_bootstrap_pvalue


def test_holm_bonferroni_adjusted_pvalues_step_function() -> None:
    pvals = [0.01, 0.04, 0.03, 0.005]
    adj = holm_bonferroni(pvals)
    assert np.all(adj >= np.array(pvals))
    assert np.all(adj <= 1.0)


def test_bh_fdr_monotone() -> None:
    pvals = [0.001, 0.008, 0.02, 0.4]
    adj = bh_fdr(pvals)
    sorted_adj = adj[np.argsort(pvals)]
    assert np.all(np.diff(sorted_adj) >= -1.0e-9)


def test_paired_bootstrap_recognises_strong_difference() -> None:
    a = np.full(50, 0.9)
    b = np.full(50, 0.1)
    p = paired_bootstrap_pvalue(a, b, iterations=100, seed=0)
    assert p < 0.05
