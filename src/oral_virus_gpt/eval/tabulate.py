from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd

from oral_virus_gpt.eval.runner import EvalResult
from oral_virus_gpt.metrics.triage import TierMetrics


def main_table_row(method: str, dataset: str, result: EvalResult) -> dict[str, Any]:
    return {
        "method": method,
        "dataset": dataset,
        "accuracy": round(result.accuracy * 100.0, 2),
        "macro_f1": round(result.macro_f1 * 100.0, 2),
        "auc": round(result.auc * 100.0, 2) if not np.isnan(result.auc) else np.nan,
        "n": result.n,
    }


def uq_ablation_row(label: str, result: EvalResult) -> dict[str, Any]:
    return {
        "config": label,
        "ece": round(result.ece, 4),
        "brier": round(result.brier, 4),
        "coverage": round(result.coverage * 100.0, 1),
        "set_size": round(result.set_size, 2),
    }


def risk_tier_table(metrics_by_tier: dict[str, TierMetrics]) -> pd.DataFrame:
    rows = []
    for tier_name, m in metrics_by_tier.items():
        rows.append(
            {
                "tier": tier_name,
                "coverage_pct": round(m.coverage * 100.0, 1),
                "accuracy_pct": (
                    round(m.accuracy * 100.0, 1) if not np.isnan(m.accuracy) else np.nan
                ),
                "sensitivity_pct": (
                    round(m.sensitivity * 100.0, 1) if not np.isnan(m.sensitivity) else np.nan
                ),
                "npv_pct": round(m.npv * 100.0, 1) if not np.isnan(m.npv) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def aggregate_seed_metrics(results: list[EvalResult]) -> dict[str, float]:
    if not results:
        return {}
    accs = np.array([r.accuracy for r in results], dtype=np.float64)
    f1s = np.array([r.macro_f1 for r in results], dtype=np.float64)
    return {
        "n_seeds": float(len(results)),
        "acc_mean": float(accs.mean()),
        "acc_std": float(accs.std(ddof=1)) if accs.size > 1 else 0.0,
        "f1_mean": float(f1s.mean()),
        "f1_std": float(f1s.std(ddof=1)) if f1s.size > 1 else 0.0,
    }


def to_records(results: dict[str, EvalResult]) -> pd.DataFrame:
    rows = [{"name": name, **asdict(r)} for name, r in results.items()]
    return pd.DataFrame(rows)
