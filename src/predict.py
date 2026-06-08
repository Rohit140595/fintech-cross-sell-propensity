"""
Inference pipeline for LendingClub LTV prediction.

Mirrors the training feature engineering pipeline to prevent train-serve skew.
"""

from __future__ import annotations

import pandas as pd

from src.features import build_feature_matrix  # noqa: F401
from src.model import load_model  # noqa: F401


def predict_single(
    member_id: str,
    loans: list[dict],
    cutoff_date: pd.Timestamp,
    model_path: str = "models/ltv_model.pkl",
) -> dict:
    """
    Score a single borrower's LTV from their loan history.

    Args:
        member_id:   Stable borrower identifier.
        loans:       List of raw loan dicts (pre-cutoff only).
        cutoff_date: Feature observation cutoff.
        model_path:  Path to saved model artifact.

    Returns:
        Dict with will_return_probability and loan_tier.
    """
    raise NotImplementedError
