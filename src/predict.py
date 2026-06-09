"""
Inference pipeline for cross-sell propensity scoring.

Mirrors the training feature engineering pipeline to prevent train-serve skew.
The pipeline artifact (FrequencyEncoder + TargetEncoder + XGBoost) handles all
stateful encoding internally — predict_single only applies the stateless
build_feature_matrix transforms before passing data to the pipeline.
"""

from __future__ import annotations

import pandas as pd

from src.features import build_feature_matrix
from src.model import load_model


def predict_single(
    customer: dict,
    model_path: str = "models/cross_sell_model.pkl",
    artifact: dict | None = None,
) -> dict:
    """
    Score a single customer's cross-sell propensity.

    Applies the same stateless feature transforms used at training time
    (log transform, interaction terms, age features), then passes the row
    through the loaded Pipeline (FrequencyEncoder + TargetEncoder + XGBoost)
    to produce a probability score and binary prediction.

    Args:
        customer:   Dict of raw customer fields matching the training schema:
                    Gender, Age, Driving_License, Region_Code,
                    Previously_Insured, Vehicle_Age, Vehicle_Damage,
                    Annual_Premium, Policy_Sales_Channel, Vintage.
                    Values must already be cleaned (numeric, no strings) —
                    same encoding as src.data.clean() output.
        model_path: Path to the saved model pickle. Used only if artifact
                    is None.
        artifact:   Pre-loaded model artifact dict (keys: pipeline, threshold).
                    Pass this when serving multiple requests to avoid
                    reloading the model from disk on every call.

    Returns:
        Dict with keys:
            probability  : float — predicted conversion probability in [0, 1].
            prediction   : int   — 1 if probability >= threshold, else 0.
            threshold    : float — decision threshold used.
    """
    if artifact is None:
        artifact = load_model(model_path)

    pipeline  = artifact["pipeline"]
    threshold = artifact["threshold"]

    row = pd.DataFrame([customer])
    row = build_feature_matrix(row)

    # Drop id if present — not a feature
    row = row.drop(columns=["id"], errors="ignore")

    prob       = float(pipeline.predict_proba(row)[0, 1])
    prediction = int(prob >= threshold)

    return {
        "probability": round(prob, 4),
        "prediction":  prediction,
        "threshold":   round(threshold, 4),
    }
