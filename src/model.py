"""
Two-stage model training, evaluation, and persistence.

Stage 1 — Return classifier (binary XGBoost):
    Predicts will_return. Primary metric: PR-AUC.
    CV: TimeSeriesSplit (temporal ordering preserved).

Stage 2 — Loan amount tier classifier (3-class XGBoost):
    Predicts Low / Mid / High next loan tier.
    Trained on returning borrowers only.
    CV: StratifiedKFold (small dataset).
"""

from __future__ import annotations

import json  # noqa: F401
import pickle  # noqa: F401
from pathlib import Path  # noqa: F401

import numpy as np
import optuna
import pandas as pd
import shap  # noqa: F401
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,  # noqa: F401
    f1_score,  # noqa: F401
    precision_recall_curve,  # noqa: F401
    roc_auc_score,  # noqa: F401
)
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit  # noqa: F401

optuna.logging.set_verbosity(optuna.logging.WARNING)

TIER_LABELS = {0: "Low", 1: "Mid", 2: "High"}


def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    drop_missing_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Drop high-missing and constant columns. Returns (X_clean, y, dropped)."""
    raise NotImplementedError


def select_features_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 20,
    cache_path: str | None = None,
) -> list[str]:
    """Select top N features by mean |SHAP|. Caches result to disk."""
    raise NotImplementedError


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    objective: str = "binary",
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Tune XGBoost with seeded Optuna TPE sampler."""
    raise NotImplementedError


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    objective: str = "binary",
) -> xgb.XGBClassifier:
    """Train final XGBoost model with tuned hyperparameters."""
    raise NotImplementedError


def tune_threshold(
    y_true: pd.Series,
    probs: np.ndarray,
    beta: float = 2.0,
) -> float:
    """Find F-beta optimal threshold. Beta>1 weights recall over precision."""
    raise NotImplementedError


def evaluate_stage1(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float | None = None,
) -> dict:
    """Evaluate Stage 1: PR-AUC, ROC-AUC, precision, recall, threshold."""
    raise NotImplementedError


def evaluate_stage2(
    model: xgb.XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate Stage 2: weighted F1 and per-tier F1."""
    raise NotImplementedError


def save_model(
    stage1_model: xgb.XGBClassifier,
    stage2_model: xgb.XGBClassifier,
    stage1_features: list[str],
    stage2_features: list[str],
    threshold: float,
    path: str,
) -> None:
    """Persist both stage models and metadata to a single pickle."""
    raise NotImplementedError


def load_model(path: str) -> dict:
    """Load model artifact from disk."""
    raise NotImplementedError
