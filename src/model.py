"""
Model training, evaluation, and persistence for cross-sell propensity.

Responsibilities:
  - Build a sklearn Pipeline: FrequencyEncoder → TargetEncoder → XGBoost.
    All stateful encoders are inside the pipeline so CV folds never leak
    encoding statistics — frequency and target rates are always fit on the
    training fold only.
  - Hyperparameter tuning via Optuna with seeded TPE sampler.
  - Threshold optimisation using F-beta score.
  - Evaluation: PR-AUC, ROC-AUC, precision, recall at optimal threshold.
  - Model persistence: save/load artifact as a single pickle.

Primary metric: PR-AUC — preferred over ROC-AUC for imbalanced targets (~12%
positive rate). A random classifier scores PR-AUC ≈ 0.12; the tuned model
achieves ~0.37.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,  # noqa: F401
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder

from src.features import FrequencyEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Columns handled by stateful encoders inside the pipeline
FREQ_COLS       = ["Policy_Sales_Channel", "Region_Code"]
TARGET_ENC_COLS = ["Policy_Sales_Channel", "Region_Code"]


# ── Pipeline builder ──────────────────────────────────────────────────────────

def build_pipeline(
    xgb_params: dict,
    target_col: str = "Response",
    random_state: int = 42,
) -> Pipeline:
    """
    Build the full sklearn Pipeline: encoders → XGBoost classifier.

    Pipeline steps:
      1. FrequencyEncoder  — replaces Policy_Sales_Channel and Region_Code
                             with their relative training-set frequencies.
      2. TargetEncoder     — replaces the same two columns with smoothed
                             per-category conversion rates (fit on train fold).
                             Uses sklearn's built-in TargetEncoder (≥1.3).
      3. XGBClassifier     — gradient boosted trees with tuned hyperparameters
                             and automatic class imbalance handling via
                             scale_pos_weight.

    Putting all stateful encoders inside the pipeline guarantees that
    cross-validation folds compute encoding statistics from the training
    split only — no leakage from validation or test data.

    Args:
        xgb_params:   XGBoost hyperparameter dict. scale_pos_weight is set
                      automatically at fit time from the training labels.
        target_col:   Name of the binary target column. Used by TargetEncoder.
        random_state: Seed for TargetEncoder and XGBoost reproducibility.

    Returns:
        Unfitted sklearn Pipeline ready for fit() / predict_proba().
    """
    freq_encoder = FrequencyEncoder(cols=FREQ_COLS)

    # TargetEncoder replaces the same columns with smoothed conversion rates.
    # cv=5 means it uses 5-fold internal cross-fitting to prevent within-fold
    # leakage when called inside an outer CV loop.
    target_encoder = ColumnTransformer(
        transformers=[
            ("target_enc", TargetEncoder(
                target_type="binary",
                smooth="auto",
                cv=5,
                random_state=random_state,
            ), TARGET_ENC_COLS),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        random_state=random_state,
        verbosity=0,
        **xgb_params,
    )

    return Pipeline([
        ("freq_encoder",   freq_encoder),
        ("target_encoder", target_encoder),
        ("model",          model),
    ])


# ── Hyperparameter tuning ─────────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Tune XGBoost hyperparameters with Optuna using stratified k-fold CV.

    Each Optuna trial builds a full Pipeline (FrequencyEncoder +
    TargetEncoder + XGBoost) so encoding statistics are refit per fold —
    no leakage. Optimises PR-AUC (average_precision_score), the primary
    metric for this imbalanced binary classification task.

    Search space:
        n_estimators     : 100 – 500
        max_depth        : 3 – 7
        learning_rate    : 0.01 – 0.3 (log scale)
        subsample        : 0.6 – 1.0
        colsample_bytree : 0.6 – 1.0
        min_child_weight : 1 – 10

    scale_pos_weight is set automatically from the training fold class ratio
    inside each CV split.

    Args:
        X_train:      Training feature matrix (after build_feature_matrix).
        y_train:      Training target series.
        n_trials:     Number of Optuna trials. Default 50.
        cv_folds:     Number of stratified CV folds. Default 5.
        random_state: Seed for Optuna TPE sampler and XGBoost.

    Returns:
        Dict of best hyperparameter name → value, ready to pass to
        build_pipeline().
    """
    cv     = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    splits = list(cv.split(X_train, y_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        scores = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
            pipeline   = build_pipeline(
                {**params, "scale_pos_weight": pos_weight},
                random_state=random_state,
            )
            pipeline.fit(X_tr, y_tr)
            probs = pipeline.predict_proba(X_val)[:, 1]
            scores.append(average_precision_score(y_val, probs))

        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best CV PR-AUC: {study.best_value:.4f}")
    return study.best_params


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    best_params: dict,
    random_state: int = 42,
) -> Pipeline:
    """
    Train the final Pipeline with tuned hyperparameters on the full training set.

    Sets scale_pos_weight automatically from the training set class ratio.
    Returns the complete fitted Pipeline (encoders + model) so inference
    only requires calling pipeline.predict_proba(X).

    Args:
        X_train:      Training feature matrix (after build_feature_matrix).
        y_train:      Training target series.
        best_params:  Hyperparameter dict from tune_hyperparameters().
        random_state: Seed for reproducibility.

    Returns:
        Fitted sklearn Pipeline ready for predict_proba() or persistence.
    """
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    pipeline   = build_pipeline(
        {**best_params, "scale_pos_weight": pos_weight},
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)
    return pipeline


# ── Threshold optimisation ────────────────────────────────────────────────────

def tune_threshold(
    y_true: pd.Series,
    probs: np.ndarray,
    beta: float = 2.0,
) -> float:
    """
    Find the probability threshold that maximises F-beta score.

    Beta > 1 weights recall more than precision. For cross-sell, missing a
    likely converter (false negative) is more costly than a wasted outreach
    (false positive), so beta=2 is a sensible default.

    Args:
        y_true: True binary labels.
        probs:  Predicted probabilities for the positive class (shape: n,).
        beta:   F-beta weight. beta=1 → F1, beta=2 → recall twice as important.

    Returns:
        Optimal threshold float in (0, 1).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f_beta = (
        (1 + beta ** 2) * precision * recall
        / (beta ** 2 * precision + recall + 1e-9)
    )
    return float(thresholds[np.argmax(f_beta[:-1])])


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    name: str,
    y_true: pd.Series,
    probs: np.ndarray,
    threshold: float | None = None,
) -> dict:
    """
    Evaluate a binary classifier and return a metrics dict.

    Computes PR-AUC and ROC-AUC from predicted probabilities, then applies
    a threshold (F2-optimal by default) to derive precision and recall.

    Args:
        name:      Model name used as the "model" key in the returned dict.
        y_true:    True binary labels.
        probs:     Predicted probabilities for the positive class (shape: n,).
        threshold: Decision threshold for precision/recall. If None, the
                   F2-optimal threshold is computed from y_true and probs.

    Returns:
        Dict with keys: model, pr_auc, roc_auc, precision, recall, threshold.
    """
    if threshold is None:
        threshold = tune_threshold(y_true, probs, beta=2.0)

    preds = (probs >= threshold).astype(int)
    tp    = int(((preds == 1) & (y_true == 1)).sum())
    fp    = int(((preds == 1) & (y_true == 0)).sum())
    fn    = int(((preds == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "model":     name,
        "pr_auc":    round(average_precision_score(y_true, probs), 4),
        "roc_auc":   round(roc_auc_score(y_true, probs), 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "threshold": round(threshold, 4),
    }


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(
    pipeline: Pipeline,
    threshold: float,
    path: str,
) -> None:
    """
    Persist the fitted Pipeline and decision threshold to a pickle file.

    The artifact stores everything needed for inference:
      - The complete fitted Pipeline (FrequencyEncoder + TargetEncoder + XGBoost).
      - The F2-optimal decision threshold.

    Args:
        pipeline:  Fitted Pipeline from train_model().
        threshold: F2-optimal decision threshold from tune_threshold().
        path:      Output file path (.pkl). Parent directories are created
                   automatically if they do not exist.

    Returns:
        None. Prints confirmation with the save path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "pipeline":  pipeline,
        "threshold": threshold,
    }
    with open(path, "wb") as f:
        pickle.dump(artifact, f)
    print(f"Model saved → {path}")


def load_model(path: str) -> dict:
    """
    Load a model artifact from disk.

    Args:
        path: Path to a pickle file saved by save_model().

    Returns:
        Dict with keys:
            pipeline  : Fitted sklearn Pipeline (encoders + XGBoost).
            threshold : F2-optimal decision threshold (float).
    """
    with open(path, "rb") as f:
        return pickle.load(f)
