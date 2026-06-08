"""
Model training, evaluation, and persistence for cross-sell propensity.

Responsibilities:
  - Baseline training: Logistic Regression and XGBoost with default params.
  - Hyperparameter tuning via Optuna with seeded TPE sampler.
  - SHAP-based feature selection.
  - Threshold optimisation using F-beta score.
  - Evaluation: PR-AUC, ROC-AUC, precision, recall at optimal threshold.
  - Model persistence: save/load artifact as a single pickle.

Primary metric: PR-AUC — preferred over ROC-AUC for imbalanced targets (~12%
positive rate). A random classifier scores PR-AUC ≈ 0.12; meaningful models
should exceed 0.50.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,  # noqa: F401
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Baseline ──────────────────────────────────────────────────────────────────

def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, object]:
    """
    Train Logistic Regression and XGBoost baselines with default parameters.

    Both models handle class imbalance explicitly:
      - Logistic Regression: class_weight='balanced'.
      - XGBoost: scale_pos_weight = neg / pos count ratio.

    No hyperparameter tuning is applied — these are intentionally simple
    baselines to establish a PR-AUC floor before feature engineering and
    Optuna tuning.

    Args:
        X_train: Feature matrix (no id or target column).
        y_train: Binary target series (0/1).

    Returns:
        Dict mapping model name → fitted model object:
            "logistic_regression" → sklearn Pipeline (StandardScaler + LR)
            "xgboost_baseline"    → xgb.XGBClassifier
    """
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])
    lr.fit(X_train, y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    return {
        "logistic_regression": lr,
        "xgboost_baseline":    xgb_model,
    }


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(
    X: pd.DataFrame,
    y: pd.Series,
    drop_missing_threshold: float = 0.99,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Clean the feature matrix before training.

    Drops columns that would degrade model quality:
      - Columns with more than drop_missing_threshold fraction of nulls.
      - Constant columns (only one unique non-null value).

    Args:
        X:                      Feature matrix (id and target already excluded).
        y:                      Target series aligned with X.
        drop_missing_threshold: Columns with missing rate above this are dropped.
                                Default 0.99 drops near-empty columns only.

    Returns:
        Tuple of:
            X_clean   : Cleaned feature matrix.
            y         : Target series (unchanged).
            drop_cols : List of column names that were dropped.
    """
    drop_cols: list[str] = []

    missing_rate = X.isnull().mean()
    high_missing = missing_rate[missing_rate > drop_missing_threshold].index.tolist()
    drop_cols.extend(high_missing)

    constant = [c for c in X.columns if X[c].nunique(dropna=True) <= 1]
    drop_cols.extend([c for c in constant if c not in drop_cols])

    X_clean = X.drop(columns=drop_cols)
    if drop_cols:
        print(f"Dropped {len(drop_cols)} columns: {drop_cols}")

    return X_clean, y, drop_cols


# ── SHAP feature selection ────────────────────────────────────────────────────

def select_features_shap(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_n: int = 20,
    cache_path: str | None = None,
) -> list[str]:
    """
    Select the top N features by mean absolute SHAP value.

    Fits a lightweight XGBoost scout model (100 trees, depth 4) purely for
    feature ranking — not the final model. SHAP values measure each feature's
    average contribution to predictions across all training samples.

    Result is cached to disk so repeated runs skip the computation. Delete
    the cache file to force re-selection after adding new features.

    Args:
        X_train:    Training feature matrix.
        y_train:    Training target series.
        top_n:      Number of top features to return.
        cache_path: Path to a JSON file for caching selected feature names.
                    If the file exists, feature names are loaded from it
                    instead of recomputing. Pass None to disable caching.

    Returns:
        List of selected feature names, ordered by descending mean |SHAP|.
        Length is min(top_n, total features).
    """
    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            features = json.load(f)
        print(f"SHAP features loaded from cache ({len(features)} features)")
        return features

    scout = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, verbosity=0, eval_metric="logloss",
    )
    scout.fit(X_train, y_train)

    explainer   = shap.TreeExplainer(scout)
    shap_values = explainer.shap_values(X_train)

    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif shap_values.ndim == 3:
        shap_values = np.abs(shap_values).mean(axis=2)

    mean_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_train.columns
    ).sort_values(ascending=False)

    features = mean_shap.head(top_n).index.tolist()
    print(f"Top {len(features)} SHAP features selected")

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(features, f)

    return features


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

    Optimises PR-AUC (average_precision_score) — the primary metric for this
    imbalanced binary classification task. Uses a seeded TPE sampler so
    results are reproducible across runs.

    Search space:
        n_estimators     : 100 – 500
        max_depth        : 3 – 7
        learning_rate    : 0.01 – 0.3 (log scale)
        subsample        : 0.6 – 1.0
        colsample_bytree : 0.6 – 1.0
        min_child_weight : 1 – 10

    Args:
        X_train:      Training feature matrix.
        y_train:      Training target series.
        n_trials:     Number of Optuna trials. More trials → better params
                      but longer runtime. 50 is a good default.
        cv_folds:     Number of stratified CV folds.
        random_state: Seed for Optuna TPE sampler and XGBoost.

    Returns:
        Dict of best hyperparameter name → value, ready to pass to
        train_model().
    """
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    splits = list(cv.split(X_train, y_train))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": pos_weight,
            "random_state":     random_state,
            "verbosity":        0,
        }
        scores = []
        for train_idx, val_idx in splits:
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            model = xgb.XGBClassifier(
                objective="binary:logistic", eval_metric="aucpr", **params
            )
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_val)[:, 1]
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
) -> xgb.XGBClassifier:
    """
    Train the final XGBoost model with tuned hyperparameters.

    Sets scale_pos_weight automatically from the training set class ratio
    so the model handles class imbalance even when best_params omits it.

    Args:
        X_train:     Training feature matrix.
        y_train:     Training target series.
        best_params: Hyperparameter dict from tune_hyperparameters().

    Returns:
        Fitted xgb.XGBClassifier ready for evaluation or persistence.
    """
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=pos_weight,
        **best_params,
    )
    model.fit(X_train, y_train)
    return model


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

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())

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
    model: xgb.XGBClassifier,
    features: list[str],
    threshold: float,
    path: str,
) -> None:
    """
    Persist the trained model and metadata to a single pickle file.

    The artifact stores everything needed for inference:
      - The fitted XGBoost model.
      - The ordered list of feature names expected at predict time.
      - The F2-optimal decision threshold.

    Args:
        model:     Fitted XGBoost classifier from train_model().
        features:  Ordered list of feature column names used during training.
        threshold: F2-optimal decision threshold from tune_threshold().
        path:      Output file path (.pkl). Parent directories are created
                   automatically if they do not exist.

    Returns:
        None. Prints confirmation with the save path.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":     model,
        "features":  features,
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
        Dict with keys: model, features, threshold.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
