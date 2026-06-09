"""
Feature engineering for cross-sell propensity prediction.

Responsibilities:
  - Stateless transforms: log, interactions, age features (no fitting needed).
  - FrequencyEncoder: a sklearn-compatible transformer that fits value_counts
    on train and applies them at transform time — integrates cleanly into a
    Pipeline so CV folds never leak encoding statistics.

Target encoding is handled by sklearn.preprocessing.TargetEncoder inside the
model Pipeline in src/model.py — keeping all fitting inside CV folds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ── Stateless transforms ──────────────────────────────────────────────────────

def add_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log1p-transformed Annual_Premium to reduce right skew.

    Annual_Premium has a heavy right tail (max ~540K vs median ~31K).
    log1p compresses the scale so linear models treat extreme premiums less
    differently from the bulk of the distribution. Tree-based models are
    scale-invariant but benefit from reduced outlier influence on splits.

    Args:
        df: DataFrame containing an Annual_Premium column (float).

    Returns:
        Copy of df with an additional premium_log column (float).
        The original Annual_Premium column is preserved.
    """
    df = df.copy()
    df["premium_log"] = np.log1p(df["Annual_Premium"])
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms capturing the strongest signal combinations from EDA.

    not_insured_x_damage:
        Previously_Insured=0 AND Vehicle_Damage=1.
        The single highest-conversion segment — customers without existing
        vehicle insurance who have experienced damage are strongly motivated
        to get covered.

    age_x_vehicle_damage:
        Age × Vehicle_Damage.
        Captures that older customers with damage history convert at higher
        rates than younger ones. Encodes both signals jointly as a product.

    Args:
        df: DataFrame containing Previously_Insured (int 0/1),
            Vehicle_Damage (int 0/1), and Age (int) columns.

    Returns:
        Copy of df with two additional columns:
            not_insured_x_damage (int 0/1)
            age_x_vehicle_damage (float)
    """
    df = df.copy()
    df["not_insured_x_damage"] = (
        (df["Previously_Insured"] == 0) & (df["Vehicle_Damage"] == 1)
    ).astype(int)
    df["age_x_vehicle_damage"] = df["Age"] * df["Vehicle_Damage"]
    return df


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add age segment flags and premium-to-age ratio.

    is_young_driver:
        1 if Age < 25. Young drivers showed below-average conversion in EDA
        and are worth flagging explicitly so the model can down-weight them.

    is_prime_age:
        1 if 35 <= Age <= 55. This band had the highest conversion rate in EDA
        and represents the most valuable cross-sell target segment.

    premium_per_age:
        Annual_Premium / Age. Proxies lifetime insurance value relative to age
        — customers paying high premiums at a younger age represent higher LTV.
        A small epsilon (1e-6) prevents division-by-zero on edge-case zero-age.

    Args:
        df: DataFrame containing Age (int) and Annual_Premium (float) columns.

    Returns:
        Copy of df with three additional columns:
            is_young_driver  (int 0/1)
            is_prime_age     (int 0/1)
            premium_per_age  (float)
    """
    df = df.copy()
    df["is_young_driver"] = (df["Age"] < 25).astype(int)
    df["is_prime_age"]    = ((df["Age"] >= 35) & (df["Age"] <= 55)).astype(int)
    df["premium_per_age"] = df["Annual_Premium"] / (df["Age"] + 1e-6)
    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all stateless feature transforms to a single DataFrame.

    This function is intentionally stateless — it requires no fitting and can
    be called independently on train, test, or any single inference row without
    risk of leakage.

    Stateful encodings (frequency and target encoding) are handled inside the
    sklearn Pipeline in src/model.py so they are always fit on the training
    fold only during cross-validation.

    Transforms applied:
      1. Log transform on Annual_Premium → premium_log.
      2. Interaction features → not_insured_x_damage, age_x_vehicle_damage.
      3. Age segment flags and ratio → is_young_driver, is_prime_age,
         premium_per_age.

    Args:
        df: Cleaned DataFrame — output of src.data.clean(). Must contain
            Annual_Premium, Previously_Insured, Vehicle_Damage, Age columns.

    Returns:
        Copy of df with all engineered feature columns appended.
        Original columns are preserved alongside new ones.
    """
    df = add_log_transform(df)
    df = add_interaction_features(df)
    df = add_age_features(df)

    new_cols = [
        "premium_log",
        "not_insured_x_damage", "age_x_vehicle_damage",
        "is_young_driver", "is_prime_age", "premium_per_age",
    ]
    print(f"Feature matrix: {df.shape[1]} columns | new: {new_cols}")
    return df


# ── FrequencyEncoder — sklearn-compatible transformer ─────────────────────────

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that replaces category values with their
    relative frequency in the training set.

    Integrating frequency encoding into a Pipeline ensures the frequency
    statistics are fit only on the training fold during cross-validation,
    preventing any leakage from validation or test data.

    Unknown categories seen at transform time (not in training) receive
    frequency 0.0.

    Attributes:
        cols:       List of column names to encode.
        freq_maps_: Dict mapping column name → {value: frequency} fitted
                    during fit(). Populated after fit() is called.

    Args:
        cols: Column names to frequency-encode. Each column is replaced
              in-place with its frequency value; no new columns are added.

    Example:
        >>> enc = FrequencyEncoder(cols=["Policy_Sales_Channel", "Region_Code"])
        >>> enc.fit(X_train)
        >>> X_train_enc = enc.transform(X_train)
        >>> X_test_enc  = enc.transform(X_test)
    """

    def __init__(self, cols: list[str]) -> None:
        self.cols = cols

    def fit(self, X: pd.DataFrame, y=None) -> "FrequencyEncoder":
        """
        Compute frequency maps from the training data.

        Args:
            X: Training DataFrame containing self.cols columns.
            y: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            self — fitted transformer.
        """
        self.freq_maps_: dict[str, dict] = {}
        for col in self.cols:
            self.freq_maps_[col] = (
                X[col].value_counts(normalize=True).to_dict()
            )
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Replace category values with fitted frequency statistics.

        Args:
            X: DataFrame to transform. Must contain self.cols columns.
            y: Ignored. Present for sklearn Pipeline compatibility.

        Returns:
            Copy of X with self.cols columns replaced by their frequency
            values (float). Unknown categories map to 0.0.
        """
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.freq_maps_[col]).fillna(0.0)
        return X
