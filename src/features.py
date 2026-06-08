"""
Feature engineering for cross-sell propensity prediction.

Responsibilities:
  - Transform raw cleaned columns into model-ready features.
  - Log transform right-skewed Annual_Premium.
  - Frequency-encode high-cardinality categoricals (channel, region).
  - Target-encode region and channel using train-only conversion rates.
  - Add interaction terms for the strongest signal combinations from EDA.
  - Add age segment flags and premium-to-age ratio.

All encoding statistics (frequencies, target rates) are fit on train_df only
and applied to test_df to prevent leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log1p-transformed Annual_Premium to reduce right skew.

    Annual_Premium has a heavy right tail (max ~540K vs median ~31K).
    log1p compresses the scale so linear models and distance-based models
    treat extreme premiums less differently from the bulk of the distribution.
    Tree-based models are scale-invariant but benefit from reduced outlier
    influence on split evaluation.

    Args:
        df: DataFrame containing an Annual_Premium column (float).

    Returns:
        Copy of df with an additional premium_log column (float).
        The original Annual_Premium column is preserved.
    """
    df = df.copy()
    df["premium_log"] = np.log1p(df["Annual_Premium"])
    return df


def add_frequency_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Frequency-encode high-cardinality categorical columns.

    Replaces each category value with its relative frequency in the training
    set. This gives the model a signal for how common each channel/region is
    without exploding dimensionality via one-hot encoding.

    Frequencies are computed only on train_df and then mapped onto test_df
    to prevent leakage. Unknown test categories get frequency 0.0.

    Args:
        train_df: Training DataFrame. Frequencies are computed from this.
        test_df:  Test DataFrame. Frequencies from train are applied here.
        cols:     Column names to frequency-encode. Each column gets a new
                  column named <col>_freq; the original column is preserved.

    Returns:
        Tuple of (train_df, test_df) with new <col>_freq columns added.
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    for col in cols:
        freq_map = (train_df[col].value_counts() / len(train_df)).to_dict()
        train_df[f"{col}_freq"] = train_df[col].map(freq_map).fillna(0.0)
        test_df[f"{col}_freq"]  = test_df[col].map(freq_map).fillna(0.0)

    return train_df, test_df


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: list[str],
    target_col: str = "Response",
    smoothing: float = 20.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Target-encode categorical columns using smoothed conversion rates.

    Replaces each category value with the smoothed mean of the target within
    that category computed on train_df only. Smoothing blends the category
    mean toward the global mean for low-frequency categories to reduce noise:

        encoded = (n * category_mean + smoothing * global_mean) / (n + smoothing)

    where n is the number of training samples in that category. This prevents
    rare categories (e.g. small regions, niche channels) from receiving noisy
    extreme rates based on very few observations.

    Args:
        train_df:   Training DataFrame. Encoding statistics are fit here.
        test_df:    Test DataFrame. Statistics from train are applied here.
        cols:       Columns to target-encode. Each gets a new <col>_target_enc
                    column; the original column is preserved.
        target_col: Binary target column used to compute conversion rates.
                    Defaults to "Response".
        smoothing:  Controls blend strength. Higher = stronger shrinkage toward
                    global mean for small groups. Default 20 works well for
                    groups with tens to hundreds of samples.

    Returns:
        Tuple of (train_df, test_df) with new <col>_target_enc columns added.
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    global_mean = train_df[target_col].mean()

    for col in cols:
        stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (
            (stats["count"] * stats["mean"] + smoothing * global_mean)
            / (stats["count"] + smoothing)
        )
        enc_map = smoothed.to_dict()
        train_df[f"{col}_target_enc"] = train_df[col].map(enc_map).fillna(global_mean)
        test_df[f"{col}_target_enc"]  = test_df[col].map(enc_map).fillna(global_mean)

    return train_df, test_df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms capturing the strongest signal combinations from EDA.

    not_insured_x_damage:
        Previously_Insured=0 AND Vehicle_Damage=1.
        The single highest-conversion segment — customers without existing
        vehicle insurance who have already experienced damage are strongly
        motivated to get covered.

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

    Three features are added based on EDA findings:

    is_young_driver:
        1 if Age < 25. Young drivers showed below-average conversion in EDA
        and are worth flagging explicitly so the model can down-weight them.

    is_prime_age:
        1 if 35 <= Age <= 55. This band had the highest conversion rate in EDA
        and represents the most valuable cross-sell target segment.

    premium_per_age:
        Annual_Premium / Age. Proxies lifetime insurance value relative to age —
        customers paying high premiums at a younger age represent higher LTV.
        A small epsilon (1e-6) is added to Age to prevent division-by-zero on
        any edge-case zero-age rows.

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


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Response",
    freq_cols: list[str] | None = None,
    target_enc_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble the full feature matrix for both train and test sets.

    Applies all feature engineering steps in order:
      1. Log transform on Annual_Premium → premium_log.
      2. Frequency encoding of Policy_Sales_Channel, Region_Code → <col>_freq.
      3. Target encoding of Policy_Sales_Channel, Region_Code → <col>_target_enc.
      4. Interaction features → not_insured_x_damage, age_x_vehicle_damage.
      5. Age segment flags and ratio → is_young_driver, is_prime_age,
         premium_per_age.

    All encoding statistics are fit on train_df only and applied to test_df
    — no leakage from test into encoding statistics.

    Args:
        train_df:        Training DataFrame (output of src.data.clean + split).
        test_df:         Test DataFrame (output of src.data.clean + split).
        target_col:      Target column name, used for target encoding.
                         Defaults to "Response".
        freq_cols:       Columns to frequency-encode. Defaults to
                         ["Policy_Sales_Channel", "Region_Code"].
        target_enc_cols: Columns to target-encode. Defaults to
                         ["Policy_Sales_Channel", "Region_Code"].

    Returns:
        Tuple of (train_df, test_df) with all engineered features added
        alongside the original columns.
    """
    if freq_cols is None:
        freq_cols = ["Policy_Sales_Channel", "Region_Code"]
    if target_enc_cols is None:
        target_enc_cols = ["Policy_Sales_Channel", "Region_Code"]

    train_df = add_log_transform(train_df)
    test_df  = add_log_transform(test_df)

    train_df, test_df = add_frequency_encoding(train_df, test_df, cols=freq_cols)

    train_df, test_df = add_target_encoding(
        train_df, test_df, cols=target_enc_cols, target_col=target_col
    )

    train_df = add_interaction_features(train_df)
    test_df  = add_interaction_features(test_df)

    train_df = add_age_features(train_df)
    test_df  = add_age_features(test_df)

    new_cols = (
        ["premium_log"]
        + [f"{c}_freq" for c in freq_cols]
        + [f"{c}_target_enc" for c in target_enc_cols]
        + ["not_insured_x_damage", "age_x_vehicle_damage"]
        + ["is_young_driver", "is_prime_age", "premium_per_age"]
    )
    print(f"Feature matrix: {train_df.shape[1]} columns | {len(new_cols)} new: {new_cols}")

    return train_df, test_df
