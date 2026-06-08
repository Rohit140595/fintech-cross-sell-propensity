"""
Feature engineering for cross-sell propensity prediction.

Responsibilities:
  - Transform raw cleaned columns into model-ready features.
  - Frequency-encode high-cardinality categoricals (channel, region).
  - Add interaction terms that capture the strongest signal combinations
    identified in EDA (Previously_Insured × Vehicle_Damage, Age × damage).
  - Apply log1p transform to right-skewed Annual_Premium.

All transformations are fit on train and applied to test to prevent leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log1p-transformed Annual_Premium to reduce right skew.

    Annual_Premium has a heavy right tail (max ~540K vs median ~31K).
    log1p compresses the scale so tree splits and linear models treat
    extreme values less differently from the bulk of the distribution.

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
    set. This gives the model a sense of how common each channel/region is
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


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction terms capturing the strongest signal combinations from EDA.

    Two interactions are added:

    not_insured_x_damage:
        Previously_Insured=0 AND Vehicle_Damage=1.
        This is the single highest-conversion segment — customers without
        existing vehicle insurance who have already experienced vehicle damage
        are strongly motivated to get covered.

    age_x_vehicle_damage:
        Age × Vehicle_Damage.
        Captures that older customers with damage history convert at higher
        rates than younger ones. The product encodes both signals jointly.

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


def build_feature_matrix(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    freq_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assemble the full feature matrix for both train and test sets.

    Applies all feature engineering steps in order:
      1. Log transform on Annual_Premium → premium_log.
      2. Frequency encoding of high-cardinality categoricals → <col>_freq.
      3. Interaction features → not_insured_x_damage, age_x_vehicle_damage.

    Frequencies are always fit on train_df only and mapped to test_df,
    so there is no leakage from test into train encoding statistics.

    Args:
        train_df:  Training DataFrame (output of src.data.clean + split).
        test_df:   Test DataFrame (output of src.data.clean + split).
        freq_cols: Columns to frequency-encode. Defaults to
                   ["Policy_Sales_Channel", "Region_Code"].

    Returns:
        Tuple of (train_df, test_df) with all engineered features added.
        Original columns are preserved alongside new ones.
    """
    if freq_cols is None:
        freq_cols = ["Policy_Sales_Channel", "Region_Code"]

    train_df = add_log_transform(train_df)
    test_df  = add_log_transform(test_df)

    train_df, test_df = add_frequency_encoding(train_df, test_df, cols=freq_cols)

    train_df = add_interaction_features(train_df)
    test_df  = add_interaction_features(test_df)

    new_cols = ["premium_log"] + [f"{c}_freq" for c in freq_cols] + \
               ["not_insured_x_damage", "age_x_vehicle_damage"]
    print(f"Feature matrix: {train_df.shape[1]} columns | new: {new_cols}")

    return train_df, test_df
