"""
Data loading and preparation for the Health Insurance Cross-Sell dataset.

Responsibilities:
  - Load train.csv from data/raw/
  - Clean and type-cast columns
  - Stratified train/test split on the Response target
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw(raw_dir: str, filename: str = "train.csv") -> pd.DataFrame:
    """
    Load the cross-sell train CSV.

    Returns:
        DataFrame with one row per customer, 12 columns including Response.
    """
    df = pd.read_csv(Path(raw_dir) / filename)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the raw DataFrame.

    Steps:
      1. Encode Gender, Vehicle_Damage as binary (1/0).
      2. Encode Vehicle_Age as ordered numeric (< 1 Year=0, 1-2 Year=1, > 2 Years=2).
      3. Cast Region_Code and Policy_Sales_Channel to int.

    Returns:
        Cleaned DataFrame — no string columns remain.
    """
    df = df.copy()

    df["Gender"]         = (df["Gender"] == "Male").astype(int)
    df["Vehicle_Damage"] = (df["Vehicle_Damage"] == "Yes").astype(int)

    vehicle_age_map = {"< 1 Year": 0, "1-2 Year": 1, "> 2 Years": 2}
    df["Vehicle_Age"] = df["Vehicle_Age"].map(vehicle_age_map)

    df["Region_Code"]          = df["Region_Code"].astype(int)
    df["Policy_Sales_Channel"] = df["Policy_Sales_Channel"].astype(int)

    print(f"Cleaned shape: {df.shape} | missing values: {df.isnull().sum().sum()}")
    return df


def split(
    df: pd.DataFrame,
    target_col: str = "Response",
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/test split on the Response target.

    Stratification preserves the ~12% positive rate in both sets.

    Returns:
        (train_df, test_df) — both include the target column.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    print(f"Train: {len(train_df):,} rows | positive rate: {train_df[target_col].mean():.1%}")
    print(f"Test:  {len(test_df):,} rows  | positive rate: {test_df[target_col].mean():.1%}")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
