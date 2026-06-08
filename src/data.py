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
    """
    Load the central YAML config file.

    Args:
        config_path: Path to the YAML config file. Defaults to config.yaml
                     in the current working directory.

    Returns:
        Dict of config values keyed by section (paths, data, split, model…).
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw(raw_dir: str, filename: str = "train.csv") -> pd.DataFrame:
    """
    Load the Health Insurance Cross-Sell CSV from disk.

    Expects train.csv from the Kaggle "Health Insurance Cross Sell Prediction"
    dataset (~381K rows, 12 columns). The test.csv from Kaggle has no Response
    column and is not used — we split train.csv ourselves via split().

    Args:
        raw_dir:  Directory containing the CSV files (e.g. "data/raw").
        filename: CSV filename to load. Defaults to "train.csv".

    Returns:
        DataFrame with one row per customer and 12 columns:
        id, Gender, Age, Driving_License, Region_Code, Previously_Insured,
        Vehicle_Age, Vehicle_Damage, Annual_Premium, Policy_Sales_Channel,
        Vintage, Response.
    """
    df = pd.read_csv(Path(raw_dir) / filename)
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and type-cast the raw DataFrame into a fully numeric form.

    Encoding decisions:
      - Gender: Male=1, Female=0 (binary).
      - Vehicle_Damage: Yes=1, No=0 (binary).
      - Vehicle_Age: ordered numeric — < 1 Year=0, 1-2 Year=1, > 2 Years=2.
        Ordered encoding preserves the monotonic relationship with conversion
        rate observed in EDA (older vehicle → higher conversion).
      - Region_Code, Policy_Sales_Channel: cast float → int (already numeric,
        Kaggle stores them as float due to CSV formatting).

    Args:
        df: Raw DataFrame from load_raw().

    Returns:
        Copy of df with all string columns replaced by numeric encodings.
        No columns are dropped — all 12 original columns are retained.
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
    Stratified train/test split preserving the target class distribution.

    Stratification ensures both sets reflect the overall ~12.3% positive rate
    rather than leaving class balance to chance on a random split.

    Args:
        df:           Cleaned DataFrame from clean().
        target_col:   Name of the binary target column. Defaults to "Response".
        test_size:    Fraction of rows assigned to the test set. Defaults to
                      0.20 (80/20 split → ~304K train, ~76K test).
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, test_df). Both DataFrames include all columns
        (features + target) and have reset integer indices.
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
