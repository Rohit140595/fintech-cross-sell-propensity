"""
Data loading and preparation layer for the LendingClub loan dataset.

Responsibilities:
  - Load raw CSV from data/raw/
  - Clean and type-cast columns
  - Identify repeat borrowers via member_id
  - Compute two-stage LTV targets:
      Stage 1 — will_return   : 1 if borrower takes another loan within
                                 horizon_months after cutoff
      Stage 2 — loan_amt_tier : Low / Mid / High next loan amount tier
                                 (tertiles among returning borrowers)
  - Chronological train/test split by first loan issue date
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load the central YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_raw(raw_dir: str) -> pd.DataFrame:
    """
    Load the LendingClub accepted loans CSV.

    Expects one of:
      - accepted_2007_to_2018Q4.csv  (full dataset, ~2.3M rows)
      - a sampled subset saved to the same directory

    Parses issue_d and earliest_cr_line as datetime.
    Drops the two descriptor rows LendingClub appends at the end of the file.

    Returns:
        Raw DataFrame with one row per loan application.
    """
    raise NotImplementedError


def clean(df: pd.DataFrame, valid_statuses: list[str]) -> pd.DataFrame:
    """
    Clean and type-cast the raw LendingClub DataFrame.

    Steps:
      1. Filter to valid_statuses (closed loans only — active loans at
         cutoff have unknown outcome, creating leakage risk).
      2. Drop columns that are >90% missing.
      3. Parse int_rate, revol_util (strip '%' and cast to float).
      4. Encode emp_length as integer years (remove 'years', '< 1 year' → 0).
      5. Encode home_ownership, purpose as categorical.
      6. Cast grade / sub_grade to ordered numeric (A=1 … G=7).

    Args:
        df:             Raw DataFrame from load_raw().
        valid_statuses: Loan statuses to keep.

    Returns:
        Cleaned DataFrame.
    """
    raise NotImplementedError


def compute_ltv_targets(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    horizon_months: int = 24,
) -> pd.DataFrame:
    """
    Compute two-stage LTV targets per borrower.

    A borrower "returns" if they take a new loan that was issued strictly
    after cutoff_date and before cutoff_date + horizon_months, AND that new
    loan started after their previous loan closed (no concurrent loans).

    Stage 1 target — will_return (binary):
        1 if borrower has a qualifying return loan, else 0.

    Stage 2 target — loan_amt_tier (3-class):
        Tertile of next_loan_amount among returning borrowers.
        0 = Low, 1 = Mid, 2 = High. Non-returners get -1.

    Args:
        df:             Cleaned DataFrame (all loans, all dates).
        cutoff_date:    Feature observation cutoff.
        horizon_months: Prediction window length in months.

    Returns:
        DataFrame with one row per borrower:
        member_id, will_return, loan_amt_tier, next_loan_amount.
    """
    raise NotImplementedError


def chronological_split(
    targets: pd.DataFrame,
    df: pd.DataFrame,
    train_frac: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split borrowers into train/test by their first loan issue date.

    The earliest train_frac of borrowers go to train; the rest to test.
    No random shuffling — preserves temporal ordering.

    Returns:
        (train_targets, test_targets)
    """
    raise NotImplementedError
