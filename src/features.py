"""
Feature engineering for LendingClub LTV prediction.

Four core feature families:
  - RFM          : Recency, Frequency, Monetary — adapted for lending
                   (months since last loan, number of loans, total borrowed).
  - Credit       : FICO, DTI, grade, revolving utilization — the strongest
                   predictors unique to lending data.
  - Behavior     : Repayment history (fully paid vs. charged off), payoff speed.
  - Profile      : Income, employment, home ownership, loan purpose, geography.

All features are computed using only loans issued strictly before cutoff_date.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_rfm_features(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    member_col: str = "member_id",
    date_col: str = "issue_d",
    amount_col: str = "loan_amnt",
) -> pd.DataFrame:
    """
    Compute lending-adapted RFM features per borrower.

    - Recency  : months since borrower's most recent loan before cutoff.
    - Frequency: total number of loans taken before cutoff.
    - Monetary : total amount borrowed before cutoff.
    - avg_loan_amount: monetary / frequency.

    Returns:
        DataFrame with one row per borrower.
    """
    raise NotImplementedError


def compute_credit_features(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    member_col: str = "member_id",
) -> pd.DataFrame:
    """
    Compute credit quality features per borrower (pre-cutoff loans only).

    These are the strongest signals unique to lending data — not available
    in e-commerce LTV models.

    Adds:
        avg_fico        : mean FICO score across pre-cutoff loans.
        avg_dti         : mean debt-to-income ratio.
        avg_grade       : mean LendingClub grade (A=1 … G=7, lower = safer).
        avg_int_rate    : mean interest rate offered.
        avg_revol_util  : mean revolving credit utilization.
        avg_open_acc    : mean number of open credit lines.
        avg_delinq_2yrs : mean number of delinquencies in past 2 years.
        credit_tenure_years : years from earliest credit line to cutoff.

    Returns:
        DataFrame with one row per borrower.
    """
    raise NotImplementedError


def compute_behavior_features(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    member_col: str = "member_id",
) -> pd.DataFrame:
    """
    Compute repayment behavior features per borrower (pre-cutoff loans only).

    Repayment history is a strong signal for repeat borrowing — borrowers
    who repaid cleanly and quickly are more likely to return.

    Adds:
        pct_fully_paid  : fraction of closed loans that were fully repaid
                          (vs. charged off or defaulted).
        ever_defaulted  : 1 if any loan was charged off or defaulted.
        avg_int_paid    : mean total interest paid (total_pymnt - loan_amnt).

    Returns:
        DataFrame with one row per borrower.
    """
    raise NotImplementedError


def compute_profile_features(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    member_col: str = "member_id",
) -> pd.DataFrame:
    """
    Compute borrower profile features from most recent pre-cutoff loan.

    Uses the most recent loan's values since these reflect the borrower's
    current financial situation most accurately.

    Adds:
        annual_inc       : reported annual income.
        emp_length       : employment length in years.
        home_ownership   : encoded home ownership status.
        top_purpose      : most frequent loan purpose (frequency encoded).
        pct_debt_consol  : fraction of loans for debt consolidation.
        state_freq       : frequency encoding of borrower's home state.

    Returns:
        DataFrame with one row per borrower.
    """
    raise NotImplementedError


def build_feature_matrix(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Assemble the full feature matrix by joining all feature families.

    Steps:
      1. RFM features (recency, frequency, monetary)
      2. Credit features (FICO, DTI, grade, utilization)
      3. Behavior features (repayment history, defaults)
      4. Profile features (income, purpose, geography)

    All features use only loans issued before cutoff_date — leak-free.

    Returns:
        DataFrame with one row per borrower and all engineered features.
    """
    raise NotImplementedError
