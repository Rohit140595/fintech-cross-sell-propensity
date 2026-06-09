"""Tests for src/features.py — stateless transforms and FrequencyEncoder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    FrequencyEncoder,
    add_age_features,
    add_interaction_features,
    add_log_transform,
    build_feature_matrix,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "Annual_Premium":       [10000.0, 50000.0, 0.0],
        "Previously_Insured":   [0, 1, 0],
        "Vehicle_Damage":       [1, 0, 1],
        "Age":                  [22, 40, 55],
        "Policy_Sales_Channel": [26, 152, 26],
        "Region_Code":          [28, 3, 11],
    })


# ── add_log_transform ─────────────────────────────────────────────────────────

def test_log_transform_adds_column(sample_df):
    result = add_log_transform(sample_df)
    assert "premium_log" in result.columns


def test_log_transform_correct_values(sample_df):
    result = add_log_transform(sample_df)
    expected = np.log1p(sample_df["Annual_Premium"])
    pd.testing.assert_series_equal(result["premium_log"], expected, check_names=False)


def test_log_transform_preserves_original(sample_df):
    result = add_log_transform(sample_df)
    assert "Annual_Premium" in result.columns
    pd.testing.assert_series_equal(result["Annual_Premium"], sample_df["Annual_Premium"])


def test_log_transform_does_not_mutate(sample_df):
    original = sample_df.copy()
    add_log_transform(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


def test_log_transform_zero_premium(sample_df):
    """log1p(0) should be 0.0, not NaN."""
    result = add_log_transform(sample_df)
    assert result.loc[sample_df["Annual_Premium"] == 0, "premium_log"].iloc[0] == 0.0


# ── add_interaction_features ──────────────────────────────────────────────────

def test_interaction_adds_columns(sample_df):
    result = add_interaction_features(sample_df)
    assert "not_insured_x_damage" in result.columns
    assert "age_x_vehicle_damage" in result.columns


def test_not_insured_x_damage_logic(sample_df):
    result = add_interaction_features(sample_df)
    # Row 0: Previously_Insured=0, Vehicle_Damage=1 → 1
    # Row 1: Previously_Insured=1, Vehicle_Damage=0 → 0
    # Row 2: Previously_Insured=0, Vehicle_Damage=1 → 1
    assert result["not_insured_x_damage"].tolist() == [1, 0, 1]


def test_age_x_vehicle_damage_logic(sample_df):
    result = add_interaction_features(sample_df)
    expected = (sample_df["Age"] * sample_df["Vehicle_Damage"]).tolist()
    assert result["age_x_vehicle_damage"].tolist() == expected


def test_interaction_does_not_mutate(sample_df):
    original = sample_df.copy()
    add_interaction_features(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


# ── add_age_features ──────────────────────────────────────────────────────────

def test_age_features_adds_columns(sample_df):
    result = add_age_features(sample_df)
    assert "is_young_driver" in result.columns
    assert "is_prime_age" in result.columns
    assert "premium_per_age" in result.columns


def test_is_young_driver_threshold(sample_df):
    result = add_age_features(sample_df)
    # Ages: 22, 40, 55 — only 22 < 25
    assert result["is_young_driver"].tolist() == [1, 0, 0]


def test_is_prime_age_range(sample_df):
    result = add_age_features(sample_df)
    # Ages: 22, 40, 55 — 40 and 55 are in [35, 55]
    assert result["is_prime_age"].tolist() == [0, 1, 1]


def test_premium_per_age_no_zero_division():
    """Even Age=0 should not raise ZeroDivisionError."""
    df = pd.DataFrame({"Age": [0], "Annual_Premium": [30000.0]})
    result = add_age_features(df)
    assert np.isfinite(result["premium_per_age"].iloc[0])


def test_age_features_does_not_mutate(sample_df):
    original = sample_df.copy()
    add_age_features(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


# ── build_feature_matrix ──────────────────────────────────────────────────────

def test_build_feature_matrix_adds_all_new_cols(sample_df):
    result = build_feature_matrix(sample_df)
    expected_new = [
        "premium_log",
        "not_insured_x_damage", "age_x_vehicle_damage",
        "is_young_driver", "is_prime_age", "premium_per_age",
    ]
    for col in expected_new:
        assert col in result.columns, f"Missing column: {col}"


def test_build_feature_matrix_preserves_originals(sample_df):
    result = build_feature_matrix(sample_df)
    for col in sample_df.columns:
        assert col in result.columns


def test_build_feature_matrix_no_nulls(sample_df):
    result = build_feature_matrix(sample_df)
    assert result.isnull().sum().sum() == 0


def test_build_feature_matrix_does_not_mutate(sample_df):
    original = sample_df.copy()
    build_feature_matrix(sample_df)
    pd.testing.assert_frame_equal(sample_df, original)


# ── FrequencyEncoder ──────────────────────────────────────────────────────────

@pytest.fixture()
def freq_data():
    train = pd.DataFrame({"channel": [1, 1, 2, 3], "region": [10, 20, 10, 30]})
    test  = pd.DataFrame({"channel": [1, 2, 99], "region": [10, 30, 99]})
    return train, test


def test_freq_encoder_fit_stores_maps(freq_data):
    train, _ = freq_data
    enc = FrequencyEncoder(cols=["channel"])
    enc.fit(train)
    assert "channel" in enc.freq_maps_
    assert enc.freq_maps_["channel"][1] == pytest.approx(0.5)


def test_freq_encoder_transform_replaces_values(freq_data):
    train, test = freq_data
    enc = FrequencyEncoder(cols=["channel"])
    enc.fit(train)
    result = enc.transform(train)
    # channel 1 appears 2/4 = 0.5 of the time
    assert result["channel"].iloc[0] == pytest.approx(0.5)


def test_freq_encoder_unknown_category_gets_zero(freq_data):
    train, test = freq_data
    enc = FrequencyEncoder(cols=["channel"])
    enc.fit(train)
    result = enc.transform(test)
    # channel 99 is unknown — should map to 0.0
    assert result["channel"].iloc[2] == 0.0


def test_freq_encoder_does_not_mutate_input(freq_data):
    train, _ = freq_data
    original = train.copy()
    enc = FrequencyEncoder(cols=["channel"])
    enc.fit(train)
    enc.transform(train)
    pd.testing.assert_frame_equal(train, original)


def test_freq_encoder_multiple_cols(freq_data):
    train, test = freq_data
    enc = FrequencyEncoder(cols=["channel", "region"])
    enc.fit(train)
    result = enc.transform(test)
    assert result["channel"].dtype == float
    assert result["region"].dtype == float


def test_freq_encoder_frequencies_sum_to_one(freq_data):
    train, _ = freq_data
    enc = FrequencyEncoder(cols=["channel"])
    enc.fit(train)
    total = sum(enc.freq_maps_["channel"].values())
    assert total == pytest.approx(1.0)
