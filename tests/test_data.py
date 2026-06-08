"""Tests for src/data.py — loading, cleaning, and splitting."""

import pandas as pd
import pytest

from src.data import clean, split


@pytest.fixture()
def raw_df():
    return pd.DataFrame({
        "id":                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Gender":                ["Male", "Female", "Male", "Female", "Male",
                                  "Male", "Female", "Male", "Female", "Male"],
        "Age":                   [30, 45, 25, 60, 35, 40, 55, 28, 50, 33],
        "Driving_License":       [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        "Region_Code":           [28.0, 3.0, 11.0, 28.0, 41.0,
                                  28.0, 3.0, 11.0, 28.0, 41.0],
        "Previously_Insured":    [0, 1, 0, 0, 1, 0, 1, 0, 0, 1],
        "Vehicle_Age":           ["< 1 Year", "1-2 Year", "> 2 Years", "< 1 Year", "1-2 Year",
                                  "< 1 Year", "1-2 Year", "> 2 Years", "< 1 Year", "1-2 Year"],
        "Vehicle_Damage":        ["Yes", "No", "Yes", "No", "Yes",
                                  "Yes", "No", "Yes", "No", "Yes"],
        "Annual_Premium":        [40000.0, 33000.0, 28000.0, 45000.0, 30000.0,
                                  40000.0, 33000.0, 28000.0, 45000.0, 30000.0],
        "Policy_Sales_Channel":  [26.0, 26.0, 13.0, 26.0, 152.0,
                                  26.0, 26.0, 13.0, 26.0, 152.0],
        "Vintage":               [217, 183, 100, 50, 300, 217, 183, 100, 50, 300],
        "Response":              [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    })


def test_clean_encodes_gender(raw_df):
    df = clean(raw_df)
    assert set(df["Gender"].unique()) <= {0, 1}
    assert df.loc[raw_df["Gender"] == "Male", "Gender"].eq(1).all()
    assert df.loc[raw_df["Gender"] == "Female", "Gender"].eq(0).all()


def test_clean_encodes_vehicle_damage(raw_df):
    df = clean(raw_df)
    assert set(df["Vehicle_Damage"].unique()) <= {0, 1}
    assert df.loc[raw_df["Vehicle_Damage"] == "Yes", "Vehicle_Damage"].eq(1).all()


def test_clean_encodes_vehicle_age(raw_df):
    df = clean(raw_df)
    assert set(df["Vehicle_Age"].unique()) <= {0, 1, 2}
    assert df.loc[raw_df["Vehicle_Age"] == "< 1 Year",  "Vehicle_Age"].eq(0).all()
    assert df.loc[raw_df["Vehicle_Age"] == "1-2 Year",  "Vehicle_Age"].eq(1).all()
    assert df.loc[raw_df["Vehicle_Age"] == "> 2 Years", "Vehicle_Age"].eq(2).all()


def test_clean_no_string_columns(raw_df):
    df = clean(raw_df)
    string_cols = [c for c in df.columns if df[c].dtype == object]
    assert string_cols == [], f"String columns remain: {string_cols}"


def test_clean_no_missing_values(raw_df):
    df = clean(raw_df)
    assert df.isnull().sum().sum() == 0


def test_split_sizes(raw_df):
    df = clean(raw_df)
    train, test = split(df, test_size=0.2, random_state=42)
    assert len(train) + len(test) == len(df)
    assert len(test) == pytest.approx(len(df) * 0.2, abs=1)


def test_split_preserves_positive_rate(raw_df):
    df = clean(raw_df)
    train, test = split(df, test_size=0.2, random_state=42)
    overall = df["Response"].mean()
    assert abs(train["Response"].mean() - overall) < 0.1
    assert abs(test["Response"].mean() - overall) < 0.1


def test_split_no_overlap(raw_df):
    df = clean(raw_df)
    train, test = split(df, test_size=0.2, random_state=42)
    assert set(train["id"]).isdisjoint(set(test["id"]))
