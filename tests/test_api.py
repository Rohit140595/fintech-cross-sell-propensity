"""Tests for src/api.py and src/predict.py."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from src.api import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_CUSTOMER = {
    "Gender":               1,
    "Age":                  35,
    "Driving_License":      1,
    "Region_Code":          28,
    "Previously_Insured":   0,
    "Vehicle_Age":          1,
    "Vehicle_Damage":       1,
    "Annual_Premium":       40000.0,
    "Policy_Sales_Channel": 26,
    "Vintage":              217,
}


def _make_mock_artifact(prob: float = 0.75, threshold: float = 0.45) -> dict:
    """Return a mock artifact dict with a pipeline that returns a fixed prob."""
    mock_pipeline = MagicMock()
    mock_pipeline.predict_proba.return_value = np.array([[1 - prob, prob]])
    return {"pipeline": mock_pipeline, "threshold": threshold}


@contextmanager
def _patched_client(artifact: dict):
    """
    Context manager that yields a TestClient with load_model patched.
    The lifespan runs inside the context manager, populating _store with
    the mock artifact instead of reading from disk.
    """
    with patch("src.api.load_model", return_value=artifact):
        with TestClient(app) as client:
            yield client


# ── Health endpoint ───────────────────────────────────────────────────────────

def test_health_model_loaded():
    """Health returns model_loaded=True after successful startup."""
    with _patched_client(_make_mock_artifact()) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True


def test_health_status_ok():
    """Health always returns status=ok."""
    with _patched_client(_make_mock_artifact()) as client:
        resp = client.get("/health")
    assert resp.json()["status"] == "ok"


# ── Predict endpoint ──────────────────────────────────────────────────────────

def test_predict_returns_200():
    """Valid request returns 200 with probability, prediction, threshold."""
    with _patched_client(_make_mock_artifact(prob=0.75, threshold=0.45)) as client:
        resp = client.post("/predict", json={"customer": VALID_CUSTOMER})
    assert resp.status_code == 200
    body = resp.json()
    assert "probability" in body
    assert "prediction" in body
    assert "threshold" in body


def test_predict_probability_range():
    """Probability is always between 0 and 1."""
    with _patched_client(_make_mock_artifact(prob=0.75, threshold=0.45)) as client:
        resp = client.post("/predict", json={"customer": VALID_CUSTOMER})
    assert 0.0 <= resp.json()["probability"] <= 1.0


def test_predict_above_threshold_returns_1():
    """Prediction is 1 when probability exceeds threshold."""
    with _patched_client(_make_mock_artifact(prob=0.75, threshold=0.45)) as client:
        resp = client.post("/predict", json={"customer": VALID_CUSTOMER})
    assert resp.json()["prediction"] == 1


def test_predict_below_threshold_returns_0():
    """Prediction is 0 when probability is below threshold."""
    with _patched_client(_make_mock_artifact(prob=0.20, threshold=0.45)) as client:
        resp = client.post("/predict", json={"customer": VALID_CUSTOMER})
    assert resp.json()["prediction"] == 0


def test_predict_422_on_missing_fields():
    """Returns 422 when required customer fields are missing."""
    with _patched_client(_make_mock_artifact()) as client:
        resp = client.post("/predict", json={"customer": {"Age": 35}})
    assert resp.status_code == 422


def test_predict_422_on_invalid_field_value():
    """Returns 422 when a field value is out of the valid range."""
    bad_customer = {**VALID_CUSTOMER, "Gender": 99}  # Gender must be 0 or 1
    with _patched_client(_make_mock_artifact()) as client:
        resp = client.post("/predict", json={"customer": bad_customer})
    assert resp.status_code == 422
