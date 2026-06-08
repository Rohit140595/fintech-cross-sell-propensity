"""
FastAPI serving layer for LendingClub LTV scoring.

Endpoints:
  POST /predict  — score a single borrower from their loan history
  GET  /health   — liveness check
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import load_model
from src.predict import predict_single


class LoanRecord(BaseModel):
    """One loan in a borrower's history."""
    issue_d:         datetime
    loan_amnt:       float = Field(..., ge=0)
    int_rate:        float = Field(..., ge=0, le=100)
    grade:           Optional[str] = None
    sub_grade:       Optional[str] = None
    purpose:         Optional[str] = None
    annual_inc:      Optional[float] = None
    dti:             Optional[float] = None
    fico_range_low:  Optional[float] = None
    fico_range_high: Optional[float] = None
    revol_util:      Optional[float] = None
    open_acc:        Optional[int]   = None
    loan_status:     Optional[str]   = None
    total_pymnt:     Optional[float] = None
    addr_state:      Optional[str]   = None


class PredictRequest(BaseModel):
    member_id:   str
    loans:       list[LoanRecord] = Field(..., min_length=1)
    cutoff_date: datetime


class PredictResponse(BaseModel):
    member_id:               str
    will_return_probability: float
    loan_tier:               str     # Non-returner | Low | Mid | High
    cutoff_date:             datetime


class HealthResponse(BaseModel):
    status: str
    loaded: bool


_store: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _store["artifact"] = load_model("models/ltv_model.pkl")
    print(f"Model loaded — threshold={_store['artifact']['threshold']:.4f}")
    yield
    _store.clear()


app = FastAPI(
    title="LendingClub LTV Prediction API",
    description="Two-stage LTV scoring: return probability + loan amount tier.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", loaded="artifact" in _store)


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if "artifact" not in _store:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    loans  = [loan.model_dump() for loan in request.loans]
    cutoff = request.cutoff_date

    try:
        result = predict_single(
            member_id=request.member_id,
            loans=loans,
            cutoff_date=cutoff,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return PredictResponse(
        member_id=request.member_id,
        will_return_probability=result["will_return_probability"],
        loan_tier=result["loan_tier"],
        cutoff_date=request.cutoff_date,
    )
