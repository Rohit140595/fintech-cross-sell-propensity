# Fintech Cross-Sell Propensity Model

End-to-end ML pipeline predicting which existing health insurance customers are likely to purchase vehicle insurance — built to demonstrate Senior DS-level thinking for roles at SoFi / Robinhood.

---

## Business Problem

Insurance companies spend heavily on outreach campaigns. Calling every customer is expensive and drives churn. This model scores each customer's conversion likelihood so the sales team can **target the top 20% of customers and capture 58% of all converters** — 2.9x more efficient than random outreach.

---

## Results

| Segment | Conversion Rate | Lift | Converters Captured |
|---|---|---|---|
| Overall (baseline) | 12.3% | 1.0x | — |
| Top 10% by score | 40.0% | 3.3x | 33% |
| Top 20% by score | 31.6% | 2.6x | 58% |
| Top 30% by score | 25.5% | 2.1x | 79% |

**Model:** XGBoost (tuned with Optuna, 50 trials) | **PR-AUC:** 0.37 | **ROC-AUC:** 0.86

---

## Architecture

```
data/raw/train.csv
    │
    ▼
src/data.py          load → clean → stratified split
    │
    ▼
src/features.py      stateless transforms (log, interactions, age flags)
    │
    ▼
src/model.py         sklearn Pipeline:
                       FrequencyEncoder
                       → TargetEncoder (sklearn, cv=5)
                       → XGBoost
                     Optuna tuning (StratifiedKFold, PR-AUC metric)
                     save_model / load_model
    │
    ▼
models/cross_sell_model.pkl   (Pipeline + threshold)
    │
    ▼
src/predict.py       predict_single — inference for one customer
src/api.py           FastAPI — POST /predict, GET /health
```

---

## Key Design Decisions

**Leak-free encoding in CV**
`FrequencyEncoder` and `TargetEncoder` live inside the sklearn `Pipeline`. During Optuna's 5-fold CV, encoding statistics (frequencies, conversion rates) are refit on each training fold — the validation fold never leaks into encoding statistics.

**PR-AUC as primary metric**
With a 12.3% positive rate, ROC-AUC is misleading (a poor model can score 0.80+). PR-AUC measures precision-recall trade-off directly on the imbalanced class.

**F2-optimal threshold**
Threshold is chosen to maximise F2 score (recall weighted 2x over precision). Missing a likely converter is more costly than a wasted outreach call.

---

## Project Structure

```
├── data/
│   └── raw/                  # Kaggle Health Insurance Cross Sell dataset
├── notebooks/
│   ├── eda.ipynb             # EDA: class imbalance, feature signals
│   ├── model_training.ipynb  # Full training pipeline
│   └── shap_lift.ipynb       # SHAP explainability + lift/decile analysis
├── src/
│   ├── data.py               # Load, clean, split
│   ├── features.py           # Stateless transforms + FrequencyEncoder
│   ├── model.py              # Pipeline, tuning, evaluation, persistence
│   ├── predict.py            # Single-customer inference
│   └── api.py                # FastAPI endpoints
├── tests/
│   ├── test_data.py          # 8 tests
│   ├── test_features.py      # 24 tests
│   └── test_api.py           # 8 tests
├── config.yaml
├── requirements.txt
└── Dockerfile
```

---

## Quickstart

**Install dependencies**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Train the model**

Open `notebooks/model_training.ipynb` and run all cells. The fitted pipeline is saved to `models/cross_sell_model.pkl`.

**Run the API**
```bash
uvicorn src.api:app --reload
```

**Score a customer**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer": {
      "Gender": 1,
      "Age": 35,
      "Driving_License": 1,
      "Region_Code": 28,
      "Previously_Insured": 0,
      "Vehicle_Age": 1,
      "Vehicle_Damage": 1,
      "Annual_Premium": 40000.0,
      "Policy_Sales_Channel": 26,
      "Vintage": 217
    }
  }'
```

**Run tests**
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Dataset

[Health Insurance Cross Sell Prediction](https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction) — Kaggle

- 381K customers, 10 features, binary target (`Response`)
- 12.3% positive rate (class imbalance handled via `scale_pos_weight`)

---

## Experiment Tracking

MLflow tracks all runs under the `cross-sell-propensity` experiment.

```bash
mlflow ui --backend-store-uri mlruns/
```
