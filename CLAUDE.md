# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes.

## 1. Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them.
- If simpler approach exists, say so.

## 2. Simplicity First
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes
- Don't improve adjacent code unless asked.
- Match existing style.
- Remove only imports/variables YOUR changes made unused.

## 4. Goal-Driven Execution
- Transform tasks into verifiable goals.
- State a brief plan for multi-step tasks.

---

## Project: Fintech Cross-Sell Propensity

### Architecture
- `src/data.py`     — load CSV, clean, stratified split
- `src/features.py` — stateless transforms + FrequencyEncoder
- `src/model.py`    — sklearn Pipeline (FrequencyEncoder → TargetEncoder → XGBoost), Optuna tuning, evaluation, persistence
- `src/predict.py`  — single-customer inference (mirrors training pipeline)
- `src/api.py`      — FastAPI: POST /predict, GET /health

### Key constraints
- **Leak-free**: FrequencyEncoder and TargetEncoder live inside the sklearn Pipeline — encoding stats fit on train fold only during CV
- **Primary metric**: PR-AUC (not accuracy or ROC-AUC) — target is 12.3% positive rate
- **Threshold**: F2-optimal (recall weighted 2x over precision)
- **Model artifact**: Pipeline + threshold saved to `models/cross_sell_model.pkl`
- **No retraining in shap_lift.ipynb** — load model from `models/`

### Data facts
- Raw file: `data/raw/train.csv` (Kaggle Health Insurance Cross Sell Prediction)
- 381K customers, 10 features, binary target (`Response`)
- 12.3% positive rate

### Don't do
- Don't fit any encoder outside the Pipeline — always fit inside CV folds
- Don't use relative paths in notebooks — use `ROOT = Path("..").resolve()`
- Don't save raw CSVs or trained models to git (both gitignored)
- Don't retrain the model in `shap_lift.ipynb` — load from `models/`
