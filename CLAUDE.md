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

## Project-Specific: LendingClub LTV

### Architecture
- `src/data.py`     — load CSV, clean, compute targets, split
- `src/features.py` — RFM, credit, behavior, profile features (all leak-free)
- `src/model.py`    — two-stage: Stage 1 return classifier + Stage 2 tier classifier
- `src/predict.py`  — inference pipeline (mirrors training exactly)
- `src/api.py`      — FastAPI serving layer

### Key constraints
- **Cutoff date**: `2016-01-01` (in config.yaml)
- **Prediction horizon**: 24 months after cutoff
- **Leak-free**: all features use only loans issued strictly before cutoff_date
- **Valid statuses**: only closed loans (Fully Paid / Charged Off / Default)
  Active loans at cutoff have unknown outcome — including them leaks target info
- **Return definition**: borrower takes a NEW loan that starts AFTER their
  previous loan closed (no concurrent loan overlap)
- **Grain**: one row per member_id (borrower), not per loan

### Data facts (fill in after EDA)
- Raw file: `data/raw/accepted_2007_to_2018Q4.csv`
- Total loans: ~2.3M
- Unique borrowers: TBD
- Repeat borrowers: TBD (~15% expected)
- Cutoff-eligible borrowers: TBD

### Don't do
- Don't use `loan_status` of active loans as a feature (leaks target)
- Don't include loans issued after cutoff_date in feature computation
- Don't use `total_pymnt` for active loans (unknown final payment amount)
- Don't random-shuffle before splitting — use chronological split
- Don't save raw CSVs or trained models to git (both gitignored)
