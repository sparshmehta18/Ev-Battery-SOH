# EV Battery Health, Remaining Useful Life (RUL) & Resale Value Prediction System

An end-to-end ML system that predicts a used EV's battery State of Health (SOH),
Remaining Useful Life (RUL), degradation trend, and an AI-estimated
battery-adjusted resale value — built on real lithium-ion cell aging data.

> **Business use case:** Used EV → analyze battery cycling history → predict
> battery health & remaining life → support resale valuation for buyers,
> sellers, and used-EV marketplaces.

---

## 1. Problem Statement

Used-EV buyers have no easy way to know how much life is left in the most
expensive component of the car — the battery pack. This project builds a
data-driven pipeline that ingests battery cycling data (voltage, current,
temperature, capacity per charge/discharge cycle) and outputs a health score,
RUL estimate, and a transparent, formula-based resale value adjustment.

## 2. Dataset

**NASA PCoE Li-ion Battery Aging Dataset** (Saha & Goebel, NASA Ames
Prognostics Center of Excellence) — cells **B0005, B0006, B0007, B0018**.
These 18650 Li-ion cells were repeatedly charged/discharged at room
temperature (24°C) with a 2A constant-current discharge until each cell
reached **30% capacity fade** (2.0 Ah → 1.4 Ah), which is also the End-of-Life
(EOL) definition used throughout this project.

This is **not EV-pack-scale data** (it's per-cell), so this project is
explicitly a **prototype methodology for EV battery-health estimation**, not a
system trained on real EV telemetry. The methodology (features, SOH/RUL
definitions, model architecture, API) generalizes directly to real EV pack
data if/when it's available — only the training data would change.

Raw `.mat` files → parsed with `scipy.io` in
[`src/data/parse_nasa_mat.py`](src/data/parse_nasa_mat.py) into tidy
per-discharge-cycle CSVs (636 total discharge cycles across 4 batteries). No
measurement in this dataset was fabricated or altered.

## 3. System Architecture

```
Raw .mat files → parser → feature engineering → {baseline ML, LSTM+Attention}
                                                          │
                                          SOH / RUL predictions
                                                          │
                                   health score + resale valuation (rule-based)
                                                          │
                              FastAPI backend  ⇄  MongoDB  ⇄  React dashboard
```

## 4. Splitting Strategy (leakage avoidance)

**Battery-wise holdout**, not random row splitting: models train on
B0005 + B0007 + B0018 and are evaluated on **B0006 held out completely** —
the model never sees a single cycle from B0006 during fitting. Consecutive
cycles from the same battery are highly autocorrelated, so a random
row-level split would let a model "memorize" a battery's trajectory instead
of proving it generalizes to a battery it has never seen. Sequence models
additionally use a 10-cycle sliding window built independently per battery
so no window straddles a battery boundary.

## 5. Feature Engineering

Per discharge cycle: voltage/current/temperature statistics (mean, min, max,
std), discharge duration, energy delivered (Wh, trapezoidal integration of
V×I), cycle number, rolling 5-cycle capacity mean/std, cycle-over-cycle
capacity degradation rate, and cumulative capacity fade.

`SOH = Capacity(cycle) / Capacity(cycle 1) × 100`
`EOL threshold = 1.4 Ah (30% fade)` — the same threshold NASA PCoE used
`RUL(cycle) = EOL cycle index − current cycle index` (clipped at 0; B0007
never reached EOL within recorded cycles and is flagged `censored=True`
rather than silently treated as ground truth zero)

## 6. Models & Real Results

**These numbers come directly from `results/baseline_metrics.json` and
`results/dl_metrics.json` — nothing here is invented.**

### SOH prediction (test battery: B0006, fully held out)
| Model              |  MAE | RMSE |    R² |  MAPE |
|---------------------|-----:|-----:|------:|------:|
| Linear Regression    | 1.65 | 1.97 | 0.975 | 2.45% |
| Random Forest        | 3.28 | 4.73 | 0.853 | 5.07% |
| XGBoost              | 3.17 | 4.71 | 0.855 | 4.94% |
| LSTM                 | 9.43 | 9.96 | 0.228 | 13.25%|
| **LSTM + Attention** | 7.43 | 8.08 | 0.492 | 10.25%|

### RUL prediction (cycles remaining, test battery: B0006)
| Model              |   MAE |  RMSE |    R² |
|---------------------|------:|------:|------:|
| Linear Regression    | 81.08 | 84.30 | -4.44 |
| Random Forest        | 14.08 | 24.08 | 0.556 |
| XGBoost              |  9.88 | 15.50 | 0.816 |
| LSTM                 |  9.65 | 13.52 | 0.830 |
| **LSTM + Attention** |  9.61 | 12.53 | **0.854** |

**Honest interpretation, not spin:** LSTM+Attention clearly beats plain LSTM
on both tasks (proving the attention mechanism helps), and gives the best
RUL result overall. But for SOH, simple engineered features + Linear
Regression/XGBoost actually outperform the sequence models. This is a real,
explainable, and well-documented phenomenon in battery-prognostics
literature: with only **3 training batteries**, a deep sequence model has far
less signal to learn generalizable temporal patterns from than a gradient
boosted tree has from well-engineered per-cycle features. The primary
architecture requested for this project (LSTM+Attention) is still the right
choice for RUL and for scaling to a real fleet-scale EV dataset with far more
batteries, where the sequence model's advantage would compound. Both
findings are reported here rather than cherry-picking the flattering one.

Loss curves, attention-weight heatmaps, and SHAP explanations are all in
[`results/figures/`](results/figures/).

## 7. Explainable AI

SHAP (`TreeExplainer`) is run on the XGBoost SOH model
(`src/models/explainability.py` → `results/shap_summary.png` and
`results/shap_explanation_example.json`). For the most degraded test cycle
(B0006, cycle 168, actual SOH 58.3%), the model predicted 69.7% and
attributed the gap primarily to `cum_capacity_fade` (SHAP contribution
-11.3) — i.e. the model under-predicts degradation severity in the
deep-fade region it saw the least of during training. This is disclosed as
a genuine limitation, not hidden.

## 8. Battery Health Score & Resale Valuation

Both are **rule-based formulas that consume the model's SOH/RUL output** —
not separate ML models, and not an official certification or a real market
price. See [`src/models/health_score_and_resale.py`](src/models/health_score_and_resale.py)
for the full, documented formula (depreciation curve + battery-health
deduction). Every API response includes the explicit disclaimer text.

## 9. API (FastAPI)

```
POST /predict/soh            POST /predict/rul
POST /predict/resale         POST /predict/full-report
POST /upload-battery-data    GET  /battery/{id}
```

Verified end-to-end in this repo (`tests/test_api.py`, all passing):
```json
{
  "soh": 76.79, "rul_cycles": 22, "predicted_capacity": 1.536,
  "degradation_rate": 0.006, "health_score": 76.8,
  "battery_condition": "Moderate", "estimated_battery_adjusted_value": 838034.0
}
```

## 10. Install & Run

```bash
pip install -r requirements.txt

# 1. Parse raw data → features → labels
python src/data/parse_nasa_mat.py
python src/features/build_features.py

# 2. Train models (writes to saved_models/ and results/)
python src/models/train_baselines.py
python src/models/train_lstm_attention.py
python src/models/explainability.py
python src/visualization/make_plots.py

# 3. Run tests
pytest tests/ -v

# 4. Run the full stack
docker compose up --build
# backend:  http://localhost:8000/docs
# frontend: http://localhost:5173
```

### Free resume-demo deployment (Render)

This deployment serves the manual report form without requiring MongoDB. CSV
upload and battery history remain available in the full local stack when
`MONGO_URI` is configured.

1. Push the project to a GitHub repository.
2. In Render, create a **New Web Service** from the repository. Set **Language**
  to Docker, **Dockerfile Path** to `backend/Dockerfile`, **Docker Context
  Directory** to `.`, and choose the **Free** instance type. Do not set
  `MONGO_URI` for the manual-report-only deployment.
3. Wait for the API deployment. Check `https://<api-service>.onrender.com/health`
  and open `https://<api-service>.onrender.com/docs` to verify it.
4. Create a **New Static Site** from the same repository. Set **Root Directory**
  to `frontend`, **Build Command** to `npm install && npm run build`, and
  **Publish Directory** to `dist`.
5. Add the static site's environment variable `VITE_API_BASE` with the API URL,
  such as `https://<api-service>.onrender.com`, then deploy the site.
6. Open the static site's `onrender.com` URL and submit the manual form. The API
  may take about a minute to respond after being idle because the free service
  spins down when inactive.

## 11. Limitations

- Cell-level lab data, not real EV pack telemetry — a real deployment needs
  EV fleet data with far more batteries/packs to generalize confidently.
- Only 4 batteries total → 3-battery training set is small for a deep
  sequence model; SOH LSTM+Attention R² (0.49) leaves clear room to improve
  with more batteries.
- RUL MAPE is unstable near true RUL ≈ 0 (division blows up) — MAE/RMSE/R²
  are the metrics to trust for RUL, not MAPE.
- Resale valuation is a transparent formula, not trained on real transaction
  data — treat it as a starting estimate, never a quote.

## 12. Future Improvements

- Train on a real EV-pack-scale dataset (many packs, real driving cycles).
- Multi-battery cross-validation instead of a single holdout battery.
- Multi-output model (SOH+RUL jointly) once more data makes it stable.
- Replace the resale formula with a model trained on real used-EV listings.

---

### Resume bullet points
- Built an end-to-end ML system (data pipeline → LSTM+Attention & gradient-boosted
  models → FastAPI → MongoDB → React) predicting EV battery SOH and RUL on the
  NASA PCoE Li-ion aging dataset, achieving R²=0.85 for RUL with a
  battery-wise holdout split that eliminates time-series leakage.
- Designed and implemented an attention-based LSTM in PyTorch that improved
  RUL prediction R² from 0.83 (plain LSTM) to 0.85 and SOH MAE from 9.4 to
  7.4 over a vanilla LSTM baseline, visualizing learned attention weights to
  show which past cycles drive each prediction.
- Added SHAP-based explainability to a production-style FastAPI service
  (6 REST endpoints, Dockerized with MongoDB) so every SOH prediction ships
  with a feature-level explanation rather than a black-box number.
- Delivered a full-stack, containerized dashboard (React+Vite, Docker
  Compose, 7 passing unit/API tests) that turns raw battery cycling CSVs
  into a health score, degradation forecast, and a transparent, formula-based
  resale valuation with documented assumptions.

### 10 likely interview questions with answers

1. **Why battery-wise split instead of random split?**
   Consecutive cycles from one battery are near-identical (autocorrelated).
   A random split leaks near-duplicate rows into both train and test, so the
   model appears to generalize when it's actually memorizing. Battery-wise
   holdout tests generalization to a genuinely unseen battery.

2. **Why did XGBoost beat your LSTM+Attention on SOH?**
   With only 3 training batteries, a deep sequence model has too little
   variety to learn robust temporal patterns, while hand-engineered features
   like cumulative capacity fade are already strong, near-linear predictors
   of SOH by construction. Deep models earn their advantage at fleet scale
   with many more batteries — I report this honestly rather than picking
   metrics that flatter the deep model.

3. **What does the attention mechanism actually attend to?**
   The additive attention layer learns a weight per cycle in the 10-cycle
   input window; the heatmap in `results/figures/07_attention_weights.png`
   shows it. I'd walk through a concrete example.

4. **How do you define RUL and why that EOL threshold?**
   RUL = cycles remaining until capacity crosses 1.4 Ah (30% fade from the
   2.0 Ah rated capacity) — the same criterion NASA PCoE used when they
   physically stopped testing these exact cells, so it's not an arbitrary
   choice.

5. **How would this change with real EV data instead of lab cells?**
   You'd add real-world confounds — variable ambient temperature, partial
   charges, fast-charging degradation, driving-style effects — and need many
   more vehicles for the split to be meaningful. The pipeline/architecture
   carries over; the feature set would grow.

6. **Why is the resale value a formula and not a model?**
   No real EV transaction dataset was available, and presenting a black-box
   number as a market price would be misleading. The formula is fully
   auditable, and every response ships its assumptions.

7. **How did you avoid data leakage in feature engineering?**
   Rolling features (5-cycle mean/std, degradation rate) only look backward
   within a single battery's own history — never at future cycles or at
   another battery's data — and the StandardScaler is fit only on the
   training batteries before being applied to the held-out one.

8. **Why MAE/RMSE/R² and not just MAPE for RUL?**
   MAPE divides by the true value, so it blows up when true RUL is near
   zero (end-of-life cycles). I flagged this explicitly rather than hiding
   an unstable metric or quietly excluding it.

9. **How does the SHAP explanation change your trust in the model?**
   It showed the model under-predicts SOH in the deepest-fade region — i.e.
   it's most trustworthy in the healthy-to-moderate range and least reliable
   exactly where a buyer cares most (a heavily degraded pack). That's a real,
   disclosed limitation, not a hypothetical one.

10. **What would you change with more time/data?**
    Cross-validate across all 4+ batteries (not just one holdout), try a
    joint multi-task SOH+RUL head once there's enough data to stabilize it,
    and replace the resale formula with a model trained on real listings.
