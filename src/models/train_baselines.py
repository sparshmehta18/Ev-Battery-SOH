"""
Baseline models for SOH and RUL regression.

Splitting strategy: BATTERY-WISE holdout (not random row splitting).
Train batteries: B0005, B0007, B0018
Test battery:     B0006  (held out entirely -- the model never sees a single
                   cycle from B0006 during training or fitting)
This avoids leakage: rows from the same battery are highly autocorrelated
(consecutive cycles look almost identical), so a random row-level split would
let the model "memorize" a battery's trajectory instead of generalizing to an
unseen one.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import json

repo_root = Path(__file__).resolve().parents[2]
PROC_DIR = repo_root / "data" / "processed"
RESULTS_DIR = repo_root / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = repo_root / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "voltage_start", "voltage_end", "voltage_mean", "voltage_min", "voltage_max",
    "voltage_std", "voltage_range", "current_mean", "current_min", "current_max",
    "temperature_mean", "temperature_max", "temperature_min",
    "discharge_duration_s", "energy_wh", "cycle_number",
    "rolling_capacity_mean_5", "rolling_capacity_std_5",
    "capacity_degradation_rate", "cum_capacity_fade",
]

TEST_BATTERY = "B0006"

df = pd.read_csv(PROC_DIR / "features_labeled.csv")
train_df = df[df.battery_id != TEST_BATTERY].reset_index(drop=True)
test_df = df[df.battery_id == TEST_BATTERY].reset_index(drop=True)

print(f"Train rows: {len(train_df)} (batteries {sorted(train_df.battery_id.unique())})")
print(f"Test rows:  {len(test_df)} (battery {TEST_BATTERY})")

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURES])
X_test = scaler.transform(test_df[FEATURES])

results = {}

def evaluate(y_true, y_pred, target, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100
    results.setdefault(target, {})[model_name] = {
        "MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4), "MAPE": round(float(mape), 4),
    }
    print(f"[{target}] {model_name:18s} MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} MAPE={mape:.2f}%")

for target in ["soh", "rul"]:
    y_train = train_df[target].values
    y_test = test_df[target].values

    lr = LinearRegression().fit(X_train, y_train)
    evaluate(y_test, lr.predict(X_test), target, "Linear Regression")

    rf = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    evaluate(y_test, rf.predict(X_test), target, "Random Forest")

    xgm = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgm.fit(X_train, y_train)
    evaluate(y_test, xgm.predict(X_test), target, "XGBoost")

    if target == "soh":
        import joblib
        joblib.dump(rf, MODEL_DIR / "rf_soh.joblib")
        joblib.dump(xgm, MODEL_DIR / "xgb_soh.joblib")
        joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
        # store feature importance for XAI section
        importances = dict(zip(FEATURES, xgm.feature_importances_.tolist()))
        json.dump(importances, open(RESULTS_DIR / "xgb_soh_feature_importance.json", "w"), indent=2)

with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved metrics to", RESULTS_DIR / "baseline_metrics.json")
