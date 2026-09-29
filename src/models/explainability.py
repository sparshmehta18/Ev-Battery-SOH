"""
Explainable AI for the SOH model: SHAP values on the XGBoost SOH regressor.
"""
import sys
from unittest.mock import MagicMock
sys.modules['numba'] = MagicMock()
sys.modules['numba.typed'] = MagicMock()
def _dummy_jit(*args, **kwargs):
    if len(args) == 1 and callable(args[0]): return args[0]
    return lambda f: f
sys.modules['numba'].njit = _dummy_jit
sys.modules['numba'].jit = _dummy_jit

import shap, joblib, json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
PROC_DIR = repo_root / "data" / "processed"
MODEL_DIR = repo_root / "saved_models"
RESULTS_DIR = repo_root / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "voltage_start", "voltage_end", "voltage_mean", "voltage_min", "voltage_max",
    "voltage_std", "voltage_range", "current_mean", "current_min", "current_max",
    "temperature_mean", "temperature_max", "temperature_min",
    "discharge_duration_s", "energy_wh", "cycle_number",
    "rolling_capacity_mean_5", "rolling_capacity_std_5",
    "capacity_degradation_rate", "cum_capacity_fade",
]

df = pd.read_csv(PROC_DIR / "features_labeled.csv")
test_df = df[df.battery_id == "B0006"].reset_index(drop=True)
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
xgm = joblib.load(MODEL_DIR / "xgb_soh.joblib")

X_test = scaler.transform(test_df[FEATURES])
explainer = shap.TreeExplainer(xgm)
shap_values = explainer(X_test)

fig = plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, features=X_test, feature_names=FEATURES, show=False)
plt.title("SHAP Summary — SOH Prediction (XGBoost)")
plt.tight_layout()
plt.savefig(FIG_DIR / "09_shap_summary.png", dpi=140)
plt.close(fig)

# per-cycle explanation for one example (last test cycle -> most degraded)
idx = len(test_df) - 1
contrib = dict(zip(FEATURES, shap_values.values[idx].tolist()))
sorted_contrib = dict(sorted(contrib.items(), key=lambda x: -abs(x[1])))
explanation = {
    "battery_id": "B0006",
    "cycle_number": int(test_df.loc[idx, "cycle_number"]),
    "predicted_soh": float(xgm.predict(X_test[idx:idx+1])[0]),
    "actual_soh": float(test_df.loc[idx, "soh"]),
    "top_feature_contributions": sorted_contrib,
}
json.dump(explanation, open(RESULTS_DIR / "shap_explanation_example.json", "w"), indent=2)
print(json.dumps(explanation, indent=2))
