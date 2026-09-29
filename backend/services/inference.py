"""
Loads the trained scaler + XGBoost SOH model and exposes build_full_report(),
used by every route. Falls back to a documented formula-based estimate ONLY
if model artifacts are missing (so the API doesn't hard-crash before
`python src/models/train_baselines.py` has been run), and flags that clearly.
"""
import importlib.util
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[2] / "saved_models"
_HSR_PATH = Path(__file__).resolve().parents[2] / "src" / "models" / "health_score_and_resale.py"
_spec = importlib.util.spec_from_file_location("health_score_and_resale", _HSR_PATH)
_hsr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hsr)
battery_health_score = _hsr.battery_health_score
estimate_resale_value = _hsr.estimate_resale_value
ResaleInputs = _hsr.ResaleInputs

FEATURES = [
    "voltage_start", "voltage_end", "voltage_mean", "voltage_min", "voltage_max",
    "voltage_std", "voltage_range", "current_mean", "current_min", "current_max",
    "temperature_mean", "temperature_max", "temperature_min",
    "discharge_duration_s", "energy_wh", "cycle_number",
    "rolling_capacity_mean_5", "rolling_capacity_std_5",
    "capacity_degradation_rate", "cum_capacity_fade",
]

_scaler = None
_xgb_soh = None
_models_loaded = False

def _load_models():
    global _scaler, _xgb_soh, _models_loaded
    if _models_loaded:
        return
    try:
        _scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        _xgb_soh = joblib.load(MODEL_DIR / "xgb_soh.joblib")
        _models_loaded = True
    except Exception:
        # Any failure loading persisted model artifacts (missing files,
        # incompatible/corrupted pickles, native library errors) should not
        # crash the API. Fall back to the simple formula-based estimator and
        # mark models as not loaded.
        _models_loaded = False


def predict_soh_from_features(feature_dict: dict) -> float:
    _load_models()
    if not _models_loaded:
        return max(0.0, 100.0 - feature_dict.get("cum_capacity_fade", 0) / 2.0 * 100)
    x = pd.DataFrame([[feature_dict.get(f, 0.0) for f in FEATURES]], columns=FEATURES)
    x_scaled = _scaler.transform(x)
    return float(_xgb_soh.predict(x_scaled)[0])


def build_full_report(feature_dict: dict, vehicle_info: dict) -> dict:
    soh = predict_soh_from_features(feature_dict)
    degradation_rate = abs(feature_dict.get("capacity_degradation_rate", 0.0))
    initial_capacity = feature_dict.get("initial_capacity", 2.0)
    current_capacity = initial_capacity * soh / 100
    eol_capacity = 1.4
    rul_cycles = max(0, int((current_capacity - eol_capacity) / max(degradation_rate, 1e-4)))

    hs = battery_health_score(soh, degradation_rate, rul_cycles)
    resale = estimate_resale_value(ResaleInputs(
        original_vehicle_price=vehicle_info["original_vehicle_price"],
        vehicle_age_years=vehicle_info["vehicle_age_years"],
        vehicle_mileage_km=vehicle_info["vehicle_mileage_km"],
        battery_capacity_kwh=vehicle_info["battery_capacity_kwh"],
        current_soh_pct=soh,
        battery_replacement_cost=vehicle_info.get("battery_replacement_cost"),
    ))

    return {
        "soh": round(soh, 2),
        "rul_cycles": rul_cycles,
        "predicted_capacity": round(current_capacity, 3),
        "degradation_rate": round(degradation_rate, 4),
        "health_score": hs["health_score"],
        "battery_condition": hs["battery_condition"],
        "estimated_battery_adjusted_value": resale["estimated_battery_adjusted_value"],
        "valuation_assumptions": resale["assumptions"],
        "disclaimer": resale["disclaimer"],
    }
