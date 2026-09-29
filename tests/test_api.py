import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"

def test_full_report_endpoint_returns_expected_keys():
    payload = {
        "battery_id": "TESTX", "cycle_number": 100,
        "voltage_start": 4.2, "voltage_end": 3.3, "voltage_mean": 3.7,
        "voltage_min": 3.2, "voltage_max": 4.2,
        "current_mean": -1.8, "current_min": -2.0, "current_max": -1.5,
        "temperature_mean": 30, "temperature_max": 35, "temperature_min": 25,
        "discharge_duration_s": 3000, "energy_wh": 6.0,
        "cum_capacity_fade": 0.3, "capacity_degradation_rate": 0.004,
        "rolling_capacity_mean_5": 1.7, "rolling_capacity_std_5": 0.01,
        "initial_capacity": 2.0,
        "vehicle_info": {
            "original_vehicle_price": 1200000, "vehicle_age_years": 2,
            "vehicle_mileage_km": 20000, "battery_capacity_kwh": 30,
        },
    }
    resp = client.post("/predict/full-report", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    for key in ["soh", "rul_cycles", "predicted_capacity", "health_score",
                "battery_condition", "estimated_battery_adjusted_value"]:
        assert key in body
