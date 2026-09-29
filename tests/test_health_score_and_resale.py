import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src" / "models"))
from health_score_and_resale import battery_health_score, estimate_resale_value, ResaleInputs

def test_health_score_healthy():
    r = battery_health_score(soh=95, degradation_rate_ah_per_cycle=0.001, rul_cycles=500)
    assert r["battery_condition"] == "Healthy"
    assert 0 <= r["health_score"] <= 100

def test_health_score_degraded():
    r = battery_health_score(soh=55, degradation_rate_ah_per_cycle=0.01, rul_cycles=5)
    assert r["battery_condition"] == "Degraded"

def test_health_score_bounds():
    r = battery_health_score(soh=150, degradation_rate_ah_per_cycle=0, rul_cycles=1000)
    assert r["health_score"] <= 100
    r2 = battery_health_score(soh=-10, degradation_rate_ah_per_cycle=0, rul_cycles=0)
    assert r2["health_score"] >= 0

def test_resale_value_non_negative():
    out = estimate_resale_value(ResaleInputs(
        original_vehicle_price=1_000_000, vehicle_age_years=10,
        vehicle_mileage_km=300000, battery_capacity_kwh=40, current_soh_pct=40,
    ))
    assert out["estimated_battery_adjusted_value"] >= 0

def test_resale_value_decreases_with_lower_soh():
    high_soh = estimate_resale_value(ResaleInputs(
        original_vehicle_price=1_500_000, vehicle_age_years=3,
        vehicle_mileage_km=45000, battery_capacity_kwh=30, current_soh_pct=95,
    ))
    low_soh = estimate_resale_value(ResaleInputs(
        original_vehicle_price=1_500_000, vehicle_age_years=3,
        vehicle_mileage_km=45000, battery_capacity_kwh=30, current_soh_pct=60,
    ))
    assert low_soh["estimated_battery_adjusted_value"] < high_soh["estimated_battery_adjusted_value"]
