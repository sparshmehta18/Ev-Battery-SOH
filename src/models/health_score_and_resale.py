"""
Battery Health Score (0-100) and Estimated Battery-Adjusted Resale Value.

Both of these are RULE-BASED / FORMULA modules that consume the ML model's
outputs (SOH, RUL) plus user-provided vehicle facts. They are NOT separate
ML models, and are explicitly documented as such -- neither is an official
industry certification nor a real market price.
"""
from dataclasses import dataclass

# ---- Health Score -----------------------------------------------------
# Thresholds documented and defensible: mirrors common EV OEM warranty
# language, where 70% SOH is the typical battery-replacement warranty floor.
def battery_health_score(soh: float, degradation_rate_ah_per_cycle: float, rul_cycles: float) -> dict:
    # Base score = SOH directly (0-100 scale by construction)
    score = soh
    # Penalize fast recent degradation (rate is Ah lost per cycle; typical
    # healthy cells lose <0.003 Ah/cycle early on, our data shows up to ~0.01)
    if degradation_rate_ah_per_cycle > 0.006:
        score -= 5
    # Penalize very low remaining useful life independent of current SOH
    if rul_cycles < 20:
        score -= 5
    score = max(0, min(100, score))

    if score >= 80:
        band, emoji = "Healthy", "🟢"
    elif score >= 60:
        band, emoji = "Moderate", "🟡"
    else:
        band, emoji = "Degraded", "🔴"

    return {"health_score": round(float(score), 1), "battery_condition": band, "indicator": emoji}


# ---- Resale valuation ---------------------------------------------------
@dataclass
class ResaleInputs:
    original_vehicle_price: float
    vehicle_age_years: float
    vehicle_mileage_km: float
    battery_capacity_kwh: float
    current_soh_pct: float
    battery_replacement_cost: float = None  # optional override

def estimate_resale_value(inputs: ResaleInputs) -> dict:
    """
    AI-BASED ESTIMATED VALUATION ONLY -- not a real market price.

    Method (transparent, explainable, NOT a black box):
      1. Start from a standard age/mileage depreciation curve applied to the
         original price (non-battery depreciation).
      2. Apply an additional battery-health-adjusted deduction: the fraction
         of the pack's health that has been "used up" (100 - SOH) is charged
         against a stated battery replacement cost, scaled down because a
         used pack at, say, 80% SOH is not worthless -- it still has years
         of life -- so we only charge the buyer for the fraction of
         replacement cost proportional to lost health, discounted 50% to
         reflect that full replacement is rarely needed immediately.
    """
    p = inputs
    # 1) standard depreciation: ~15%/yr declining balance, plus mileage penalty
    age_factor = (0.85) ** p.vehicle_age_years
    mileage_penalty = min(0.25, p.vehicle_mileage_km / 200000 * 0.25)
    base_value = p.original_vehicle_price * age_factor * (1 - mileage_penalty)

    # 2) battery health adjustment
    replacement_cost = p.battery_replacement_cost or (p.battery_capacity_kwh * 9000)  # INR/kWh assumption, stated
    health_lost_fraction = max(0.0, (100 - p.current_soh_pct) / 100)
    battery_deduction = replacement_cost * health_lost_fraction * 0.5

    estimated_value = max(0.0, base_value - battery_deduction)

    return {
        "estimated_battery_adjusted_value": round(estimated_value, 0),
        "assumptions": {
            "depreciation_model": "15%/yr declining balance + mileage penalty (max 25% at 200,000km)",
            "battery_replacement_cost_used": round(replacement_cost, 0),
            "battery_replacement_cost_basis": "INR 9,000/kWh assumption -- override with real quote if available",
            "health_lost_fraction": round(health_lost_fraction, 3),
            "battery_deduction_applied": round(battery_deduction, 0),
        },
        "disclaimer": ("AI-based estimated valuation only. Actual resale price depends on "
                        "vehicle model, location, demand, accident history, service record, "
                        "warranty status and negotiation -- this figure is not a market quote."),
    }

if __name__ == "__main__":
    hs = battery_health_score(soh=69.7, degradation_rate_ah_per_cycle=0.008, rul_cycles=15)
    print("Health score example:", hs)
    rv = estimate_resale_value(ResaleInputs(
        original_vehicle_price=1_500_000, vehicle_age_years=3, vehicle_mileage_km=45000,
        battery_capacity_kwh=40, current_soh_pct=69.7,
    ))
    print("Resale example:", rv)
