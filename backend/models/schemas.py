from pydantic import BaseModel
from typing import Optional

class VehicleInfo(BaseModel):
    ev_model: Optional[str] = "Generic EV"
    original_vehicle_price: float
    vehicle_age_years: float
    vehicle_mileage_km: float
    battery_capacity_kwh: float
    battery_replacement_cost: Optional[float] = None

class ManualBatteryInput(BaseModel):
    battery_id: str
    cycle_number: int
    voltage_start: float
    voltage_end: float
    voltage_mean: float
    voltage_min: float
    voltage_max: float
    current_mean: float
    current_min: float
    current_max: float
    temperature_mean: float
    temperature_max: float
    temperature_min: float
    discharge_duration_s: float
    energy_wh: float
    cum_capacity_fade: float = 0.0
    capacity_degradation_rate: float = 0.0
    rolling_capacity_mean_5: float = 0.0
    rolling_capacity_std_5: float = 0.0
    initial_capacity: float = 2.0
    vehicle_info: VehicleInfo

class SOHRequest(BaseModel):
    features: dict

class FullReportResponse(BaseModel):
    soh: float
    rul_cycles: int
    predicted_capacity: float
    degradation_rate: float
    health_score: float
    battery_condition: str
    estimated_battery_adjusted_value: float
