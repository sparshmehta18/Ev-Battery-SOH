import { useState } from "react";

const initial = {
  battery_id: "MY_EV_01", cycle_number: 150,
  voltage_start: 4.2, voltage_end: 3.2, voltage_mean: 3.7, voltage_min: 3.1, voltage_max: 4.2,
  current_mean: -1.8, current_min: -2.0, current_max: -1.5,
  temperature_mean: 32, temperature_max: 38, temperature_min: 26,
  discharge_duration_s: 3200, energy_wh: 6.1,
  cum_capacity_fade: 0.4, capacity_degradation_rate: 0.005,
  rolling_capacity_mean_5: 1.6, rolling_capacity_std_5: 0.02, initial_capacity: 2.0,
  vehicle_info: {
    ev_model: "Tata Nexon EV", original_vehicle_price: 1500000,
    vehicle_age_years: 3, vehicle_mileage_km: 45000, battery_capacity_kwh: 30,
  },
};

export default function ManualInputForm({ onSubmit, isSubmitting = false }) {
  const [form, setForm] = useState(initial);

  const update = (path, value) => {
    setForm((prev) => {
      const next = { ...prev };
      if (path.startsWith("vehicle_info.")) {
        next.vehicle_info = { ...next.vehicle_info, [path.split(".")[1]]: value };
      } else {
        next[path] = value;
      }
      return next;
    });
  };

  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit(form); }}
          style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, maxWidth: 500 }}>
      <label>Battery age (cycles)
        <input type="number" value={form.cycle_number}
               onChange={(e) => update("cycle_number", Number(e.target.value))} />
      </label>
      <label>Current SOH-related fade (Ah)
        <input type="number" step="0.01" value={form.cum_capacity_fade}
               onChange={(e) => update("cum_capacity_fade", Number(e.target.value))} />
      </label>
      <label>Vehicle age (years)
        <input type="number" value={form.vehicle_info.vehicle_age_years}
               onChange={(e) => update("vehicle_info.vehicle_age_years", Number(e.target.value))} />
      </label>
      <label>Mileage (km)
        <input type="number" value={form.vehicle_info.vehicle_mileage_km}
               onChange={(e) => update("vehicle_info.vehicle_mileage_km", Number(e.target.value))} />
      </label>
      <label>Original price (₹)
        <input type="number" value={form.vehicle_info.original_vehicle_price}
               onChange={(e) => update("vehicle_info.original_vehicle_price", Number(e.target.value))} />
      </label>
      <label>Battery capacity (kWh)
        <input type="number" value={form.vehicle_info.battery_capacity_kwh}
               onChange={(e) => update("vehicle_info.battery_capacity_kwh", Number(e.target.value))} />
      </label>
      <button type="submit" disabled={isSubmitting} style={{ gridColumn: "span 2", padding: 10 }}>
        {isSubmitting ? "Generating report..." : "Generate Full Report"}
      </button>
    </form>
  );
}
