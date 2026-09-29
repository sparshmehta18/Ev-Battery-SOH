"""
Parses the raw NASA PCoE Li-ion Battery Aging dataset (.mat files, MATLAB
struct format, as published by Saha & Goebel, NASA Ames PCoE) into a tidy
per-discharge-cycle CSV.

Source: NASA Prognostics Center of Excellence, Battery Data Set,
"Randomized Battery Usage" / "Battery Aging ARC-FY08Q4" collection.
Batteries B0005, B0006, B0007, B0018 were cycled (charge/discharge/impedance)
at room temperature (24C) until reaching 30% capacity fade (rated 2Ah -> 1.4Ah
end-of-life threshold), which is the standard EOL definition used in the
NASA PCoE literature (Saha & Goebel, 2007).

We do NOT fabricate any measurement here -- every row below is read directly
out of the original .mat structure.
"""
import scipy.io as sio
import numpy as np
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
RAW_DIR = repo_root / "data" / "raw"
OUT_DIR = repo_root / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BATTERY_IDS = ["B0005", "B0006", "B0007", "B0018"]
RATED_CAPACITY_AH = 2.0
EOL_CAPACITY_AH = 1.4  # 30% fade threshold used by NASA PCoE for this cell family

def matlab_datenum_or_vec_to_str(t):
    try:
        # NASA files store cycle start time as a 6-element date vector
        # [year, month, day, hour, min, sec]
        y, mo, d, h, mi, s = [float(x) for x in np.array(t).flatten()[:6]]
        return f"{int(y):04d}-{int(mo):02d}-{int(d):02d} {int(h):02d}:{int(mi):02d}:{s:05.2f}"
    except Exception:
        return None

def parse_battery(battery_id):
    mat = sio.loadmat(RAW_DIR / f"{battery_id}.mat", squeeze_me=False, struct_as_record=False)
    root = mat[battery_id][0, 0]
    cycles = root.cycle[0]

    rows = []
    discharge_idx = 0
    for cyc in cycles:
        ctype = str(cyc.type[0]) if hasattr(cyc.type, "__getitem__") else str(cyc.type)
        ambient_temp = float(np.array(cyc.ambient_temperature).flatten()[0])
        timestamp = matlab_datenum_or_vec_to_str(cyc.time)
        data = cyc.data[0, 0]

        if ctype == "discharge":
            discharge_idx += 1
            voltage = np.array(data.Voltage_measured).flatten().astype(float)
            current = np.array(data.Current_measured).flatten().astype(float)
            temperature = np.array(data.Temperature_measured).flatten().astype(float)
            time_s = np.array(data.Time).flatten().astype(float)
            capacity = float(np.array(data.Capacity).flatten()[0])

            if len(voltage) < 3:
                continue

            # discharge duration = time until voltage crosses below cutoff / end of record
            discharge_duration = float(time_s[-1] - time_s[0])
            # energy delivered (Wh), trapezoidal integration of V*I over time (I is negative during discharge)
            power_w = np.abs(voltage * current)
            energy_wh = float(np.trapezoid(power_w, time_s) / 3600.0)

            rows.append({
                "battery_id": battery_id,
                "cycle_number": discharge_idx,
                "timestamp": timestamp,
                "ambient_temperature": ambient_temp,
                "voltage_start": voltage[0],
                "voltage_end": voltage[-1],
                "voltage_mean": float(np.mean(voltage)),
                "voltage_min": float(np.min(voltage)),
                "voltage_max": float(np.max(voltage)),
                "voltage_std": float(np.std(voltage)),
                "current_mean": float(np.mean(current)),
                "current_min": float(np.min(current)),
                "current_max": float(np.max(current)),
                "temperature_mean": float(np.mean(temperature)),
                "temperature_max": float(np.max(temperature)),
                "temperature_min": float(np.min(temperature)),
                "discharge_duration_s": discharge_duration,
                "energy_wh": energy_wh,
                "capacity": capacity,
            })
    df = pd.DataFrame(rows)
    df["cycle_number"] = range(1, len(df) + 1)  # renumber sequential discharge cycles
    return df

def main():
    # If there are no .mat files, warn and exit gracefully so the rest of
    # the pipeline (which may already include processed CSVs) can continue.
    if not RAW_DIR.exists() or not any(RAW_DIR.glob('*.mat')):
        print(f"No .mat files found in {RAW_DIR}. Skipping parse step.")
        return

    all_frames = []
    for bid in BATTERY_IDS:
        df = parse_battery(bid)
        if df.empty:
            print(f"{bid}: no discharge cycles parsed")
            continue
        print(f"{bid}: {len(df)} discharge cycles parsed, "
              f"capacity range {df.capacity.min():.3f} -> {df.capacity.max():.3f} Ah")
        df.to_csv(OUT_DIR / f"{bid}_discharge_cycles.csv", index=False)
        all_frames.append(df)

    if not all_frames:
        print("No data parsed from any battery .mat files.")
        return

    full = pd.concat(all_frames, ignore_index=True)
    full.to_csv(OUT_DIR / "all_batteries_discharge_cycles.csv", index=False)
    print("\nSaved:", OUT_DIR / "all_batteries_discharge_cycles.csv", "-- total rows:", len(full))
    print(full.head())


if __name__ == '__main__':
    main()

