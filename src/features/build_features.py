"""
Feature engineering + SOH/RUL labeling for the NASA PCoE battery dataset.

SOH definition (documented, standard in the NASA PCoE literature):
    SOH(cycle) = Capacity(cycle) / Capacity(cycle=1) * 100

End-of-life (EOL) definition:
    EOL is reached when capacity falls to 1.4 Ah (70% of the nominal 2.0 Ah
    rated capacity for this cell family -- i.e. 30% fade), the threshold
    NASA PCoE used when they stopped cycling these specific cells.

RUL definition:
    RUL(cycle) = (cycle index at which capacity first drops to/below the
    EOL threshold) - (current cycle index), in discharge cycles.
    For a battery that never reaches EOL within the recorded cycles, RUL at
    each point is defined relative to its own last recorded cycle and is
    marked as right-censored (flagged, not silently treated as ground truth).
"""
import pandas as pd
import numpy as np
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
PROC_DIR = repo_root / "data" / "processed"
EOL_CAPACITY_AH = 1.4

# allow the script to exit gracefully if the processed CSV doesn't exist
csv_path = PROC_DIR / "all_batteries_discharge_cycles.csv"
if not csv_path.exists():
    print(f"Processed CSV not found at {csv_path}. Run the parser or place CSV in data/processed.")
    raise SystemExit(1)

df = pd.read_csv(csv_path)
df = df.sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)

feat_rows = []
for bid, g in df.groupby("battery_id"):
    g = g.sort_values("cycle_number").reset_index(drop=True)
    initial_capacity = g.loc[0, "capacity"]
    g["soh"] = g["capacity"] / initial_capacity * 100.0

    # Determine EOL cycle (first cycle index where capacity <= threshold)
    eol_hits = g.index[g["capacity"] <= EOL_CAPACITY_AH].tolist()
    censored = len(eol_hits) == 0
    eol_cycle = g.loc[eol_hits[0], "cycle_number"] if eol_hits else g["cycle_number"].max()

    g["rul"] = eol_cycle - g["cycle_number"]
    g["rul"] = g["rul"].clip(lower=0)
    g["censored"] = censored

    # rolling / derived degradation features (careful: only use past+current, no leakage)
    g["capacity_retention_pct"] = g["soh"]
    g["voltage_range"] = g["voltage_max"] - g["voltage_min"]
    g["rolling_capacity_mean_5"] = g["capacity"].rolling(5, min_periods=1).mean()
    g["rolling_capacity_std_5"] = g["capacity"].rolling(5, min_periods=1).std().fillna(0)
    g["capacity_degradation_rate"] = g["capacity"].diff().fillna(0)  # Ah lost vs previous cycle
    g["cum_capacity_fade"] = initial_capacity - g["capacity"]
    g["cycle_number_norm"] = g["cycle_number"] / g["cycle_number"].max()

    feat_rows.append(g)

full = pd.concat(feat_rows, ignore_index=True)
full.to_csv(PROC_DIR / "features_labeled.csv", index=False)

print("Feature matrix shape:", full.shape)
print("\nPer-battery summary:")
print(full.groupby("battery_id").agg(
    n_cycles=("cycle_number", "max"),
    start_capacity=("capacity", "first"),
    end_capacity=("capacity", "last"),
    end_soh=("soh", "last"),
    censored=("censored", "first"),
))
print("\nColumns:", list(full.columns))
