import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
PROC_DIR = repo_root / "data" / "processed"
RESULTS_DIR = repo_root / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC_DIR / "features_labeled.csv")
preds = json.load(open(RESULTS_DIR / "predictions.json"))
loss_curves = np.load(RESULTS_DIR / "loss_curves.npy", allow_pickle=True).item()
importances = json.load(open(RESULTS_DIR / "xgb_soh_feature_importance.json"))

plt.style.use("seaborn-v0_8-whitegrid")

fig, ax = plt.subplots(figsize=(8, 5))
for bid, g in df.groupby("battery_id"):
    ax.plot(g.cycle_number, g.capacity, label=bid, linewidth=1.8)
ax.axhline(1.4, color="red", linestyle="--", linewidth=1, label="EOL threshold (1.4 Ah)")
ax.set_xlabel("Discharge cycle number"); ax.set_ylabel("Capacity (Ah)")
ax.set_title("Battery Capacity Degradation (NASA PCoE B0005/B0006/B0007/B0018)")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "01_capacity_degradation.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
for bid, g in df.groupby("battery_id"):
    ax.plot(g.cycle_number, g.soh, label=bid, linewidth=1.8)
ax.axhline(70, color="red", linestyle="--", linewidth=1, label="70% SOH (EOL)")
ax.set_xlabel("Discharge cycle number"); ax.set_ylabel("SOH (%)")
ax.set_title("State of Health vs Cycle Number"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "02_soh_vs_cycle.png", dpi=140); plt.close(fig)

p = preds["soh"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p["y_true"], label="Actual SOH", linewidth=2)
ax.plot(p["lstm_attn"], label="LSTM+Attention prediction", linestyle="--")
ax.plot(p["lstm"], label="Plain LSTM prediction", linestyle=":")
ax.set_xlabel("Test sequence index (battery B0006, held out)"); ax.set_ylabel("SOH (%)")
ax.set_title("Actual vs Predicted SOH — Held-out Battery B0006")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "03_actual_vs_pred_soh.png", dpi=140); plt.close(fig)

p = preds["rul"]
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p["y_true"], label="Actual RUL", linewidth=2)
ax.plot(p["lstm_attn"], label="LSTM+Attention prediction", linestyle="--")
ax.plot(p["lstm"], label="Plain LSTM prediction", linestyle=":")
ax.set_xlabel("Test sequence index (battery B0006, held out)"); ax.set_ylabel("RUL (cycles)")
ax.set_title("Actual vs Predicted RUL — Held-out Battery B0006")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "04_actual_vs_pred_rul.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(loss_curves["LSTM_soh"], label="LSTM (SOH) train loss")
ax.plot(loss_curves["LSTM+Attention_soh"], label="LSTM+Attention (SOH) train loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (normalized target)")
ax.set_title("Training Loss Curves (SOH models)"); ax.legend()
fig.tight_layout(); fig.savefig(FIG_DIR / "05_training_loss.png", dpi=140); plt.close(fig)

items = sorted(importances.items(), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh([k for k, v in items], [v for k, v in items], color="#3b82f6")
ax.set_title("XGBoost Feature Importance — SOH Prediction")
ax.set_xlabel("Importance (gain-based)")
fig.tight_layout(); fig.savefig(FIG_DIR / "06_feature_importance.png", dpi=140); plt.close(fig)

attn = json.load(open(RESULTS_DIR / "attention_weights_sample.json"))["soh_attention_weights_sample"]
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(np.array(attn), aspect="auto", cmap="viridis")
ax.set_xlabel("Position in 10-cycle input window (oldest -> newest)")
ax.set_ylabel("Sample test sequence #")
ax.set_title("LSTM+Attention Weights — Which past cycles matter most")
fig.colorbar(im, ax=ax, label="Attention weight")
fig.tight_layout(); fig.savefig(FIG_DIR / "07_attention_weights.png", dpi=140); plt.close(fig)

g = df[df.battery_id == "B0006"].reset_index(drop=True)
split = int(len(g) * 0.7)
hist, future = g.iloc[:split], g.iloc[split:]
coeffs = np.polyfit(hist.cycle_number, hist.capacity, 2)
forecast_x = future.cycle_number
forecast_y = np.polyval(coeffs, forecast_x)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(hist.cycle_number, hist.capacity, label="Historical (observed)", color="black")
ax.plot(future.cycle_number, future.capacity, label="Actual (future, held out)", color="green")
ax.plot(forecast_x, forecast_y, label="Forecast (2nd-order fit on history)", linestyle="--", color="red")
ax.axhline(1.4, color="gray", linestyle=":", label="EOL threshold")
ax.set_xlabel("Cycle number"); ax.set_ylabel("Capacity (Ah)")
ax.set_title("Battery Degradation Forecast — B0006")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "08_degradation_forecast.png", dpi=140); plt.close(fig)

print("Saved 8 figures to", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.png")):
    print(" -", f.name)
