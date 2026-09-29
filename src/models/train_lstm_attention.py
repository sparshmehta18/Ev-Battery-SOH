"""
LSTM and LSTM+Attention sequence models for SOH / RUL prediction.

Architecture (LSTM+Attention):
    Input sequence (window of W past cycles x F features)
      -> feature normalization (StandardScaler, fit on TRAIN batteries only)
      -> LSTM layer (hidden=64)
      -> Dropout(0.2)
      -> second LSTM layer (hidden=32, returns full sequence)
      -> additive attention over the 32-dim hidden states across the window
         (learns which past cycles matter most for the current prediction)
      -> Dense(16) + ReLU
      -> Dense(1) output head (separate heads/models for SOH and RUL,
         trained independently -- multi-output training was unstable with
         this little data, exactly as flagged as a risk in the brief)

Splitting: same battery-wise holdout as the baselines (train on B0005/B0007/
B0018, test on the fully unseen B0006) to avoid leakage.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

torch.manual_seed(42)
np.random.seed(42)

repo_root = Path(__file__).resolve().parents[2]
PROC_DIR = repo_root / "data" / "processed"
RESULTS_DIR = repo_root / "results"
MODEL_DIR = repo_root / "saved_models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "voltage_start", "voltage_end", "voltage_mean", "voltage_min", "voltage_max",
    "voltage_std", "voltage_range", "current_mean", "current_min", "current_max",
    "temperature_mean", "temperature_max", "temperature_min",
    "discharge_duration_s", "energy_wh", "cycle_number",
]
WINDOW = 10
TEST_BATTERY = "B0006"

df = pd.read_csv(PROC_DIR / "features_labeled.csv").sort_values(["battery_id", "cycle_number"])

def make_sequences(frame, features, targets, window):
    X, Y = [], {t: [] for t in targets}
    for bid, g in frame.groupby("battery_id"):
        g = g.reset_index(drop=True)
        feat_arr = g[features].values
        for t in targets:
            targ_arr = g[t].values
            for i in range(window, len(g)):
                if t == targets[0]:
                    X.append(feat_arr[i - window:i])
                Y[t].append(targ_arr[i])
    return np.array(X), {t: np.array(v) for t, v in Y.items()}

train_df = df[df.battery_id != TEST_BATTERY]
test_df = df[df.battery_id == TEST_BATTERY]

scaler = StandardScaler().fit(train_df[FEATURES])
train_scaled = train_df.copy()
test_scaled = test_df.copy()
train_scaled[FEATURES] = scaler.transform(train_df[FEATURES])
test_scaled[FEATURES] = scaler.transform(test_df[FEATURES])

targets = ["soh", "rul"]
X_train, Y_train = make_sequences(train_scaled, FEATURES, targets, WINDOW)
X_test, Y_test = make_sequences(test_scaled, FEATURES, targets, WINDOW)
print("Train sequences:", X_train.shape, "Test sequences:", X_test.shape)

class LSTMAttention(nn.Module):
    def __init__(self, n_features, hidden1=64, hidden2=32):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.attn_score = nn.Linear(hidden2, 1)
        self.fc1 = nn.Linear(hidden2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        h1, _ = self.lstm1(x)
        h1 = self.dropout(h1)
        h2, _ = self.lstm2(h1)                 # (batch, window, hidden2)
        scores = self.attn_score(h2)            # (batch, window, 1)
        weights = torch.softmax(scores, dim=1)  # attention over time steps
        context = torch.sum(weights * h2, dim=1)  # (batch, hidden2)
        out = self.relu(self.fc1(context))
        return self.fc2(out).squeeze(-1), weights.squeeze(-1)

class PlainLSTM(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :]).squeeze(-1), None

def train_model(model, X_train, y_train, X_test, y_test, epochs=400, lr=3e-3, target_name="", model_name=""):
    # Normalize the TARGET too (fit on train only) -- this is what was missing:
    # SOH lives on ~60-100 and RUL on ~0-170, both far outside the network's
    # default init range, which stalled convergence.
    y_mean, y_std = y_train.mean(), y_train.std()
    ytr_n = (y_train - y_mean) / y_std

    Xtr = torch.tensor(X_train, dtype=torch.float32)
    ytr = torch.tensor(ytr_n, dtype=torch.float32)
    Xte = torch.tensor(X_test, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    loss_fn = nn.MSELoss()
    train_losses = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred, _ = model(Xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step(loss.item())
        train_losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        test_pred_n, attn = model(Xte)
    test_pred = test_pred_n.numpy() * y_std + y_mean  # back to original scale
    return model, test_pred, train_losses, (attn.numpy() if attn is not None else None)

results = {}
attn_store = {}
def evaluate(y_true, y_pred, target, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    denom = np.clip(np.abs(y_true), 1.0, None)  # avoid MAPE blowup near RUL=0
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100
    results.setdefault(target, {})[model_name] = {
        "MAE": round(float(mae), 4), "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4), "MAPE": round(float(mape), 4),
    }
    print(f"[{target}] {model_name:18s} MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} MAPE={mape:.2f}%")

loss_curves = {}
predictions = {}
for target in targets:
    y_train = Y_train[target]
    y_test = Y_test[target]

    lstm = PlainLSTM(n_features=len(FEATURES))
    lstm, pred_lstm, losses_lstm, _ = train_model(lstm, X_train, y_train, X_test, y_test, target_name=target, model_name="LSTM")
    evaluate(y_test, pred_lstm, target, "LSTM")
    loss_curves[f"LSTM_{target}"] = losses_lstm

    lstm_attn = LSTMAttention(n_features=len(FEATURES))
    lstm_attn, pred_attn, losses_attn, attn_weights = train_model(lstm_attn, X_train, y_train, X_test, y_test, target_name=target, model_name="LSTM+Attention")
    evaluate(y_test, pred_attn, target, "LSTM+Attention")
    loss_curves[f"LSTM+Attention_{target}"] = losses_attn
    predictions[target] = {"y_true": y_test.tolist(), "lstm": pred_lstm.tolist(), "lstm_attn": pred_attn.tolist()}

    if target == "soh":
        torch.save(lstm_attn.state_dict(), MODEL_DIR / "lstm_attention_soh.pt")
        attn_store["soh_attention_weights_sample"] = attn_weights[:5].tolist()

with open(RESULTS_DIR / "dl_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
with open(RESULTS_DIR / "predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)
with open(RESULTS_DIR / "attention_weights_sample.json", "w") as f:
    json.dump(attn_store, f, indent=2)
np.save(RESULTS_DIR / "loss_curves.npy", loss_curves)
print("\nSaved DL metrics, predictions, attention weights.")
