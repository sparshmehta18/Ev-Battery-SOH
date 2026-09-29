Evaluation logic (MAE/RMSE/R²/MAPE) lives inline in
`src/models/train_baselines.py` and `src/models/train_lstm_attention.py`
(see the `evaluate()` function in each) so metrics are computed at training
time, right next to the model that produced them, and written to
`results/baseline_metrics.json` / `results/dl_metrics.json`.
