Analysis in this project was done as standalone, reviewable scripts under
`src/` rather than notebooks, so every step (parsing, features, training,
evaluation, plots) runs reproducibly end-to-end via
`python src/<step>/<script>.py`. If you prefer notebooks for the
presentation/demo, the quickest path is to paste each script's contents
into `01_data_exploration.ipynb` ... `05_attention_model.ipynb` in this
folder, cell by cell — the underlying logic and real results are already
verified in `results/`.
