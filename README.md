# Goalkeeper Positioning Analysis

Counterfactual goalkeeper-positioning model from StatsBomb open data: trains a CNN on a rasterized pitch to estimate `V(x, g) = P(goal | x, g)`, where `g` is the goalkeeper's position and `x` is everything else. Sweeping `g` across the pitch produces a danger heatmap; the optimum is `g* = argmin_g V(x, g)`.

**Module:** Mathematics and Deep Learning — University of Leipzig
**Authors:** Lauren Pommer, Hannes [TBD]
**Baselines:** StatsBomb xG (`shot_statsbomb_xg`); Anzer & Bauer (2021) RPS=0.197

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Notebook-only extras
pip install mplsoccer squarify wordcloud
```

No data download is required — `mplsoccer.Sbopen` fetches StatsBomb open data at runtime.

## Project structure

```
src/
  data/
    build_splits.py      # builds train/val/test + transfer manifests
    rasterize.py         # rasterize_shot, filter_rasterizable_shots
    dataset.py           # GoalkeeperShotsDataset
  training/
    dataloaders.py       # make_dataloaders
    metrics.py           # AUC, Brier, log-loss, ECE
    train.py             # train_model — main training loop
    evaluate.py          # evaluate_model — held-out evaluation
  models/
    danger_cnn.py        # DangerCNN (102k params, 5-channel input)
data/
  raw/                   # shots_master_df.csv, freeze_master_df.csv (gitignored)
  processed/splits/      # *_shot_ids.csv manifests per split
models/checkpoints/      # per-run artifacts (gitignored)
results/                 # eval predictions, smoke-test figures (mostly gitignored)
notebooks/               # exploratory analysis
```

## Pipeline — run in this order

### 1. Build the master shot/freeze CSVs

The training pipeline expects `data/raw/shots_master_df.csv` and `data/raw/freeze_master_df.csv` (one row per shot / per freeze-frame player). These are produced from `mplsoccer.Sbopen` in `notebooks/01_eda.ipynb`. Run that notebook end-to-end the first time you set up the project.

### 2. Build the splits

```bash
python src/data/build_splits.py
```

Writes five manifests to `data/processed/splits/`:

| split | content | size |
|---|---|---|
| `train` | men 2015/16, top-5 leagues | ~30k |
| `val` | men 2015/16, top-5 leagues (held out) | ~6.4k |
| `test` | men 2015/16, top-5 leagues (held out) | ~6.5k |
| `transfer_women` | all women's competitions | ~12.6k |
| `transfer_men_other` | other men's competitions/seasons | ~25.8k |

Manifests are just `shot_id` lists; the actual data lives in the master CSVs.

### 3. Train

```bash
python src/training/train.py
```

Defaults: `run_name="baseline_v1"`, batch 64, Adam lr=1e-3, BCEWithLogitsLoss (no `pos_weight` so the model stays calibrated to the ~10% goal base rate), `ReduceLROnPlateau` on val loss, early stopping after 10 epochs without improvement, train cached in memory (~2.4 GB).

Or programmatically with a custom config:

```python
from src.training.train import train_model
train_model({"run_name": "v2_widercrop", "epochs": 30, "learning_rate": 5e-4})
```

`train_model` will:

1. Load shot/freeze masters and run `filter_rasterizable_shots` on each manifest (idempotent — only rewrites a manifest if shots were dropped).
2. Build `GoalkeeperShotsDataset` per split via `make_dataloaders`.
3. Train `DangerCNN`, logging per-epoch metrics to `models/checkpoints/<run_name>/history.json`.
4. Save `best.pt` (lowest val loss) and `last.pt` after each epoch.
5. Write `training_curves.png` (loss + val AUC + val Brier).

### 4. Evaluate

```bash
python src/training/evaluate.py models/checkpoints/baseline_v1/best.pt test
# valid splits: test, transfer_women, transfer_men_other
```

Or programmatically:

```python
from src.training.evaluate import evaluate_model
result = evaluate_model("models/checkpoints/baseline_v1/best.pt", "test")
```

Returns model metrics alongside StatsBomb xG metrics on the same shots, and writes per-shot predictions to `results/eval/<run_name>/<split>_predictions.csv` (`id, y_true, y_pred, y_statsbomb_xg`).

The training/val/test splits are for model selection. The two `transfer_*` splits are held out for post-hoc generalization checks — only run them once you have a final model.

## Component reference

For interactive use (notebooks, ad-hoc scripts) the public functions/classes are:

```python
# Rasterization (data/rasterize.py)
rasterize_shot(shot_row, freeze_rows) -> torch.Tensor       # (5, 80, 60)
filter_rasterizable_shots(shot_ids, shots_df, freeze_df)    # -> (kept, dropped)

# Datasets and loaders (data/dataset.py, training/dataloaders.py)
GoalkeeperShotsDataset(manifest_path, shots_df, freeze_df, cache_in_memory=False)
make_dataloaders(batch_size=64, num_workers=4, cache_in_memory=False)
    # -> {"train": DataLoader, "val": DataLoader, "test": DataLoader}

# Model (models/danger_cnn.py)
DangerCNN(in_channels=5, dropout_p=0.3)
model.forward_logits(x)   # for training (use BCEWithLogitsLoss)
model(x)                  # for inference; returns sigmoid probabilities

# Metrics (training/metrics.py)
compute_metrics(y_true, y_pred_probs)
    # -> {auc, brier, log_loss, accuracy_at_0.5, expected_calibration_error}
```

## Input representation

5-channel raster, x ∈ [60, 120] m × y ∈ [0, 80] m, 1 m/cell → tensor shape `(5, 80, 60)`:

| ch | content | combine |
|---|---|---|
| 0 | shooter | single Gaussian |
| 1 | ball | single Gaussian (= shooter for now) |
| 2 | attacking teammates (excl. shooter) | elementwise max of per-player Gaussians |
| 3 | defenders (excl. GK) | elementwise max |
| 4 | goalkeeper | single Gaussian |

Sigma = 1.5 m. Elementwise max (not sum) keeps values in `[0, 1]` and emphasizes occupancy. Off-crop shooter raises `ShotOutsideCropError` (filtered upstream); other off-crop players are clip-rendered.

## Current status (paused 2026-05-07)

Baseline v1 (`models/checkpoints/baseline_v1/`):

| metric | DangerCNN | StatsBomb xG |
|---|---|---|
| AUC | 0.8032 | 0.8209 |
| Brier | 0.0750 | 0.0698 |
| ECE | 0.0097 | 0.0079 |

23 epochs, early-stopped, best at epoch 13. Mean prediction (0.098) tracks the test base rate (0.096); decile reliability matches predicted probabilities to within ~1.5 pp across all bins. Next: counterfactual `V(x, g)` heatmap sweep.

## Commands

```bash
pytest                  # all tests
pytest tests/path/...   # single test file
ruff check .
ruff format .
jupyter notebook
```

## License

_(TBD)_
