# Goalkeeper Positioning Analysis

Counterfactual goalkeeper-positioning model from StatsBomb open data: trains a CNN on a rasterized pitch to estimate `V(x, g) = P(goal | x, g)`, where `g` is the goalkeeper's position and `x` is everything else. Sweeping `g` across the pitch produces a danger heatmap; the optimum is `g* = argmin_g V(x, g)`.

**Module:** Mathematics and Deep Learning — University of Leipzig
**Authors:** Lauren Pommer, Hannes [TBD]
**Baselines:** StatsBomb xG (`shot_statsbomb_xg`); Anzer & Bauer (2021) RPS=0.197

## TL;DR — which model to use

The project has two distinct goals (a *positioning recommender* and a *goal predictor*), and they turned out to trade off. Use the model that matches your goal:

- **Positioning / counterfactual `g*` → use `baseline_v2`.** Goal-geometry channels + a spatial-preserving pool. Produces sensible, grid-independent, coaching-consistent `g*` (anti-coaching optima 0.2% ± 0.45% across seeds; holds on transfer).
- **Pure goal prediction vs StatsBomb xG → cite `baseline_v3`.** v2 + a parallel scalar-feature head. Matches xG on AUC (0.820 ≈ 0.821) but its counterfactual `g*` regresses badly (~19.5% anti-coaching). See "Results".

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Notebook-only extras (EDA plots)
pip install squarify wordcloud
```

No data download is required — `mplsoccer.Sbopen` fetches StatsBomb open data at runtime.

## Project structure

```
src/
  data/
    build_splits.py        # builds train/val/test + transfer manifests
    rasterize.py           # rasterize_shot (5 or 10 ch), geometry_channels, filter
    dataset.py             # GoalkeeperShotsDataset (raster [+ scalars] + label)
  features/
    scalar_features.py     # compute_scalar_features (Tier 2 shot/keeper geometry)
  models/
    danger_cnn.py          # DangerCNN — configurable channels / pool / scalar head
  training/
    dataloaders.py         # make_dataloaders
    metrics.py             # AUC, Brier, log-loss, ECE
    train.py               # train_model — main training loop
    evaluate.py            # evaluate_model — held-out / transfer evaluation
    multiseed_v2.py        # multi-seed confidence intervals for v2
  analysis/
    counterfactual.py      # V(x, g) sweep, g* heatmaps, regret
    regret_distribution.py # aggregate regret / g*-quality over a split sample
data/
  raw/                     # shots_master_df.csv, freeze_master_df.csv (gitignored)
  processed/splits/        # *_shot_ids.csv manifests per split
models/checkpoints/        # per-run artifacts (*.pt kept; v2_seeds/ gitignored)
results/                   # eval predictions, counterfactual PNGs/CSVs
notebooks/                 # exploratory analysis
docs/                      # supervisor notes, presentation, theory background
```

## Model variants

All variants share the `DangerCNN` class; they differ only by config flags. v2 is the recommended deliverable.

| run | input | head | scalar head | params | test AUC |
|---|---|---|---|---|---|
| `baseline_v1` | 5 player channels | global avg pool | — | 102k | 0.803 |
| **`baseline_v2`** | + 5 goal-geometry channels | (4,3) spatial pool | — | 194k | **0.814 ± 0.003** |
| `baseline_v3` | v2 channels | (4,3) pool | 9 features ("all") | 196k | 0.820 |
| `baseline_v3b` | v2 channels | (4,3) pool | 6 features ("context") | 195k | 0.815 |

Reproduce any variant by editing the config in `train.py`'s `__main__` (the default there is currently `baseline_v3b`):

```python
# v1: include_geometry=False, in_channels=5,  pool_size=1, include_scalars=False, scalar_dim=0
# v2: include_geometry=True,  in_channels=10, pool_size=[4,3], include_scalars=False, scalar_dim=0
# v3: ...as v2 plus include_scalars=True, scalar_feature_set="all",     scalar_dim=9
# v3b:...as v2 plus include_scalars=True, scalar_feature_set="context", scalar_dim=6
```

> Note: `pool_size=[4,3]` is non-square because the pre-pool feature map is 20×15 and Apple-MPS requires the adaptive-pool output to divide the input evenly (20/4, 15/3). It also matches the pitch (depth × width).

## Pipeline — run in this order

### 1. Build the master shot/freeze CSVs

The pipeline expects `data/raw/shots_master_df.csv` and `data/raw/freeze_master_df.csv` (one row per shot / per freeze-frame player). These are produced from `mplsoccer.Sbopen` in `notebooks/Data_Loading_and_Exploration.ipynb`. Run that notebook end-to-end the first time you set up the project.

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
python src/training/train.py            # uses the config in __main__ (currently baseline_v3b)
```

Or programmatically with a custom config:

```python
from src.training.train import train_model
train_model({
    "run_name": "baseline_v2", "epochs": 50, "batch_size": 64,
    "learning_rate": 1e-3, "weight_decay": 1e-5, "optimizer": "adam",
    "early_stopping_patience": 10, "device": "auto",
    "checkpoint_dir": "models/checkpoints", "seed": 42,
    "include_geometry": True, "in_channels": 10, "pool_size": [4, 3],
    "include_scalars": False, "scalar_dim": 0,
    "cache_train_in_memory": True, "num_workers": 0, "log_every_n_batches": 0,
})
```

`BCEWithLogitsLoss` with no `pos_weight` (keeps the model calibrated to the ~10% base rate), `ReduceLROnPlateau` on val loss, early stopping after 10 epochs. Writes `best.pt` / `last.pt` / `history.json` / `training_curves.png` to `models/checkpoints/<run_name>/`, plus `config.json` (channels/pool/scalar settings are read back from it at eval/sweep time, so v1–v3b all load correctly).

### 4. Evaluate

```bash
python src/training/evaluate.py models/checkpoints/baseline_v2/best.pt test
# valid splits: test, transfer_women, transfer_men_other
```

Returns model metrics alongside StatsBomb xG metrics on the same shots, and writes per-shot predictions to `results/eval/<run_name>/<split>_predictions.csv`.

### 5. Counterfactual sweep + regret

```bash
# 8 curated example shots -> 4-panel heatmaps + summary.csv
python src/analysis/counterfactual.py        # edit __main__ for the checkpoint

# aggregate g*-quality over a 200-shot sample of a split
python src/analysis/regret_distribution.py models/checkpoints/baseline_v2/best.pt
```

`compute_regret_distribution(..., split=...)` accepts `test`, `transfer_women`, or `transfer_men_other` and reports median regret, % of `g*` pinned to the grid edge, and % on the far (anti-coaching) side of the shooter.

### 6. Multi-seed confidence intervals (v2)

```bash
python src/training/multiseed_v2.py          # seeds 42 1 2 3 4 by default
```

Retrains v2 across seeds and reports mean ± std for both predictive and counterfactual metrics → `results/eval/v2_multiseed_summary.csv`. (Per-seed checkpoints under `models/checkpoints/v2_seeds/` are gitignored.)

## Component reference

```python
# Rasterization (data/rasterize.py)
rasterize_shot(shot_row, freeze_rows, include_geometry=False)  # (5 or 10, 80, 60)
geometry_channels()                                            # static (5, 80, 60)
filter_rasterizable_shots(shot_ids, shots_df, freeze_df)       # -> (kept, dropped)

# Scalar features (features/scalar_features.py)
compute_scalar_features(shot_row, freeze_rows, gk_pos=None, feature_set="all")
    # feature_set in {"all" (9), "context" (6 g-independent)}; gk_pos sweeps g

# Datasets and loaders (data/dataset.py, training/dataloaders.py)
GoalkeeperShotsDataset(manifest_path, shots_df, freeze_df,
                       cache_in_memory=False, include_geometry=False,
                       include_scalars=False, scalar_feature_set="all")
    # item is (raster, label) or (raster, scalars, label) when include_scalars
make_dataloaders(batch_size=64, num_workers=4, cache_in_memory=False)

# Model (models/danger_cnn.py)
DangerCNN(in_channels=5, dropout_p=0.3, pool_size=1, scalar_dim=0)
model.forward_logits(x, scalars=None)   # for training (BCEWithLogitsLoss)
model(x, scalars=None)                  # inference; sigmoid probabilities

# Metrics (training/metrics.py)
compute_metrics(y_true, y_pred_probs)
    # -> {auc, brier, log_loss, expected_calibration_error}

# Counterfactual analysis (analysis/)
sweep_gk_positions(model, shot_row, freeze_rows, ...)          # danger grid + g*
run_sweep_on_examples(checkpoint_path, shot_ids=None, ...)     # PNGs + summary
compute_regret_distribution(checkpoint_path, split="test", ...) # aggregate g*-quality
```

## Input representation

Raster: x ∈ [60, 120] m × y ∈ [0, 80] m, 1 m/cell → `(C, 80, 60)`. The goalkeeper is always channel 4, so the counterfactual sweep can overwrite that one channel as it moves `g`.

**Player channels (always present), σ = 1.5 m:**

| ch | content | combine |
|---|---|---|
| 0 | shooter | single Gaussian |
| 1 | ball | single Gaussian (= shooter for now) |
| 2 | attacking teammates (excl. shooter) | elementwise max of per-player Gaussians |
| 3 | defenders (excl. GK) | elementwise max |
| 4 | goalkeeper | single Gaussian |

**Goal-geometry channels (v2+, `include_geometry=True`) — static, identical for every shot:**

| ch | content |
|---|---|
| 5 | `goal_frame` — Gaussian ridge on the goal-line segment (the goal mouth) |
| 6 | `dist_to_goal` — normalized distance from each cell to the goal centre |
| 7 | `goal_angle` — angle the posts subtend from each cell (the xG "view angle") |
| 8–9 | `coord_x`, `coord_y` — normalized coordinates (CoordConv) |

These give the CNN explicit knowledge of the goal frame, which is why v2's `g*` stops sliding off the grid edge (see "Results"). Elementwise max (not sum) keeps player values in `[0, 1]`. Off-crop shooter raises `ShotOutsideCropError` (filtered upstream); other off-crop players are clip-rendered.

**Scalar features (v3/v3b, parallel head):** `dist_to_goal`, `shot_angle`, `gk_coverage`, `gk_perp_offset`, `gk_depth`, body-part one-hots, `under_pressure`. The `"context"` subset (v3b) drops the three GK-dependent ones.

## Results (updated 2026-06-24)

### Predictive (test split)

| model | AUC | Brier | ECE |
|---|---|---|---|
| baseline_v1 | 0.803 | 0.075 | 0.010 |
| **baseline_v2** (5-seed) | **0.814 ± 0.003** | 0.0716 ± 0.0001 | 0.0077 ± 0.0008 |
| baseline_v3 | 0.820 | 0.0705 | 0.011 |
| baseline_v3b | 0.815 | 0.0708 | 0.0065 |
| StatsBomb xG | 0.821 | 0.070 | 0.008 |

### Counterfactual `g*` quality (200-shot sweep)

| model | median regret | y-edge pinned | far-side (anti-coaching) |
|---|---|---|---|
| baseline_v1 | — | up to 7/8 examples | grid-dependent, pervasive |
| **baseline_v2** (5-seed) | 0.0099 ± 0.0011 | 0.7% ± 0.45% | **0.2% ± 0.45%** |
| baseline_v3 | 0.015 | 4.5% | 19.5% |
| baseline_v3b | 0.008 | 1.5% | 14% |

### Transfer (baseline_v2, out-of-domain)

| split | AUC | far-side g* |
|---|---|---|
| test (in-domain) | 0.809* | 0% |
| transfer_women | 0.781 | 0% |
| transfer_men_other | 0.797 | 1% |

\* single-seed; the in-domain multi-seed mean is 0.814.

### Key findings

- **v1 → v2:** v1 had no structural knowledge of the goal frame, so `V` dropped monotonically as the synthetic keeper left the goal mouth — `g*` pinned to the sweep-grid edge and recommended anti-coaching far-post positions. Adding goal-geometry channels + a spatial pool resolved this; `g*` is now grid-independent and coaching-consistent.
- **v3/v3b — a predictive vs counterfactual tradeoff.** Adding a scalar head closes the StatsBomb xG gap but degrades `g*`. Notably this happens even for v3b's *g-independent* features (which mathematically cannot change `argmin_g`): the degradation is a **training-interference** effect — the auxiliary head reshapes the conv trunk so the spatial branch becomes a worse function of keeper position. A single model can't be both the best predictor and the best positioning recommender with this architecture.
- **v2 generalizes for positioning.** Predictive AUC drops modestly out-of-domain, but `g*` quality is nearly domain-invariant, because the goal-frame geometry driving it is the same in every league and in the women's game.

See `docs/notes_for_supervisor_2026-05-08.md` for the decision-point write-up and resolution, and `CLAUDE.md` for the detailed status log.

## Tests / lint

There is no `pytest` suite; each module has a `__main__` smoke test that runs and self-checks:

```bash
python src/models/danger_cnn.py        # param count, forward/backward, grad check
python src/data/rasterize.py           # renders a 10-channel panel + validates
python src/data/dataset.py             # builds the dataset, prints item shapes
python src/features/scalar_features.py  # prints the feature vector + g-sensitivity
jupyter notebook                       # exploratory analysis
```

`ruff check .` / `ruff format .` are the intended linters but are not currently in the venv (`pip install ruff` to use them).

## License

_(TBD)_
