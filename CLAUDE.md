# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status (updated 2026-06-30)

**Docs:** current consolidated supervisor briefing is
`docs/project_overview_2026-06-30.md` (mid-semester talk delivered 2026-05-28;
its briefing doc has been retired — see git history if needed).

**Headline:** Option A worked. baseline_v2 adds goal-frame geometry channels +
a spatial-preserving pool; the counterfactual sweep now produces sensible,
grid-independent g* recommendations. The grid-dependence / anti-coaching
pathology of v1 is resolved. Tier 2 (scalar head, v3/v3b) closed the StatsBomb
xG gap on predictive accuracy but *degraded* counterfactual quality — so
**v2 is the recommended positioning model; v3 is the predictive benchmark.**
The predictive-vs-counterfactual tension is itself a result (see below).

**Done (v1 baseline + diagnosis):**
- EDA: 81k open-play shots with freeze frames; goal rate 9.7-10.3% across splits
- Splits built (option 3): men 2015/16 top 5 leagues = train/val/test (30k/6.4k/6.5k), women + men_other = transfer eval (12.6k + 25.8k)
- Rasterization: 5 channels, 80×60 grid, sigma=1.5m, max-pooled Gaussians for crowd channels — verified visually
- PyTorch Dataset (GoalkeeperShotsDataset) + DataLoader (make_dataloaders) — smoke test passes
- Filter for un-rasterizable shots applied to manifests; 0.12% of train dropped
- DangerCNN baseline_v1 trained: test AUC 0.803, Brier 0.075, ECE 0.010
- Decile calibration excellent (predicted vs actual within 1.5pp across all 10 buckets)
- Pearson correlation with StatsBomb xG = 0.74 (agreement in direction, independent in absolute terms)
- Training plateaus around epoch 5-7; later epochs overfit train without improving val. Diagnosis: saturated input representation, not broken model
- Counterfactual sweep V(x, g) runs end-to-end on baseline_v1: 8 example shots, wide + constrained grids, regret CSV
- v1 finding: g* is grid-dependent (wide grid 7/8 pin behind posts; constrained 2/8 pin at edge). v1 has no structural knowledge of the goal frame, so V drops monotonically as the synthetic GK moves *away* from the goal mouth — a spurious "GKs in goal mouth correlate with goals" signal. Decision-point doc: docs/notes_for_supervisor_2026-05-08.md (options A/B/C)

**Done (v2 = Option A, 2026-06-24):**
- Tier 1 feature engineering shipped. rasterize_shot(include_geometry=True) appends 5 static channels after the GK channel: goal_frame (Gaussian ridge on the goal-line segment), dist_to_goal, goal_angle (posts' subtended view angle), coord_x, coord_y (CoordConv). GK stays at channel index 4 in both 5- and 10-channel layouts.
- DangerCNN gained a `pool_size` arg (int or (h,w) tuple). v2 uses a (4,3) adaptive pool instead of global average pooling, so absolute spatial layout survives into the head. (4,3) chosen because the pre-pool map is 20×15 and MPS needs the output to divide it evenly; square 3/4 fails on MPS.
- baseline_v2 config: include_geometry=True, in_channels=10, pool_size=[4,3], 194k params (within the 100-200k budget). Trained 24 epochs (early-stopped, best epoch 14).
- v2 test metrics (5-seed mean ± std): AUC 0.814 ± 0.003, Brier 0.0716 ± 0.0001, ECE 0.0077 ± 0.0008 (v1 single-run 0.803/0.075/0.010). Closer to StatsBomb xG (AUC 0.821) than the original seed-42-only number (0.809) suggested; calibration preserved. NOTE: earlier docs/commits quoted AUC 0.809 — that was seed 42, the weakest of the 5; the mean is 0.814.
- Counterfactual sweep is now channel-count-agnostic (overwrites the GK channel rather than concat-at-end); _load_model and evaluate.py read in_channels/pool_size/include_geometry from the saved config, so v1 and v2 both load correctly.
- v2 sweep on the same 8 example shots (constrained grid): 0/8 pinned to edge (v1 had 2/8), and the anti-coaching far-post pull is gone (e.g. shot 0c5620fc, shooter y=45.5: v1 → far post y=34; v2 → near post (112.5, 45.5), advanced off the line to cut the angle). PNGs in results/counterfactual/v2_geom/.
- v2 aggregate over 200 seeded test shots (src/analysis/regret_distribution.py): median regret 0.0097 (goals 0.051, saves 0.008), max 0.238. y-edge pinned 1/200 (0.5%); far-side optima 0/200 (0.0%); post-side split 182 centre / 18 near / 0 far. Table: results/counterfactual/v2_geom/regret_200.csv
- Also dropped the misleading accuracy_at_0.5 metric (frozen at 1−base_rate; a calibrated rare-event model rightly rarely predicts >0.5). Confirmed no class-imbalance pathology — the val plateau is overfitting, not imbalance starvation; balancing would only break calibration.

**Done (Tier 2 = parallel scalar head, 2026-06-24):**
- src/features/scalar_features.py: 9-feature vector — dist_to_goal, shot_angle, gk_coverage (fraction of goal occluded from shooter), gk_perp_offset, gk_depth, body-part one-hots (R/L/Head), under_pressure. Three are GK-dependent (GK_DEPENDENT_IDX = 2,3,4). Named subsets via FEATURE_SETS: "all" (9) and "context" (6 g-independent: drops gk_coverage/perp/depth).
- DangerCNN gained `scalar_dim`: a parallel MLP (scalar_dim→32→32) concatenated with the CNN's 64-d embedding before the final layer. scalar_dim=0 is byte-identical to v1/v2. Dataset/train/evaluate thread include_scalars + scalar_feature_set and handle the (raster, scalars, label) 3-tuple; the sweep recomputes per-grid scalars (constant across g for "context").
- **baseline_v3** (scalar_feature_set="all", 196k params): test AUC 0.820 ≈ StatsBomb xG 0.821 (gap closed from 0.012), Brier 0.0705, ECE 0.011. BUT counterfactuals regressed: 200-shot far-side/anti-coaching optima 39/200 (19.5%) vs v2's 0%; y-edge pinned 4.5%; post-side 67 centre/94 near/39 far.
- **baseline_v3b** (scalar_feature_set="context", 195k params): test AUC 0.815, Brier 0.0708, ECE 0.0065 (best of all). Counterfactuals better than v3 but still not v2: far-side 28/200 (14%), pinned 1.5%, post-side 79 centre/93 near/28 far.
- **Key finding — predictive vs counterfactual tension is a *training* effect, not an inference one.** The "context" scalars are g-independent, so they add a constant to every grid cell's logit and *cannot* change argmin_g — yet v3b's g* still degraded vs v2. The only difference is the CNN branch's learned weights: adding any auxiliary predictive head lets the conv layers offload variance onto the scalar MLP, so the spatial branch becomes a worse function of keeper position. Conclusion: a single model can't be both the best xG predictor and the best positioning recommender with this architecture.
- Artifacts: results/counterfactual/v3_scalar/, results/counterfactual/v3b_context/ (8 PNGs + regret_200.csv each); models/checkpoints/baseline_v3{,b}/.

**Done (v2 validation, 2026-06-24):**
- Multi-seed CIs (src/training/multiseed_v2.py, 5 seeds): test AUC 0.814 ± 0.003, Brier 0.0716 ± 0.0001, ECE 0.0077 ± 0.0008. Counterfactual on the fixed 200-shot sample: median regret 0.0099 ± 0.0011, y-edge pinned 0.7% ± 0.45%, far-side/anti-coaching 0.2% ± 0.45% (range 0–1%). The clean-counterfactual property is reproducible, not a lucky seed; v2 0.2% vs v3 19.5% far-side is ~40σ apart. Per-seed checkpoints gitignored (models/checkpoints/v2_seeds/); kept: results/eval/v2_multiseed_summary.csv + results/counterfactual/v2_seeds/seed*_regret.csv.
- Transfer eval of baseline_v2 (predictive): test AUC 0.809 / women 0.781 / men_other 0.797; the gap to StatsBomb xG is stable across domains (~0.012–0.014 AUC). ECE stays ≤1.7pp out-of-domain. Brier rises partly from higher transfer base rates (women 10.3%, men_other 11.25%). Predictions in results/eval/baseline_v2/transfer_*_predictions.csv.
- Transfer counterfactual quality (200-shot, src/analysis/regret_distribution.py now takes split=): far-side stays ≈0 out-of-domain (women 0%, men_other 1%), pinning ≤3%. v2's g* recommendations generalize — the goal-frame geometry that drives g* is domain-invariant even though raw P(goal) accuracy drops. Tables: results/counterfactual/v2_transfer_{women,men_other}/regret_200.csv.

**Done (position-jitter augmentation experiment, 2026-06-24):**
- baseline_v2_jitter = v2 + train-only Gaussian position jitter (σ=0.75 m) on every player, re-sampled each epoch (forces caching off → ~41s/epoch vs ~12s). rasterize_shot/dataset/train gained a `jitter_sigma` flag (default 0.0, byte-identical to before when off; geometry channels and the off-crop check use unperturbed positions).
- Motivation: the sweep evaluates V(x,g) at GK positions never seen in training, so jitter should align train/eval and smooth V(x,g). **Result: neutral.** Predictive AUC 0.810 (within v2's 0.809–0.817 multi-seed range), ECE 0.0060 (slightly better). Counterfactual: far-side 0%, pinned 1.0%, median regret 0.0076 — same as v2. Crucially, **direct landscape roughness (mean total-variation of the danger grid) is identical (0.0053 vs 0.0053)** — the smoothing hypothesis did NOT hold.
- Why: v2's goal-geometry channels already give a smooth, clean V(x,g); there was no brittleness for jitter to fix. Useful as (a) an empirical answer to the supervisor's augmentation suggestion — not a needed lever here — and (b) a robustness confirmation that v2's clean g* survives perturbing every training position. Single-seed; not worth a multi-seed run given roughness is flat. Artifacts: results/counterfactual/v2_jitter/, models/checkpoints/baseline_v2_jitter/.

**Done (v3 multi-seed CIs + latent embedding, 2026-07-02):**
- v3 multi-seed CIs (5 seeds, src/training/multiseed_v2.py generalized to take a model tag: `python src/training/multiseed_v2.py v3`). v3 test AUC 0.819 ± 0.001, Brier 0.0704 ± 0.0001, ECE 0.0097 ± 0.0025. Both v2 and v3 now 5-seed on matched seeds → **paired** comparison: mean ΔAUC = +0.0050 in v3's favour, paired t=2.80, **p ≈ 0.049** — real but only marginally significant, driven mostly by seed 42 (Δ +0.011; seed 3 Δ ≈ 0). So v3 ≈ StatsBomb xG (0.821) is firm, but v3 ≫ v2 is NOT: the predictive edge is ~0.5 AUC pts and borderline. v3's counterfactuals are also unstable across seeds (far-side 16.3% ± 7.3%, range 6–26%; pinned 9.9% ± 15.5%, one seed 37.5%) vs v2's tight 0.2%/0.7%. Summary: results/eval/v3_multiseed_summary.csv (+ per-seed regret CSVs in results/counterfactual/v3_seeds/; throwaway checkpoints gitignored, mirroring v2).
- Latent embedding analysis (src/analysis/latent_embedding.py): hooks v2's 64-d ReLU(fc1) embedding, runs the test split, PCA→2D. PC1 explains 48.7% of variance and is a **danger axis** — r = −0.85 with predicted P(goal), −0.72 with StatsBomb xG, +0.68 with shooter→goal distance; goals cluster in its low tail. Confirms the v2 representation is organized around danger/geometry. Figure: reports/figures/latent_embedding_v2.png. Reads include_geometry/include_scalars from the checkpoint config so it also runs on v1/v3.

**Recommendation / model selection:**
- **Positioning recommender (project goal): use v2.** Only model with clean counterfactuals (0.2% ± 0.45% anti-coaching across seeds), and they hold on transfer.
- **Predictive benchmark vs StatsBomb xG: cite v3** (5-seed AUC 0.819 ± 0.001 ≈ 0.821 — now firm). But v3 is NOT a decisive upgrade over v2: paired edge only +0.005 AUC (p ≈ 0.049), bought with a large, seed-unstable counterfactual penalty. Net: v2 for positioning, v3 only for xG-parity benchmarking.
- The v2/v3/v3b sweep is a reportable result about auxiliary-head training interference.

**Next (not yet done):**
- Possible: revisit best-of-both via training tricks (e.g. stop-gradient between scalar head and conv trunk, or a two-model setup — v2 for g*, v3 for absolute P(goal)); a defenders-in-cone *spatial* channel is a promising counterfactual-clean way to test whether it's the parallel head (not the extra info) that breaks g*
- The written report (now the main remaining deliverable)
- Separate ball channel from shooter (rasterize.py still copies it); wider crop x ∈ [60,122]; y-flip augmentation

**Open questions:**
- Is the constrained grid y ∈ [34, 46] a defensible canonical eval region? (Less load-bearing now that v2 doesn't pin to the edge regardless of grid.)
- Worth pursuing best-of-both (stop-gradient / two-model), or report the tension as-is and move to transfer eval? Awaiting Hannes / supervisor steer.

## Project

University of Leipzig research project (Mathematics and Deep Learning module) analyzing goalkeeper positioning in football using StatsBomb event data, deep learning (PyTorch), and applied mathematics.

## Research Direction

**Module:** Math and Machine Learning Praktikum, University of Leipzig. **Collaborator:** Hannes.

**Goal:** Build a counterfactual goalkeeper positioning model `V(x, g) = P(goal | x, g)` that estimates the probability of a goal given the non-goalkeeper context `x` and goalkeeper position `g`. At inference, `x` is fixed and `g` is swept across a pitch grid to produce a danger heatmap; the optimal position is `g* = argmin_g V(x, g)`.

**Input decomposition from freeze frame:**
- `g` — goalkeeper (x, y) position (the variable being optimized)
- `x` — everything else: shooter/ball position, attacking teammates, outfield defenders (goalkeeper excluded)

**Architecture:** CNN on a rasterized pitch (80×60 grid). v1: 5 player channels (shooter, ball, attacking teammates, outfield defenders, goalkeeper) rendered as 2D Gaussian blobs, global-average-pool head. v2 (recommended for positioning): the same 5 player channels + 5 static goal-geometry channels (goal_frame, dist_to_goal, goal_angle, coord_x, coord_y) and a (4,3) spatial pool. v3/v3b: v2 plus a parallel scalar-feature head (`scalar_dim`) — better xG prediction, worse counterfactuals. Toggle via `include_geometry` / `in_channels` / `pool_size` / `include_scalars` / `scalar_feature_set` / `scalar_dim`.

**Scope (phase 1):** Open-play shots only. Excludes set pieces, penalties, and direct free kicks (`sub_type_name == 'Open Play'` filter on `df_event`).

**Baselines:**
- Anzer & Bauer (2021): RPS = 0.197 on Bundesliga data
- StatsBomb built-in xG: `shot_statsbomb_xg` column in `df_event`

### Data Notes

`freeze_master_df` uses StatsBomb's **older shot-event freeze frame format**, not StatsBomb 360. Key properties:
- Manually annotated, scoped to players in the vicinity of the shot — not full-pitch tracking data
- No `visible_area` polygon: a player absent from the freeze frame means "not annotated near the shot," not "confirmed absent from the pitch"
- GK coverage is 99.9% in the open-play sample, so the goalkeeper position signal is reliable
- Median 13 visible players per shot (range 1–21), meaning context completeness varies across shots
- Practical implication: model inputs will have more variance in context completeness than tracking-data-based work like Anzer & Bauer (2021), who used full optical tracking

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Notebook-only EDA extras (not in requirements.txt):
pip install squarify wordcloud
```

## Commands

```bash
# Run all tests
pytest

# Run a single test file
pytest tests/path/to/test_file.py

# Lint
ruff check .

# Format
ruff format .

# Start Jupyter
jupyter notebook
```

## Architecture

```
src/
  data/          # Data loading and StatsBomb API access
  features/      # Feature engineering for shots/freeze frames
  models/        # PyTorch model definitions
  visualization/ # Pitch plotting utilities
configs/         # YAML files for experiment hyperparameters
data/
  raw/           # Unmodified StatsBomb data (not versioned)
  processed/     # Cleaned/feature-engineered data (not versioned)
  external/      # Third-party data (not versioned)
models/          # Saved model weights (not versioned)
notebooks/       # Exploratory analysis
reports/figures/ # Output visualizations
```

## Data

Data comes from the **StatsBomb open data API** accessed via `mplsoccer.Sbopen`. No local files need to be downloaded; the parser fetches data at runtime.

```python
from mplsoccer import Sbopen
parser = Sbopen()

df_competition = parser.competition()                          # All competitions
df_match = parser.match(competition_id=9, season_id=281)      # Matches for a competition/season
df_event, df_related, df_freeze, df_tactics = parser.event(match_id)
```

**Key dataframes per match:**
- `df_event`: All match events (~4000 rows per match). Filter to `type_name == 'Shot'` for shot analysis. Relevant shot columns include `x`, `y`, `end_x`, `end_y`, `end_z`, `shot_statsbomb_xg`, `outcome_name`, `body_part_name`, `technique_name`, `under_pressure`, `goalkeeper_position_name`.
- `df_freeze`: Player positions at the moment of each shot — critical for goalkeeper positioning analysis. Columns: `id` (links to shot event), `x`, `y`, `player_id`, `player_name`, `position_name`, `teammate`.
- `df_related`: Related event IDs (e.g., shot linked to Goal Keeper event). Less relevant for the xG/positioning model.
- `df_tactics`: Formation and lineup data at match start.

**Pitch coordinate system:** StatsBomb uses 120 (length) × 80 (width). When using `mplsoccer.VerticalPitch` with `pitch_type='statsbomb'`, coordinates are used as-is. For custom pitch dimensions, use `pitch_type='custom'` with `pitch_length=120, pitch_width=80`.

**Competitions with 360 data** (freeze frames for all visible players, not just shot): check `match_available_360` column in competition data.
