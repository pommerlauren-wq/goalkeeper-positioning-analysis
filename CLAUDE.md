# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

University of Leipzig research project (Mathematics and Deep Learning module) analyzing goalkeeper positioning in football using StatsBomb event data, deep learning (PyTorch), and applied mathematics.

## Research Direction

**Module:** Math and Machine Learning Praktikum, University of Leipzig. **Collaborator:** Hannes.

**Goal:** Build a counterfactual goalkeeper positioning model `V(x, g) = P(goal | x, g)` that estimates the probability of a goal given the non-goalkeeper context `x` and goalkeeper position `g`. At inference, `x` is fixed and `g` is swept across a pitch grid to produce a danger heatmap; the optimal position is `g* = argmin_g V(x, g)`.

**Input decomposition from freeze frame:**
- `g` — goalkeeper (x, y) position (the variable being optimized)
- `x` — everything else: shooter/ball position, attacking teammates, outfield defenders (goalkeeper excluded)

**Architecture:** CNN on a rasterized pitch (~60×40 grid), 5 channels — one each for shooter+ball, attacking teammates, outfield defenders, goalkeeper. Player positions rendered as 2D Gaussian blobs.

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
# Additional packages used in notebooks (not yet in requirements.txt):
pip install mplsoccer squarify wordcloud
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
