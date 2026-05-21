"""
Partition open-play shots into train/val/test and transfer evaluation sets.

Outputs (data/processed/splits/):
  train_shot_ids.csv
  val_shot_ids.csv
  test_shot_ids.csv
  transfer_women_shot_ids.csv
  transfer_men_other_shot_ids.csv
  README.md
"""

import os
import textwrap

import pandas as pd
from mplsoccer import Sbopen
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
SHOTS_PATH = "data/raw/shots_master_df.csv"
FREEZE_PATH = "data/raw/freeze_master_df.csv"
OUT_DIR = "data/processed/splits"
os.makedirs(OUT_DIR, exist_ok=True)

WOMEN_COMPETITIONS = {
    "FA Women's Super League",
    "Women's World Cup",
    "UEFA Women's Euro",
    "NWSL",
}

MEN_2015_16_TOP5 = {
    ("La Liga", "2015/2016"),
    ("Premier League", "2015/2016"),
    ("Serie A", "2015/2016"),
    ("Ligue 1", "2015/2016"),
    ("1. Bundesliga", "2015/2016"),
}


# ---------------------------------------------------------------------------
# Step 1: Load and filter shots
# ---------------------------------------------------------------------------

print("Loading shots and freeze frames...")
shots = pd.read_csv(SHOTS_PATH)
freeze = pd.read_csv(FREEZE_PATH, usecols=["id", "teammate", "position_name"])

open_play = shots[shots["sub_type_name"] == "Open Play"].copy()
print(f"  Open-play shots          : {len(open_play):,}")

freeze_ids = set(freeze["id"].unique())
open_play = open_play[open_play["id"].isin(freeze_ids)]
print(f"  With freeze frame        : {len(open_play):,}")

gk_ids = freeze.loc[
    (freeze["position_name"] == "Goalkeeper") & (freeze["teammate"] == False), "id"
].unique()
open_play = open_play[open_play["id"].isin(gk_ids)].copy()
print(f"  With identifiable GK     : {len(open_play):,}")

open_play["is_goal"] = (open_play["outcome_name"] == "Goal").astype(int)


# ---------------------------------------------------------------------------
# Step 2 & 3: Fetch match metadata and assign tier
# ---------------------------------------------------------------------------

print("\nFetching match metadata from StatsBomb API...")
parser = Sbopen()
df_comp = parser.competition()

meta_rows = []
for _, row in df_comp.iterrows():
    try:
        m = parser.match(row["competition_id"], row["season_id"])
        m["competition_name"] = row["competition_name"]
        m["season_name"] = row["season_name"]
        m["competition_gender"] = row["competition_gender"]
        meta_rows.append(m[["match_id", "competition_name", "season_name", "competition_gender"]])
    except Exception:
        continue

meta = pd.concat(meta_rows, ignore_index=True)
meta = meta.drop_duplicates("match_id")

open_play = open_play.merge(
    meta[["match_id", "competition_name", "season_name", "competition_gender"]],
    on="match_id",
    how="left",
)

missing_meta = open_play["competition_name"].isna().sum()
if missing_meta:
    print(f"  WARNING: {missing_meta} shots could not be matched to competition metadata")

# Gender: use API field; fall back to competition name inference
open_play["gender"] = open_play.apply(
    lambda r: "female"
    if r["competition_gender"] == "female"
    or r["competition_name"] in WOMEN_COMPETITIONS
    else "male",
    axis=1,
)

# Competition tier
def assign_tier(row):
    if row["gender"] == "female":
        return "women"
    if (row["competition_name"], row["season_name"]) in MEN_2015_16_TOP5:
        return "men_2015_16_top5"
    return "men_tournament_other"

open_play["tier"] = open_play.apply(assign_tier, axis=1)

tier_counts = open_play["tier"].value_counts()
print("\nShots by tier:")
for tier, n in tier_counts.items():
    print(f"  {tier:<25} {n:,}")


# ---------------------------------------------------------------------------
# Step 4: Build splits
# ---------------------------------------------------------------------------

MANIFEST_COLS = ["id", "match_id", "competition_name", "season_name", "is_goal"]

# --- Main split: men_2015_16_top5 ---
main = open_play[open_play["tier"] == "men_2015_16_top5"].copy()

# Compute per-match goal rate for stratification
match_stats = (
    main.groupby("match_id")["is_goal"]
    .agg(shots_n="count", goals_n="sum")
    .assign(goal_rate=lambda d: d["goals_n"] / d["shots_n"])
    .reset_index()
)
# Bin into quartiles so train_test_split can stratify at match level
match_stats["rate_bin"] = pd.qcut(
    match_stats["goal_rate"], q=4, labels=False, duplicates="drop"
)

train_match_ids, temp_match_ids = train_test_split(
    match_stats["match_id"],
    test_size=0.30,
    stratify=match_stats["rate_bin"],
    random_state=RANDOM_SEED,
)
temp_stats = match_stats[match_stats["match_id"].isin(temp_match_ids)]
val_match_ids, test_match_ids = train_test_split(
    temp_stats["match_id"],
    test_size=0.50,
    stratify=temp_stats["rate_bin"],
    random_state=RANDOM_SEED,
)

train_shots = main[main["match_id"].isin(train_match_ids)][MANIFEST_COLS]
val_shots   = main[main["match_id"].isin(val_match_ids)][MANIFEST_COLS]
test_shots  = main[main["match_id"].isin(test_match_ids)][MANIFEST_COLS]

# --- Transfer sets ---
transfer_women     = open_play[open_play["tier"] == "women"][MANIFEST_COLS]
transfer_men_other = open_play[open_play["tier"] == "men_tournament_other"][MANIFEST_COLS]


# ---------------------------------------------------------------------------
# Step 5: Save manifests
# ---------------------------------------------------------------------------

splits = {
    "train":              train_shots,
    "val":                val_shots,
    "test":               test_shots,
    "transfer_women":     transfer_women,
    "transfer_men_other": transfer_men_other,
}

file_map = {
    "train":              "train_shot_ids.csv",
    "val":                "val_shot_ids.csv",
    "test":               "test_shot_ids.csv",
    "transfer_women":     "transfer_women_shot_ids.csv",
    "transfer_men_other": "transfer_men_other_shot_ids.csv",
}

for key, df in splits.items():
    path = os.path.join(OUT_DIR, file_map[key])
    df.to_csv(path, index=False)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Step 6: Summary table
# ---------------------------------------------------------------------------

def competitions_str(df):
    return ", ".join(
        sorted(f"{c} {s}" for c, s in df[["competition_name", "season_name"]].drop_duplicates().itertuples(index=False))
    )

print("\n" + "=" * 70)
print(f"{'Split':<22} {'Shots':>8} {'Goals':>7} {'Rate':>7} {'Matches':>8}")
print("=" * 70)

for key, df in splits.items():
    n = len(df)
    g = df["is_goal"].sum()
    rate = g / n if n else 0
    m = df["match_id"].nunique()
    print(f"{key:<22} {n:>8,} {g:>7,} {rate:>7.1%} {m:>8,}")

print("=" * 70)

print("\nCompetitions per split:")
for key, df in splits.items():
    comps = ", ".join(
        sorted(df["competition_name"].unique())
    )
    print(f"  {key}: {comps}")


# ---------------------------------------------------------------------------
# Step 8: Write README.md
# ---------------------------------------------------------------------------

readme_lines = [
    "# Data Splits",
    "",
    "Generated by `src/data/build_splits.py` (seed=42).",
    "",
    "## Filtering",
    "",
    "Starting from `shots_master_df.csv`, shots were kept if they satisfy all of:",
    "- `sub_type_name == 'Open Play'`",
    "- Has at least one entry in `freeze_master_df.csv`",
    "- Has an identifiable opponent goalkeeper in the freeze frame",
    "  (`position_name == 'Goalkeeper'` and `teammate == False`)",
    "",
    f"**Total usable shots: {len(open_play):,}**",
    "",
    "## Competition Tiers",
    "",
    "| Tier | Description |",
    "|---|---|",
    "| `men_2015_16_top5` | La Liga, Premier League, Serie A, Ligue 1, Bundesliga — all 2015/16 season |",
    "| `men_tournament_other` | All other men's matches (World Cups, Euros, Champions League, historical, etc.) |",
    "| `women` | All women's competitions (FA WSL, Women's World Cup, UEFA Women's Euro, NWSL) |",
    "",
    "## Split Logic",
    "",
    "The **main model** is trained and evaluated on `men_2015_16_top5` only.",
    "Splits are at **match level** — all shots from the same match go to the same split,",
    "preventing data leakage from temporal and tactical correlations within matches.",
    "Matches were stratified by binned per-match goal rate (quartiles) before splitting.",
    "",
    "| Split | Ratio | Shots | Goals | Rate | Matches |",
    "|---|---|---|---|---|---|",
]

ratio_map = {"train": "70%", "val": "15%", "test": "15%", "transfer_women": "100%", "transfer_men_other": "100%"}
for key, df in splits.items():
    n = len(df)
    g = int(df["is_goal"].sum())
    rate = g / n if n else 0
    m = df["match_id"].nunique()
    readme_lines.append(
        f"| `{file_map[key]}` | {ratio_map[key]} | {n:,} | {g:,} | {rate:.1%} | {m:,} |"
    )

readme_lines += [
    "",
    "## Transfer Evaluation Sets",
    "",
    "These sets are held out entirely during training. They test generalisation across:",
    "- **`transfer_women_shot_ids.csv`**: domain shift (women's football — different body size,",
    "  positioning tendencies, shot profiles). Do not tune on this set.",
    "- **`transfer_men_other_shot_ids.csv`**: distribution shift across competition formats",
    "  (knockout tournaments, historical data, non-top-5 leagues). Do not tune on this set.",
    "",
    "## Intended Workflow",
    "",
    "1. Train on `train_shot_ids.csv`; tune hyperparameters using `val_shot_ids.csv`.",
    "2. Report final in-distribution performance on `test_shot_ids.csv` (run once).",
    "3. Report transfer performance on `transfer_women_shot_ids.csv` and",
    "   `transfer_men_other_shot_ids.csv` after the model is finalised.",
]

readme_path = os.path.join(OUT_DIR, "README.md")
with open(readme_path, "w") as f:
    f.write("\n".join(readme_lines) + "\n")
print(f"\nSaved {readme_path}")
