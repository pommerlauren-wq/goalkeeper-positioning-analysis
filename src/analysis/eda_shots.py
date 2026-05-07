"""
EDA for shots_master_df.csv and freeze_master_df.csv.

Outputs:
  results/eda/eda_summary.md
  results/eda/shot_heatmap.png
  results/eda/xg_distribution.png
  results/eda/players_per_shot.png
"""

import os
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import VerticalPitch

OUT_DIR = "results/eda"
os.makedirs(OUT_DIR, exist_ok=True)

SHOTS_PATH = "data/raw/shots_master_df.csv"
FREEZE_PATH = "data/raw/freeze_master_df.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pct(n, total):
    return f"{n:,} ({100 * n / total:.1f}%)" if total else "0"


def section(title):
    bar = "=" * 60
    print(f"\n{bar}\n{title}\n{bar}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

section("Loading data")
shots = pd.read_csv(SHOTS_PATH)
freeze = pd.read_csv(FREEZE_PATH)
print(f"Shots  : {shots.shape[0]:,} rows × {shots.shape[1]} cols")
print(f"Freeze : {freeze.shape[0]:,} rows × {freeze.shape[1]} cols")


# ---------------------------------------------------------------------------
# 1. Basic info
# ---------------------------------------------------------------------------

section("1. Basic info — shots")
print(shots.dtypes.to_string())
missing_shots = shots.isnull().sum()
missing_shots = missing_shots[missing_shots > 0].sort_values(ascending=False)
print("\nMissing values (shots):")
print(missing_shots.to_string())

section("1. Basic info — freeze frames")
print(freeze.dtypes.to_string())
missing_freeze = freeze.isnull().sum()
missing_freeze = missing_freeze[missing_freeze > 0].sort_values(ascending=False)
print("\nMissing values (freeze):")
print(missing_freeze.to_string() if len(missing_freeze) else "  none")


# ---------------------------------------------------------------------------
# 2. Shot counts and sub-type breakdown
# ---------------------------------------------------------------------------

section("2. Shot sub-type breakdown")
total_shots = len(shots)
print(f"Total shots: {total_shots:,}")

subtype_counts = shots["sub_type_name"].value_counts(dropna=False)
print("\nBreakdown by sub_type_name:")
for name, count in subtype_counts.items():
    print(f"  {str(name):<25} {pct(count, total_shots)}")


# ---------------------------------------------------------------------------
# 3. Goal conversion rates
# ---------------------------------------------------------------------------

section("3. Goal conversion rates")
shots["is_goal"] = shots["outcome_name"] == "Goal"

overall_rate = shots["is_goal"].mean()
print(f"Overall conversion rate: {overall_rate:.1%}")

print("\nConversion rate by sub_type_name:")
conv_by_subtype = (
    shots.groupby("sub_type_name")["is_goal"]
    .agg(shots_n="count", goals_n="sum")
    .assign(conversion=lambda d: d["goals_n"] / d["shots_n"])
    .sort_values("shots_n", ascending=False)
)
print(conv_by_subtype.to_string())


# ---------------------------------------------------------------------------
# 4a. Shot location heatmap
# ---------------------------------------------------------------------------

section("4a. Shot location heatmap")
open_play = shots[shots["sub_type_name"] == "Open Play"].copy()
print(f"Open-play shots for heatmap: {len(open_play):,}")

pitch = VerticalPitch(
    pitch_type="statsbomb",
    half=True,
    line_color="black",
    line_zorder=2,
)
fig, ax = pitch.draw(figsize=(6, 8))

bin_stat = pitch.bin_statistic(open_play["x"], open_play["y"], bins=(40, 40))
pcm = pitch.heatmap(bin_stat, ax=ax, cmap="Reds", edgecolors="none")
fig.colorbar(pcm, ax=ax, fraction=0.03, label="Shot count")
ax.set_title("Open-play shot locations (StatsBomb data)", fontsize=13)

heatmap_path = f"{OUT_DIR}/shot_heatmap.png"
fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {heatmap_path}")


# ---------------------------------------------------------------------------
# 4b. xG distribution
# ---------------------------------------------------------------------------

section("4b. xG distribution")
xg = shots["shot_statsbomb_xg"].dropna()
print(f"xG available for {len(xg):,} / {total_shots:,} shots")
print(f"  mean={xg.mean():.4f}  median={xg.median():.4f}  max={xg.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(xg, bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
ax.set_xlabel("StatsBomb xG")
ax.set_ylabel("Count")
ax.set_title("Distribution of StatsBomb xG across all shots")
ax.axvline(xg.median(), color="red", linestyle="--", label=f"median={xg.median():.3f}")
ax.legend()
fig.tight_layout()

xg_path = f"{OUT_DIR}/xg_distribution.png"
fig.savefig(xg_path, dpi=150)
plt.close(fig)
print(f"Saved: {xg_path}")


# ---------------------------------------------------------------------------
# 5. Freeze frame coverage
# ---------------------------------------------------------------------------

section("5. Freeze frame coverage")
freeze_shot_ids = freeze["id"].unique()
shots_with_freeze = shots["id"].isin(freeze_shot_ids).sum()
print(f"Shots with at least one freeze frame entry : {pct(shots_with_freeze, total_shots)}")
print(f"Shots without freeze frames                : {pct(total_shots - shots_with_freeze, total_shots)}")

players_per_shot = freeze.groupby("id").size()
print(f"\nVisible players per shot:")
print(f"  min={players_per_shot.min()}  "
      f"median={players_per_shot.median():.0f}  "
      f"mean={players_per_shot.mean():.1f}  "
      f"max={players_per_shot.max()}")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(players_per_shot.values, bins=range(1, players_per_shot.max() + 2),
        color="steelblue", edgecolor="white", linewidth=0.3, align="left")
ax.set_xlabel("Visible players in freeze frame")
ax.set_ylabel("Number of shots")
ax.set_title("Distribution of visible players per shot freeze frame")
ax.axvline(players_per_shot.median(), color="red", linestyle="--",
           label=f"median={players_per_shot.median():.0f}")
ax.legend()
fig.tight_layout()

players_path = f"{OUT_DIR}/players_per_shot.png"
fig.savefig(players_path, dpi=150)
plt.close(fig)
print(f"Saved: {players_path}")


# ---------------------------------------------------------------------------
# 6. Goalkeeper identifiability in open-play freeze frames
# ---------------------------------------------------------------------------

section("6. Goalkeeper identifiability (open-play shots)")
open_play_ids = shots.loc[shots["sub_type_name"] == "Open Play", "id"]
freeze_open_play = freeze[freeze["id"].isin(open_play_ids)]

open_play_with_freeze = freeze_open_play["id"].nunique()
total_open_play = len(open_play_ids)
print(f"Open-play shots              : {total_open_play:,}")
print(f"  with freeze frame          : {pct(open_play_with_freeze, total_open_play)}")

gk_mask = (freeze_open_play["position_name"] == "Goalkeeper") & (freeze_open_play["teammate"] == False)
shots_with_gk = freeze_open_play.loc[gk_mask, "id"].nunique()
print(f"  with identifiable opp. GK : {pct(shots_with_gk, total_open_play)}")
print(f"    (% of those with freeze) : {100 * shots_with_gk / open_play_with_freeze:.1f}%"
      if open_play_with_freeze else "")


# ---------------------------------------------------------------------------
# 7. Final modeling sample (joined, open-play, GK visible)
# ---------------------------------------------------------------------------

section("7. Final modeling sample")
gk_shot_ids = freeze_open_play.loc[gk_mask, "id"].unique()
modeling_shots = shots[shots["id"].isin(gk_shot_ids)].copy()
n_modeling = len(modeling_shots)

modeling_goal_rate = modeling_shots["is_goal"].mean()
print(f"Final sample size            : {n_modeling:,}")
print(f"Goal conversion rate         : {modeling_goal_rate:.1%}")

if "match_id" in modeling_shots.columns:
    n_matches = modeling_shots["match_id"].nunique()
    print(f"Unique matches               : {n_matches:,}")

# Sub-type sanity check (should be all Open Play)
print("\nSub-type breakdown in final sample:")
print(modeling_shots["sub_type_name"].value_counts().to_string())

# Recommendation
TARGET_LOW, TARGET_HIGH = 5_000, 10_000
if n_modeling >= TARGET_HIGH:
    verdict = (f"YES — {n_modeling:,} samples comfortably exceeds the {TARGET_HIGH:,} target. "
               "Proceed with CNN training.")
elif n_modeling >= TARGET_LOW:
    verdict = (f"MARGINAL — {n_modeling:,} samples meets the minimum {TARGET_LOW:,} threshold "
               "but is below the comfortable {TARGET_HIGH:,}. "
               "Consider data augmentation or broadening competition scope before full CNN training.")
else:
    verdict = (f"NO — {n_modeling:,} samples is below the minimum {TARGET_LOW:,} threshold. "
               "Expand the dataset (more competitions/seasons) before proceeding with the CNN.")

print(f"\nRECOMMENDATION:\n  {verdict}")


# ---------------------------------------------------------------------------
# 8. Write eda_summary.md
# ---------------------------------------------------------------------------

summary_path = f"{OUT_DIR}/eda_summary.md"
with open(summary_path, "w") as f:
    f.write("# EDA Summary\n\n")

    f.write("## Dataset sizes\n\n")
    f.write(f"| Dataset | Rows | Columns |\n|---|---|---|\n")
    f.write(f"| shots_master_df | {shots.shape[0]:,} | {shots.shape[1]} |\n")
    f.write(f"| freeze_master_df | {freeze.shape[0]:,} | {freeze.shape[1]} |\n\n")

    f.write("## Shot sub-type breakdown\n\n")
    f.write("| Sub-type | Count | % of total |\n|---|---|---|\n")
    for name, count in subtype_counts.items():
        f.write(f"| {name} | {count:,} | {100*count/total_shots:.1f}% |\n")
    f.write("\n")

    f.write("## Goal conversion rates by sub-type\n\n")
    f.write("| Sub-type | Shots | Goals | Conversion |\n|---|---|---|---|\n")
    for subtype, row in conv_by_subtype.iterrows():
        f.write(f"| {subtype} | {int(row.shots_n):,} | {int(row.goals_n):,} | {row.conversion:.1%} |\n")
    f.write("\n")

    f.write("## xG distribution\n\n")
    f.write(f"- Available for: {len(xg):,} / {total_shots:,} shots\n")
    f.write(f"- Mean: {xg.mean():.4f}\n")
    f.write(f"- Median: {xg.median():.4f}\n")
    f.write(f"- Max: {xg.max():.4f}\n\n")

    f.write("## Freeze frame coverage\n\n")
    f.write(f"- Shots with freeze frames: {shots_with_freeze:,} / {total_shots:,} "
            f"({100*shots_with_freeze/total_shots:.1f}%)\n")
    f.write(f"- Visible players per shot: min={players_per_shot.min()}, "
            f"median={players_per_shot.median():.0f}, max={players_per_shot.max()}\n\n")

    f.write("## Goalkeeper identifiability (open-play)\n\n")
    f.write(f"- Open-play shots: {total_open_play:,}\n")
    f.write(f"- With freeze frame: {open_play_with_freeze:,} ({100*open_play_with_freeze/total_open_play:.1f}%)\n")
    f.write(f"- With identifiable opponent GK: {shots_with_gk:,} "
            f"({100*shots_with_gk/total_open_play:.1f}% of open-play, "
            f"{100*shots_with_gk/open_play_with_freeze:.1f}% of those with freeze frames)\n\n")

    f.write("## Final modeling sample\n\n")
    f.write(f"- **Sample size: {n_modeling:,}**\n")
    f.write(f"- Goal conversion rate: {modeling_goal_rate:.1%}\n")
    if "match_id" in modeling_shots.columns:
        f.write(f"- Unique matches: {n_matches:,}\n")
    f.write("\n")

    f.write("## Recommendation\n\n")
    f.write(textwrap.fill(verdict, width=80) + "\n")

print(f"\nSaved: {summary_path}")
section("Done")
