"""
Report-ready figures for the write-up / supervisor briefing.

Two figures, both saved into reports/figures/:

1. cf_v2_vs_v3_<shot>.png — the headline finding. The SAME shot swept under
   baseline_v2 and baseline_v3, side by side on a shared colour scale. v2 keeps
   g* on the shooter's (near) side; v3's auxiliary scalar head pulls g* to the
   far post — the anti-coaching pathology. This is the v2/v3 counterfactual
   tension made visual; in the 200-shot sample it is 0% (v2) vs 19.5% (v3).

2. far_side_by_model.png — bar chart of the far-side / anti-coaching optimum
   rate across v2 / v3 / v3b, read straight from each model's regret_200.csv.

Run:
    python src/analysis/report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.counterfactual import (
    FREEZE_PATH,
    SHOTS_PATH,
    _draw_pitch_overlays,
    _load_model,
    _select_device,
    sweep_gk_positions,
)

FIG_DIR = _REPO_ROOT / "reports/figures"

# Default contrast shot: wide shot (shooter y=27.5). v2 keeps g* near-post side
# (y=36.5); v3 pulls it to the far post (y=46.0). Picked from the 200-shot sample
# as the clearest v3-far / v2-not-far case (see regret_200.csv cross-reference).
DEFAULT_CONTRAST_SHOT = "1e582bc0-e319-4c39-9322-9bf6954b19d0"

GOAL_CENTRE_Y = 40.0


def _panel(ax, fig, sweep, shooter_y, title, vmin, vmax):
    """Draw one V(x,g) heatmap panel with actual GK, g*, and post labels."""
    gx, gy = sweep["gk_grid_x"], sweep["gk_grid_y"]
    danger = sweep["danger_grid"]
    a_gx, a_gy = sweep["actual_gk_pos"]
    o_gx, o_gy = sweep["optimal_gk_pos"]

    im = ax.pcolormesh(gx, gy, danger, cmap="Reds", shading="auto", vmin=vmin, vmax=vmax)
    _draw_pitch_overlays(ax)
    ax.scatter([a_gx], [a_gy], s=170, color="#2ca02c", marker="o",
               edgecolors="black", linewidths=1.0, label="actual GK", zorder=6)
    ax.scatter([o_gx], [o_gy], s=320, color="gold", marker="*",
               edgecolors="black", linewidths=1.2, label="g* (model optimum)", zorder=7)

    # Which post is nearer the shooter? (posts at y=36 and y=44)
    near_post_y = 36.0 if shooter_y < GOAL_CENTRE_Y else 44.0
    far_post_y = 44.0 if shooter_y < GOAL_CENTRE_Y else 36.0
    ax.annotate("near post", (120.0, near_post_y), xytext=(118.4, near_post_y),
                fontsize=8, ha="right", va="center", color="black",
                fontweight="bold", zorder=8)
    ax.annotate("far post", (120.0, far_post_y), xytext=(118.4, far_post_y),
                fontsize=8, ha="right", va="center", color="black", zorder=8)

    ax.set_xlim(gx.min(), gx.max())
    ax.set_ylim(gy.max(), gy.min())  # invert y to match StatsBomb pitch
    ax.set_xlabel("GK x (m)  — penalty spot → goal line")
    ax.set_ylabel("GK y (m)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    return im


def make_comparison_figure(shot_id: str = DEFAULT_CONTRAST_SHOT) -> Path:
    device = _select_device()
    v2 = _load_model(_REPO_ROOT / "models/checkpoints/baseline_v2/best.pt", device)
    v3 = _load_model(_REPO_ROOT / "models/checkpoints/baseline_v3/best.pt", device)

    shots_df = pd.read_csv(SHOTS_PATH)
    freeze_df = pd.read_csv(FREEZE_PATH)
    shot = shots_df[shots_df["id"] == shot_id].iloc[0]
    fr = freeze_df[freeze_df["id"] == shot_id]
    shooter_x, shooter_y = float(shot["x"]), float(shot["y"])

    s2 = sweep_gk_positions(v2, shot, fr, device=device)
    s3 = sweep_gk_positions(v3, shot, fr, device=device)

    vmin = min(s2["danger_grid"].min(), s3["danger_grid"].min())
    vmax = max(s2["danger_grid"].max(), s3["danger_grid"].max())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6))
    _panel(axL, fig, s2, shooter_y,
           f"baseline_v2  →  g* = ({s2['optimal_gk_pos'][0]:.1f}, "
           f"{s2['optimal_gk_pos'][1]:.1f})   [near-post side]", vmin, vmax)
    im = _panel(axR, fig, s3, shooter_y,
                f"baseline_v3  →  g* = ({s3['optimal_gk_pos'][0]:.1f}, "
                f"{s3['optimal_gk_pos'][1]:.1f})   [far post — anti-coaching]",
                vmin, vmax)

    cbar = fig.colorbar(im, ax=[axL, axR], fraction=0.046, pad=0.02)
    cbar.set_label("V(x, g) = P(goal | x, g)")

    is_goal = "GOAL" if int(shot.get("outcome_name") == "Goal") else "no goal"
    fig.suptitle(
        f"Same shot, two models: wide shot from (x={shooter_x:.0f}, y={shooter_y:.0f}), "
        f"{is_goal}.  v2 keeps the keeper on the shooter's side to cut the angle; "
        f"v3's scalar head pulls g* to the far post.\n"
        f"Across 200 shots this far-side pathology is 0% (v2) vs 19.5% (v3) — "
        f"the cost of the auxiliary predictive head.",
        fontsize=10.5, y=1.02,
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"cf_v2_vs_v3_{shot_id[:8]}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


def make_far_side_barchart() -> Path:
    """Bar chart of far-side / pinned optimum rates across v2, v3, v3b."""
    models = {
        "baseline_v2\n(geometry only)": "results/counterfactual/v2_geom/regret_200.csv",
        "baseline_v3\n(all scalars)": "results/counterfactual/v3_scalar/regret_200.csv",
        "baseline_v3b\n(context scalars)": "results/counterfactual/v3b_context/regret_200.csv",
    }
    labels, far_pct, pin_pct = [], [], []
    for label, rel in models.items():
        df = pd.read_csv(_REPO_ROOT / rel)
        n = len(df)
        labels.append(label)
        far_pct.append(100 * (df.post_side == "far").sum() / n)
        pin_pct.append(100 * df.y_pinned.sum() / n)

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    b1 = ax.bar(x - w / 2, far_pct, w, label="far-side g* (anti-coaching)",
                color="#d62728")
    b2 = ax.bar(x + w / 2, pin_pct, w, label="g* pinned to grid edge",
                color="#7f7f7f")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("share of 200-shot test sample (%)")
    ax.set_ylim(0, max(far_pct) * 1.25)
    ax.set_title("Counterfactual pathology rate by model\n"
                 "Lower is better — a good positioning recommender keeps g* "
                 "sensible, not just P(goal) accurate", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "far_side_by_model.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    make_comparison_figure()
    make_far_side_barchart()
