"""
Scalar shot/keeper geometry features for the DangerCNN's parallel head (Tier 2).

The position-only raster captures *where* players are but expresses shot
difficulty and keeper-angle geometry only implicitly. This module computes a
small fixed-length vector of explicit features that a parallel MLP consumes
alongside the CNN embedding.

Three features depend on the goalkeeper position `g` (gk_coverage,
gk_perp_offset, gk_depth) and so must be recomputed when `g` is swept in the
counterfactual analysis; their indices are exposed as GK_DEPENDENT_IDX. The
rest depend only on the fixed context `x` and the shot itself.

All features are scaled to roughly O(1) with fixed constants (no train-set
statistics needed), so the vector is reproducible without a fitted scaler.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.data.rasterize import (
    GOAL_CENTER,
    GOAL_LINE_X,
    POST_LEFT_Y,
    POST_RIGHT_Y,
    _identify_goalkeeper,
)

SCALAR_FEATURE_NAMES = [
    "dist_to_goal",     # ball -> goal centre, / 60
    "shot_angle",       # angle subtended by the posts from the ball, / pi
    "gk_coverage",      # fraction of the goal the GK occludes from the shooter
    "gk_perp_offset",   # GK distance off the ball->goal-centre line, / 10
    "gk_depth",         # GK -> goal centre, / 15
    "bp_right",         # body part one-hots (Other -> all zero)
    "bp_left",
    "bp_head",
    "under_pressure",
]
# gk_coverage, gk_perp_offset, gk_depth depend on the keeper position.
GK_DEPENDENT_IDX = (2, 3, 4)

# Named subsets (indices into the full vector above):
#   "all"     — every feature (v3).
#   "context" — only the g-INDEPENDENT shot-difficulty features (v3b). Dropping
#               the GK-dependent scalars removes the second, non-spatial route
#               to encode keeper position that degraded v3's counterfactuals.
FEATURE_SETS: dict[str, tuple[int, ...]] = {
    "all": tuple(range(len(SCALAR_FEATURE_NAMES))),
    "context": (0, 1, 5, 6, 7, 8),
}


def scalar_dim(feature_set: str = "all") -> int:
    return len(FEATURE_SETS[feature_set])


def feature_names(feature_set: str = "all") -> list[str]:
    return [SCALAR_FEATURE_NAMES[i] for i in FEATURE_SETS[feature_set]]


# Backwards-compatible default (full set).
SCALAR_DIM = scalar_dim("all")

# Effective half-width (m) of a keeper's reach when computing angular coverage;
# ~1.6 m total span.
GK_HALF_WIDTH = 0.8


def _angle_between(ax: float, ay: float, bx: float, by: float) -> float:
    """Unsigned angle (radians) between vectors a and b."""
    dot = ax * bx + ay * by
    cross = ax * by - ay * bx
    return math.atan2(abs(cross), dot)


def _gk_coverage(sx: float, sy: float, gx: float, gy: float) -> float:
    """Fraction of the goal's angular width (from the shooter) the GK blocks."""
    # Goal angular interval as seen from the shooter.
    aL = math.atan2(POST_LEFT_Y - sy, GOAL_LINE_X - sx)
    aR = math.atan2(POST_RIGHT_Y - sy, GOAL_LINE_X - sx)
    goal_lo, goal_hi = min(aL, aR), max(aL, aR)
    goal_w = goal_hi - goal_lo
    if goal_w <= 1e-6:
        return 0.0
    # GK angular interval: centre direction +/- half-width subtended at range.
    dist_sg = math.hypot(gx - sx, gy - sy)
    if dist_sg <= 1e-6:
        return 1.0
    aG = math.atan2(gy - sy, gx - sx)
    half_w = math.atan2(GK_HALF_WIDTH, dist_sg)
    gk_lo, gk_hi = aG - half_w, aG + half_w
    overlap = max(0.0, min(gk_hi, goal_hi) - max(gk_lo, goal_lo))
    return float(min(1.0, overlap / goal_w))


def _perp_offset(sx: float, sy: float, gx: float, gy: float) -> float:
    """Perpendicular distance (m) of G from the ball->goal-centre line."""
    cx, cy = GOAL_CENTER
    dx, dy = cx - sx, cy - sy
    denom = math.hypot(dx, dy)
    if denom <= 1e-6:
        return 0.0
    return abs(dx * (gy - sy) - dy * (gx - sx)) / denom


def compute_scalar_features(
    shot_row: pd.Series,
    freeze_rows: pd.DataFrame,
    gk_pos: tuple[float, float] | None = None,
    feature_set: str = "all",
) -> np.ndarray:
    """Return the float32 feature vector for one shot, sliced to `feature_set`.

    If `gk_pos` is given it overrides the freeze-frame keeper position (used to
    sweep g in the counterfactual analysis); otherwise the actual GK is used.
    For feature_set="context" the GK-dependent entries are computed then dropped,
    so the result is invariant to gk_pos.
    """
    sx, sy = float(shot_row["x"]), float(shot_row["y"])
    if gk_pos is None:
        gk_row = _identify_goalkeeper(freeze_rows)
        gx, gy = float(gk_row["x"]), float(gk_row["y"])
    else:
        gx, gy = float(gk_pos[0]), float(gk_pos[1])

    cx, cy = GOAL_CENTER
    dist_to_goal = math.hypot(cx - sx, cy - sy)
    shot_angle = _angle_between(
        GOAL_LINE_X - sx, POST_LEFT_Y - sy, GOAL_LINE_X - sx, POST_RIGHT_Y - sy
    )
    gk_coverage = _gk_coverage(sx, sy, gx, gy)
    gk_perp_offset = _perp_offset(sx, sy, gx, gy)
    gk_depth = math.hypot(cx - gx, cy - gy)

    bp = shot_row.get("body_part_name", None)
    bp_right = 1.0 if bp == "Right Foot" else 0.0
    bp_left = 1.0 if bp == "Left Foot" else 0.0
    bp_head = 1.0 if bp == "Head" else 0.0

    up = shot_row.get("under_pressure", float("nan"))
    under_pressure = 1.0 if up == 1 or up == 1.0 else 0.0

    full = np.array(
        [
            dist_to_goal / 60.0,
            shot_angle / math.pi,
            gk_coverage,
            gk_perp_offset / 10.0,
            gk_depth / 15.0,
            bp_right,
            bp_left,
            bp_head,
            under_pressure,
        ],
        dtype=np.float32,
    )
    return full[list(FEATURE_SETS[feature_set])]


if __name__ == "__main__":
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    shots = pd.read_csv(repo / "data/raw/shots_master_df.csv")
    freeze = pd.read_csv(repo / "data/raw/freeze_master_df.csv")
    sid = pd.read_csv(repo / "data/processed/splits/train_shot_ids.csv")["id"].iloc[0]
    shot = shots[shots["id"] == sid].iloc[0]
    fr = freeze[freeze["id"] == sid]

    v = compute_scalar_features(shot, fr)
    print(f"SCALAR_DIM={SCALAR_DIM}  GK_DEPENDENT_IDX={GK_DEPENDENT_IDX}")
    for name, val in zip(SCALAR_FEATURE_NAMES, v):
        print(f"  {name:<16} {val:+.4f}")
    # Sweeping g should change only the GK-dependent entries.
    v2 = compute_scalar_features(shot, fr, gk_pos=(116.0, 40.0))
    changed = [SCALAR_FEATURE_NAMES[i] for i in range(SCALAR_DIM) if abs(v[i] - v2[i]) > 1e-9]
    print(f"changed when g moved: {changed}")
