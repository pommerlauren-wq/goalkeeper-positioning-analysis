"""
Aggregate counterfactual regret over a sample of test shots.

`run_sweep_on_examples` sweeps 200 shots only to pick 8 examples and discards
the full table. This script keeps it: it sweeps the same seeded 200-shot sample,
writes the per-shot regret CSV, and prints summary statistics that matter for
judging whether the model's g* recommendations are sensible at scale:

  - regret distribution (median / p90), split by goal vs no-goal
  - share of optima pinned to the y-edge of the grid (the v1 failure mode:
    g* slides behind the posts because V drops monotonically out of the mouth)
  - near / centre / far-post breakdown relative to the shooter (a far-side
    optimum is the anti-coaching pull v1 exhibited)

Usage:
    python src/analysis/regret_distribution.py models/checkpoints/baseline_v2/best.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.analysis.counterfactual import (
    DEFAULT_GRID_X,
    DEFAULT_GRID_Y,
    FREEZE_PATH,
    SHOTS_PATH,
    TEST_MANIFEST,
    _load_model,
    _select_device,
    sweep_gk_positions,
)

GOAL_CENTRE_Y = 40.0


def _post_side(shooter_y: float, gk_y: float) -> str:
    """Classify g* as near/centre/far relative to the shooter's side of goal."""
    if abs(gk_y - GOAL_CENTRE_Y) <= 2.0:
        return "centre"
    same_side = np.sign(shooter_y - GOAL_CENTRE_Y) == np.sign(gk_y - GOAL_CENTRE_Y)
    return "near" if same_side else "far"


def compute_regret_distribution(
    checkpoint_path: Path,
    sample_size: int = 200,
    seed: int = 42,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _REPO_ROOT / checkpoint_path

    device = _select_device()
    model = _load_model(checkpoint_path, device)
    print(f"Loaded {checkpoint_path.parent.name}/{checkpoint_path.name} on {device}")

    shots_df = pd.read_csv(SHOTS_PATH)
    freeze_df = pd.read_csv(FREEZE_PATH)
    shots_by_id = shots_df.set_index("id")

    # Reproduce the exact seeded sample run_sweep_on_examples draws.
    rng = np.random.default_rng(seed)
    test_ids = pd.read_csv(TEST_MANIFEST)["id"].tolist()
    idx = rng.choice(len(test_ids), size=min(sample_size, len(test_ids)), replace=False)
    sample_ids = [test_ids[i] for i in idx]

    y_lo, y_hi = float(DEFAULT_GRID_Y.min()), float(DEFAULT_GRID_Y.max())
    x_hi = float(DEFAULT_GRID_X.max())

    rows = []
    print(f"Sweeping {len(sample_ids)} shots on grid "
          f"x[{DEFAULT_GRID_X.min()},{x_hi}] y[{y_lo},{y_hi}]...")
    for i, sid in enumerate(sample_ids, start=1):
        shot = shots_df[shots_df["id"] == sid].iloc[0]
        fr = freeze_df[freeze_df["id"] == sid]
        r = sweep_gk_positions(model, shot, fr, device=device)
        gx, gy = r["optimal_gk_pos"]
        sy = float(shots_by_id.loc[sid, "x"]), float(shots_by_id.loc[sid, "y"])
        rows.append({
            "shot_id": sid,
            "is_goal": r["is_goal"],
            "shooter_x": sy[0],
            "shooter_y": sy[1],
            "optimal_gk_x": gx,
            "optimal_gk_y": gy,
            "actual_v": r["actual_v"],
            "optimal_v": r["optimal_v"],
            "regret": r["actual_v"] - r["optimal_v"],
            "y_pinned": bool(gy <= y_lo + 1e-6 or gy >= y_hi - 1e-6),
            "on_goal_line": bool(gx >= x_hi - 1e-6),
            "post_side": _post_side(sy[1], gy),
        })
        if i % 50 == 0 or i == len(sample_ids):
            print(f"  {i}/{len(sample_ids)} swept")

    df = pd.DataFrame(rows)
    if output_csv is None:
        output_csv = _REPO_ROOT / "results/counterfactual/v2_geom/regret_200.csv"
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    _print_summary(df, output_csv)
    return df


def _print_summary(df: pd.DataFrame, output_csv: Path) -> None:
    n = len(df)
    goals = df[df.is_goal == 1]
    saves = df[df.is_goal == 0]

    def q(s, p):
        return float(np.percentile(s, p)) if len(s) else float("nan")

    print(f"\n=== Regret distribution (n={n}) ===")
    print(f"{'group':<12}{'n':>5}{'median':>10}{'p90':>10}{'max':>10}")
    for name, g in (("all", df), ("goals", goals), ("saves", saves)):
        print(f"{name:<12}{len(g):>5}{g.regret.median():>10.4f}"
              f"{q(g.regret, 90):>10.4f}{g.regret.max():>10.4f}")

    pinned = int(df.y_pinned.sum())
    far = int((df.post_side == "far").sum())
    on_line = int(df.on_goal_line.sum())
    side_counts = df.post_side.value_counts().to_dict()
    print(f"\ng* pinned to y-edge (v1 failure mode): {pinned}/{n} "
          f"({100*pinned/n:.1f}%)")
    print(f"g* on far side of shooter (anti-coaching): {far}/{n} "
          f"({100*far/n:.1f}%)")
    print(f"post-side breakdown: {side_counts}")
    print(f"g* on the goal line (x={df.optimal_gk_x.max():.1f}): {on_line}/{n} "
          f"({100*on_line/n:.1f}%)")
    print(f"\nSaved per-shot table to {output_csv}")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/checkpoints/baseline_v2/best.pt"
    compute_regret_distribution(ckpt)
