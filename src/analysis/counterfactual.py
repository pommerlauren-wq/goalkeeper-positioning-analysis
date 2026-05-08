"""
Counterfactual goalkeeper-position sweep.

For a single shot, hold the non-GK context (shooter, ball, attacking teammates,
defenders) fixed and evaluate V(x, g) = P(goal | x, g) over a 2D grid of
candidate goalkeeper positions g. The optimal position is g* = argmin V(x, g),
and regret = V(x, g_actual) - V(x, g*).

Default GK grid covers the 6-yard area in front of goal:
  x in [110, 120] (1m past 18-yard line through the goal line), step 0.5m
  y in [30, 50]   (centered on the goal mouth y in [36, 44]),    step 0.5m

Run as a script to produce 8 example visualizations on the test split:
    python src/analysis/counterfactual.py
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
import torch
from mplsoccer import Pitch

from src.data.rasterize import (
    GRID_H,
    GRID_W,
    SIGMA,
    rasterize_shot,
    render_gaussian,
)
from src.models.danger_cnn import DangerCNN

SHOTS_PATH = _REPO_ROOT / "data/raw/shots_master_df.csv"
FREEZE_PATH = _REPO_ROOT / "data/raw/freeze_master_df.csv"
TEST_MANIFEST = _REPO_ROOT / "data/processed/splits/test_shot_ids.csv"

DEFAULT_GRID_X = np.linspace(110.0, 120.0, 21)
DEFAULT_GRID_Y = np.linspace(30.0, 50.0, 41)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _actual_gk_pos(freeze_rows: pd.DataFrame) -> tuple[float, float]:
    gk = freeze_rows[
        (freeze_rows["position_name"] == "Goalkeeper") & (~freeze_rows["teammate"])
    ].iloc[0]
    return float(gk["x"]), float(gk["y"])


def sweep_gk_positions(
    model: DangerCNN,
    shot_row: pd.Series,
    freeze_rows: pd.DataFrame,
    gk_grid_x: np.ndarray | None = None,
    gk_grid_y: np.ndarray | None = None,
    device: torch.device | None = None,
) -> dict:
    """Hold non-GK context fixed; sweep GK over a 2D grid; return the danger map.

    Returns a dict:
      danger_grid       (n_y, n_x) array of V values
      gk_grid_x, gk_grid_y   coordinate arrays (world meters)
      actual_gk_pos     (x, y) where the GK actually stood
      actual_v          V(x, g_actual) from the real freeze-frame raster
      optimal_gk_pos    (x, y) of g* = argmin_g V on the grid
      optimal_v         V(x, g*)
      shot_id, is_goal
    """
    if gk_grid_x is None:
        gk_grid_x = DEFAULT_GRID_X
    if gk_grid_y is None:
        gk_grid_y = DEFAULT_GRID_Y
    if device is None:
        device = next(model.parameters()).device

    base = rasterize_shot(shot_row, freeze_rows)  # (5, 80, 60)
    context = base[:4].unsqueeze(0).to(device)  # (1, 4, 80, 60)

    n_y, n_x = len(gk_grid_y), len(gk_grid_x)
    n = n_y * n_x

    gk_channels = np.empty((n, GRID_H, GRID_W), dtype=np.float32)
    k = 0
    for gy in gk_grid_y:
        for gx in gk_grid_x:
            gk_channels[k] = render_gaussian(GRID_H, GRID_W, float(gx), float(gy), SIGMA)
            k += 1
    gk_tensor = torch.from_numpy(gk_channels).unsqueeze(1).to(device)  # (N, 1, 80, 60)
    batch = torch.cat([context.expand(n, -1, -1, -1), gk_tensor], dim=1)  # (N, 5, 80, 60)

    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model.forward_logits(batch)).cpu().numpy()
        actual_v = float(
            torch.sigmoid(model.forward_logits(base.unsqueeze(0).to(device))).cpu().item()
        )
    danger_grid = probs.reshape(n_y, n_x)

    yi, xi = np.unravel_index(np.argmin(danger_grid), danger_grid.shape)
    optimal_gk_pos = (float(gk_grid_x[xi]), float(gk_grid_y[yi]))
    optimal_v = float(danger_grid[yi, xi])

    return {
        "danger_grid": danger_grid,
        "gk_grid_x": gk_grid_x,
        "gk_grid_y": gk_grid_y,
        "actual_gk_pos": _actual_gk_pos(freeze_rows),
        "actual_v": actual_v,
        "optimal_gk_pos": optimal_gk_pos,
        "optimal_v": optimal_v,
        "shot_id": shot_row["id"],
        "is_goal": int(shot_row.get("outcome_name") == "Goal"),
    }


def visualize_sweep(
    sweep_result: dict,
    shot_row: pd.Series,
    freeze_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    """4-panel: pitch view, V heatmap, V contour, and a text summary."""
    grid_x = sweep_result["gk_grid_x"]
    grid_y = sweep_result["gk_grid_y"]
    danger = sweep_result["danger_grid"]
    a_gx, a_gy = sweep_result["actual_gk_pos"]
    o_gx, o_gy = sweep_result["optimal_gk_pos"]
    is_goal = bool(sweep_result["is_goal"])
    regret = sweep_result["actual_v"] - sweep_result["optimal_v"]

    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    # --- Panel 1: pitch with players ---
    pitch = Pitch(pitch_type="statsbomb", half=True, line_color="black")
    ax1 = fig.add_subplot(gs[0, 0])
    pitch.draw(ax=ax1)

    sx, sy = float(shot_row["x"]), float(shot_row["y"])
    teammates = freeze_rows[
        (freeze_rows["teammate"]) & (freeze_rows["position_name"] != "Goalkeeper")
    ]
    defenders = freeze_rows[
        (~freeze_rows["teammate"]) & (freeze_rows["position_name"] != "Goalkeeper")
    ]
    pitch.scatter(teammates["x"], teammates["y"], ax=ax1, s=80, color="#1f77b4",
                  alpha=0.7, edgecolors="black", linewidths=0.5, label="Attackers", zorder=3)
    pitch.scatter(defenders["x"], defenders["y"], ax=ax1, s=80, color="#7f7f7f",
                  alpha=0.7, edgecolors="black", linewidths=0.5, label="Defenders", zorder=3)
    pitch.scatter([sx], [sy], ax=ax1, s=240, color="red", marker="*",
                  edgecolors="black", linewidths=0.5, label="Shooter", zorder=5)
    pitch.scatter([a_gx], [a_gy], ax=ax1, s=180, color="#2ca02c", marker="o",
                  edgecolors="black", linewidths=1.0, label="Actual GK", zorder=6)
    pitch.scatter([o_gx], [o_gy], ax=ax1, s=320, color="gold", marker="*",
                  edgecolors="black", linewidths=1.0, label="Optimal g*", zorder=7)
    ax1.set_title(f"Shot {sweep_result['shot_id']} — {'GOAL' if is_goal else 'no goal'}")
    ax1.legend(loc="lower left", fontsize=8, framealpha=0.9)

    # --- Panel 2: heatmap ---
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.pcolormesh(grid_x, grid_y, danger, cmap="Reds", shading="auto")
    fig.colorbar(im, ax=ax2, label="V(x, g)")
    ax2.scatter([a_gx], [a_gy], s=180, color="#2ca02c", marker="o",
                edgecolors="black", linewidths=1.0, label="Actual", zorder=5)
    ax2.scatter([o_gx], [o_gy], s=280, color="gold", marker="*",
                edgecolors="black", linewidths=1.0, label="g*", zorder=6)
    # Goal posts y in [36, 44] for reference
    for gy_post in (36.0, 44.0):
        ax2.axhline(gy_post, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("GK x (m)")
    ax2.set_ylabel("GK y (m)")
    ax2.set_title("V(x, g) heatmap")
    ax2.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax2.invert_yaxis()  # match StatsBomb pitch (y=0 at top)

    # --- Panel 3: contour ---
    ax3 = fig.add_subplot(gs[1, 0])
    cs = ax3.contourf(grid_x, grid_y, danger, levels=15, cmap="Reds")
    ax3.contour(grid_x, grid_y, danger, levels=8, colors="black", linewidths=0.5, alpha=0.5)
    fig.colorbar(cs, ax=ax3, label="V(x, g)")
    ax3.scatter([a_gx], [a_gy], s=180, color="#2ca02c", marker="o",
                edgecolors="black", linewidths=1.0, label="Actual", zorder=5)
    ax3.scatter([o_gx], [o_gy], s=280, color="gold", marker="*",
                edgecolors="black", linewidths=1.0, label="g*", zorder=6)
    for gy_post in (36.0, 44.0):
        ax3.axhline(gy_post, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax3.set_xlabel("GK x (m)")
    ax3.set_ylabel("GK y (m)")
    ax3.set_title("V(x, g) contour")
    ax3.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax3.invert_yaxis()

    # --- Panel 4: text summary ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    xg = shot_row.get("shot_statsbomb_xg", np.nan)
    xg_str = f"{xg:.4f}" if pd.notna(xg) else "n/a"
    lines = [
        f"shot_id        {sweep_result['shot_id']}",
        f"outcome        {'GOAL' if is_goal else 'no goal'}",
        f"StatsBomb xG   {xg_str}",
        "",
        f"actual GK pos  ({a_gx:.1f}, {a_gy:.1f})",
        f"actual V       {sweep_result['actual_v']:.4f}",
        "",
        f"optimal g*     ({o_gx:.1f}, {o_gy:.1f})",
        f"optimal V      {sweep_result['optimal_v']:.4f}",
        "",
        f"regret         {regret:+.4f}",
        "               (actual_v - optimal_v)",
    ]
    ax4.text(0.05, 0.95, "\n".join(lines), fontsize=12, family="monospace",
             va="top", transform=ax4.transAxes)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _load_model(checkpoint_path: Path, device: torch.device) -> DangerCNN:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = DangerCNN().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _pick_examples(
    regret_df: pd.DataFrame, per_bucket: int, seed: int
) -> pd.DataFrame:
    """Pick `per_bucket` examples each from: high-regret-goal, high-regret-no-goal,
    low-regret, random. No shot is picked twice."""
    chosen_parts = []
    seen: set = set()

    def _take(df: pd.DataFrame, label: str) -> None:
        df = df[~df["shot_id"].isin(seen)]
        if len(df) == 0:
            return
        df = df.head(per_bucket).assign(category=label)
        chosen_parts.append(df)
        seen.update(df["shot_id"].tolist())

    _take(
        regret_df[regret_df["is_goal"] == 1].sort_values("regret", ascending=False),
        "high-regret-goal",
    )
    _take(
        regret_df[regret_df["is_goal"] == 0].sort_values("regret", ascending=False),
        "high-regret-no-goal",
    )
    _take(regret_df.sort_values("regret", ascending=True), "low-regret")
    remaining = regret_df[~regret_df["shot_id"].isin(seen)]
    if len(remaining) > 0:
        chosen_parts.append(
            remaining.sample(n=min(per_bucket, len(remaining)), random_state=seed)
            .assign(category="random")
        )
    return pd.concat(chosen_parts, ignore_index=True)


def run_sweep_on_examples(
    checkpoint_path: Path,
    shot_ids: list | None = None,
    n_examples: int = 8,
    output_dir: Path = Path("results/counterfactual"),
    sample_size: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Run sweep on a curated set of test shots and write 4-panel PNGs.

    Returns the summary table (also saved to <output_dir>/summary.csv).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _REPO_ROOT / checkpoint_path
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = _REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device()
    model = _load_model(checkpoint_path, device)
    print(f"Loaded {checkpoint_path.name} on {device}")

    print("Loading shot/freeze masters...")
    shots_df = pd.read_csv(SHOTS_PATH)
    freeze_df = pd.read_csv(FREEZE_PATH)

    if shot_ids is None:
        rng = np.random.default_rng(seed)
        test_ids = pd.read_csv(TEST_MANIFEST)["id"].tolist()
        idx = rng.choice(len(test_ids), size=min(sample_size, len(test_ids)), replace=False)
        sample_ids = [test_ids[i] for i in idx]
        print(f"Sweeping {len(sample_ids)} test shots to compute regret...")
        results = []
        for i, sid in enumerate(sample_ids, start=1):
            shot = shots_df[shots_df["id"] == sid].iloc[0]
            fr = freeze_df[freeze_df["id"] == sid]
            results.append(sweep_gk_positions(model, shot, fr, device=device))
            if i % 25 == 0 or i == len(sample_ids):
                print(f"  {i}/{len(sample_ids)} swept")
        regret_df = pd.DataFrame(
            [
                {
                    "shot_id": r["shot_id"],
                    "is_goal": r["is_goal"],
                    "actual_v": r["actual_v"],
                    "optimal_v": r["optimal_v"],
                    "regret": r["actual_v"] - r["optimal_v"],
                }
                for r in results
            ]
        )
        per_bucket = max(1, n_examples // 4)
        chosen = _pick_examples(regret_df, per_bucket=per_bucket, seed=seed)
        results_by_id = {r["shot_id"]: r for r in results}
    else:
        print(f"Sweeping {len(shot_ids)} user-specified shots...")
        results_by_id = {}
        records = []
        for sid in shot_ids:
            shot = shots_df[shots_df["id"] == sid].iloc[0]
            fr = freeze_df[freeze_df["id"] == sid]
            r = sweep_gk_positions(model, shot, fr, device=device)
            results_by_id[sid] = r
            records.append(
                {
                    "shot_id": sid,
                    "is_goal": r["is_goal"],
                    "actual_v": r["actual_v"],
                    "optimal_v": r["optimal_v"],
                    "regret": r["actual_v"] - r["optimal_v"],
                    "category": "user",
                }
            )
        chosen = pd.DataFrame(records)

    print()
    header = (
        f"{'category':<22}{'shot_id':>14}{'goal':>6}"
        f"{'actual_v':>11}{'optimal_v':>11}{'regret':>10}"
    )
    print(header)
    print("-" * len(header))
    for _, row in chosen.iterrows():
        sid = row["shot_id"]
        r = results_by_id[sid]
        shot = shots_df[shots_df["id"] == sid].iloc[0]
        fr = freeze_df[freeze_df["id"] == sid]
        png_path = output_dir / f"{row['category']}_{sid}.png"
        visualize_sweep(r, shot, fr, png_path)
        print(
            f"{row['category']:<22}{str(sid):>14}{int(row['is_goal']):>6}"
            f"{row['actual_v']:>11.4f}{row['optimal_v']:>11.4f}"
            f"{row['regret']:>+10.4f}"
        )

    summary_path = output_dir / "summary.csv"
    chosen.to_csv(summary_path, index=False)
    print(f"\nSaved {len(chosen)} visualizations and summary.csv to {output_dir}")
    return chosen


if __name__ == "__main__":
    run_sweep_on_examples(_REPO_ROOT / "models/checkpoints/baseline_v1/best.pt")
