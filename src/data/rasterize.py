"""
Rasterize a single shot + its freeze frame into a multi-channel tensor for the CNN.

Output shape: (5, 80, 60) — channels x H (y) x W (x after crop).

Coordinate convention:
  StatsBomb pitch is 120 (length) x 80 (width), with shots already normalized
  to attack toward x=120. We crop to the attacking half x in [60, 120], y in
  [0, 80], at 1m per cell -> grid (H=80, W=60). Cell (h, w) is centered at
  world (x=60+w+0.5, y=h+0.5). World y maps to grid H, world x maps to grid W.

Channels:
  0 shooter           single Gaussian at the shot location
  1 ball              identical to shooter for now (TODO: aerial passes /
                      first-time shots will use the pass-end position)
  2 teammates         attacking teammates excluding the shooter, combined by
                      ELEMENTWISE MAX (not sum) of per-player Gaussians. This
                      keeps values in [0, 1], makes occupancy the primary
                      signal, and keeps these channels semantically comparable
                      to the single-Gaussian channels (0, 1, 4). Density can
                      be added later as a separate channel if needed.
  3 defenders         defending team excluding the goalkeeper, same combine.
  4 goalkeeper        single Gaussian at the GK position.

Off-crop handling:
  - Shooter outside [60, 120] x [0, 80] -> raises ShotOutsideCropError.
  - Other players outside the crop are clip-rendered: only the in-grid tail
    of their Gaussian is kept. This preserves a 'just off-frame' signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

X_MIN, X_MAX = 60.0, 120.0
Y_MIN, Y_MAX = 0.0, 80.0
GRID_W = int(X_MAX - X_MIN)
GRID_H = int(Y_MAX - Y_MIN)
SIGMA = 1.5
GOAL_CENTER = (120.0, 40.0)


class ShotOutsideCropError(ValueError):
    """Raised when the shooter's position is outside the cropped attacking half."""


def render_gaussian(
    grid_h: int,
    grid_w: int,
    center_x: float,
    center_y: float,
    sigma: float,
    x_offset: float = X_MIN,
) -> np.ndarray:
    """Render an isotropic 2D Gaussian into a (grid_h, grid_w) array.

    center_x, center_y are in StatsBomb world coordinates. The grid covers
    world x in [x_offset, x_offset + grid_w] and world y in [0, grid_h], at
    1m per cell. Cell (h, w) is centered at world (x_offset + w + 0.5, h + 0.5).

    Centers outside the grid are handled implicitly: only the in-grid portion
    of the Gaussian is retained.
    """
    col_centers = np.arange(grid_w) + 0.5 + x_offset
    row_centers = np.arange(grid_h) + 0.5
    dx = col_centers[None, :] - center_x
    dy = row_centers[:, None] - center_y
    d2 = dx ** 2 + dy ** 2
    return np.exp(-d2 / (2.0 * sigma ** 2))


def _identify_goalkeeper(freeze_rows: pd.DataFrame) -> pd.Series:
    """Return the defending goalkeeper row from a shot's freeze frame.

    GK = position_name == 'Goalkeeper' AND teammate == False. If multiple
    candidates match, the one closest to the goal at (120, 40) wins. If none
    match, raise — these shots should have been filtered earlier.
    """
    candidates = freeze_rows[
        (freeze_rows["position_name"] == "Goalkeeper")
        & (~freeze_rows["teammate"].astype(bool))
    ]
    if len(candidates) == 0:
        raise ValueError("No defending goalkeeper found in freeze frame.")
    if len(candidates) == 1:
        return candidates.iloc[0]
    gx, gy = GOAL_CENTER
    d2 = (candidates["x"] - gx) ** 2 + (candidates["y"] - gy) ** 2
    return candidates.loc[d2.idxmin()]


def _max_combine(rows: pd.DataFrame) -> np.ndarray:
    """Render each row's Gaussian and combine by elementwise max."""
    out = np.zeros((GRID_H, GRID_W), dtype=np.float64)
    for _, row in rows.iterrows():
        g = render_gaussian(GRID_H, GRID_W, float(row["x"]), float(row["y"]), SIGMA)
        np.maximum(out, g, out=out)
    return out


def rasterize_shot(shot_row: pd.Series, freeze_rows: pd.DataFrame) -> torch.Tensor:
    """Rasterize one shot + its freeze frame into a (5, 80, 60) float tensor.

    See module docstring for channel layout and coordinate convention.

    Raises
    ------
    ShotOutsideCropError
        Shooter position is outside [60, 120] x [0, 80].
    ValueError
        No defending goalkeeper found in the freeze frame.
    """
    sx, sy = float(shot_row["x"]), float(shot_row["y"])
    if not (X_MIN <= sx <= X_MAX and Y_MIN <= sy <= Y_MAX):
        raise ShotOutsideCropError(
            f"Shot {shot_row.get('id', '?')} shooter at ({sx:.1f}, {sy:.1f}) "
            f"is outside crop x[{X_MIN},{X_MAX}] y[{Y_MIN},{Y_MAX}]."
        )

    shooter = render_gaussian(GRID_H, GRID_W, sx, sy, SIGMA)
    ball = shooter.copy()

    gk_row = _identify_goalkeeper(freeze_rows)
    gk = render_gaussian(
        GRID_H, GRID_W, float(gk_row["x"]), float(gk_row["y"]), SIGMA
    )

    teammate_mask = freeze_rows["teammate"].astype(bool)
    shooter_pid = shot_row.get("player_id", None)
    teammates_df = freeze_rows[teammate_mask]
    if shooter_pid is not None and not pd.isna(shooter_pid):
        teammates_df = teammates_df[teammates_df["player_id"] != shooter_pid]
    teammates = _max_combine(teammates_df)

    defenders_df = freeze_rows[(~teammate_mask) & (freeze_rows.index != gk_row.name)]
    defenders = _max_combine(defenders_df)

    stack = np.stack([shooter, ball, teammates, defenders, gk], axis=0)
    return torch.from_numpy(stack).float()


def filter_rasterizable_shots(
    shot_ids,
    shots_df: pd.DataFrame,
    freeze_df: pd.DataFrame,
):
    """Filter shot IDs to those that successfully rasterize.

    Returns (kept_ids, dropped) where dropped is a list of (shot_id, reason).
    Reasons cover off-crop shooter, missing GK, and missing freeze frame.
    """
    shots_by_id = shots_df.set_index("id")
    freeze_by_id = freeze_df.groupby("id")
    kept: list = []
    dropped: list[tuple] = []
    for sid in shot_ids:
        try:
            shot_row = shots_by_id.loc[sid]
            freeze_rows = freeze_by_id.get_group(sid)
        except KeyError as e:
            dropped.append((sid, f"KeyError: {e}"))
            continue
        try:
            rasterize_shot(shot_row, freeze_rows)
        except (ShotOutsideCropError, ValueError) as e:
            dropped.append((sid, f"{type(e).__name__}: {e}"))
            continue
        kept.append(sid)
    return kept, dropped


def _save_panels(tensor: torch.Tensor, path: Path, shot_id) -> None:
    import matplotlib.pyplot as plt

    arr = tensor.numpy()
    titles = ["shooter", "ball", "teammates", "defenders", "goalkeeper"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    for ax, ch, title in zip(axes, arr, titles):
        ax.imshow(
            ch,
            origin="lower",
            extent=(X_MIN, X_MAX, Y_MIN, Y_MAX),
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"Shot {shot_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    import random

    REPO_ROOT = Path(__file__).resolve().parents[2]
    shots = pd.read_csv(REPO_ROOT / "data/raw/shots_master_df.csv")
    freeze = pd.read_csv(REPO_ROOT / "data/raw/freeze_master_df.csv")
    train_ids = pd.read_csv(
        REPO_ROOT / "data/processed/splits/train_shot_ids.csv"
    )["id"].tolist()

    shots_by_id = shots.set_index("id")
    freeze_by_id = freeze.groupby("id")

    rng = random.Random(0)

    # Save a 5-panel viz for one training shot (retry on rasterization errors).
    sample_id = None
    for sid in [train_ids[0]] + rng.sample(train_ids, 50):
        try:
            t = rasterize_shot(shots_by_id.loc[sid], freeze_by_id.get_group(sid))
        except (ShotOutsideCropError, ValueError, KeyError):
            continue
        sample_id = sid
        break
    if sample_id is None:
        raise RuntimeError("Could not rasterize any of the first 51 candidate shots.")
    out = REPO_ROOT / "results/rasterize_check" / f"{sample_id}.png"
    _save_panels(t, out, sample_id)
    print(f"Saved {out}")

    # Validate on 5 random training shots: shape (5,80,60), no NaN, in [0, 1].
    sample = rng.sample(train_ids, 5)
    print("Validation on 5 random training shots:")
    for sid in sample:
        try:
            t = rasterize_shot(shots_by_id.loc[sid], freeze_by_id.get_group(sid))
        except (ShotOutsideCropError, ValueError, KeyError) as e:
            print(f"  {sid}: SKIPPED ({type(e).__name__}: {e})")
            continue
        assert t.shape == (5, GRID_H, GRID_W), f"bad shape {tuple(t.shape)} for {sid}"
        assert not torch.isnan(t).any(), f"NaN found in {sid}"
        assert (t >= 0.0).all() and (t <= 1.0).all(), f"out-of-range in {sid}"
        print(
            f"  {sid}: OK shape={tuple(t.shape)} "
            f"min={t.min():.3f} max={t.max():.3f}"
        )
