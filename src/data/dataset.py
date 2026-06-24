"""
PyTorch Dataset wrapping the rasterized representation of shots.

Each item is (raster_tensor, is_goal) where raster_tensor is the (5, 80, 60)
output of `rasterize_shot` and is_goal is a 0.0/1.0 scalar tensor.

The Dataset trusts that the manifest has been cleaned via
`filter_rasterizable_shots`. If a shot in the manifest fails rasterization
(off-crop shooter, missing GK, missing freeze frame), __getitem__ raises.
With `cache_in_memory=True`, all shots are rasterized at __init__, so any
failures surface immediately rather than mid-epoch.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/data/dataset.py` from the repo root by ensuring the repo
# root is on sys.path so `from src.data.rasterize import ...` resolves.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.rasterize import (
    GRID_H,
    GRID_W,
    N_GEOMETRY_CHANNELS,
    N_PLAYER_CHANNELS,
    rasterize_shot,
)


class GoalkeeperShotsDataset(Dataset):
    """One split of shots, exposing rasterized tensors and is_goal labels."""

    def __init__(
        self,
        manifest_path: Path,
        shots_df: pd.DataFrame,
        freeze_df: pd.DataFrame,
        cache_in_memory: bool = False,
        include_geometry: bool = False,
    ) -> None:
        self.include_geometry = include_geometry
        n_channels = N_PLAYER_CHANNELS + (N_GEOMETRY_CHANNELS if include_geometry else 0)
        bytes_per_tensor = n_channels * GRID_H * GRID_W * 4  # float32
        manifest = pd.read_csv(manifest_path)
        missing = {"id", "is_goal"} - set(manifest.columns)
        if missing:
            raise ValueError(
                f"Manifest {manifest_path} missing required columns {missing}; "
                f"got {list(manifest.columns)}."
            )
        self.shot_ids: list[str] = manifest["id"].tolist()
        self.labels: torch.Tensor = torch.tensor(
            manifest["is_goal"].astype(float).values, dtype=torch.float32
        )

        self._shots_by_id = shots_df.set_index("id")
        self._freeze_by_id = freeze_df.groupby("id")

        self._cache: list[torch.Tensor] | None = None
        if cache_in_memory:
            n = len(self.shot_ids)
            est_mb = n * bytes_per_tensor / (1024 * 1024)
            print(
                f"GoalkeeperShotsDataset({manifest_path.name}): caching {n} shots "
                f"in memory ({bytes_per_tensor / 1024:.1f} KB/shot, "
                f"~{est_mb:.0f} MB total)."
            )
            self._cache = [self._rasterize(i) for i in range(n)]

    def _rasterize(self, idx: int) -> torch.Tensor:
        sid = self.shot_ids[idx]
        return rasterize_shot(
            self._shots_by_id.loc[sid],
            self._freeze_by_id.get_group(sid),
            include_geometry=self.include_geometry,
        )

    def __len__(self) -> int:
        return len(self.shot_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._cache[idx] if self._cache is not None else self._rasterize(idx)
        return x, self.labels[idx]


if __name__ == "__main__":
    import random

    from src.data.rasterize import ShotOutsideCropError

    shots = pd.read_csv(_REPO_ROOT / "data/raw/shots_master_df.csv")
    freeze = pd.read_csv(_REPO_ROOT / "data/raw/freeze_master_df.csv")
    manifest = _REPO_ROOT / "data/processed/splits/train_shot_ids.csv"

    ds = GoalkeeperShotsDataset(manifest, shots, freeze, cache_in_memory=False)
    print(f"Train Dataset length: {len(ds)}")
    print(f"Label mean (goal rate): {ds.labels.mean().item():.3f}")

    # Pick 5 random items; retry on the rare unfiltered off-crop shot.
    rng = random.Random(0)
    drawn = 0
    attempts = 0
    while drawn < 5 and attempts < 50:
        idx = rng.randrange(len(ds))
        attempts += 1
        try:
            x, y = ds[idx]
        except (ShotOutsideCropError, ValueError, KeyError) as e:
            print(f"  idx={idx}: SKIPPED ({type(e).__name__}: {e})")
            continue
        print(
            f"  idx={idx} id={ds.shot_ids[idx]} "
            f"shape={tuple(x.shape)} label={y.item():.1f}"
        )
        drawn += 1
