"""
DataLoader factory for the train/val/test splits.

The transfer-evaluation splits (transfer_women, transfer_men_other) are not
included here — they're held out for post-hoc evaluation, not used in the
main training loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python src/training/dataloaders.py` from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GoalkeeperShotsDataset

SHOTS_PATH = _REPO_ROOT / "data/raw/shots_master_df.csv"
FREEZE_PATH = _REPO_ROOT / "data/raw/freeze_master_df.csv"
SPLITS_DIR = _REPO_ROOT / "data/processed/splits"

SPLITS = {
    "train": SPLITS_DIR / "train_shot_ids.csv",
    "val": SPLITS_DIR / "val_shot_ids.csv",
    "test": SPLITS_DIR / "test_shot_ids.csv",
}


def make_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    cache_in_memory: bool = False,
) -> dict[str, DataLoader]:
    """Build train/val/test DataLoaders sharing a single load of the master CSVs.

    Train shuffles; val and test do not. pin_memory and persistent_workers are
    enabled when num_workers > 0 to keep training-loop overhead low.
    """
    shots = pd.read_csv(SHOTS_PATH)
    freeze = pd.read_csv(FREEZE_PATH)

    datasets = {
        name: GoalkeeperShotsDataset(path, shots, freeze, cache_in_memory=cache_in_memory)
        for name, path in SPLITS.items()
    }

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }


if __name__ == "__main__":
    # Smoke test: build loaders and pull one batch from each.
    # num_workers=0 so the rare unfiltered off-crop shot would surface here
    # rather than in a worker process (and to keep the test fast on macOS).
    loaders = make_dataloaders(batch_size=8, num_workers=0, cache_in_memory=False)
    for name, loader in loaders.items():
        x, y = next(iter(loader))
        print(
            f"{name}: dataset_size={len(loader.dataset)} "
            f"batch x={tuple(x.shape)} y={tuple(y.shape)} "
            f"label_mean={y.mean().item():.3f}"
        )
