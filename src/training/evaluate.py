"""
Evaluate a saved DangerCNN checkpoint on a held-out split.

Computes metrics for the model and (where available) for the StatsBomb xG
column on the same shots, so we can compare apples-to-apples. Predictions
go to results/eval/<run_name>/<split_name>_predictions.csv.

Usage:
    from src.training.evaluate import evaluate_model
    result = evaluate_model("models/checkpoints/baseline_v1/best.pt", "test")
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import GoalkeeperShotsDataset
from src.models.danger_cnn import DangerCNN
from src.training.metrics import compute_metrics

SHOTS_PATH = _REPO_ROOT / "data/raw/shots_master_df.csv"
FREEZE_PATH = _REPO_ROOT / "data/raw/freeze_master_df.csv"
SPLITS_DIR = _REPO_ROOT / "data/processed/splits"
VALID_SPLITS = ("test", "transfer_women", "transfer_men_other")


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_model(checkpoint_path: Path | str, split_name: str) -> dict:
    """Run the checkpoint on `split_name` and compare to StatsBomb xG.

    Returns a dict with keys:
        split, n_shots, n_with_xg,
        model_metrics, statsbomb_xg_metrics (None if no xG column entries),
        predictions_path.
    """
    if split_name not in VALID_SPLITS:
        raise ValueError(
            f"split_name must be one of {VALID_SPLITS}, got {split_name!r}"
        )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _REPO_ROOT / checkpoint_path

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    run_name = config.get("run_name", checkpoint_path.parent.name)

    device = _select_device()
    model = DangerCNN(
        in_channels=config.get("in_channels", 5),
        pool_size=config.get("pool_size", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint {checkpoint_path.name} from epoch {ckpt['epoch']} "
        f"(val_loss {ckpt['val_loss']:.4f}); device={device}"
    )

    print(f"Loading shots/freeze masters and building loader for split={split_name}...")
    shots_df = pd.read_csv(SHOTS_PATH)
    freeze_df = pd.read_csv(FREEZE_PATH)
    manifest_path = SPLITS_DIR / f"{split_name}_shot_ids.csv"
    ds = GoalkeeperShotsDataset(
        manifest_path, shots_df, freeze_df, cache_in_memory=False,
        include_geometry=config.get("include_geometry", False),
    )
    loader = DataLoader(
        ds, batch_size=config.get("batch_size", 64), shuffle=False, num_workers=0
    )

    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model.forward_logits(x)
            all_y.append(y.numpy())
            all_p.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)

    xg_lookup = shots_df.set_index("id")["shot_statsbomb_xg"]
    y_xg = xg_lookup.loc[ds.shot_ids].to_numpy()
    valid_xg = ~np.isnan(y_xg)

    model_metrics = compute_metrics(y_true, y_pred)
    if valid_xg.any():
        xg_metrics = compute_metrics(y_true[valid_xg], y_xg[valid_xg])
    else:
        xg_metrics = None

    out_dir = _REPO_ROOT / "results/eval" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(
        {
            "id": ds.shot_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_statsbomb_xg": y_xg,
        }
    )
    predictions_path = out_dir / f"{split_name}_predictions.csv"
    pred_df.to_csv(predictions_path, index=False)
    print(f"Saved {len(pred_df):,} predictions to {predictions_path}")

    return {
        "split": split_name,
        "n_shots": len(y_true),
        "n_with_xg": int(valid_xg.sum()),
        "model_metrics": model_metrics,
        "statsbomb_xg_metrics": xg_metrics,
        "predictions_path": str(predictions_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a DangerCNN checkpoint.")
    parser.add_argument(
        "checkpoint", type=Path, help="Path to a .pt checkpoint (e.g. .../best.pt)."
    )
    parser.add_argument(
        "split", choices=VALID_SPLITS, help="Which evaluation split to run on."
    )
    args = parser.parse_args()

    result = evaluate_model(args.checkpoint, args.split)
    print(
        f"\n{result['split']} ({result['n_shots']:,} shots, "
        f"{result['n_with_xg']:,} with xG):"
    )
    print(f"  {'metric':>30s}  {'DangerCNN':>10s}  {'StatsBomb xG':>14s}")
    if result["statsbomb_xg_metrics"] is not None:
        for k in (
            "auc", "brier", "log_loss", "expected_calibration_error",
        ):
            print(
                f"  {k:>30s}  {result['model_metrics'][k]:>10.4f}  "
                f"{result['statsbomb_xg_metrics'][k]:>14.4f}"
            )
    else:
        for k, v in result["model_metrics"].items():
            print(f"  {k:>30s}  {v:>10.4f}  {'(xG missing)':>14s}")
