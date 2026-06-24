"""
Training loop for DangerCNN.

Usage:
    python src/training/train.py            # uses the default config in __main__

Run artifacts (under models/checkpoints/<run_name>/):
    config.json              the exact config used
    history.json             per-epoch metrics (updated every epoch)
    best.pt                  state_dict + epoch + val_loss for the lowest-val-loss epoch
    last.pt                  state_dict + epoch + val_loss for the most recent epoch
    training_curves.png      train/val loss + val AUC + val Brier curves

We use BCEWithLogitsLoss (no pos_weight) so predictions stay calibrated to
the ~10% goal base rate, keeping Brier/ECE meaningful for comparison with
StatsBomb xG and Anzer & Bauer (2021).
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import GoalkeeperShotsDataset
from src.data.rasterize import filter_rasterizable_shots
from src.models.danger_cnn import DangerCNN
from src.training.metrics import compute_metrics

SHOTS_PATH = _REPO_ROOT / "data/raw/shots_master_df.csv"
FREEZE_PATH = _REPO_ROOT / "data/raw/freeze_master_df.csv"
SPLITS_DIR = _REPO_ROOT / "data/processed/splits"
ALL_SPLIT_NAMES = ("train", "val", "test", "transfer_women", "transfer_men_other")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device(spec: str) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ensure_clean_manifests(
    shots_df: pd.DataFrame, freeze_df: pd.DataFrame, split_paths: list[Path]
) -> None:
    """Drop unrasterizable shots from each split CSV in place. Idempotent."""
    for path in split_paths:
        if not path.exists():
            print(f"  {path.name}: not found, skipping")
            continue
        manifest = pd.read_csv(path)
        kept, dropped = filter_rasterizable_shots(
            manifest["id"].tolist(), shots_df, freeze_df
        )
        if not dropped:
            print(f"  {path.name}: already clean ({len(kept):,} shots)")
            continue
        cleaned = manifest[manifest["id"].isin(set(kept))].copy()
        cleaned.to_csv(path, index=False)
        print(
            f"  {path.name}: removed {len(dropped):,} unrasterizable, "
            f"kept {len(kept):,}/{len(manifest):,}"
        )


def _build_dataloaders(
    config: dict, shots_df: pd.DataFrame, freeze_df: pd.DataFrame
) -> tuple[DataLoader, DataLoader]:
    cache_train = config["cache_train_in_memory"]
    include_geometry = config["include_geometry"]
    include_scalars = config.get("include_scalars", False)
    scalar_feature_set = config.get("scalar_feature_set", "all")
    # Position-jitter augmentation is train-only; val/test are always clean.
    jitter_sigma = config.get("jitter_sigma", 0.0)
    train_ds = GoalkeeperShotsDataset(
        SPLITS_DIR / "train_shot_ids.csv",
        shots_df, freeze_df, cache_in_memory=cache_train,
        include_geometry=include_geometry, include_scalars=include_scalars,
        scalar_feature_set=scalar_feature_set, jitter_sigma=jitter_sigma,
    )
    # If we're caching train, val (~615 MB) is a small extra cost and avoids
    # validation dominating per-epoch time.
    val_ds = GoalkeeperShotsDataset(
        SPLITS_DIR / "val_shot_ids.csv",
        shots_df, freeze_df, cache_in_memory=cache_train,
        include_geometry=include_geometry, include_scalars=include_scalars,
        scalar_feature_set=scalar_feature_set,
    )
    common = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": config["num_workers"] > 0,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader


def _unpack_batch(batch, device):
    """Return (x, scalars, y) on device; scalars is None for 2-tuple batches."""
    if len(batch) == 3:
        x, scalars, y = batch
        scalars = scalars.to(device, non_blocking=True)
    else:
        x, y = batch
        scalars = None
    return x.to(device, non_blocking=True), scalars, y


def _run_validation(
    model: DangerCNN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    model.eval()
    losses_weighted = 0.0
    all_y, all_p = [], []
    with torch.no_grad():
        for batch in loader:
            # Capture labels on CPU before any device transfer. MPS with
            # non_blocking=True can return garbage from a tensor that was
            # moved over and then read back via .cpu() while other work is
            # queued in between (verified failure mode on torch 2.11).
            all_y.append(batch[-1].numpy())
            x, scalars, y = _unpack_batch(batch, device)
            y = y.to(device)
            logits = model.forward_logits(x, scalars)
            losses_weighted += criterion(logits, y).item() * x.size(0)
            all_p.append(torch.sigmoid(logits).cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_p)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = losses_weighted / y_true.size
    return metrics


def _save_checkpoint(path: Path, model: nn.Module, epoch: int, val_loss: float, config: dict) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_loss": val_loss,
            "config": config,
        },
        path,
    )


def _plot_curves(history: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_auc"] for h in history], color="C2")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val AUC")
    axes[1].set_title("Validation AUC")
    axes[1].set_ylim(0.5, 1.0)

    axes[2].plot(epochs, [h["val_brier"] for h in history], color="C3")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("val Brier")
    axes[2].set_title("Validation Brier")

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def train_model(config: dict) -> dict:
    """Train DangerCNN per `config`. See module docstring for artifact layout."""
    _set_seeds(config["seed"])
    device = _select_device(config["device"])
    print(f"Device: {device}")

    print("Loading shots and freeze master CSVs...")
    shots_df = pd.read_csv(SHOTS_PATH)
    freeze_df = pd.read_csv(FREEZE_PATH)

    print("Verifying manifests are rasterization-clean:")
    _ensure_clean_manifests(
        shots_df, freeze_df,
        [SPLITS_DIR / f"{n}_shot_ids.csv" for n in ALL_SPLIT_NAMES],
    )

    print("Building DataLoaders...")
    train_loader, val_loader = _build_dataloaders(config, shots_df, freeze_df)

    model = DangerCNN(
        in_channels=config["in_channels"],
        pool_size=config["pool_size"],
        scalar_dim=config.get("scalar_dim", 0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: DangerCNN, {n_params:,} params "
        f"(in_channels={config['in_channels']}, pool_size={config['pool_size']}, "
        f"scalar_dim={config['scalar_dim']})"
    )

    criterion = nn.BCEWithLogitsLoss()
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']!r}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    ckpt_dir = Path(config["checkpoint_dir"]) / config["run_name"]
    if not ckpt_dir.is_absolute():
        ckpt_dir = _REPO_ROOT / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    log_every = config["log_every_n_batches"]

    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_n = 0
        for batch_idx, batch in enumerate(train_loader):
            x, scalars, y = _unpack_batch(batch, device)
            y = y.to(device, non_blocking=True)
            logits = model.forward_logits(x, scalars)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            running_n += x.size(0)
            if log_every and (batch_idx + 1) % log_every == 0:
                print(
                    f"  epoch {epoch} batch {batch_idx + 1}/{len(train_loader)} "
                    f"running_loss={running_loss / running_n:.4f}"
                )

        train_loss = running_loss / running_n
        val_metrics = _run_validation(model, val_loader, device, criterion)
        scheduler.step(val_metrics["loss"])
        elapsed = time.perf_counter() - epoch_start
        lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_brier": val_metrics["brier"],
            "val_log_loss": val_metrics["log_loss"],
            "val_ece": val_metrics["expected_calibration_error"],
            "lr": lr,
            "epoch_time_s": elapsed,
        }
        history.append(record)
        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(
            f"epoch {epoch:>3}/{config['epochs']} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} "
            f"val_brier={val_metrics['brier']:.4f} "
            f"val_ece={val_metrics['expected_calibration_error']:.4f} "
            f"lr={lr:.2e} "
            f"({elapsed:.1f}s)"
        )

        _save_checkpoint(ckpt_dir / "last.pt", model, epoch, val_metrics["loss"], config)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                ckpt_dir / "best.pt", model, epoch, val_metrics["loss"], config
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config["early_stopping_patience"]:
                print(
                    f"Early stopping at epoch {epoch}: no val_loss improvement for "
                    f"{config['early_stopping_patience']} epochs "
                    f"(best was epoch {best_epoch}, val_loss={best_val_loss:.4f})."
                )
                break

    _plot_curves(history, ckpt_dir / "training_curves.png")

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_dir": str(ckpt_dir),
    }


if __name__ == "__main__":
    # baseline_v2 = goal-geometry channels (in_channels=10) + (4,3) spatial pool,
    # no scalar head. This is the recommended positioning model (clean, stable,
    # transferable counterfactual g*); running this file as-is reproduces it.
    # Variants via flags:
    #   v1   -> include_geometry=False, in_channels=5, pool_size=1
    #   v3   -> include_scalars=True, scalar_feature_set="all",     scalar_dim=9
    #   v3b  -> include_scalars=True, scalar_feature_set="context", scalar_dim=6
    #   jitter augmentation -> jitter_sigma=0.75 (train-only; found neutral here,
    #     see CLAUDE.md — does not improve g*, ~3x slower since caching is off)
    config = {
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "optimizer": "adam",
        "early_stopping_patience": 10,
        "device": "auto",
        "checkpoint_dir": "models/checkpoints",
        "run_name": "baseline_v2",
        "include_geometry": True,
        "in_channels": 10,
        "pool_size": [4, 3],
        "include_scalars": False,
        "scalar_dim": 0,
        "jitter_sigma": 0.0,
        "log_every_n_batches": 0,
        "cache_train_in_memory": True,
        "num_workers": 0,
        "seed": 42,
    }
    result = train_model(config)
    print(
        f"\nTraining complete. Best epoch: {result['best_epoch']} "
        f"(val_loss {result['best_val_loss']:.4f}). "
        f"Artifacts in {result['checkpoint_dir']}."
    )

    from src.training.evaluate import evaluate_model

    print("\nEvaluating best checkpoint on test split...")
    test_result = evaluate_model(
        Path(result["checkpoint_dir"]) / "best.pt", split_name="test"
    )
    print(f"\nTest set ({test_result['n_shots']:,} shots):")
    print(f"  {'metric':>30s}  {'DangerCNN':>10s}  {'StatsBomb xG':>14s}")
    if test_result["statsbomb_xg_metrics"] is not None:
        for k in ("auc", "brier", "log_loss", "expected_calibration_error"):
            print(
                f"  {k:>30s}  {test_result['model_metrics'][k]:>10.4f}  "
                f"{test_result['statsbomb_xg_metrics'][k]:>14.4f}"
            )
    else:
        for k, v in test_result["model_metrics"].items():
            print(f"  {k:>30s}  {v:>10.4f}  {'(xG missing)':>14s}")
