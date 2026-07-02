"""
Multi-seed confidence intervals for baseline_v2 / baseline_v3.

Retrains a config across several seeds and, for each, records both:
  - predictive metrics on the test split (AUC / Brier / ECE), and
  - counterfactual-quality metrics on the *same fixed* 200-shot sample
    (median regret, % g* pinned to the y-edge, % g* on the far side of the
    shooter, post-side breakdown).

Two model tags are provided: "v2" (goal-geometry channels + (4,3) pool, no
scalar head) and "v3" (v2 + the parallel scalar head, all 9 features). v3's
counterfactual metrics are known to be poor; the run logs their spread but the
point of the v3 sweep is to firm up the *predictive* v2-vs-v3 comparison, which
was single-seed until now.

The counterfactual sample seed is held fixed (42) across all model seeds, so the
spread in those metrics reflects model variation, not sample variation. Results
go to results/eval/<tag>_multiseed_summary.csv and print as mean +/- std.

Usage:
    python src/training/multiseed_v2.py               # v2, seeds 42,1,2,3,4
    python src/training/multiseed_v2.py v3            # v3, seeds 42,1,2,3,4
    python src/training/multiseed_v2.py v3 42 1 2     # v3, custom seeds
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from src.analysis.regret_distribution import compute_regret_distribution
from src.features.scalar_features import SCALAR_DIM
from src.training.evaluate import evaluate_model
from src.training.train import train_model

DEFAULT_SEEDS = [42, 1, 2, 3, 4]

V2_CONFIG = {
    "epochs": 50,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "optimizer": "adam",
    "early_stopping_patience": 10,
    "device": "auto",
    "checkpoint_dir": "models/checkpoints",
    "include_geometry": True,
    "in_channels": 10,
    "pool_size": [4, 3],
    "include_scalars": False,
    "scalar_dim": 0,
    "log_every_n_batches": 0,
    "cache_train_in_memory": True,
    "num_workers": 0,
}

# v3 = v2 + the parallel scalar head (all 9 features). Everything else identical.
V3_CONFIG = {
    **V2_CONFIG,
    "include_scalars": True,
    "scalar_dim": SCALAR_DIM,
    "scalar_feature_set": "all",
}

CONFIGS = {"v2": V2_CONFIG, "v3": V3_CONFIG}


def _summary_path(tag: str) -> Path:
    return _REPO_ROOT / f"results/eval/{tag}_multiseed_summary.csv"


def run(seeds: list[int], tag: str = "v2", config: dict | None = None) -> pd.DataFrame:
    config = config if config is not None else CONFIGS[tag]
    rows = []
    for i, seed in enumerate(seeds, start=1):
        run_name = f"{tag}_seeds/seed{seed}"
        print(f"\n{'='*70}\n[{i}/{len(seeds)}] Training {tag} seed={seed} ({run_name})\n{'='*70}")
        cfg = {**config, "seed": seed, "run_name": run_name}
        result = train_model(cfg)

        best = Path(result["checkpoint_dir"]) / "best.pt"
        ev = evaluate_model(best, "test")
        m = ev["model_metrics"]

        df = compute_regret_distribution(
            best,
            output_csv=_REPO_ROOT / f"results/counterfactual/{tag}_seeds/seed{seed}_regret.csv",
        )
        n = len(df)
        rows.append({
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "test_auc": m["auc"],
            "test_brier": m["brier"],
            "test_ece": m["expected_calibration_error"],
            "cf_median_regret": float(df.regret.median()),
            "cf_pinned_pct": 100.0 * float(df.y_pinned.mean()),
            "cf_far_pct": 100.0 * float((df.post_side == "far").mean()),
            "cf_centre": int((df.post_side == "centre").sum()),
            "cf_near": int((df.post_side == "near").sum()),
            "cf_far": int((df.post_side == "far").sum()),
            "cf_n": n,
        })
        gc.collect()

    summary = pd.DataFrame(rows)
    summary_path = _summary_path(tag)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    _print_ci(summary, tag)
    return summary


def _print_ci(summary: pd.DataFrame, tag: str = "v2") -> None:
    cols = [
        "test_auc", "test_brier", "test_ece",
        "cf_median_regret", "cf_pinned_pct", "cf_far_pct",
    ]
    print(f"\n{'='*70}\nbaseline_{tag} multi-seed (n={len(summary)}): mean +/- std\n{'='*70}")
    for c in cols:
        vals = summary[c].to_numpy()
        # ddof=1 sample std; guard n=1.
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {c:<18} {vals.mean():.4f} +/- {std:.4f}   "
              f"(min {vals.min():.4f}, max {vals.max():.4f})")
    print(f"\nPer-seed table and summary saved to {_summary_path(tag)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    tag = "v2"
    if args and args[0] in CONFIGS:
        tag, args = args[0], args[1:]
    seeds = [int(s) for s in args] or DEFAULT_SEEDS
    run(seeds, tag=tag)
