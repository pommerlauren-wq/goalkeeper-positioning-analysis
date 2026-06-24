"""
Multi-seed confidence intervals for baseline_v2.

Retrains the v2 config (goal-geometry channels + (4,3) pool, no scalar head)
across several seeds and, for each, records both:
  - predictive metrics on the test split (AUC / Brier / ECE), and
  - counterfactual-quality metrics on the *same fixed* 200-shot sample
    (median regret, % g* pinned to the y-edge, % g* on the far side of the
    shooter, post-side breakdown).

The counterfactual sample seed is held fixed (42) across all model seeds, so the
spread in those metrics reflects model variation, not sample variation. Results
go to results/eval/v2_multiseed_summary.csv and are printed as mean +/- std.

Usage:
    python src/training/multiseed_v2.py            # seeds 42,1,2,3,4
    python src/training/multiseed_v2.py 42 1 2     # custom seeds
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

SUMMARY_PATH = _REPO_ROOT / "results/eval/v2_multiseed_summary.csv"


def run(seeds: list[int]) -> pd.DataFrame:
    rows = []
    for i, seed in enumerate(seeds, start=1):
        run_name = f"v2_seeds/seed{seed}"
        print(f"\n{'='*70}\n[{i}/{len(seeds)}] Training v2 seed={seed} ({run_name})\n{'='*70}")
        cfg = {**V2_CONFIG, "seed": seed, "run_name": run_name}
        result = train_model(cfg)

        best = Path(result["checkpoint_dir"]) / "best.pt"
        ev = evaluate_model(best, "test")
        m = ev["model_metrics"]

        df = compute_regret_distribution(
            best,
            output_csv=_REPO_ROOT / f"results/counterfactual/v2_seeds/seed{seed}_regret.csv",
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
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    _print_ci(summary)
    return summary


def _print_ci(summary: pd.DataFrame) -> None:
    cols = [
        "test_auc", "test_brier", "test_ece",
        "cf_median_regret", "cf_pinned_pct", "cf_far_pct",
    ]
    print(f"\n{'='*70}\nbaseline_v2 multi-seed (n={len(summary)}): mean +/- std\n{'='*70}")
    for c in cols:
        vals = summary[c].to_numpy()
        # ddof=1 sample std; guard n=1.
        std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        print(f"  {c:<18} {vals.mean():.4f} +/- {std:.4f}   "
              f"(min {vals.min():.4f}, max {vals.max():.4f})")
    print(f"\nPer-seed table and summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    seeds = [int(s) for s in sys.argv[1:]] or DEFAULT_SEEDS
    run(seeds)
