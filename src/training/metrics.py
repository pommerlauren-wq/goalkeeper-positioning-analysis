"""Classification metrics for goal-probability predictions."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def expected_calibration_error(
    y_true: np.ndarray, y_pred_probs: np.ndarray, n_bins: int = 10
) -> float:
    """Equal-width-binning ECE: weighted average gap between |bin accuracy - bin confidence|."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_pred_probs, edges[1:-1]), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_pred_probs[mask].mean())
    return float(ece)


def compute_metrics(y_true: np.ndarray, y_pred_probs: np.ndarray) -> dict:
    """Return auc, brier, log_loss, accuracy_at_0.5, expected_calibration_error."""
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(y_pred_probs).astype(float)
    p_clip = np.clip(p, 1e-7, 1.0 - 1e-7)
    return {
        "auc": float(roc_auc_score(y_true, p)),
        "brier": float(np.mean((p - y_true) ** 2)),
        "log_loss": float(log_loss(y_true, p_clip)),
        "accuracy_at_0.5": float(np.mean(y_true == (p >= 0.5).astype(float))),
        "expected_calibration_error": expected_calibration_error(y_true, p),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y_true = (rng.random(1000) < 0.1).astype(float)
    # Predictions correlated with truth + base rate prior.
    y_pred = np.clip(0.1 + 0.4 * y_true + 0.1 * rng.standard_normal(1000), 0.01, 0.99)
    m = compute_metrics(y_true, y_pred)
    print("Synthetic-data metrics (n=1000, ~10% positives):")
    for k, v in m.items():
        print(f"  {k:>30s}: {v:.4f}")
