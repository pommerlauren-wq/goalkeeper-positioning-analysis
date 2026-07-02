"""
Latent-embedding analysis of the DangerCNN representation.

Extracts the 64-d embedding the model builds just before its output layer
(h = ReLU(fc1(pooled conv features)) in DangerCNN.forward_logits, i.e. the
learned feature vector that fc2 turns into a logit), runs it over the test
split, projects to 2-D with PCA, and colours the projection by:

  - predicted P(goal)          (what the model itself outputs)
  - StatsBomb xG               (an external danger estimate)
  - distance shooter -> goal   (raw geometry)
  - outcome (goal vs save)     (ground truth)

If the representation is organized around danger/geometry rather than raw player
density, PC1 should track P(goal) / distance smoothly. This is a qualitative
confirmation of the v2 story (the geometry channels make the model reason about
where the goal is), and a report figure.

Targets baseline_v2 by default (the recommended positioning model, scalar_dim=0),
but reads include_geometry / include_scalars from the checkpoint config so it also
works on v1 (5-channel) or v3 (scalar-head) checkpoints.

Run (after any training job frees the device):
    python src/analysis/latent_embedding.py
    python src/analysis/latent_embedding.py models/checkpoints/baseline_v3/best.pt
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
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.analysis.counterfactual import (
    FREEZE_PATH,
    SHOTS_PATH,
    TEST_MANIFEST,
    _load_model,
    _select_device,
)
from src.data.dataset import GoalkeeperShotsDataset

FIG_DIR = _REPO_ROOT / "reports/figures"
GOAL_CENTRE = (120.0, 40.0)


def _extract_embeddings(
    checkpoint_path: Path, batch_size: int = 256
) -> pd.DataFrame:
    """Run the model over the test split, returning a per-shot table with the
    64-d embedding, predicted P(goal), label, and shot_id."""
    device = _select_device()
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    include_geometry = config.get("include_geometry", False)
    include_scalars = config.get("include_scalars", False)
    scalar_feature_set = config.get("scalar_feature_set", "all")

    model = _load_model(checkpoint_path, device)

    # Capture ReLU(fc1(x)) — the CNN embedding feeding the output layer. The
    # forward hook on fc1 gives the pre-activation linear output; apply ReLU to
    # match the h used downstream (dropout is identity in eval mode).
    captured: list[torch.Tensor] = []

    def _hook(_module, _inp, out):
        captured.append(F.relu(out).detach().cpu())

    handle = model.fc1.register_forward_hook(_hook)

    shots_df = pd.read_csv(SHOTS_PATH)
    freeze_df = pd.read_csv(FREEZE_PATH)
    ds = GoalkeeperShotsDataset(
        TEST_MANIFEST, shots_df, freeze_df,
        include_geometry=include_geometry,
        include_scalars=include_scalars,
        scalar_feature_set=scalar_feature_set,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if include_scalars:
                x, scalars, y = batch
                scalars = scalars.to(device)
            else:
                x, y = batch
                scalars = None
            logits = model.forward_logits(x.to(device), scalars)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(y.numpy())
    handle.remove()

    emb = torch.cat(captured, dim=0).numpy()  # (N, 64), test order (shuffle=False)
    table = pd.DataFrame({
        "shot_id": ds.shot_ids,
        "p_goal": np.concatenate(preds),
        "is_goal": np.concatenate(labels).astype(int),
    })
    # Join external references (StatsBomb xG) and raw geometry.
    ref = shots_df.set_index("id")
    table["statsbomb_xg"] = table.shot_id.map(ref["shot_statsbomb_xg"])
    sx = table.shot_id.map(ref["x"]).to_numpy()
    sy = table.shot_id.map(ref["y"]).to_numpy()
    table["dist_to_goal"] = np.hypot(GOAL_CENTRE[0] - sx, GOAL_CENTRE[1] - sy)
    return table, emb


def make_latent_figure(
    checkpoint_path: Path | str = "models/checkpoints/baseline_v2/best.pt",
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _REPO_ROOT / checkpoint_path
    tag = checkpoint_path.parent.name  # e.g. baseline_v2

    print(f"Extracting embeddings from {tag}...")
    table, emb = _extract_embeddings(checkpoint_path)
    print(f"  {emb.shape[0]} test shots, {emb.shape[1]}-d embedding")

    # Standardize (StandardScaler maps zero-variance/dead ReLU units to scale 1,
    # so no div-by-zero) then PCA to 2 components.
    z = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(z)
    evr = pca.explained_variance_ratio_
    table["pc1"], table["pc2"] = xy[:, 0], xy[:, 1]

    # Correlations that back the interpretation text.
    def _corr(a, b):
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])

    c_pgoal = _corr(table.pc1.to_numpy(), table.p_goal.to_numpy())
    c_dist = _corr(table.pc1.to_numpy(), table.dist_to_goal.to_numpy())
    c_xg = _corr(table.pc1.to_numpy(), table.statsbomb_xg.to_numpy())

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    panels = [
        ("p_goal", "predicted P(goal)", "Reds", axes[0, 0]),
        ("statsbomb_xg", "StatsBomb xG", "Reds", axes[0, 1]),
        ("dist_to_goal", "shooter → goal distance (m)", "viridis_r", axes[1, 0]),
    ]
    order = np.argsort(table.p_goal.to_numpy())  # draw dangerous shots on top
    for col, label, cmap, ax in panels:
        sc = ax.scatter(
            table.pc1.to_numpy()[order], table.pc2.to_numpy()[order],
            c=table[col].to_numpy()[order], cmap=cmap, s=8, alpha=0.7,
            linewidths=0,
        )
        fig.colorbar(sc, ax=ax, label=label, fraction=0.046, pad=0.02)
        ax.set_title(f"coloured by {label}", fontsize=11)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}% var)")

    # Outcome panel (categorical).
    ax = axes[1, 1]
    saves = table[table.is_goal == 0]
    goals = table[table.is_goal == 1]
    ax.scatter(saves.pc1, saves.pc2, s=8, alpha=0.4, color="#1f77b4",
               linewidths=0, label=f"no goal (n={len(saves)})")
    ax.scatter(goals.pc1, goals.pc2, s=14, alpha=0.8, color="#d62728",
               linewidths=0, label=f"goal (n={len(goals)})")
    ax.set_title("coloured by outcome", fontsize=11)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.0f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.0f}% var)")
    ax.legend(fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"{tag}: PCA of the 64-d CNN embedding over the test split "
        f"({emb.shape[0]} shots)\n"
        f"PC1 vs predicted P(goal) r={c_pgoal:+.2f}, vs shooter→goal distance "
        f"r={c_dist:+.2f}, vs StatsBomb xG r={c_xg:+.2f}  "
        f"— the representation is organized around danger/geometry",
        fontsize=11, y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"latent_embedding_{tag.replace('baseline_', '')}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPC explained variance: PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%")
    print(f"PC1 correlations: P(goal) {c_pgoal:+.3f}, distance {c_dist:+.3f}, "
          f"StatsBomb xG {c_xg:+.3f}")
    print(f"Wrote {out}")
    return out


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "models/checkpoints/baseline_v2/best.pt"
    make_latent_figure(ckpt)
