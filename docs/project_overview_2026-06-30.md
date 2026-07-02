# Project overview — counterfactual goalkeeper positioning

**Date:** 2026-06-30
**Author:** Lauren Pommer
**Audience:** Hannes; supervisor
**Module:** Math and Machine Learning Praktikum, University of Leipzig
**Status:** Core modelling complete; v2 is the recommended deliverable. Open
work is analysis + write-up, not new architecture.

This is the single current-state briefing. It supersedes the (now retired)
mid-semester talk briefing and folds in the three update notes since
(`docs/notes_for_supervisor_2026-05-08.md`).

---

## 1. Goal (unchanged)

Build a counterfactual goalkeeper-positioning model

> `V(x, g) = P(goal | x, g)`

where `g` is the goalkeeper's (x, y) position and `x` is everything else in the
freeze frame (shooter + ball, attacking teammates, outfield defenders). At
inference `x` is fixed and `g` is swept over a pitch grid to produce a danger
heatmap; the recommendation is `g* = argmin_g V(x, g)`, and a keeper's "regret"
is `V(x, g_actual) − V(x, g*)`.

**Scope:** open-play shots only. **Benchmarks:** StatsBomb built-in xG
(`shot_statsbomb_xg`); Anzer & Bauer (2021), RPS 0.197 on full tracking data.

---

## 2. Where we were at the mid-semester talk

`baseline_v1` (5 player channels, global-average pool, 102k params) was
**calibrated and competitive with StatsBomb xG** (AUC 0.803, Brier 0.075,
ECE 0.010) using purely spatial inputs. But the counterfactual sweep exposed a
**structural failure**: with no knowledge of the goal frame, `V` fell
monotonically as the synthetic keeper moved *away* from the goal mouth, so `g*`
pinned to the grid edge — and for wide-angle shots it recommended the keeper
stand behind the *far* post, the opposite of coaching intuition. We presented
this as a discovered limitation and an A/B/C decision point, not a bug.

**That decision has now been made and executed: Option A (add structural
priors). It worked.**

---

## 3. What we did since (the resolution)

### 3.1 baseline_v2 — goal-frame geometry (the fix)

Two changes, both inductive biases rather than hand-tuning:

- **Five static goal-geometry channels** appended to the raster: a goal-frame
  Gaussian ridge on the goal-line segment, distance-to-goal, the angle subtended
  by the two posts ("view angle"), and two normalized coordinate channels
  (CoordConv). The model can now *see* where the goal is.
- **Spatial-preserving head:** the global-average pool is replaced by a small
  (4×3) spatial pool, so absolute location survives into the classifier instead
  of being averaged away. (Global pooling was an independent second reason v1
  reasoned about player *density* rather than *geometry*.)

10 input channels, 194k params (within the 100–200k budget). Calibration
deliberately preserved — no class rebalancing, no `pos_weight`.

### 3.2 Tier 2 — scalar head (v3 / v3b), explored, not adopted

A parallel hand-engineered scalar head (distance, angle, GK angular coverage,
body part, pressure) on top of v2. **`v3`** (all 9 features) closes the xG gap
but its counterfactuals regress badly; **`v3b`** (6 g-independent "context"
features) calibrates best but counterfactuals are still worse than v2.

### 3.3 Validation of v2

Multi-seed CIs (5 seeds), transfer evaluation to held-out domains
(women, other men's competitions), and a transfer counterfactual quality check.

### 3.4 Position-jitter augmentation — tested, neutral

The supervisor's augmentation suggestion, implemented and measured directly.
Neutral result (see §6).

---

## 4. Results

### 4.1 Predictive accuracy (test split, men 2015/16 top 5)

| Model | AUC | Brier | ECE | Notes |
|---|---|---|---|---|
| StatsBomb xG | 0.821 | — | — | external benchmark |
| baseline_v1 | 0.803 | 0.075 | 0.010 | spatial only, global pool |
| **baseline_v2** | **0.814 ± 0.003** | **0.0716 ± 0.0001** | **0.0077 ± 0.0008** | 5-seed mean ± std; **recommended** |
| baseline_v3 (all scalars) | 0.819 ± 0.001 | 0.0704 ± 0.0001 | 0.0097 ± 0.0025 | 5-seed mean ± std; ≈ StatsBomb xG; bad counterfactuals |
| baseline_v3b (context scalars) | 0.815 | 0.0708 | 0.0065 | single seed; best calibration |

v2 closes most of the v1→xG gap (residual ~0.007 AUC) while keeping calibration
within 1pp.

**Is v3 actually a better predictor than v2?** Both are now 5-seed, on matched
seeds, so we can compare properly (paired). Mean paired ΔAUC = **+0.0050** in
v3's favour, paired *t* = 2.80, **p ≈ 0.049** — real, but *only marginally*
significant, and driven substantially by one seed (seed 42: Δ = +0.011; seed 3:
Δ ≈ 0). So v3's predictive edge over v2 is ~0.5 AUC points and borderline; it is
**not** a decisive predictive win, and it comes at a large counterfactual cost
(§4.2). The v3 ≈ StatsBomb xG (0.821) claim holds; the v3 ≫ v2 claim does not.

### 4.2 Counterfactual quality (200-shot seeded sample — the property that matters)

This is what makes a *positioning recommender* rather than just an xG model. We
measure how often `g*` is pathological: pinned to the grid edge, or on the
"far" side (behind the post opposite the shooter — the anti-coaching signal).

| Model | far-side / anti-coaching | edge-pinned | post-side split (centre / near / far) |
|---|---|---|---|
| **baseline_v2** | **0.2% ± 0.45%** (5-seed) | 0.7% ± 0.45% | ~183 / 17 / 0 |
| baseline_v3 (all) | **16.3% ± 7.3%** (5-seed) | 9.9% ± 15.5% | 67 / 94 / 39 (seed 42) |
| baseline_v3b (context) | 14% (single seed) | 1.5% | 79 / 93 / 28 |

v2 vs v3 on the far-side metric is ~40σ apart, and the 5-seed v3 run shows the
penalty is not only large but **unstable**: far-side ranges 6–26% and edge-pinning
1.5–37.5% across seeds (one seed pins 37.5% of optima to the grid edge). v2's
clean counterfactuals, by contrast, are tight and reproducible (far-side 0–1%,
pinning ≤1.5%) — not a lucky run. So the auxiliary head doesn't just trade a bit
of `g*` quality for accuracy; it makes the positioning policy erratic.

### 4.3 The headline finding — a predictive vs counterfactual *training* tension

The v3b "context" scalars are **g-independent**: they add the same constant to
every grid cell's logit and therefore **cannot mathematically change which `g`
is optimal**. Yet v3b's `g*` still degraded relative to v2. The only thing that
differs is the CNN trunk's learned weights — **adding any auxiliary predictive
head lets the conv layers offload variance onto the scalar MLP, so the spatial
branch becomes a worse *function of keeper position*.** Predictive accuracy and
counterfactual-policy validity trade off, and the cause is **training
interference, not the inference-time features.** We propose reporting this
comparison itself as a contribution. Practical consequence: one model cannot
be simultaneously the best xG predictor and the best positioning recommender
under this architecture.

### 4.4 Transfer (generalization of v2)

| Split | AUC | gap to xG | ECE | far-side cf |
|---|---|---|---|---|
| test (in-domain) | 0.809 | ~0.012 | ≤1.7pp | 0% |
| women (12.6k) | 0.781 | ~0.014 | ≤1.7pp | 0% |
| men_other (25.8k) | 0.797 | ~0.013 | ≤1.7pp | 1% |

The gap to StatsBomb xG is *stable* across domains and the counterfactual `g*`
stays clean out-of-domain — **the goal-frame geometry that drives `g*` is
domain-invariant even though raw P(goal) accuracy drops** on harder splits.

---

## 5. Model selection / recommendation

- **Positioning recommender (the project's actual goal): use `baseline_v2`.**
  It is the only model with clean, reproducible, transfer-stable counterfactuals.
- **Predictive benchmark vs StatsBomb xG: cite `baseline_v3`** (5-seed AUC
  0.819 ± 0.001 ≈ 0.821). This claim is now firm.
- **But v3 is *not* a decisive predictive upgrade over v2.** With both at 5 seeds,
  the paired edge is only +0.005 AUC (p ≈ 0.049), while v3's counterfactual policy
  is far worse and erratic across seeds. So the recommendation stands and is now
  quantified: **v2 for positioning, v3 only as the xG-parity benchmark** — you are
  giving up ~0.5 borderline AUC points, not a real predictive advantage, to keep a
  clean and stable `g*`.
- The v2/v3/v3b comparison is a reportable result about auxiliary-head training
  interference.

---

## 6. Response to the supervisor's four suggestions

| Suggestion | Outcome |
|---|---|
| Augment features with distance-to-goal, coordinates, etc. | ✅ Done — geometry channels (v2) + scalar features (v3). |
| Inductive biases | ✅ Done — goal-frame channel + CoordConv + spatial pool. **The decisive fix.** |
| Augment with slightly perturbed positions | ✅ Tested → **neutral.** Train-only jitter (σ=0.75 m); predictive AUC and counterfactual metrics unchanged, and measured directly, landscape roughness is identical (mean total-variation 0.0053 vs 0.0053). v2's geometry channels already give a smooth `V(x,g)`; there was no brittleness to fix. Good robustness check: clean `g*` survives perturbing every training position. |
| Balance the dataset | ⛔ Deliberately **not done.** It would break calibration, which the `V = P(goal)` interpretation depends on, and there is no imbalance pathology (the val plateau is overfitting, not starvation; a calibrated rare-event model rightly rarely predicts >0.5). |

---

## 7. Open questions / next steps

Analysis and write-up, not new architecture:

- ✅ **v3 multi-seed CIs** — done. Firms up the v2-vs-v3 predictive comparison:
  v3 0.819 ± 0.001, paired edge over v2 only +0.005 AUC (p ≈ 0.049). See §4.1.
- ✅ **Latent embedding analysis** — done. PCA of v2's 64-d embedding; PC1 is a
  danger axis (r = −0.85 with P(goal)). Figure `reports/figures/latent_embedding_v2.png`.
- **Best-of-both?** — worth chasing (stop-gradient between scalar head and conv
  trunk, or simply two models for two purposes), or report the tension as-is?
  *This is the main steer we'd like from you.*
- Smaller ideas: separate ball channel from shooter; defenders-in-cone channel;
  wider crop x ∈ [60, 122]; y-flip augmentation.
- **The written report** — now the main remaining deliverable.

---

## 8. Artefacts to look at

**Report-ready figures (`reports/figures/`, regenerate via `python src/analysis/report_figures.py`):**
- `cf_v2_vs_v3_1e582bc0.png` — the headline finding: the *same* wide shot swept
  under v2 and v3 on a shared scale. v2 keeps `g*` on the shooter's (near) side;
  v3's scalar head pulls `g*` to the far post. The 0% vs 19.5% tension, in one image.
- `far_side_by_model.png` — far-side / edge-pinned optimum rate across v2 / v3 / v3b.
- `latent_embedding_v2.png` — PCA of v2's 64-d CNN embedding, coloured by P(goal) /
  xG / distance / outcome; PC1 is a danger axis (regenerate via
  `python src/analysis/latent_embedding.py`).

**Raw outputs:**
- `README.md` — pipeline, model-variant table, results.
- `CLAUDE.md` — full chronological status log (v1 → v2 → v3/v3b → validation → jitter).
- `results/eval/v2_multiseed_summary.csv`, `results/eval/v3_multiseed_summary.csv` — per-seed predictive + counterfactual numbers.
- `results/counterfactual/v2_geom/` — v2 example heatmaps + `regret_200.csv`.
- `results/counterfactual/v3_scalar/`, `.../v3b_context/` — the tradeoff, side by side.
- `results/counterfactual/v2_transfer_{women,men_other}/` — transfer counterfactuals.
- `models/checkpoints/baseline_v2/training_curves.png`.
- `src/analysis/counterfactual.py` — the sweep; `src/analysis/regret_distribution.py` — the 200-shot metrics.
