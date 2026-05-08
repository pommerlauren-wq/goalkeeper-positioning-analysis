# Counterfactual goalkeeper sweep — what we found, and an open question

**Date:** 2026-05-08
**Author:** Lauren Pommer
**Audience:** Hannes; supervisor
**Status:** decision point — looking for input before continuing

## TL;DR

The counterfactual sweep `V(x, g) = P(goal | x, g)` runs end-to-end on the
DangerCNN baseline (`baseline_v1`, test AUC 0.803, Brier 0.075, ECE 0.010).
Mechanically it works. But the recommended goalkeeper position `g*` depends
**fundamentally on the choice of sweep grid**, and the underlying reason is
a structural limitation of the v1 model. Before doing a full test-set
counterfactual analysis I want to align on what we should actually claim
from this model.

## Setup

For a fixed shot context `x` (shooter, ball, attackers, defenders), we
sweep the goalkeeper position `g` over a 2D grid and evaluate the model's
goal probability `V(x, g)`. The optimum is `g* = argmin_g V(x, g)` and the
"regret" of the actual keeper is `V(x, g_actual) − V(x, g*)`.

Two grids tested:

- **Wide:** x in [110, 120], y in [30, 50], 0.5 m step
- **Constrained:** x in [108, 120], y in [34, 46], 0.5 m step
  (penalty spot to goal line; goalposts at 36 and 44 plus 2 m diving margin)

## Empirical finding

On 8 example shots (2 high-regret goals, 2 high-regret saves, 2 low-regret,
2 random; chosen from a 200-shot test sample):

- Under the **wide** grid, 7/8 optima land at the y boundary, almost all
  at y in {30, 48} — i.e. 2–4 m **behind the goal posts**.
- Under the **constrained** grid, optima for 6/8 shots fall inside the
  goal mouth as expected. 2/8 still pin at y = 34, the constrained edge.
- Widening just for those two pinned shots to y in [32, 48] sees `g*`
  walk all the way to y = 32. The finite difference `V(33) − V(35)` at the
  optimal x is negative for both shots, meaning V is still falling at the
  edge — not a corner solution but a truncation.

For one of the pinned shots: shooter at (111.2, 45.5), goal mouth y in
[36, 44]. The model wants the keeper at y = 32 (behind the **far** post,
opposite the shooter). A real keeper would shift toward the **near** post
(y ≈ 44) to cut the angle. The model is recommending the opposite of what
football coaching would say.

## Diagnosis

The DangerCNN has no structural knowledge that y in [36, 44] defines the
goal mouth, or that the goal posts are at (120, 36) and (120, 44). It only
sees a 5-channel pitch raster of player positions. From the training data
(81k shots), it learned a useful but **spurious** correlation:

> Goalkeepers who actually conceded goals were standing inside the goal
> mouth at the moment the shot was taken. So a GK Gaussian inside that
> region predicts higher goal probability.

The counterfactual sweep then exploits this directly: V drops monotonically
as the synthetic GK is moved *away* from the goal mouth, because the model
sees less GK density in the danger region. The minimum is therefore
wherever the grid forces the keeper to remain — not a learned positional
optimum.

This is a v1 design limitation, not a bug, and it is consistent with the
saturation we saw at training time (val plateaued at epoch 5–7; train kept
improving without val gain). The model has extracted what it can from the
position-only raster.

## Three directions

I see three options. They are not mutually exclusive, but each implies a
different research story.

**A. Add structural priors / goal-aware inputs to v2.**
Encode the goal frame explicitly: a binary "goal mouth" channel, distance
and angle from shooter to goal centre as scalar features through a parallel
head, or a learned shooter-to-goal-vector input. Most likely to produce
sensible `g*` recommendations and close the residual gap to StatsBomb xG.
Cost: model redesign, retraining, longer timeline. This is the path most
consistent with publishable counterfactual claims.

**B. Restrict analysis to evaluating actual GK positions only.**
Drop the `g*` recommendation entirely. Compute `V(x, g_actual)` per shot,
compare to StatsBomb xG, and study how the value depends on the GK's
observed position via the model. Honest, well-scoped, and we already have
all the artefacts. Cost: gives up the counterfactual heatmap, which was
the main novel deliverable.

**C. Frame the artefact as a finding about the limits of pure data-driven
counterfactual GK modelling.**
Lean into what we observed. Keep the sweep, document the grid-dependence
and the y = 32 pull, and write up the message that position-only CNNs
without explicit goal-frame priors do not learn counterfactually valid
goalkeeper-positioning policies. Modest in technical scope, but a clean
empirical contribution about model–task mismatch in football analytics.
Cost: less original modelling work; depends on whether this null-ish
result is interesting enough on its own.

## What I want input on

1. Which of A / B / C (or combination) is most appropriate for the
   Praktikum scope and timeline?
2. If A: are we allowed to add hand-engineered features, or is the
   methodological appeal of the project that the model is "purely
   spatial"? That would push us toward a structural prior (e.g. goal-mouth
   channel) rather than scalar features.
3. Independently of the above, before any further analysis run: is the
   constrained grid `y in [34, 46]` defensible as the canonical evaluation
   region given what we now know it represents (the model's preference is
   "outside the goal mouth, opposite the shooter")?

## Artefacts you can look at

- `results/counterfactual/` — 8 example PNGs, original wide-grid sweep
- `results/counterfactual/v2/` — same 8 shots with the constrained grid
- `results/counterfactual/v2/diff_v1_vs_v2.csv` — head-to-head movement of `g*`
- `models/checkpoints/baseline_v1/training_curves.png` — training curves
- `src/analysis/counterfactual.py` — the sweep / visualisation module
