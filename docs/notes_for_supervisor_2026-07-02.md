# Counterfactual goalkeeper sweep — what we found, and an open question

**Date:** 2026-07-02 (originally 2026-05-08; maintained as work progressed)
**Author:** Lauren Pommer
**Audience:** Hannes; supervisor
**Status:** RESOLVED — Option A implemented (baseline_v2); Tier 2 (scalar head)
explored. See updates below.

---

## Update 2 (2026-06-24): Tier 2 scalar head — closes xG gap, but trades off counterfactuals

We added a parallel scalar-feature head on top of v2 (Tier 2). Two variants:

- **v3** ("all" 9 features incl. GK-coverage/offset/depth): test AUC **0.820**,
  matching StatsBomb xG (0.821) — the predictive gap is closed. But the 200-shot
  sweep regressed badly: anti-coaching (far-side) optima went from **0%** (v2) to
  **19.5%**, edge-pinning 0.5% → 4.5%.
- **v3b** ("context" 6 g-independent features only): test AUC **0.815**, best
  calibration of any model (ECE 0.0065). Counterfactuals better than v3 but still
  worse than v2 (far-side 14%).

**The important bit (a real finding).** The "context" features are *g-independent*,
so they add a constant to every grid cell's V and **cannot mathematically change
which g is optimal**. Yet v3b's g* still degraded relative to v2. The only thing
that differs is the CNN branch's learned weights: adding any auxiliary predictive
head lets the conv trunk offload variance onto the scalar MLP, so the spatial
branch becomes a worse *function of keeper position*. **Predictive accuracy and
counterfactual-policy validity trade off, and the cause is training interference,
not the inference-time features.**

**Model selection going forward:**
- **Positioning recommender (our actual goal): v2** — the only model with clean
  counterfactuals (0% anti-coaching), AUC 0.809.
- **Predictive benchmark vs StatsBomb xG: v3** (AUC 0.820 ≈ 0.821).
- We propose reporting the v2/v3/v3b comparison itself as a contribution.

**New question for the supervisor:** is it worth chasing "best of both" (e.g.
stop-gradient between the scalar head and the conv trunk, or simply keeping two
models for two purposes), or do we report this tension and move on to transfer
eval (women / men_other) on v2?

Artefacts: `results/counterfactual/v3_scalar/`, `results/counterfactual/v3b_context/`,
`models/checkpoints/baseline_v3{,b}/`.

---

## Update 3 (2026-06-24): position-jitter augmentation — tested, neutral

You suggested data augmentation with slightly perturbed positions. We tried it
(v2 + train-only Gaussian jitter σ=0.75 m on every player, re-sampled per epoch)
and the result is **neutral**: predictive AUC 0.810 (inside v2's multi-seed
range), counterfactual far-side/pinning/regret unchanged, and — measuring it
directly — the V(x,g) landscape roughness is **identical** (mean total-variation
0.0053 vs 0.0053). The smoothing we hoped for didn't materialize, because v2's
goal-geometry channels already give a smooth, clean landscape; there was no
brittleness left to fix. So: a good idea in general, but not a needed lever here
— the inductive-bias work already covered it. (It's a nice robustness check
though: v2's clean g* survives perturbing every training position.)

For reference, your other suggestions map to: distance-to-goal / coordinates →
done (geometry channels + scalar features); inductive biases → done (goal-frame
channel + CoordConv + spatial pool, the decisive fix); balance the dataset →
deliberately not done (it would break calibration, which the V=P(goal)
interpretation depends on, and there is no imbalance pathology).

---

## Update 1 (2026-06-24): Option A implemented, and it worked

We took **Option A** (add structural / goal-aware inputs) and it resolved the
grid-dependence problem. Summary for the supervisor:

**What changed (Tier 1 feature engineering):**
- Five static goal-geometry channels appended to the raster: a goal-frame
  Gaussian ridge on the goal-line segment, distance-to-goal, the angle
  subtended by the two posts ("view angle"), and two normalized coordinate
  channels (CoordConv). The model can now *see* where the goal is.
- The global-average-pool head was replaced with a small (4×3) spatial pool, so
  absolute location survives into the classifier rather than being averaged away
  — this was a second, independent reason v1 reasoned about density not geometry.
- baseline_v2: 10 input channels, 194k params (within budget), trained in 24
  epochs (early-stopped). Calibration deliberately preserved (no class
  rebalancing, no pos_weight).

**Results:**
- Test metrics improved and calibration held: AUC 0.803 → **0.809**, Brier
  0.075 → **0.072**, ECE 0.010 → **0.007**. Closer to StatsBomb xG (AUC 0.821).
- Counterfactual sweep, 8 example shots: **0/8 optima pinned to the grid edge**
  (v1 had 2/8 on the constrained grid, 7/8 on the wide grid). The anti-coaching
  far-post pull is gone — e.g. the shot from (111.2, 45.5)-type wide angles now
  send g* toward the *near* post and *off the line* to cut the angle, as a coach
  would.
- 200-shot aggregate (seeded): median regret 0.010 (goals 0.051, saves 0.008);
  y-edge pinned **1/200 (0.5%)**; far-side / anti-coaching optima **0/200
  (0.0%)**; post-side split 182 centre / 18 near / **0 far**.

**Answer to the original Q3 (is the constrained grid defensible?):** less
load-bearing now — v2 no longer pins to the edge regardless of grid, so g* is
not an artefact of where we draw the box.

**Open for the supervisor (new):**
- Q2 from below still stands: Tier 2 would add a *hand-engineered scalar head*
  (distance / angle / GK angular coverage). It is the most likely way to close
  the residual ~0.012 AUC gap and sharpen near/far-post discrimination (v2
  optima are 91% "centre"), but it trades away some of the "purely spatial"
  methodological appeal. Proceed to Tier 2, or move to transfer eval first?

The original decision-point write-up is preserved below for reference.

---

## TL;DR (original, 2026-05-08)

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
