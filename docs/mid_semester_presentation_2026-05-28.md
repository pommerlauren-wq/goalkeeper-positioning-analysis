# Mid-semester presentation — current position

**Date:** 2026-05-28
**Presenter:** Lauren Pommer
**Module:** Math and Machine Learning Praktikum, University of Leipzig
**Collaborator:** Hannes
**Topic:** Counterfactual goalkeeper positioning from StatsBomb event data

> **Status: DELIVERED — historical record (talk given 2026-05-28).**
> This briefing captures the project at the `baseline_v1` stage, when the
> counterfactual `g*` failure was an open finding and the path forward (A / B / C)
> was a pending supervisor decision. **Everything in §7 has since been resolved:**
> we took Option A, shipped `baseline_v2` (clean counterfactuals), explored a
> scalar head (`v3`/`v3b`), and validated v2 with multi-seed CIs + transfer eval.
> For the post-presentation state, see **`docs/project_overview_2026-06-30.md`**
> (consolidated current overview) and `docs/notes_for_supervisor_2026-05-08.md`
> (Updates 1–3). This file is left unchanged as a snapshot of the talk.

This file is a single-source briefing for the mid-semester talk: what we have done,
what works, what doesn't, what's next. Visual artefacts are referenced by path so
they can be dropped straight into slides.

---

## 1. Project goal (one slide)

Build a counterfactual goalkeeper positioning model

> `V(x, g) = P(goal | x, g)`

where `g` is the goalkeeper's (x, y) position and `x` is everything else in the
freeze frame (shooter + ball, attacking teammates, outfield defenders).

At inference, `x` is fixed and `g` is swept across a pitch grid to produce a
danger heatmap. The optimal position is

> `g* = argmin_g V(x, g)`.

**Scope (phase 1):** open-play shots only. Excludes set pieces, penalties, free kicks.

**Baselines we benchmark against:**
- Anzer & Bauer (2021): RPS = 0.197 on Bundesliga (uses full optical tracking)
- StatsBomb built-in xG (`shot_statsbomb_xg`)

---

## 2. Data (one slide)

Source: **StatsBomb open data** via `mplsoccer.Sbopen` (no local downloads).

Important data property — **shot-event freeze frames**, not StatsBomb 360:
- Manually annotated, scoped to players near the shot
- A missing player ≠ "not on the pitch" — only "not annotated near the shot"
- GK coverage 99.9% in open play → GK position signal reliable
- Median 13 visible players per shot (range 1–21)
- Implication: more variance in context completeness than tracking-data work like Anzer & Bauer

**EDA outputs:**
- `results/eda/eda_summary.md` — full table
- `results/eda/shot_heatmap.png` — where shots happen on the pitch
- `results/eda/players_per_shot.png` — visible-player distribution
- `results/eda/xg_distribution.png` — StatsBomb xG distribution

| Metric | Value |
|---|---|
| Total shots | 87,111 |
| Open-play shots | 81,551 (93.6%) |
| Open-play conversion | 10.3% |
| With identifiable opponent GK | 81,453 (99.9%) |
| Final modelling sample | **81,453** |

**Splits (option 3):**
- Train / val / test = men 2015/16 top 5 leagues = **30k / 6.4k / 6.5k**
- Transfer eval (held out) = women (12.6k) + men_other (25.8k)
- Goal rate consistent across splits (9.7%–10.3%)

---

## 3. Method — how we visualise and learn the problem

### 3.1 Input representation (one slide)

Each shot is rasterised to a **5-channel pitch image** (80 × 60 grid, sigma ≈ 1.5 m):

1. Shooter + ball
2. Attacking teammates
3. Outfield defenders
4. Goalkeeper
5. (5th channel — see code)

Players are rendered as 2D Gaussian blobs; crowd channels are max-pooled.

**Visual:** `results/rasterize_check/ba46e9d6-e828-4599-952c-39c1f7d22659.png`
→ shows the rasterised 5 channels side by side for one real shot. Use this slide
to *visualise the prediction problem* — the model only sees these blobs.

### 3.2 Model

**DangerCNN baseline_v1** — 102k parameters, CNN backbone over the 5-channel
raster, BCE loss. Source: `src/models/danger_cnn.py`. Training curves:
`models/checkpoints/baseline_v1/training_curves.png`.

---

## 4. What works — baseline_v1 results (one slide)

Test set (men 2015/16 top 5, n = 6.5k):

| Metric | baseline_v1 | What it means |
|---|---|---|
| **AUC** | **0.803** | Ranks shots well |
| **Brier** | **0.075** | Probabilities are sharp |
| **ECE** | **0.010** | Excellent calibration (within 1pp) |
| Decile calibration | predicted vs actual within 1.5pp across all 10 buckets | Honest probabilities |
| Pearson vs StatsBomb xG | **0.74** | Same direction, but independent in absolute terms |

**Headline:** competitive with StatsBomb xG using *only spatial inputs* — no
distance, no angle, no body part, no pressure flag.

**Training behaviour:** val plateaus around epoch 5–7; train keeps improving →
diagnosis is a **saturated input representation**, not a broken model. v1 has
extracted what it can from the position-only raster.

---

## 5. The actual research contribution — counterfactual sweep

### 5.1 What we ran

For 8 example shots (2 high-regret goals, 2 high-regret saves, 2 low-regret,
2 random; chosen from a 200-shot test sample) we swept `g` and computed
`V(x, g)` + regret `V(x, g_actual) − V(x, g*)`.

Two grids:
- **v1 wide:** x ∈ [110, 120], y ∈ [30, 50], 0.5 m step
- **v2 constrained:** x ∈ [108, 120], y ∈ [34, 46], 0.5 m step (penalty spot
  → goal line; goalposts at 36 / 44 plus 2 m diving margin)

### 5.2 What we found — the key finding for this talk

**`g*` is grid-dependent in a way that exposes a structural model limitation.**

| Grid | Result |
|---|---|
| Wide  | **7 / 8 optima pin at the y-boundary** (y ∈ {30, 48}) — *2–4 m behind the goal posts* |
| Constrained | 6 / 8 optima fall inside the goal mouth as expected; **2 / 8 still pin at the edge (y = 34)** |
| Widening the 2 pinned shots further | `g*` walks all the way to y = 32. Finite difference `V(33) − V(35)` is still negative → not a corner solution, a **truncation**. |

For one of the pinned shots: shooter at (111.2, 45.5), goal mouth y ∈ [36, 44].
The model wants the keeper at **y = 32** — behind the **far** post, opposite the
shooter. A real keeper would shift toward the **near** post (y ≈ 44) to cut the
angle. **The model recommends the opposite of what football coaching would say.**

### 5.3 Diagnosis (the slide that ties it together)

DangerCNN has no structural knowledge that y ∈ [36, 44] is the goal mouth or that
the posts sit at (120, 36) and (120, 44). It only sees a 5-channel raster of
player positions. From 81k training shots it learned a useful but **spurious**
correlation:

> Goalkeepers who actually conceded goals were standing inside the goal mouth
> at the moment the shot was taken. So a GK Gaussian inside that region
> predicts higher goal probability.

The counterfactual sweep exploits this directly: V drops monotonically as the
synthetic GK is moved *away* from the goal mouth, so the minimum lands wherever
the grid forces the keeper to stop — **not a learned positional optimum**.

This is consistent with the training saturation we saw at epoch 5–7. v1 has
no concept of the goal frame.

### 5.4 Visual artefacts to show

Pick 2–3 of these for slides — the contrast between high-regret and low-regret
shots makes the story land:

- `results/counterfactual/high-regret-goal_9d58e332-….png` — model predicted high goal probability at actual GK; goal scored
- `results/counterfactual/high-regret-no-goal_cb22793f-….png` — model predicted high goal probability but the keeper saved
- `results/counterfactual/low-regret_2b44a51a-….png` — actual GK position essentially optimal
- `results/counterfactual/random_2e571682-….png` — middling case
- `results/counterfactual/v2/…` — same 8 shots under the constrained grid
- `results/counterfactual/v2/diff_v1_vs_v2.csv` — head-to-head movement of `g*`

**Suggested slide pair:**
1. *One shot under wide grid* → `g*` pins behind the post
2. *Same shot under constrained grid* → `g*` shifts but for the wrong reason
   (the model's preference is still "outside the goal mouth")

### 5.5 Hard numbers (regret table, summary.csv)

| Category | shot_id (short) | is_goal | V(x, g_actual) | V(x, g*) | regret |
|---|---|---|---|---|---|
| high-regret-goal | 9d58…b103c4 | 1 | 0.440 | 0.058 | **0.382** |
| high-regret-goal | 0c56…dee1a5 | 1 | 0.447 | 0.067 | 0.380 |
| high-regret-no-goal | ff82…b6fc | 0 | 0.391 | 0.049 | 0.341 |
| high-regret-no-goal | cb22…06b22 | 0 | 0.386 | 0.080 | 0.307 |
| low-regret | 2b44…ca14 | 0 | 0.007 | 0.007 | ~0 |
| low-regret | 6380…7601 | 0 | 0.006 | 0.006 | ~0 |
| random | 2e57…e95a | 0 | 0.148 | 0.026 | 0.123 |
| random | 3324…0513 | 0 | 0.071 | 0.030 | 0.040 |

Note the bimodal structure: high-regret shots have huge V(actual) (~0.4),
low-regret shots have tiny V(actual) (~0.006). The model is confidently
identifying which shots are dangerous — but its *recommendation* about where the
keeper should have stood is the part we can't trust yet.

---

## 6. What works vs what doesn't (one summary slide)

**Works**
- End-to-end pipeline: rasterisation → CNN → calibrated probabilities
- baseline_v1 is competitive with StatsBomb xG on rank / Brier / calibration
- Counterfactual sweep + regret pipeline runs end-to-end
- The model honestly identifies high-danger vs low-danger contexts

**Doesn't work yet**
- `g*` recommendations are not counterfactually valid
- Optima pin to grid edges → "outside the goal mouth"
- For some shots, model points to the **opposite** of football coaching intuition
- Root cause: no structural knowledge of the goal frame in the input

**Why the failure is informative, not fatal**
- The training saturation already told us v1 had extracted what it could
- The sweep makes the limitation legible and pointable-at — it gives us the next research step

---

## 7. Next steps and future directions (one slide)

Currently **blocked on a supervisor decision** documented in
`docs/notes_for_supervisor_2026-05-08.md`. Three options:

**A. Add structural priors / goal-aware inputs to v2.**
A binary "goal mouth" channel; or distance / angle scalar head; or a learned
shooter-to-goal-vector input. Most likely to produce sensible `g*` and close
the residual gap to StatsBomb xG. *Most consistent with publishable
counterfactual claims.* Cost: model redesign, retraining.

**B. Drop `g*` recommendation; evaluate `V(x, g_actual)` only.**
Honest, well-scoped, all artefacts already exist. Cost: gives up the
counterfactual heatmap — the original novel deliverable.

**C. Frame v1 as a finding about the limits of pure data-driven counterfactual
GK modelling.** Lean into the observation. Modest scope, but a clean empirical
contribution about model–task mismatch in football analytics.

**Downstream of A / B:**
- Transfer evaluation on women + men_other splits (12.6k + 25.8k held out)
- Latent embedding analysis from pre-MLP features

**v2 levers if we go path A:**
- Parallel scalar head with distance / angle / body part
- Goal-mouth structural channel
- Wider pitch crop x ∈ [60, 122] to fit GK Gaussian fully
- Y-axis flip augmentation
- Heavier dropout / weight decay

---

## 8. Suggested slide order

1. Title + project goal (`V(x, g) = P(goal | x, g)`, sweep diagram)
2. Data — StatsBomb open data, sample sizes, the freeze-frame caveat
3. The prediction problem visualised — rasterisation example (`rasterize_check`)
4. **Theory I — supervised learning setup** (from `docs/theory_background_2026-05-28.md` §1–2): BCE as Bernoulli NLL → calibrated probabilities; CNN inductive biases (locality + translation equivariance); the *missing* prior (no goal-frame structure)
5. Model — DangerCNN, 5 channels, training curves
6. Baseline_v1 results — AUC / Brier / ECE / vs StatsBomb xG
7. **Theory II — counterfactual sweep and regret** (theory doc §3–4): definition of `g* = argmin_g V(x, g)`; regret `R = V(x, g_actual) − V(x, g*)`; why regret factors out intrinsic shot difficulty
8. Counterfactual sweep — what we ran (grids), one example heatmap
9. **The key finding** — grid-dependence, behind-the-post recommendation, diagnosis as observational/causal failure
10. What works / what doesn't (the summary table from §6)
11. Next steps — A / B / C decision and downstream work

Total: 11 slides; pace ~1.3 min/slide for a ~15 min talk. If time pressure, theory I and theory II can each be compressed to a half-slide alongside their adjacent application slide (4+5 merged, 7+8 merged) → back to 9 slides.

---

## 9. Honest-framing tips for the talk

- Lead with the headline that baseline_v1 is **calibrated and competitive with
  StatsBomb xG** — this earns credibility before the counterfactual caveat.
- Present the counterfactual failure as **a discovered structural limitation,
  not a bug**. The pipeline works; the model lacks the right inductive bias.
- The fact that we identified this *before* doing a full 6.5k-shot sweep is the
  research-process win — it saved a lot of wasted compute on a model we now
  know would produce systematically biased `g*`.
- Be explicit that the next step is a *decision*, not just more code: A vs B
  vs C have very different research stories.
