# Theoretical background — for the mid-semester talk

**Date:** 2026-07-02 (originally 2026-05-28)
**Presenter:** Lauren Pommer
**Module:** Math and Machine Learning Praktikum, University of Leipzig

This file is a self-contained theory primer to read before the talk. It covers
(1) what the model actually is from a statistics/ML standpoint, (2) what the
counterfactual sweep is and why it is the research contribution, and (3) the
regret metric and how to interpret it.

---

## 1. The learning problem — supervised binary classification

We estimate a conditional probability

$$V(x, g) = P(Y = 1 \mid x, g)$$

where:
- $Y \in \{0, 1\}$ is the shot outcome (1 = goal),
- $g \in \mathbb{R}^2$ is the goalkeeper's $(x, y)$ position (the variable being optimised),
- $x$ is everything else in the freeze frame: shooter + ball, attacking teammates, outfield defenders.

This is a **binary classification** problem. The CNN $f_\theta$ outputs a logit, passed through a sigmoid $\sigma$, trained with **binary cross-entropy**:

$$\mathcal{L}(\theta) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log \sigma(f_\theta(z_i)) + (1-y_i)\log(1 - \sigma(f_\theta(z_i))) \right]$$

where $z_i$ is the 5-channel raster of shot $i$.

BCE is the **negative log-likelihood of a Bernoulli model**. The population minimiser, under a rich enough function class, is the **true conditional probability** $P(Y = 1 \mid z)$. This is why we can read $\sigma(f_\theta(z))$ as a probability and why calibration metrics (Brier, ECE) are meaningful — not just discrimination metrics like AUC.

### Why the metrics we report

| Metric | What it measures | baseline_v1 |
|---|---|---|
| **AUC** | Ranking — does the model order shots by danger? | 0.803 |
| **Brier** | Mean squared error of probabilities — combines calibration + sharpness | 0.075 |
| **ECE** | Calibration — does "30% predicted" actually mean ~30% observed? | 0.010 |
| **Pearson vs StatsBomb xG** | Agreement with an independent baseline | 0.74 |

AUC alone is insufficient for a counterfactual model: we need the *absolute* probability to be trustworthy, not just the ranking, because we later take $\arg\min$ over hypothetical inputs.

---

## 2. Input representation — rasterisation as a soft kernel density

Each player is rendered onto an 80×60 pitch grid as a 2D isotropic Gaussian:

$$K_\sigma(u; p) = \exp\!\left(-\frac{\|u - p\|^2}{2\sigma^2}\right), \quad \sigma \approx 1.5\,\text{m}$$

with positional uncertainty $\sigma$ absorbing annotation noise.

**Five channels:**

1. Shooter
2. Ball
3. Attacking teammates (max-pooled Gaussians)
4. Outfield defenders (max-pooled Gaussians)
5. Goalkeeper

Crowd channels (teammates, defenders) are **max-pooled** rather than summed: this keeps the amplitude bounded in $[0, 1]$ regardless of how many players are present (median 13, range 1–21 in our data).

### Why a CNN — the inductive bias

The CNN embeds two priors that match the physics of the problem:

- **Locality** — danger is determined by *local spatial relationships* (shooter ↔ GK line, defender ↔ ball channel).
- **Translation equivariance** — those relationships should be evaluated the same way wherever they appear on the pitch.

The architecture (`src/models/danger_cnn.py`, ~102k params): three Conv→BN→ReLU blocks with max-pool, then global average pool, two FC layers, sigmoid.

**Crucial prior the CNN does *not* embed:** structural knowledge of the goal frame. The CNN sees a generic pitch raster, not a goal at $x = 120, y \in [36, 44]$. The counterfactual sweep is what exposes this.

---

## 3. The counterfactual sweep



Formally: fix $x$ (shooter, ball, teammates, defenders), and evaluate $V(x, g)$ on a grid of hypothetical $g$. The **optimal counterfactual position** is

$$g^*(x) = \arg\min_{g \in \mathcal{G}} V(x, g).$$

In Pearl's notation this is $V(x, do(G = g))$ restricted to interventions on a single variable. It is only a valid counterfactual if (a) the model has learned the true conditional probability and (b) extrapolation to $(x, g)$ pairs unsupported by the training distribution is well-behaved. The key finding is that **(b) fails** for baseline_v1.

### Pipeline

1. Pick a shot from the test split. Extract the freeze frame.
2. Rasterise everything **except** the GK once → fixed 4-channel context $x$.
3. Choose a grid $\mathcal{G}$:
   - **v1 wide:** $x \in [110, 120]$, $y \in [30, 50]$, 0.5 m step (625 points)
   - **v2 constrained:** $x \in [108, 120]$, $y \in [34, 46]$, 0.5 m step (penalty spot → goal line; goal posts at $y \in \{36, 44\}$ plus 2 m diving margin)
4. For each $g \in \mathcal{G}$: render the GK channel as a Gaussian centred at $g$, concatenate with $x$, forward pass → $V(x, g)$.
5. The output is a 2D **danger heatmap** over the pitch. Read off $g^*$ and compute regret against $g_\text{actual}$.

Cost: ~625 forward passes per shot. Cheap (~5 MFLOPs/shot).

### Why the sweep is the actual research contribution

A calibrated xG-style model only tells you *how dangerous a shot was* — that's **descriptive**.

The sweep upgrades the output to a **prescriptive** statement:

> "Given everything we observed, the keeper should have stood at $g^*$."

That is the deliverable that distinguishes this project from StatsBomb xG.

### What we ran

8 example shots from a 200-shot test sample, stratified into 4 categories:

- 2 **high-regret goals** — V(actual) ≈ 0.44, V(g*) ≈ 0.06, regret ≈ 0.38
- 2 **high-regret saves** — model said "dangerous", keeper saved anyway
- 2 **low-regret** — actual GK essentially optimal, V tiny
- 2 **random**

Each shot run under both grids → 16 heatmaps total.

### Key finding

$g^*$ is **grid-dependent**:

| Grid | Result |
|---|---|
| Wide | 7/8 optima pin at the y-boundary (2–4 m behind the goal posts) |
| Constrained | 6/8 fall inside the goal mouth as expected; 2/8 still pin at the edge |
| Further widening | $g^*$ keeps walking outward. Finite difference $V(33) - V(35) < 0$ → not a corner solution, a **truncation**. |

**Diagnosis.** The model has no inductive bias for the goal frame. It learned the spurious observational correlation:

> Keepers who actually conceded goals were inside the goal mouth at the moment the shot was taken → a GK Gaussian inside that region predicts higher goal probability.

Moving the synthetic GK *outside* that region therefore decreases $V(x, g)$ regardless of football reality. This is a textbook **causal failure of an observational model**: $V(x, g)$ is a good conditional probability on the training support, but $\arg\min_g V$ pushes us off that support, and the model's extrapolation has the wrong sign relative to the underlying physics.

---

## 4. The regret metric

### Definition

For a single shot with actual GK position $g_\text{actual}$:

$$R(x, g_\text{actual}) = V(x, g_\text{actual}) - V(x, g^*) = V(x, g_\text{actual}) - \min_{g \in \mathcal{G}} V(x, g)$$

**Units:** probability points. $R \in [0, 1]$.

- $R = 0$ → the real keeper was at the model's optimum. Nothing to gain by moving.
- $R$ large → according to the model, the real keeper was badly positioned and standing at $g^*$ would have made the shot much less dangerous.

It is a **non-negative** quantity by construction, since $V(x, g^*)$ is the minimum over the grid.

### Why regret, not just $V(x, g_\text{actual})$

$V(x, g_\text{actual})$ tells you *the shot was dangerous* — but a dangerous shot may be unavoidable. A striker breaking through 1-on-1 might score from any keeper position.

Regret factors out shot difficulty: it asks how much of the danger was **attributable to positioning** vs. **intrinsic to the shot**. A 1-on-1 might have $V(x, g_\text{actual}) = 0.6$ and regret = 0.02 (the keeper was as well placed as possible). A scrappy shot from the byline might have $V(x, g_\text{actual}) = 0.4$ and regret = 0.35 (a different position would have killed the danger almost entirely).

### Interpretation — three bands

| Band | $R$ range | Meaning |
|---|---|---|
| Low regret | $R \lesssim 0.05$ | Keeper essentially at the optimum. The danger (if any) is intrinsic to the shot. |
| Moderate | $0.05 < R < 0.2$ | Some positioning improvement possible. |
| High regret | $R \gtrsim 0.2$ | According to the model, a different position would have made a large difference. |

### What we observed

The regret table from the 8 example shots:

| Category | shot_id (short) | is_goal | V(actual) | V(g*) | regret |
|---|---|---|---|---|---|
| high-regret-goal | 9d58…b103c4 | 1 | 0.440 | 0.058 | **0.382** |
| high-regret-goal | 0c56…dee1a5 | 1 | 0.447 | 0.067 | 0.380 |
| high-regret-no-goal | ff82…b6fc | 0 | 0.391 | 0.049 | 0.341 |
| high-regret-no-goal | cb22…06b22 | 0 | 0.386 | 0.080 | 0.307 |
| low-regret | 2b44…ca14 | 0 | 0.007 | 0.007 | ~0 |
| low-regret | 6380…7601 | 0 | 0.006 | 0.006 | ~0 |
| random | 2e57…e95a | 0 | 0.148 | 0.026 | 0.123 |
| random | 3324…0513 | 0 | 0.071 | 0.030 | 0.040 |

Note the **bimodal structure**: high-regret shots have $V(\text{actual}) \approx 0.4$, low-regret shots have $V(\text{actual}) \approx 0.006$. The model is confidently identifying which shots are dangerous, but its *recommendation* about where the keeper should have stood is the part we can't trust yet — see §3.

### Important caveat

Regret is only as trustworthy as $g^*$ is. Because baseline_v1's $g^*$ pins to grid boundaries (the goal-frame inductive bias problem from §3), the **magnitude** of the reported regret is an upper bound on what a goal-aware model would produce — the optimum the model is finding is unphysical, so $V(x, g^*)$ is artificially low and regret is artificially high.

What is still informative:
- **Comparative regret** across shots (high vs low) is meaningful: the bimodal split above is real.
- **Sign and direction** — "this shot had some positional slack" vs "this shot was essentially optimal" — is meaningful.

What is **not** yet trustworthy:
- The exact magnitude of regret.
- The literal $g^*$ recommendation that produced it.

Both become trustworthy once option A (goal-aware inputs) is implemented.

---

## 5. How it all ties together (the one-paragraph summary)

We train a CNN $f_\theta$ to estimate $V(x, g) = P(\text{goal} \mid x, g)$ from a 5-channel pitch raster, using binary cross-entropy on 30k men's open-play shots. The model is well calibrated (Brier 0.075, ECE 0.010) and ranks shots competitively with StatsBomb xG (AUC 0.803, Pearson 0.74). To turn this descriptive model into a prescriptive one, we run a **counterfactual sweep**: hold the non-GK context $x$ fixed and evaluate $V(x, g)$ on a grid of synthetic GK positions, returning $g^* = \arg\min_g V(x, g)$ and **regret** $R = V(x, g_\text{actual}) - V(x, g^*)$. The sweep is what produces the danger heatmap and the positioning recommendation. The key finding is that $g^*$ pins to grid boundaries: the model learned to associate "GK inside the goal mouth" with conceded goals (a spurious observational correlation), so $V$ drops monotonically as the synthetic GK is pushed away from the mouth. This is a **causal failure of an observational model** lacking the right inductive bias (no goal-frame channel), and the fix is structural rather than statistical — adding goal-aware inputs in v2.
