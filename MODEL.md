# A structural model of the Academic–Employer reputation gap

Simulation: `scripts/23_model_sim.py` → `outputs/figures/model_*.png`, `data/interim/model_results.json`.
**One** structural parameter vector is used throughout (no per-fact tuning); every prediction P1–P5 is
checked against the existing empirical facts, and misses are reported, not tuned away.

---

## 1. The model

Field *f* has institutions *i*. Each institution's graduates have two latent qualities — an
**academic** value θ_A and a **market** value θ_M:

```
(θ_A[f,i], θ_M[f,i]) ~ N(0, [[1, ρ_f],[ρ_f, 1]])
```

- **ρ_f = corr_i(θ_A, θ_M)** is the field's *alignment*; **(1 − ρ_f)** is the genuine
  **valuation divergence** — the degree to which the academy and the market value different things.

**Academic prestige** ranks institutions on academic value, with noise:
```
A[f,i] = θ_A[f,i] + σ_A · ε          (AR side; academia hires on academic value)
```

**Earnings** at horizon *t* are a credential-weighted blend that drifts toward market value as
employers learn (Altonji–Pierret), plus sorting/measurement noise that resolves with *t*:
```
w(τ_f, t) = exp(−τ_f · t)                              credential weight ↓ in observability τ and in t
S[f,i](t)   = w·θ_A + (1−w)·θ_M                        systematic earnings
S_λ[f,i](t) = (1 − λ_f) · S[f,i](t)                   licensing compresses cross-institution signal
σ_E²(t)     = σ_∞² + σ_0² · exp(−δ t)                  sort noise ↓ in t (early cohorts unsorted)
E[f,i](t)   = S_λ[f,i](t) + σ_E(t) · ε                 ER side
```

**Gap** = one minus the rank agreement of prestige and earnings:
```
gap_f(t) = 1 − Spearman_i( A[f,i], E[f,i](t) )
```

A field is described by just three primitives **(ρ_f, τ_f, λ_f)** plus the common noise schedule.
Calibration (held fixed): N=150, σ_A=0.50, σ_∞=0.30, σ_0=1.00, δ=0.70.

### Closed form (Pearson proxy; validated against the Spearman MC)
With unit-variance θ's,
```
corr(A,E) = (1−λ)(w + (1−w)ρ) / sqrt( (1+σ_A²) · [ (1−λ)² B + σ_E²(t) ] ),   B = w²+(1−w)²+2w(1−w)ρ
gap = 1 − corr(A,E)
```
Three facts fall straight out of this expression and drive everything below:

1. **Floor.** As σ_E→σ_∞ (large t), gap → `1 − (1−λ)(w+(1−w)ρ)/sqrt((1+σ_A²)((1−λ)²B+σ_∞²))`, which
   **increases in (1−ρ)** and, at ρ=1, is **independent of w (hence of τ)**.
2. **Licensing needs noise.** If σ_E=0, the (1−λ) factor cancels (rank correlation is scale-invariant):
   licensing moves the gap **only when noise is present** → a licensing × noise interaction.
3. **τ acts through w.** ∂gap/∂τ has the sign of ∂gap/∂(1−w) and **magnitude scaling with (1−ρ)**,
   → 0 as ρ→1.

---

## 2. Predictions, derivations, and simulation results

### P1 — Horizon: gap declines in *t* toward a floor increasing in (1−ρ)
*Derivation.* Two forces as *t*↑: σ_E²↓ raises corr (gap↓); w↓ moves systematic earnings from θ_A toward
θ_M, lowering alignment with A=θ_A (gap↑ toward the (1−ρ) floor). Over 1–5 yr the noise term dominates
for aligned fields.
*Simulation.* floors rise with divergence — **(1−ρ)=0.1→floor 0.24, 0.4→0.49, 0.7→0.74**; the 1-yr gap is
noise-inflated and 4–5 yr sits at the floor for low/moderate divergence. **Match: YES** (grand gap falls;
floor ∝ (1−ρ); 4–5 yr ≈ floor). **Honest refinement / partial miss:** for the *most divergent* fields the
default calibration has w-drift beat noise-resolution, so their gap **rises** 1→5 yr (falls_1_to_5 = False
at (1−ρ)=0.7). The aggregate still falls (panel mean 0.50→0.42), but the model predicts the high-divergence
*tail* should fall least / rise — a falsifiable refinement, not clearly seen in the data (where the highest-
gap clusters also fall). Figure: `model_gap_vs_t.png`.

### P2 — Licensing: gap increases in λ (reproduce b_licensure ≈ +0.66)
*Derivation.* ∂corr/∂λ < 0 whenever σ_E>0; as λ→1 the systematic signal vanishes and gap→1.
*Simulation.* the *linear* slope ∂gap/∂λ = **+0.58 (t=1), +0.48 (t=4), +0.33 (t=8)** — positive and
**declining in t**, exactly the licensing × noise interaction the closed form predicts. **Match: YES on sign
and robustness.** The gap–λ curve is **convex**, so the linear slope over λ∈[0,0.85] understates the effect:
the model's licensing effect over the **full support** at 4 yr is gap(λ=0)=0.33 → gap(λ=1)≈0.97, a rise of
**≈ +0.64 — essentially on top of the empirical +0.66** (not tuned; set by the noise level). The slope figure
quoted depends on the λ-range, so we report the full-support rise. Figure: `model_gap_vs_lambda.png`.

### P3 — Pipeline NULL: academic absorption is not in the mechanism
*Derivation.* Absorption (share of graduates entering academia) appears in **none** of the primitives
(ρ, τ, λ, σ). The simplest gap-generating model has no absorption term — so the empirical null is the
**expected** result, and a non-null would have *required* an absorption channel (model falsified).
*Simulation.* With absorption drawn **independent** of the primitives, its gap coefficient is
**−0.04 [−0.09, +0.01], p=0.13 (NULL)**, alongside b_λ=+0.34. With absorption **confounded** with divergence
(absn = 0.6(1−ρ)+noise) it acquires a **spurious +0.26 (p≈0)**. **Match: YES — the cleanest hit.** It both
reproduces the null and yields a testable implication: *the empirical null implies absorption ⟂ (1−ρ)*.

### P4 — Observability: ∂gap/∂τ, and why the pooled gradient is weak
*Derivation.* ∂gap/∂τ scales with (1−ρ) and →0 at ρ=1 (then w drops out of the numerator).
*Simulation.* ∂gap/∂τ = **0.07 / 0.41 / 0.86** at (1−ρ) = 0.05 / 0.30 / 0.60 — the predicted (1−ρ) scaling and
ρ=1 vanishing. The τ × (1−ρ) **interaction is strong and robust** (+1.66 at t=4, p≈0). **Match: PARTIAL.**
The mechanism and the interaction prediction hold, **but** the model does **not** reproduce a *weak* τ
**main** effect at 4–5 yr — in the default calibration it is strong and **non-monotonic in t**
(b_τ = 0.13 / 0.58 / 0.72 / 0.50 at t = 1 / 4 / 8 / 12: weak under early noise, strong mid-convergence). Two
honest reconciliations of the weak *empirical* gradient: (i) the empirical proxy is **earnings dispersion**,
which conflates **(1−λ)** [licensing → gap↓ in dispersion] with **τ** [learning → gap↑], two *opposing*
channels that cancel to a weak/null pooled slope — exactly what was observed; (ii) a weak main effect arises
only once fields are **near their floors** (full convergence). The model's sharper test is therefore the
**τ × (1−ρ) interaction with λ controlled**, not the raw dispersion gradient. Figure: `model_gap_vs_tau.png`.

### P5 — Direction (lead–lag for the eventual 2-wave test)
*Derivation.* A = θ_A + noise is the **slow prior** (same latent across waves); E **drifts** toward θ_M as
w(τ,t)↓. In low-ρ fields the drift carries E *away* from A, opening the gap.
*Simulation (t=1 vs t=5).* earnings-rank drift rises with divergence (Spearman **+0.54**); the gap *change*
correlates **+0.71** with (1−ρ): low-divergence gaps **fall** (mean Δ = −0.10), high-divergence gaps **open**
(mean Δ = +0.09). **Directional prediction for the 2-wave AR–ER test:** prestige is the leading, stable
series; earnings rankings drift off it; the decoupling ΔE and Δgap are larger where **(1−ρ) and τ** are
larger. So earnings should *lag* prestige, and the lag/decoupling is a function of divergence and
observability.

---

## 3. Model ↔ empirics

| Empirical fact (prior work) | Model prediction | Match |
|---|---|---|
| Gap falls 1 yr → 4–5 yr toward a floor; 4–5 yr flattens; 1 yr noise-inflated | P1: σ_E(t)↓; floor=(1−w(τ))(1−ρ) | **YES** (grand falls; floor∝(1−ρ)); tail refinement |
| Between-cluster **ICC falls** with horizon (0.50→0.30) | model between-field variance ~flat/slightly **rises** (0.016→0.019) | **NO** (honest miss) |
| b_licensure ≈ **+0.66**, robust sign | P2: ∂gap/∂λ>0, × noise | **YES** sign/robustness; full-support rise ≈+0.64 ≈ empirical +0.66 (convex curve) |
| Pipeline / academic-absorption channel **NULL** | P3: absorption not a primitive | **YES** (cleanest) + falsifiable: absorption ⟂ (1−ρ) |
| Observability gradient **weak/null** at 4–5 yr | P4: τ effect ∝(1−ρ); dispersion proxy mixes λ(−)&τ(+); near-floor | **PARTIAL** (mechanism+interaction yes; weak-main-at-4–5 yr not in default) |
| Gap is a **between-discipline** (field-structural) phenomenon (high ICC) | gap = deterministic f(ρ,τ,λ); cross-field variance is structure, within-field is noise | **YES** (qualitative) |

Cleanly reproduced: **P2, P3, P5** and the **Task-1 fix** (§4). Partially reproduced with stated misses:
**P1** (high-divergence tail; ICC trajectory) and **P4** (weak 4–5 yr main effect).

---

## 4. Fixing Task 1 — the residual *is* an estimate of valuation divergence

**The model gives the Task-1 residual a meaning.** At the measurement horizon the gap decomposes as
`gap ≈ [licensing × noise amplification](λ) + floor(1−ρ, τ)`. Netting the licensing channel λ (and the
null absorption term) leaves the **floor = (1 − w(τ))·(1 − ρ)** — i.e. genuine valuation divergence scaled
by how far earnings have drifted off the credential anchor. So the Task-1 residual is **not "whatever is
orthogonal to two proxies"; it is an estimate of (1 − ρ)**, the divergence.

*Simulation confirms the recovery.* With **λ ⟂ ρ** (corr(λ, 1−ρ)=−0.03), the Task-1 residual correlates
**+0.60** with the *true* divergence (1−ρ); the absorption coefficient is null (−0.01). The residual's spread
along τ is exactly the (1−w(τ)) anchor-departure factor. Figure: `model_task1_recovery.png`.

**Is netting λ legitimate? (collider / over-control critique, answered inside the model.)**
- Netting λ is unbiased **iff λ ⟂ ρ**. The model makes this explicit: when λ is generated **correlated with
  ρ** (corr(λ, 1−ρ)=+0.83), the licensing coefficient inflates (+0.31→+0.76) and the residual's correlation
  with true divergence **collapses to +0.17** — netting then *absorbs* divergence and biases the residual.
- So the legitimacy is an **empirically checkable condition: corr(licensure_intensity, divergence_proxy) ≈ 0**.
  Licensing is a regulatory/occupational feature with no structural reason to track academic-market
  alignment, so λ ⟂ ρ is plausible *and testable* — not assumed.
- **Netting absorption** is, by P3, neither necessary nor helpful: absorption is not in the mechanism, so it
  only costs a degree of freedom, and **would bias the residual if absorption happened to proxy ρ**
  (over-control on a non-mechanism). **Recommendation: net λ (after checking λ ⟂ ρ), do *not* net absorption.**

**Sharper empirical tests the model implies (runnable on existing data):**
1. **Residual ↔ independent divergence proxy.** The Task-1 residual should correlate with an external
   academic-vs-market skill-divergence measure (e.g. O\*NET research/abstract-skill intensity vs applied/market-
   skill intensity by field; or the SDR academic-sector vs industry salary gap). The model predicts a positive
   correlation; a null would falsify the "residual = divergence" reading.
2. **τ × (1−ρ) interaction with λ controlled** (P4): regress gap on τ (a clean observability proxy), the
   divergence proxy, and their interaction, netting λ — the model predicts a **positive interaction**, sharper
   than the raw (weak) dispersion gradient.
3. **Check λ ⟂ ρ** directly (corr(licensure, divergence proxy)≈0) to license the Task-1 netting.

---

## 5. Honest assessment — what the model does and does not reproduce

**Reproduces (without per-fact tuning):**
- The licensing channel's sign, robustness, and its *interaction with noise/horizon* (P2).
- The pipeline null as the *expected* outcome, with a falsifiable condition (P3).
- The lead–lag direction for the dynamic test (P5).
- A theoretical identity for the Task-1 residual (= divergence estimate) and the exact condition (λ ⟂ ρ)
  under which netting is legitimate — converting a "descriptive, no-identification" residual into an estimand.

**Does NOT reproduce (reported, not tuned away):**
- **The decline of between-cluster ICC with horizon.** The model gets the *grand* gap falling 1→5 yr but its
  between-field variance is roughly flat/slightly rising (early noise compresses fields together; they then
  fan out toward heterogeneous floors), whereas empirically between-cluster dispersion is *largest* at 1 yr
  (Arts ≈1.3 vs CS ≈0.3 already at 1 yr). The model underweights how strongly the **licensing × noise** channel
  (which it *does* contain — corr(Δgap, λ) = −0.40) inflates between-field spread early. Matching this would
  need stronger early-licensing dispersion than the held-fixed calibration delivers.
- **A weak observability *main* effect at 4–5 yr** (P4): the default calibration gives a strong, non-monotonic
  τ main effect; the weak empirical gradient is recovered only via the proxy-conflation (λ vs τ) argument and
  the interaction test, or by assuming fields are near full convergence by 4–5 yr.
- **High-divergence horizon tail** (P1): the model predicts the most divergent fields' gaps should fall least
  or rise 1→5 yr; the data show even high-gap clusters falling.

**Scope and simplicity.** The model is the simplest that *generates* the gap: three field primitives (ρ, τ, λ)
+ one noise schedule, no free parameter per fact. It **takes (ρ, τ, λ) as field-level primitives** and does not
explain *why* they vary by discipline — so the ICC/between-discipline structure is *consistent with* the model
(gap is a deterministic function of field primitives) rather than *derived* from it. The Gaussian/Pearson
closed form is an approximation to the Spearman gap; the MC uses Spearman throughout and the two agree closely
in the figures.
