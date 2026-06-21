# TASK 2 — Validating the structural model on open data

The model (MODEL.md) predicts the Task-1 residual (gap net of licensing) is an estimate of valuation divergence (1-rho), so it should correlate POSITIVELY with an INDEPENDENT divergence proxy. B1 is make-or-break. Run: `python scripts/25_model_validation.py`. Outcome-agnostic.

Residual = gap - +0.650*licensure_strict (net licensing only, per the model's recommendation to net lambda but not absorption).

## B1 [make-or-break] — residual vs independent divergence proxies

| proxy | construction | Spearman(residual, proxy) [95% CI] | n | (vs raw gap) |
|---|---|---|---|---|
| O*NET | applied − abstract skill intensity of field's occupations | **+0.41** [+0.08, +0.68] | 48 | +0.42 |
| SDR | \|academic − industry salary\|/mean (price wedge) | **-0.61** [-0.88, -0.11] | 17 | -0.42 |

**Model prediction: POSITIVE.** Verdict: **MIXED / UNRESOLVED — neither confirmed nor cleanly falsified.** The two independent proxies give OPPOSITE signs and are themselves **negatively correlated (-0.64)** — they are not two noisy reads of one divergence construct, they measure different things and disagree. (i) **O*NET skill-content**: residual↔divergence **+0.41** [+0.08, +0.68] (positive, and robust — see below). (ii) **SDR salary-wedge**: **-0.61** [-0.88, -0.11] (robustly negative). On the SAME 17 fields where both exist, O*NET is +0.75 while SDR is -0.61. There is **no principled, non-circular basis to prefer one**: the B3 licensure-orthogonality tie-breaker does NOT rescue O*NET, because netting licensing makes the SDR correlation *more* negative (-0.42 raw → -0.61; partialling licensure out leaves -0.57) — SDR's contradiction is independent of any licensure contamination. **Verdict: the residual-as-divergence prediction is NOT supported by the open-data evidence — it is contested, not confirmed.**

**Two caveats that further weaken B1 (adversarial review):**
- **Netting licensing is nearly inert here:** the residual ≈ the raw gap (Spearman +0.87 / +0.90), so B1 is essentially testing the *raw gap* against the proxies, not a licensing-purged quantity.
- **Field-size leak in O*NET:** controlling institutions-per-field attenuates the O*NET correlation +0.41 → +0.34 (and +0.33 controlling size + licensure + dispersion) — positive but partly a field-size confound.

## B2 — interaction tau x divergence (licensing controlled)

tau proxy = within-field earnings dispersion. **CIRCULARITY FLAG: tau is earnings-derived, so it shares measurement with the gap; B2 is secondary and not independent of the gap construct.**

| proxy | b(tau x divergence) | p | n |
|---|---|---|---|
| onet | +0.448 | 0.036 | 46 |
| sdr | -1.339 | 0.443 | 17 |

Model predicts a POSITIVE interaction. Reported as-is; given the circularity flag this is corroborative at best.

## B3 — corr(licensure, divergence) (licenses the Task-1 netting)

Netting licensing is unbiased only if licensure ⟂ divergence. 
- corr(licensure, onet) = **-0.03** [-0.32, +0.25]
- corr(licensure, sdr) = **+0.39** [-0.09, +0.78]  — **materially non-zero: residual partly contaminated**

## Adversarial self-check (what each proxy conflates)

- **O*NET (applied − abstract).** Conflates the *level* of a field's average task-abstractness with the model's (1-rho), which is a *cross-institution correlation* of academic vs market value — different objects. It also can't see within-field heterogeneity (all institutions in a field get the same occupation vector). A null here is as consistent with 'the proxy is wrong' as with 'the residual is not divergence' — it cannot cleanly confirm, only fail to.
- **SDR price wedge.** Measured on *doctorate-holders* (academic vs industry salary), while the residual is a *bachelor's*-gap quantity — cross-level. |wedge| also reflects sector composition and pay scales, not only valuation divergence; SEH-only coverage shrinks n.
- **B2 tau.** Earnings-derived → circular with the gap (flagged); not an independent test.

## What a miss implies

The two independent divergence proxies CONTRADICT each other (negatively correlated, opposite signs on the same fields), so the open-data evidence does **not** confirm the residual-as-divergence reading — it is **contested/unresolved**. The model's *internal* logic (residual = floor term) stands, but its key *external* prediction is not borne out: either no available open proxy measures the cross-institution academic-vs-market value correlation the model means, or the residual is not that divergence. Combined with the near-inertness of the licensing-netting (residual ≈ raw gap) and the field-size leak, B1 is **downgraded from 'estimated' to 'unconfirmed'**, and that is the honest headline finding.
