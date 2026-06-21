# TASK 2 — Validating the structural model on open data

The model (MODEL.md) predicts the Task-1 residual (gap net of licensing) is an estimate of valuation divergence (1-rho), so it should correlate POSITIVELY with an INDEPENDENT divergence proxy. B1 is make-or-break. Run: `python scripts/25_model_validation.py`. Outcome-agnostic.

Residual = gap - +0.650*licensure_strict (net licensing only, per the model's recommendation to net lambda but not absorption).

## B1 [make-or-break] — residual vs independent divergence proxies

| proxy | construction | Spearman(residual, proxy) [95% CI] | n | (vs raw gap) |
|---|---|---|---|---|
| O*NET | applied − abstract skill intensity of field's occupations | **+0.41** [+0.08, +0.68] | 48 | +0.42 |
| SDR | \|academic − industry salary\|/mean (price wedge) | **-0.61** [-0.88, -0.11] | 17 | -0.42 |

**Model prediction: POSITIVE.** Verdict: **MIXED / CONTESTED — the two independent proxies DISAGREE.** The **O*NET skill-content** proxy SUPPORTS the model (residual↔divergence **+0.41** [+0.08, +0.68], CI>0), and it is the cleaner test by the model's OWN criterion (B3 below: licensure ⟂ this proxy, -0.03). The **SDR salary-wedge** proxy CONTRADICTS it (**-0.61** [-0.88, -0.11], CI<0). The contradiction is **robust, not a contamination artifact**: netting licensing made the SDR correlation *more* negative (-0.42 raw → -0.61 residual), so it cannot be dismissed. So the residual-as-divergence claim is **partially supported on skill-content divergence and contradicted on salary-wedge divergence** — it is NOT cleanly confirmed; which proxy one trusts decides it.

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

At least one independent proxy moves with the residual in the predicted direction, giving partial open-data support for the residual-as-divergence reading; the caveats above bound how strong that support is.
