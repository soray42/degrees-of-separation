# Tier 0 — Recalibrated Generated-Regressor Inference on P1

Run: `python scripts/09_recalibrate.py`. Same reliable-field sets and gate as the field-expansion run; point estimates unchanged. Real data only. Date 2026-06-20.

## Task 0 — Wapman deposit inventory

| item                                              | status         | detail                                                                                                      |
|:--------------------------------------------------|:---------------|:------------------------------------------------------------------------------------------------------------|
| (a) per-institution rank SE / CI                  | NOT FOUND      | no SE/CI/std columns in any file                                                                            |
| (a) bootstrap replicates / ensemble / posterior   | NOT FOUND      | no replicate/sample files                                                                                   |
| (a) continuous SpringRank SCORE per inst×field    | FOUND          | institution-stats.csv `PrestigeRank` (0–1 scaled); joinable to names via (field, Rank==OrdinalPrestigeRank) |
| (b) SpringRank analysis code                      | NOT FOUND      | deposit is data-only; no .py/.ipynb                                                                         |
| (b) documented hyperparameters (α, preprocessing) | NOT DOCUMENTED | README defines only the scaled score; no α / self-loop / weighting / thresholding given                     |

→ No reported uncertainty (so literal Option-1-with-reported-SE is unavailable), but the **continuous score is present**. Path is set by Task 1.

## Task 1 — reproduction diagnostic (canonical SpringRank, cdebacco/LarremoreLab pkg)

Canonical SpringRank on the SAME public aggregated edges, α × weighting sweep, Spearman vs Wapman's published ranks (mean over reliable fields):

|   alpha |   binary |   raw |
|--------:|---------:|------:|
|     0   |    0.708 | 0.714 |
|     0.1 |    0.753 | 0.738 |
|     0.5 |    0.772 | 0.761 |
|     1   |    0.766 | 0.768 |
|     2   |    0.746 | 0.768 |

**Best: α=0.5, binary weights → mean ρ = 0.772.** The canonical implementation tops out at ≈ 0.80 — the **same ceiling as our from-scratch SpringRank** (0.72–0.84). α and binarization do not push past it.

**Verdict: DATA CEILING, not a code/hyperparameter issue.** The public aggregated edge lists are lossy relative to the proprietary AARC census Wapman actually ranked, so NO estimator re-run on the public edges can match the published ranks beyond ≈0.80. Re-estimating ranks each draw (old Option 2) therefore injects irreducible implementation/data-mismatch error that attenuates P1 — which is exactly the artifact to remove.

## Task 2 — recalibrated bootstrap (published-rank ANCHORED)

Since the published continuous score exists but no SE is reported, we take the **Option-1 path**: ANCHOR at the published ranks (consistent with the point estimate) and inject only a data-driven sampling spread — each matched institution's **edge-sampling rank SD** (estimated by multinomially resampling edges and re-running the canonical SpringRank), applied to the published percentile rank, then re-ranked. Center = published (no attenuation); spread = generated-regressor uncertainty. Earnings cohort-SE and ACS-dispersion bootstrap-SE added as before. Side by side with naive / permutation / the old (over-attenuated) from-scratch bootstrap:

|   threshold | estimator                                  |   n |   point |       lo |       hi |       p |
|------------:|:-------------------------------------------|----:|--------:|---------:|---------:|--------:|
|        0.5  | naive Fisher (published ranks)             |  16 |  -0.491 |   -0.794 |   +0.006 | nan     |
|        0.5  | permutation (published ranks)              |  16 |  -0.491 | +nan     | +nan     |   0.053 |
|        0.5  | ANCHORED gen-reg (NEW, fixed fields)       |  16 |  -0.447 |   -0.665 |   -0.218 |   0.000 |
|        0.5  | ANCHORED full (NEW, + small-n)             |  16 |  -0.438 |   -0.812 |   +0.144 |   0.140 |
|        0.5  | old from-scratch gen-reg (over-attenuated) |  16 |  -0.356 |   -0.574 |   -0.123 |   0.005 |
|        0.5  | old from-scratch full (over-attenuated)    |  16 |  -0.338 |   -0.771 |   +0.288 |   0.297 |
|        0.65 | naive Fisher (published ranks)             |  13 |  -0.736 |   -0.916 |   -0.312 | nan     |
|        0.65 | permutation (published ranks)              |  13 |  -0.736 | +nan     | +nan     |   0.007 |
|        0.65 | ANCHORED gen-reg (NEW, fixed fields)       |  13 |  -0.632 |   -0.780 |   -0.439 |   0.000 |
|        0.65 | ANCHORED full (NEW, + small-n)             |  13 |  -0.613 |   -0.907 |   -0.049 |   0.042 |
|        0.65 | old from-scratch gen-reg (over-attenuated) |  13 |  -0.538 |   -0.720 |   -0.275 |   0.000 |
|        0.65 | old from-scratch full (over-attenuated)    |  13 |  -0.516 |   -0.863 |   +0.099 |   0.085 |

## Verdict (3–4 sentences)

Removing the implementation-mismatch attenuation moves the generated-regressor point from the over-attenuated -0.36 (old from-scratch) back to **-0.45** (anchored), essentially recovering the naive -0.49 — confirming the old bootstrap was artificially weakening P1. With the attenuation gone, SpringRank estimation noise alone does not overturn P1 (anchored gen-reg CI excludes 0) at the 0.50 gate. But the two-level **full** CI at 0.50 still SPANS 0 — so even un-attenuated, the binding limitation is the small reliable-field count (n), not the generated regressor. Net: the recalibration removes an artificial weakening and confirms the honest status of P1 — directionally robust, surviving the generated-regressor noise, but still n-limited at the 0.50 gate (significant only at the cleaner 0.65 gate).

## Figure
`outputs/figures/p1_recalibrated_forest.png`