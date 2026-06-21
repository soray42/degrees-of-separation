# TASK 3 — Static Statistical Hardening

Three robustness checks on the CIP-2 cluster-decomposition headline. Run: `python scripts/22_hardening.py`. Outcome-agnostic.

## (a) Cluster ranking vs per-cluster n

Spearman's variance shrinks with the number of institutions, so a cluster's gap could rank high merely because it is measured on few institutions. We pool each CIP-2 cluster (canonical SpringRank on pooled child edges + cohort-weighted earnings), then subsample every eligible cluster to a **common n = 20** institutions and recompute the gap.

- Cluster institution counts range 23-155 (median 100); corr(cluster gap, cluster n) = **-0.54**.
- Full-n vs common-n (20) cluster-gap **rank Spearman = +0.98** over the 18 clusters with n ≥ 20. The integrated-vs-decoupled order is **stable** to equalising n.

| name                |   n |   gap_full |   gap_commonN |
|:--------------------|----:|-----------:|--------------:|
| Health              | 104 |      1.058 |         1.040 |
| Social Work         |  51 |      0.876 |         0.882 |
| Arts                |  89 |      0.747 |         0.754 |
| Architecture        |  29 |      0.728 |         0.737 |
| Philosophy/Religion |  31 |      0.716 |         0.717 |
| Nat. Resources      |  52 |      0.690 |         0.699 |
| Biological Sci      | 149 |      0.680 |         0.689 |
| Physical Sci        |  86 |      0.581 |         0.600 |
| Agriculture         |  33 |      0.527 |         0.537 |
| Education           |  23 |      0.527 |         0.531 |
| English             | 129 |      0.522 |         0.551 |
| Business            | 115 |      0.418 |         0.427 |
| Psychology          | 155 |      0.412 |         0.429 |
| Engineering         | 126 |      0.409 |         0.424 |
| History             | 121 |      0.405 |         0.432 |
| Math & Stats        | 105 |      0.403 |         0.425 |
| Computer/Info       |  96 |      0.395 |         0.409 |
| Social Sci          | 150 |      0.305 |         0.331 |

## (b) Measurement-corrected ICC

The raw within-cluster variance double-counts gap sampling noise. Subtracting the mean field-gap measurement variance (mean SE²) gives the true within-cluster heterogeneity.

- between-cluster τ² (REML, already noise-free) = **0.0231**
- within-cluster variance: observed **0.0549**, measurement (mean SE²) **0.0267**, corrected true **0.0282**
- **ICC raw = 0.30 → measurement-corrected ICC = 0.45**. Correcting for sampling noise *raises* the between-discipline share (the raw ICC was conservative): more of the genuine cross-field variance is between disciplines than the uncorrected headline stated.

## (c) AR-rank uncertainty propagation

The published Wapman prestige ranks are treated as fixed in the gap bootstrap. We inject prestige-rank sampling noise by multinomially resampling each field's Wapman placement edges, re-fitting canonical SpringRank, and using the per-institution percentile-rank SD to perturb ranks inside the gap bootstrap. Baseline = institution resample, AR fixed; full = institution resample + AR-rank perturbation.

- Over **16 reliable fields**: median gap-CI width ratio (full / baseline) = **1.04** (mean 1.05); i.e. propagating AR-rank uncertainty widens the gap CIs by ~4%.
- Reliability flips (a field whose AR-propagated CI now overlaps the noise band): **0/16**. The reliability screen is **robust** to AR-rank uncertainty.

| name                | field                   |   n |   gap |   mean_se_pct |   w_base |   w_full |   width_ratio | flips   |
|:--------------------|:------------------------|----:|------:|--------------:|---------:|---------:|--------------:|:--------|
| Social Sci          | economics               |  97 | 0.313 |         0.076 |    0.191 |    0.225 |          1.18 | False   |
| Business            | finance                 |  57 | 0.413 |         0.103 |    0.268 |    0.313 |          1.17 | False   |
| Social Sci          | political_science       | 111 | 0.343 |         0.073 |    0.176 |    0.198 |          1.12 | False   |
| Computer/Info       | computer_science        |  96 | 0.292 |         0.082 |    0.188 |    0.208 |          1.11 | False   |
| English             | english                 | 127 | 0.506 |         0.065 |    0.236 |    0.261 |          1.10 | False   |
| Psychology          | psychology              | 153 | 0.429 |         0.076 |    0.208 |    0.228 |          1.10 | False   |
| Business            | management              |  73 | 0.525 |         0.102 |    0.380 |    0.409 |          1.07 | False   |
| Business            | marketing               |  52 | 0.482 |         0.108 |    0.382 |    0.404 |          1.06 | False   |
| Philosophy/Religion | philosophy              |  31 | 0.657 |         0.090 |    0.544 |    0.551 |          1.01 | False   |
| Math & Stats        | mathematics             |  99 | 0.384 |         0.071 |    0.224 |    0.226 |          1.01 | False   |
| Health              | nursing                 |  51 | 0.927 |         0.091 |    0.553 |    0.550 |          1.00 | False   |
| Business            | accounting              |  72 | 0.488 |         0.103 |    0.356 |    0.350 |          0.98 | False   |
| History             | history                 | 121 | 0.413 |         0.070 |    0.255 |    0.251 |          0.98 | False   |
| Biological Sci      | biology                 | 119 | 0.650 |         0.089 |    0.323 |    0.314 |          0.97 | False   |
| Engineering         | electrical_engineering  |  70 | 0.487 |         0.087 |    0.414 |    0.393 |          0.95 | False   |
| Health              | communication_disorders |  10 | 1.208 |         0.077 |    1.503 |    1.362 |          0.91 | False   |

## Adversarial self-check

**Strongest referee objection.** (a) Pooling a cluster's child fields into one SpringRank mixes heterogeneous sub-fields, so the 'cluster gap' is itself a construct; subsampling to common n controls Spearman variance but not the deeper differences in WHICH institutions populate each cluster. (b) The measurement correction assumes the bootstrap SE² is the only noise and that τ² is noise-free, but REML τ² can absorb model misspecification, so the corrected ICC could be biased UP. (c) The edge-bootstrap SD understates true AR-rank uncertainty — it captures finite-placement sampling noise on the PUBLIC aggregated edges, not the public-vs-AARC-census data gap (the ~0.8 reproduction ceiling), so the real CI widening is a lower bound.
**Does it survive?** (a) the cluster ordering is robust to equalising n (rank Spearman +0.98; gap-vs-n corr -0.54, so the ranking is partly n-linked); (b) the headline survives correction — the corrected ICC (0.45) is higher than the raw (0.30), so noise-correction strengthens, not weakens, the between-discipline finding; (c) AR-rank uncertainty widens gap CIs by ~4% (median) and flips 0/16 reliability flags — a real but bounded cost; the qualitative structure holds, though individual field CIs should be read as wider than the AR-fixed bootstrap implies.
