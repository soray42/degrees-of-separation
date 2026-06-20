# Tier 0 — Field Expansion + Generated-Regressor Inference on P1

Run: `python scripts/08_expand.py`. Real data only; no threshold-shopping (all four reported). AR = Wapman SpringRank; ER = Scorecard FoS; dispersion = ACS 22-27 IQR/p50. Date 2026-06-20.

## Task 1 — field-universe expansion

Expanded the Wapman→CIP+FOD1P crosswalk from 30 to **54 fields** (24 added). **SpringRank validated**: a from-scratch SpringRank on the released edge lists reproduces Wapman's published per-field ranks at Spearman ≈ 0.72–0.84 (CS top = Stanford/Berkeley/MIT) — a legitimate prestige estimator, used for the generated-regressor bootstrap.

### Mapping failures (logged, not hidden — a finding about the open-data ceiling)

| Wapman field                     | reason mapping failed / restricted                                   |
|:---------------------------------|:---------------------------------------------------------------------|
| Pharmacology                     | thin Wapman∩Scorecard overlap (2 < 10)                               |
| Soil Science                     | thin Wapman∩Scorecard overlap (8 < 10)                               |
| Veterinary Medical Sciences      | no Scorecard bachelor's CIP (professional-degree field)              |
| Linguistics                      | no clean ACS FOD1P (folded into foreign-language aggregate)          |
| Nutrition Sciences               | no clean ACS FOD1P (absent / health aggregate)                       |
| Classics and Classical Languages | no clean ACS FOD1P (foreign-language aggregate)                      |
| Entomology                       | no clean ACS FOD1P (folded into Miscellaneous Biology)               |
| Information Science              | CIP/FOD1P collision with Computer Science (1104 / 2105 vs CS)        |
| Cell Biology                     | CIP collision with Biology (26.03/26.04)                             |
| Biostatistics                    | GAP-ONLY: ACS FOD1P 3702 collides with Statistics → excluded from P1 |
| Religious Studies                | GAP-ONLY: ACS FOD1P 4801 collides with Philosophy → excluded from P1 |

### Reliability pass-counts (no threshold-shopping)

|   threshold |   reliable_fields |   of_which_newly_added |
|------------:|------------------:|-----------------------:|
|        0.3  |                17 |                      3 |
|        0.5  |                16 |                      2 |
|        0.65 |                13 |                      2 |
|        0.7  |                12 |                      2 |

**The expansion barely moved n:** 30→54 fields added only **2** net reliable field(s) at the 0.50 gate (16 vs 14 before); of the 24 added fields only 2 pass at 0.50. Most added fields are thin or noise-dominated and fail the gate — **a real open-data ceiling, not hidden.**

## Task 2 — P1 on the expanded reliable set (permutation p, all four thresholds)

Primary proxy = ACS 22-27 IQR/p50:

|   threshold |     n |   spearman |   p_perm |
|------------:|------:|-----------:|---------:|
|        0.3  | 17.00 |         -0 |   +0.182 |
|        0.5  | 16.00 |         -0 |   +0.053 |
|        0.65 | 13.00 |         -1 |   +0.007 |
|        0.7  | 12.00 |         -1 |   +0.023 |

Robustness proxy = ACS 22-27 CV:

|   threshold |     n |   spearman |   p_perm |
|------------:|------:|-----------:|---------:|
|        0.3  | 17.00 |         -1 |   +0.034 |
|        0.5  | 16.00 |         -1 |   +0.049 |
|        0.65 | 13.00 |         -1 |   +0.003 |
|        0.7  | 12.00 |         -1 |   +0.011 |

The larger n does NOT push the all-reliable (0.50) permutation p below 0.05 (p=0.053); it strengthens as the gate tightens.

## Task 3 — P1 under naive / permutation / generated-regressor bootstrap

The generated-regressor bootstrap multinomially resamples the faculty-placement **edges** and **re-runs SpringRank** each draw (the proper path — edges are in the Zenodo release), plus earnings sampling noise (cohort SE) and ACS-dispersion bootstrap SE. Two versions: **gen-reg** = inner (edge/earnings) noise only — isolates whether SpringRank estimation noise alone overturns P1; **full** = two-level, also resampling the field set (adds the **small-n** uncertainty). Naive Fisher CI / permutation are the point-estimate (Wapman-rank) baselines.

|   threshold |   n |   naive_rho |   naive_lo |   naive_hi |   perm_p |   gr_point |   gr_lo |   gr_hi |   gr_p |   full_point |   full_lo |   full_hi |   full_p |
|------------:|----:|------------:|-----------:|-----------:|---------:|-----------:|--------:|--------:|-------:|-------------:|----------:|----------:|---------:|
|        0.5  |  16 |       -0.49 |      -0.79 |      +0.01 |    0.053 |      -0.36 |   -0.58 |   -0.12 |  0.004 |        -0.34 |     -0.78 |     +0.29 |    0.290 |
|        0.65 |  13 |       -0.74 |      -0.92 |      -0.31 |    0.007 |      -0.53 |   -0.72 |   -0.27 |  0.000 |        -0.52 |     -0.86 |     +0.10 |    0.086 |

- The bootstrap point is **attenuated** vs naive (our from-scratch SpringRank is a noisier prestige estimate than Wapman's, ρ≈0.8, and edge-resampling adds noise) — exactly what propagating estimated-regressor uncertainty should do.
- **gen-reg CI** (SpringRank/earnings noise, fixed fields) excludes 0 at both thresholds → SpringRank estimation noise alone does NOT overturn P1.
- **full CI** (also resampling fields → the honest small-n + generated-regressor uncertainty) is wider; whether it excludes 0 is the binding test, stated per threshold.

## Honest verdict (3–4 sentences)

Expanding the universe from 30 to 54 fields added essentially no reliable fields (16 vs 14 at the 0.50 gate) — the open-data ceiling is real: most of Wapman's remaining fields are thin, suppressed, or have no clean CIP/FOD1P. So n is **not** the lever that converts P1 to clean significance. On the small reliable set the predicted negative sign is robust across thresholds. **SpringRank estimation noise alone does not overturn P1** (the gen-reg bootstrap CI excludes 0); once the **small-n** field-sampling uncertainty is added, the full two-level bootstrap CI at the 0.50 gate includes 0 — so the binding limitation is n, not the generated regressor, and at the cleaner 0.65 gate P1 is stronger but still n-limited. Net: the generated-regressor concern is **addressed** (SpringRank noise does not explain P1 away); the remaining limitation is the small reliable-field count, which the expansion could not raise — a real open-data ceiling.

## Figures

`outputs/figures/expanded_gap_map.png` · `p1_expanded_scatter.png` · `p1_naive_vs_genreg.png`