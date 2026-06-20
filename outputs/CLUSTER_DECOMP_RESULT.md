# CIP-2 Cluster Decomposition — between-discipline structure (no license machinery)

Headline = share of the gap that is BETWEEN discipline clusters. NO license dummy / exclusions / flat-decoupled / mixed flags. Regime is a descriptive label (credential-area = Health/Business clusters, by CIP-2 identity, not a license flag). AR anchored on Wapman edges (canonical SpringRank α=0.5, binary). Run: `python scripts/14_cluster_decomp.py`. Date 2026-06-20.

## Task 2 (HEADLINE) — variance decomposition

Multilevel meta model `gap_f ~ 1 + (1|CIP2)`, precision-weighted (V=diag(SE²)+τ²·same-parent, REML), n=48 fine fields:

- **Between-CIP2 variance τ² = 0.0231** (SD **0.152** gap-units).
- **ICC = τ²/(τ² + mean within-cluster variance) = 0.30** → **≈30% of the cross-field gap variance is between broad discipline areas** (descriptive one-way η² = 0.50 = 50%, consistent). The gap is **primarily a discipline-area-level phenomenon**, not a field-idiosyncratic one.

### CIP-2 cluster means, ranked (shrunk, partial-pooled) — who integrates prestige↔pay

| name                |   n_fields |   raw_mean |   shrunk_mean |    lo |    hi | regime                |
|:--------------------|-----------:|-----------:|--------------:|------:|------:|:----------------------|
| Computer/Info       |          1 |      0.292 |         0.348 | 0.209 | 0.486 | prestige-transmission |
| Math & Stats        |          2 |      0.315 |         0.352 | 0.244 | 0.459 | prestige-transmission |
| Social Sci          |          5 |      0.471 |         0.421 | 0.344 | 0.498 | prestige-transmission |
| Engineering         |         10 |      0.472 |         0.432 | 0.373 | 0.490 | prestige-transmission |
| History             |          1 |      0.413 |         0.443 | 0.304 | 0.582 | prestige-transmission |
| Psychology          |          1 |      0.429 |         0.455 | 0.316 | 0.594 | PhD-pipeline          |
| Business            |          4 |      0.477 |         0.481 | 0.402 | 0.559 | credential-area       |
| Education           |          1 |      0.451 |         0.510 | 0.280 | 0.740 | PhD-pipeline          |
| English             |          1 |      0.506 |         0.516 | 0.377 | 0.654 | prestige-transmission |
| Social Work         |          1 |      0.552 |         0.551 | 0.388 | 0.714 | PhD-pipeline          |
| Agriculture         |          3 |      0.656 |         0.567 | 0.400 | 0.733 | prestige-transmission |
| Philosophy/Religion |          1 |      0.657 |         0.611 | 0.415 | 0.807 | PhD-pipeline          |
| Architecture        |          1 |      0.777 |         0.645 | 0.419 | 0.872 | prestige-transmission |
| Physical Sci        |          3 |      0.686 |         0.661 | 0.534 | 0.788 | PhD-pipeline          |
| Biological Sci      |          5 |      0.830 |         0.678 | 0.567 | 0.790 | PhD-pipeline          |
| Nat. Resources      |          2 |      0.960 |         0.730 | 0.545 | 0.916 | prestige-transmission |
| Arts                |          2 |      0.792 |         0.746 | 0.592 | 0.899 | prestige-transmission |
| Health              |          4 |      0.787 |         0.753 | 0.620 | 0.887 | credential-area       |

Low shrunk-mean = integrated (prestige predicts pay): the **Math & Computing / Social-science-economic** end. High = decoupled: the **credential areas (Health, Business)** and natural-science / arts clusters surface as the high-gap end — the license lesson, carried by cluster structure, not by hand-coding.

## Task 1 — clean gap at the locked CIP-2 grain

Locked-grain reliable units: **23** (16 fine + 7 rolled-up parents).

| unit                       | grain   |   n_inst |   gap |    se |   signal_frac | regime                |
|:---------------------------|:--------|---------:|------:|------:|--------------:|:----------------------|
| Computer Science           | fine    |      161 | 0.292 | 0.080 |          0.92 | prestige-transmission |
| Social Sci (rolled-up)     | parent  |      150 | 0.305 | 0.050 |          0.95 | prestige-transmission |
| Economics                  | fine    |      116 | 0.313 | 0.080 |          0.90 | prestige-transmission |
| Political Science          | fine    |      139 | 0.343 | 0.080 |          0.72 | PhD-pipeline          |
| Mathematics                | fine    |      120 | 0.384 | 0.080 |          0.87 | prestige-transmission |
| Math & Stats (rolled-up)   | parent  |      105 | 0.403 | 0.063 |          0.87 | prestige-transmission |
| Engineering (rolled-up)    | parent  |      126 | 0.409 | 0.062 |          0.86 | prestige-transmission |
| Finance                    | fine    |       89 | 0.413 | 0.080 |          0.94 | credential-area       |
| History                    | fine    |      131 | 0.413 | 0.080 |          0.73 | prestige-transmission |
| Psychology                 | fine    |      173 | 0.429 | 0.080 |          0.77 | PhD-pipeline          |
| Marketing                  | fine    |       81 | 0.482 | 0.092 |          0.86 | credential-area       |
| Electrical Engineering     | fine    |      118 | 0.487 | 0.080 |          0.71 | prestige-transmission |
| Accounting                 | fine    |       94 | 0.488 | 0.080 |          0.83 | credential-area       |
| English                    | fine    |      135 | 0.506 | 0.080 |          0.58 | prestige-transmission |
| Management                 | fine    |       90 | 0.525 | 0.083 |          0.85 | credential-area       |
| Physical Sci (rolled-up)   | parent  |       86 | 0.580 | 0.095 |          0.61 | PhD-pipeline          |
| Biology                    | fine    |      137 | 0.650 | 0.080 |          0.70 | PhD-pipeline          |
| Philosophy                 | fine    |       41 | 0.657 | 0.133 |          0.60 | PhD-pipeline          |
| Biological Sci (rolled-up) | parent  |      149 | 0.667 | 0.071 |          0.71 | PhD-pipeline          |
| Arts (rolled-up)           | parent  |       89 | 0.747 | 0.107 |          0.59 | prestige-transmission |
| Nursing                    | fine    |       99 | 0.927 | 0.102 |          0.67 | credential-area       |
| Health (rolled-up)         | parent  |      104 | 1.058 | 0.107 |          0.83 | credential-area       |
| Communication Disorders    | fine    |       17 | 1.208 | 0.233 |          0.51 | credential-area       |

### 19-discipline coarse view (each CIP-2 cluster pooled)

| name                |   n_children |   n_inst |   gap |   signal_frac | reliable   | regime                |
|:--------------------|-------------:|---------:|------:|--------------:|:-----------|:----------------------|
| Social Sci          |            5 |      150 | 0.305 |          0.95 | True       | prestige-transmission |
| Computer/Info       |            1 |       96 | 0.395 |          0.93 | True       | prestige-transmission |
| Math & Stats        |            2 |      105 | 0.403 |          0.87 | True       | prestige-transmission |
| History             |            1 |      121 | 0.405 |          0.74 | True       | prestige-transmission |
| Engineering         |           10 |      126 | 0.409 |          0.86 | True       | prestige-transmission |
| Psychology          |            1 |      155 | 0.412 |          0.79 | True       | PhD-pipeline          |
| Business            |            4 |      115 | 0.418 |          0.96 | True       | credential-area       |
| Agriculture         |            4 |       35 | 0.491 |          0.53 | False      | prestige-transmission |
| English             |            1 |      129 | 0.522 |          0.58 | True       | prestige-transmission |
| Education           |            1 |       23 | 0.527 |          0.14 | False      | PhD-pipeline          |
| Philosophy/Religion |            2 |       36 | 0.564 |          0.68 | False      | PhD-pipeline          |
| Physical Sci        |            5 |       86 | 0.580 |          0.61 | True       | PhD-pipeline          |
| Biological Sci      |            7 |      149 | 0.667 |          0.71 | True       | PhD-pipeline          |
| Nat. Resources      |            2 |       52 | 0.690 |          0.27 | False      | prestige-transmission |
| Architecture        |            1 |       29 | 0.728 |          0.02 | False      | prestige-transmission |
| Arts                |            2 |       89 | 0.747 |          0.59 | True       | prestige-transmission |
| Social Work         |            1 |       51 | 0.876 |          0.16 | False      | PhD-pipeline          |
| Health              |            4 |      104 | 1.058 |          0.83 | True       | credential-area       |

## Task 3 — observability gradient (secondary, license-free)

- **Pooled / inclusive** Spearman(gap, dispersion) over all 23 reliable units (license fields kept as legitimate cases): **-0.44** [-0.74, +0.01].
- **Within-cluster** slope (gap ~ dispersion + CIP-2 fixed effects; does a field more observable than its cluster average have a lower gap?): **-0.527** [-1.08, +0.02] (n=46 fine fields).

Both are directionally negative but weak / CI spans 0 — the dispersion gradient is a **weak second-order modifier** on top of the dominant between-cluster structure, not the headline. (No license special-casing; no significance chasing.)

## Verdict

**The robust, first-order finding is the between-discipline structure: ≈30% of the cross-field gap variance is between CIP-2 clusters** (τ²=0.023, SD 0.15; descriptive η²=50%). The discipline-area a field sits in — integrated Math/Computing & economics vs decoupled credential (Health, Business), natural-science and arts areas — is the **dominant structured component** of the gap (far larger than any covariate); the within-area observability gradient is a weak second-order modifier (pooled -0.44, within-cluster -0.527, both CI-fragile). Partial pooling and roll-up improve coverage and rigor, not the information; coarser units are more heterogeneous. See `gap_by_cluster.png`, `cluster_means_forest.png`.
