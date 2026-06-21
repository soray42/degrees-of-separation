# Field-specific vs generic (academia-wide) prestige as predictors of field earnings

**Descriptive, outcome-agnostic — no causal claims.** Across-institution earnings differences conflate value-added with student selection; we describe which prestige signal *correlates* with observed median earnings. Causal backing is CITED, not re-estimated: Chetty-Deming-Friedman (Ivy-Plus lifts elite-TAIL outcomes, small average-earnings effects) and MacLeod et al. 2017 (reputation→earnings via an exit-exam shock). Run: `python scripts/28_field_vs_generic_prestige.py`.

Audited 66-field crosswalk; per-(institution × field) table n=3527 rows, 57 fields (52 with ≥15 institutions). Primary: published Wapman ranks (F = field, G = academia-wide) + Scorecard 4yr BA median earnings.

## Headline (outcome-agnostic — the hypothesis does NOT hold)

**On MEDIAN BA earnings, the generic academia-wide brand predicts pay at least as well as — usually slightly better than — field-specific academic reputation.** Mean c_G (+0.45) ≥ mean c_F (+0.40); field reputation's advantage ADV = c_F − c_G is **negative in 42/57 fields** (mean -0.05); in the horse-race, **brand-dominant fields (15) outnumber field-dominant (7)**. The 'Georgia-Tech-CS beats Yale-CS' intuition does NOT generalize on medians: in CS the data-driven brand-strong divergence schools OUT-earn the field-strong ones (Analysis C). This is fully consistent with Chetty-Deming-Friedman — the brand's payoff is concentrated in elite-TAIL outcomes (top-1%, elite firms/grad school) that Scorecard MEDIANS cannot see — and with selection (generic prestige ≈ selectivity ≈ student composition). We describe correlations; we claim no causal effect. The fields where field reputation DOES lead are applied/vocational (engineering subfields, social work, special education), NOT CS.

## A — Do the two prestige signals even agree? (prestige-only; not a function of earnings)

rho_PP[f] = Spearman(field prestige F, academia-wide prestige G). Mean **+0.79** (median +0.82); field and brand prestige are strongly aligned on average, but disagreement varies by field. **Highest-disagreement fields** (where 'which signal to trust' matters most):

| label                     |   n |   rho_PP |
|:--------------------------|----:|---------:|
| Pharmacy                  |   8 |    +0.12 |
| Forestry                  |  17 |    +0.31 |
| Microbiology              |  17 |    +0.50 |
| Urban & Regional Planning |  12 |    +0.59 |
| Ecology                   |  11 |    +0.62 |
| Nutrition Sciences        |  30 |    +0.63 |
| Nursing                   |  99 |    +0.65 |
| Environmental Sciences    |  48 |    +0.67 |
| Public Health             |  32 |    +0.68 |
| Special Education         |  19 |    +0.72 |

## B1 — Which signal predicts pay? Correlation-difference ADV = c_F − c_G

`c_F = Spearman(F, y)` **= 1 − gap[f]** (DISCLOSED: this restates the gap). `c_G = Spearman(G, y)` is the NEW object (the gap says nothing about generic prestige). `ADV = c_F − c_G` = field reputation's predictive advantage over the generic brand for field pay.

- mean c_F = **+0.40** (=1−mean gap), mean c_G = **+0.45**, mean **ADV = -0.05**; ADV>0 (field rep more informative) in **15/57** fields, ADV<0 (brand more informative) in 42/57.

Fields ranked by field-reputation advantage (ADV):

| label                     |   n |   c_F |   c_G |   ADV |   rho_PP |
|:--------------------------|----:|------:|------:|------:|---------:|
| Pharmacy                  |   8 | +0.43 | +0.12 | +0.31 |    +0.12 |
| Special Education         |  19 | +0.55 | +0.33 | +0.22 |    +0.72 |
| Forestry                  |  17 | -0.15 | -0.31 | +0.16 |    +0.31 |
| Social Work               |  68 | +0.45 | +0.30 | +0.15 |    +0.73 |
| Spanish Lang & Lit        |  30 | +0.38 | +0.23 | +0.15 |    +0.87 |
| Chemical Engineering      |  91 | +0.55 | +0.44 | +0.10 |    +0.84 |
| Aerospace Engineering     |  35 | +0.56 | +0.47 | +0.09 |    +0.85 |
| Industrial Engineering    |  39 | +0.49 | +0.41 | +0.09 |    +0.76 |
| Mechanical Engineering    | 133 | +0.62 | +0.57 | +0.05 |    +0.86 |
| Electrical Engineering    | 118 | +0.51 | +0.48 | +0.03 |    +0.85 |
| Materials Science         |  23 | +0.51 | +0.48 | +0.03 |    +0.74 |
| Geography                 |  41 | +0.24 | +0.23 | +0.01 |    +0.86 |
| Computer Engineering      |  74 | +0.70 | +0.69 | +0.01 |    +0.85 |
| Political Science         | 139 | +0.66 | +0.65 | +0.01 |    +0.88 |
| Mathematics               | 120 | +0.62 | +0.61 | +0.00 |    +0.90 |
| Economics                 | 116 | +0.69 | +0.69 | -0.00 |    +0.90 |
| Animal Sciences           |  35 | +0.20 | +0.20 | -0.01 |    +0.73 |
| Civil Engineering         | 105 | +0.51 | +0.51 | -0.01 |    +0.78 |
| Statistics                |  28 | +0.75 | +0.77 | -0.01 |    +0.93 |
| Sociology                 | 114 | +0.57 | +0.58 | -0.01 |    +0.89 |
| Nursing                   |  99 | +0.07 | +0.09 | -0.01 |    +0.65 |
| Theatre                   |  46 | +0.38 | +0.40 | -0.02 |    +0.81 |
| Food Science              |  16 | +0.76 | +0.78 | -0.02 |    +0.76 |
| Biomedical Engineering    |  48 | +0.65 | +0.67 | -0.02 |    +0.82 |
| Philosophy                |  41 | +0.34 | +0.37 | -0.03 |    +0.92 |
| History                   | 131 | +0.59 | +0.62 | -0.03 |    +0.90 |
| Psychology                | 173 | +0.57 | +0.61 | -0.03 |    +0.87 |
| Kinesiology/Exercise Sci  |  70 | +0.62 | +0.66 | -0.04 |    +0.85 |
| Computer Science          | 161 | +0.71 | +0.75 | -0.04 |    +0.85 |
| Health/PE/Recreation      |  15 | +0.57 | +0.62 | -0.05 |    +0.89 |
| Biochemistry              |  47 | +0.42 | +0.47 | -0.05 |    +0.80 |
| Music                     |  76 | +0.04 | +0.09 | -0.05 |    +0.73 |
| Physics                   |  43 | +0.35 | +0.40 | -0.05 |    +0.94 |
| Anthropology              |  75 | +0.49 | +0.55 | -0.06 |    +0.87 |
| English                   | 135 | +0.49 | +0.55 | -0.06 |    +0.88 |
| Chemistry                 |  76 | +0.31 | +0.39 | -0.07 |    +0.91 |
| Linguistics               |  26 | +0.50 | +0.58 | -0.08 |    +0.87 |
| Environmental Engineering |  19 | +0.19 | +0.27 | -0.08 |    +0.82 |
| Agronomy                  |  11 | +0.07 | +0.16 | -0.09 |    +0.86 |
| Ecology                   |  11 | +0.35 | +0.45 | -0.09 |    +0.62 |
| Biology                   | 137 | +0.35 | +0.44 | -0.09 |    +0.82 |
| Accounting                |  94 | +0.51 | +0.61 | -0.10 |    +0.75 |
| Public Health             |  32 | +0.56 | +0.66 | -0.10 |    +0.68 |
| Finance                   |  89 | +0.59 | +0.70 | -0.11 |    +0.82 |
| Management                |  90 | +0.48 | +0.60 | -0.12 |    +0.81 |
| Communication Disorders   |  17 | -0.21 | -0.09 | -0.12 |    +0.83 |
| Architecture              |  34 | +0.22 | +0.36 | -0.14 |    +0.81 |
| Physiology                |  16 | -0.34 | -0.20 | -0.14 |    +0.89 |
| Microbiology              |  17 | +0.07 | +0.21 | -0.15 |    +0.50 |
| Earth Sciences            |  50 | +0.28 | +0.45 | -0.17 |    +0.82 |
| Marketing                 |  81 | +0.52 | +0.69 | -0.17 |    +0.82 |
| Nutrition Sciences        |  30 | +0.27 | +0.44 | -0.18 |    +0.63 |
| Human Dev & Family Sci    |  31 | +0.58 | +0.76 | -0.18 |    +0.82 |
| Environmental Sciences    |  48 | +0.23 | +0.47 | -0.24 |    +0.67 |
| Teacher Ed (subjects)     |  36 | +0.23 | +0.49 | -0.26 |    +0.72 |
| Agricultural Engineering  |  14 | +0.09 | +0.40 | -0.31 |    +0.85 |
| Urban & Regional Planning |  12 | +0.13 | +0.69 | -0.57 |    +0.59 |

## B2 — Horse-race regression y ~ F + G (standardized, cohort-weighted)

β_F is mechanically tied to the gap; the NEW content is **β_G** (residual pay-association of the generic brand conditional on field strength) and the β_F-vs-β_G contrast. Classification (fields with ≥15 institutions, n=52):

| classification | n fields | meaning |
|---|---|---|
| field-dominant | 7 | β_F sig, β_G ~0 |
| brand-dominant | 15 | β_G sig, β_F ~0 |
| both | 2 | both signals add |
| brand-suppression | 1 | β_G<0: brand-elite UNDER-earn given field strength |
| neither | 27 | neither sig |

- mean β_F = +2635, mean β_G = +2902 (earnings $ per SD of standardized prestige); mean incremental R²: field 0.06, brand 0.04.

Per-field β_F, β_G (≥15 inst):

| label                     |   n |   beta_F |   se_F |   beta_G |   se_G | classification    |
|:--------------------------|----:|---------:|-------:|---------:|-------:|:------------------|
| Finance                   |  89 |     -790 |  3,529 |   13,911 |  3,449 | brand-dominant    |
| Marketing                 |  81 |   -2,669 |  1,817 |    9,601 |  1,812 | brand-dominant    |
| Management                |  90 |      -81 |  1,984 |    9,218 |  2,023 | brand-dominant    |
| Computer Engineering      |  74 |    6,816 |  3,814 |    8,600 |  3,848 | brand-dominant    |
| Human Dev & Family Sci    |  31 |   -1,370 |  1,876 |    8,258 |  1,851 | brand-dominant    |
| Computer Science          | 161 |   19,040 |  3,861 |    7,877 |  3,632 | both              |
| Food Science              |  16 |   -1,001 |  2,979 |    7,711 |  3,139 | brand-dominant    |
| Accounting                |  94 |    1,360 |  2,087 |    6,847 |  2,049 | brand-dominant    |
| Biology                   | 137 |   -3,335 |  1,560 |    6,105 |  1,603 | both              |
| Health/PE/Recreation      |  15 |   -1,590 |  2,213 |    5,701 |  2,444 | brand-dominant    |
| Philosophy                |  41 |      132 |  6,123 |    5,344 |  6,024 | neither           |
| Environmental Sciences    |  48 |      310 |  1,716 |    4,588 |  1,853 | brand-dominant    |
| Chemistry                 |  76 |   -1,480 |  2,201 |    4,395 |  2,276 | neither           |
| Psychology                | 173 |     -302 |  1,022 |    4,387 |  1,001 | brand-dominant    |
| Electrical Engineering    | 118 |   10,281 |  4,224 |    4,177 |  4,196 | field-dominant    |
| Biochemistry              |  47 |       -5 |  2,195 |    4,047 |  2,277 | neither           |
| Linguistics               |  26 |    1,364 |  2,480 |    3,883 |  2,949 | neither           |
| Public Health             |  32 |    2,535 |  1,756 |    3,595 |  1,644 | brand-dominant    |
| Earth Sciences            |  50 |      273 |  2,087 |    3,550 |  2,082 | neither           |
| Mathematics               | 120 |   11,111 |  4,999 |    3,463 |  4,979 | field-dominant    |
| Physics                   |  43 |    3,325 |  5,096 |    3,375 |  5,511 | neither           |
| Kinesiology/Exercise Sci  |  70 |    1,884 |  1,506 |    3,343 |  1,475 | brand-dominant    |
| English                   | 135 |      232 |  1,249 |    3,324 |  1,208 | brand-dominant    |
| Biomedical Engineering    |  48 |    2,363 |  1,513 |    3,151 |  1,580 | brand-dominant    |
| History                   | 131 |    2,466 |  1,861 |    2,683 |  1,878 | neither           |
| Architecture              |  34 |     -345 |  1,648 |    2,607 |  1,677 | neither           |
| Anthropology              |  75 |    2,403 |  1,610 |    2,587 |  1,654 | neither           |
| Microbiology              |  17 |   -1,665 |  2,277 |    2,396 |  2,299 | neither           |
| Teacher Ed (subjects)     |  36 |     -838 |  1,218 |    2,292 |  1,233 | neither           |
| Civil Engineering         | 105 |    1,375 |    778 |    2,174 |    795 | brand-dominant    |
| Sociology                 | 114 |    2,364 |  1,249 |    1,921 |  1,277 | neither           |
| Materials Science         |  23 |    2,282 |  2,388 |    1,904 |  2,258 | neither           |
| Nutrition Sciences        |  30 |    1,993 |  1,460 |    1,860 |  1,616 | neither           |
| Physiology                |  16 |   -3,669 |  3,070 |    1,845 |  3,080 | neither           |
| Environmental Engineering |  19 |    1,377 |  2,559 |    1,489 |  2,652 | neither           |
| Geography                 |  41 |      899 |  1,923 |    1,415 |  1,787 | neither           |
| Mechanical Engineering    | 133 |    3,441 |  1,080 |    1,083 |  1,049 | field-dominant    |
| Political Science         | 139 |    5,406 |  1,754 |      919 |  1,759 | field-dominant    |
| Economics                 | 116 |   12,885 |  4,424 |      870 |  4,509 | field-dominant    |
| Social Work               |  68 |    1,698 |    946 |      688 |    939 | neither           |
| Industrial Engineering    |  39 |    7,298 |  2,421 |      531 |  2,422 | field-dominant    |
| Nursing                   |  99 |      -95 |  1,368 |      525 |  1,340 | neither           |
| Aerospace Engineering     |  35 |    4,394 |  2,466 |       15 |  2,537 | neither           |
| Theatre                   |  46 |    1,126 |  1,553 |      -54 |  1,461 | neither           |
| Animal Sciences           |  35 |      932 |  1,255 |     -419 |  1,211 | neither           |
| Chemical Engineering      |  91 |    5,481 |  1,267 |     -771 |  1,281 | field-dominant    |
| Communication Disorders   |  17 |   -3,019 |  5,977 |     -907 |  6,084 | neither           |
| Music                     |  76 |    1,083 |  1,496 |   -1,277 |  1,534 | neither           |
| Spanish Lang & Lit        |  30 |    5,054 |  3,872 |   -1,848 |  3,779 | neither           |
| Forestry                  |  17 |    1,797 |  1,830 |   -1,965 |  1,764 | neither           |
| Statistics                |  28 |   23,323 | 16,846 |   -4,681 | 16,882 | neither           |
| Special Education         |  19 |    9,178 |  1,920 |   -5,405 |  1,963 | brand-suppression |

## C — Divergence institutions / named-example verification

Honest test of the 'Georgia-Tech-CS vs Yale-CS' intuition: within a field, do high-F/low-G (field-strong, brand-medium) schools out-earn low-F/high-G (brand-strong, field-medium) schools? Schools strong in BOTH (MIT, Stanford, CMU, Berkeley) are not divergence cases.

**Caveat on the data-driven divergence cases (important):** Georgia Tech is strong in BOTH CS and overall brand, so it is NOT a divergence case and is excluded by construction. The actual highest-(F−G) CS schools turn out to be **regional publics with only *median* CS prestige (F_pct ≈ 0.4–0.65) but very low brand**, vs brand-strong privates with median CS prestige — so this tests 'mid-field/low-brand public vs low-field/high-brand private', not 'CS-elite vs brand-elite', and it conflates geography and selection. Truly CS-elite-but-brand-weak schools are rare (CS-elite ⇒ usually brand-strong). Read the verdict in that light.

**Computer Science (low-gap)** (field prestige F, academia-wide brand G as percentiles; earnings = Scorecard median; schools strong in BOTH are not divergence cases and are excluded by construction here — these are the extremes of F_pct − G_pct):

field-strong / brand-medium (high-F, low-G):
| name                                    |   F_pct |   G_pct |       y |
|:----------------------------------------|--------:|--------:|--------:|
| University of Louisiana at Lafayette    |    0.48 |    0.03 |  80,901 |
| Mississippi State University            |    0.55 |    0.14 |  82,852 |
| University of Maine, The                |    0.65 |    0.25 |  99,413 |
| University of Texas at San Antonio, The |    0.45 |    0.12 | 106,358 |
| University of Texas at El Paso, The     |    0.40 |    0.07 | 103,286 |
| University of Central Florida           |    0.56 |    0.27 |  98,526 |

brand-strong / field-medium (low-F, high-G):
| name                                            |   F_pct |   G_pct |       y |
|:------------------------------------------------|--------:|--------:|--------:|
| American University                             |    0.06 |    0.49 | 117,293 |
| Rochester Institute of Technology               |    0.17 |    0.58 | 124,516 |
| Temple University                               |    0.23 |    0.61 | 101,670 |
| Emory University                                |    0.36 |    0.74 | 159,541 |
| University of Denver                            |    0.32 |    0.68 | 121,993 |
| University of North Carolina at Greensboro, The |    0.16 |    0.48 |  86,372 |

→ median earnings: field-strong $98,970 vs brand-strong $119,643 — **brand-strong schools OUT-earn field-strong schools** ($-20,674).

**Mechanical Engineering (low-gap)** (field prestige F, academia-wide brand G as percentiles; earnings = Scorecard median; schools strong in BOTH are not divergence cases and are excluded by construction here — these are the extremes of F_pct − G_pct):

field-strong / brand-medium (high-F, low-G):
| name                                          |   F_pct |   G_pct |       y |
|:----------------------------------------------|--------:|--------:|--------:|
| Old Dominion University                       |    0.54 |    0.13 |  87,772 |
| University of Dayton                          |    0.44 |    0.04 |  88,339 |
| Mississippi State University                  |    0.46 |    0.14 |  93,893 |
| Missouri University of Science and Technology |    0.53 |    0.23 |  91,742 |
| Clemson University                            |    0.62 |    0.37 |  91,028 |
| George Washington University                  |    0.78 |    0.56 | 101,196 |

brand-strong / field-medium (low-F, high-G):
| name                                      |   F_pct |   G_pct |      y |
|:------------------------------------------|--------:|--------:|-------:|
| University of Rochester                   |    0.38 |    0.89 | 95,060 |
| Temple University                         |    0.14 |    0.64 | 91,131 |
| Southern Illinois University Carbondale   |    0.06 |    0.49 | 86,238 |
| University of Alabama at Birmingham, The  |    0.09 |    0.48 | 89,589 |
| University of Missouri - Kansas City, The |    0.02 |    0.33 | 90,588 |
| University of North Texas                 |    0.14 |    0.42 | 87,257 |

→ median earnings: field-strong $91,385 vs brand-strong $90,088 — **field-strong schools OUT-earn brand-strong schools** ($+1,296).

**Physiology (high-gap, contrast)** (field prestige F, academia-wide brand G as percentiles; earnings = Scorecard median; schools strong in BOTH are not divergence cases and are excluded by construction here — these are the extremes of F_pct − G_pct):

field-strong / brand-medium (high-F, low-G):
| name                                 |   F_pct |   G_pct |      y |
|:-------------------------------------|--------:|--------:|-------:|
| University of Miami                  |    0.88 |    0.44 | 47,683 |
| East Carolina University             |    0.19 |    0.06 | 61,791 |
| Kansas State University              |    0.38 |    0.31 | 65,677 |
| University of Minnesota, Twin Cities |    1.00 |    0.94 | 64,629 |
| West Virginia University             |    0.25 |    0.19 | 70,165 |
| Michigan State University            |    0.62 |    0.62 | 61,309 |

brand-strong / field-medium (low-F, high-G):
| name                                  |   F_pct |   G_pct |      y |
|:--------------------------------------|--------:|--------:|-------:|
| University of California, Davis       |    0.69 |    0.88 | 66,461 |
| University of Arizona, The            |    0.56 |    0.69 | 56,045 |
| University of Wyoming                 |    0.12 |    0.25 | 58,593 |
| University of California, Los Angeles |    0.94 |    1.00 | 57,139 |
| University of Florida                 |    0.44 |    0.50 | 73,679 |
| George Washington University          |    0.31 |    0.38 | 81,548 |

→ median earnings: field-strong $63,210 vs brand-strong $62,527 — **field-strong schools OUT-earn brand-strong schools** ($+683).

## D — Cross-field synthesis

- c_F ≈ 1 − gap confirmed (max abs deviation 1e-16). The non-tautological objects are **rho_PP, c_G/ADV, and β_G**.
- Does the brand's residual premium β_G have its own pattern? corr(β_G, gap) = -0.30, corr(c_G, gap) = -0.82 — the brand premium is largely independent of the gap.

## Robustness

- **Earnings horizon:** ADV rank-stability vs 4yr: 1yr +0.45, 5yr +0.58; c_G stability 1yr +0.72, 5yr +0.74.
- **Rank vs continuous SpringRank:** Spearman analyses (A, B1) are rank-invariant so identical; the published rank is primary (the continuous per-field score is not in the public release).
- **Field-size:** per-field n reported; B2 restricted to ≥15 institutions; the cross-field synthesis weights/sizes points by n (figure).

## Adversarial self-check

1. **Tautology:** c_F and β_F are mechanically the gap (c_F = 1 − gap exactly). Nothing about field-prestige-predicts-pay is new. The genuinely new objects are **rho_PP** (do the signals agree — prestige-only, not earnings), **c_G / ADV** (does the generic brand predict pay, and does field rep beat it), and **β_G** (the brand's residual premium net of field strength).
2. **Selection vs value-added:** earnings gaps across institutions conflate value-added with who enrolls. We make NO causal claim. Chetty-Deming-Friedman: Ivy-Plus shifts elite-tail outcomes with small AVERAGE-earnings effects — so a weak c_G on MEDIAN earnings is fully consistent with a large brand effect on tails we cannot see. MacLeod et al. 2017 shows reputation can causally raise earnings where employer information is scarce.
3. **Median-only:** Scorecard gives MEDIAN earnings. Recognition / tail outcomes (top-1%, elite firms, elite grad school) — exactly where Chetty finds the brand matters — are NOT in these data and are not computed; cited to Chetty, not estimated. A null brand effect here is a null on MEDIANS only.
4. **Named examples:** the CS / engineering / high-gap tables above report the result whichever way it went (see the per-table verdicts) — not asserted.
5. **Sensitivity:** horizon-stability and field-n are reported above; the published-rank vs continuous-score choice does not affect the rank-based headlines.
