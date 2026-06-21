# Mispriced programs — institution × field over-/under-valuation (descriptive)

Two valuation systems price the same human capital (institution × field); per field we surface where they DISAGREE most. **Residual** = earnings rank-residualized on field prestige within field, standardized: **r>0 market-overperforming** (paid above academic standing), **r<0 academically over-valued** (academic standing exceeds market pay).

> **HARD CAVEAT — read first.** The residual conflates institutional value-added with student **SELECTION**. A 'market-overperforming' program may simply enrol abler / richer / better-connected students — this is **where the two valuations DISAGREE**, NOT evidence that a program 'adds value' or is 'underrated'. We make **no causal claim**. Chetty-Deming-Friedman: the brand's payoff is concentrated in the elite TAIL that median earnings cannot see; MacLeod et al. 2017 shows reputation can causally move pay where employer information is scarce. **Named programs below are ILLUSTRATIVE** — institution×field earnings are imprecise; the **DISTRIBUTION / systematic pattern is the robust object**, individual names are not.

Run: `python scripts/30_mispriced_programs.py`. 3444 programs across 52 fields (≥15 inst/field); named programs require Scorecard earnings-cohort ≥ 30.

## Most ACADEMICALLY OVER-VALUED programs (r<0 — academic standing exceeds market pay)

*Illustrative, not a verdict on any program (selection caveat above).*

| institution_name                           | label                  |   F_pct |   G_pct |      y |   resid |   cohort_n |
|:-------------------------------------------|:-----------------------|--------:|--------:|-------:|--------:|-----------:|
| University of Alabama at Birmingham, The   | Public Health          |    0.94 |    0.41 | 44,398 |   -2.96 |         94 |
| Texas Tech University                      | Computer Engineering   |    0.76 |    0.51 | 92,615 |   -2.82 |         34 |
| University of Iowa, The                    | Computer Science       |    0.71 |    0.81 | 88,264 |   -2.58 |        130 |
| University of Vermont, The                 | Computer Science       |    0.61 |    0.41 | 81,160 |   -2.55 |         74 |
| Iowa State University                      | Mechanical Engineering |    0.71 |    0.66 | 85,938 |   -2.50 |        509 |
| University of Vermont, The                 | Mechanical Engineering |    0.61 |    0.44 | 83,438 |   -2.43 |         74 |
| University of Louisiana at Lafayette       | Electrical Engineering |    0.70 |    0.04 | 84,883 |   -2.37 |         37 |
| University of Illinois at Chicago          | Economics              |    0.60 |    0.60 | 63,429 |   -2.35 |        115 |
| University of New Orleans, The             | Finance                |    0.64 |    0.38 | 54,153 |   -2.34 |         39 |
| University of California, Santa Cruz       | Mathematics            |    0.72 |    0.71 | 59,214 |   -2.33 |         70 |
| University of Illinois at Urbana-Champaign | History                |    0.85 |    0.89 | 46,718 |   -2.32 |         48 |
| Illinois Institute of Technology           | Mechanical Engineering |    0.80 |    0.62 | 87,789 |   -2.31 |         76 |

## Most MARKET-OVERPERFORMING programs (r>0 — paid above academic standing)

*Illustrative, not a verdict on any program (selection caveat above).*

| institution_name                                    | label                    |   F_pct |   G_pct |       y |   resid |   cohort_n |
|:----------------------------------------------------|:-------------------------|--------:|--------:|--------:|--------:|-----------:|
| Santa Clara University                              | Computer Engineering     |    0.31 |    0.55 | 163,765 |   +2.96 |         69 |
| Wake Forest University                              | Kinesiology/Exercise Sci |    0.23 |    0.40 |  79,599 |   +2.85 |         37 |
| Lehigh University                                   | Marketing                |    0.23 |    0.68 |  98,320 |   +2.41 |         57 |
| Wake Forest University                              | Political Science        |    0.29 |    0.27 |  85,668 |   +2.39 |         63 |
| Baylor University                                   | Computer Science         |    0.15 |    0.09 | 128,411 |   +2.32 |         42 |
| Colorado School of Mines                            | Computer Science         |    0.14 |    0.46 | 127,217 |   +2.30 |         98 |
| Emory University                                    | Computer Science         |    0.36 |    0.74 | 159,541 |   +2.23 |         34 |
| Rice University                                     | Kinesiology/Exercise Sci |    0.49 |    0.91 |  87,404 |   +2.22 |         38 |
| American University                                 | Finance                  |    0.38 |    0.58 | 134,332 |   +2.17 |         39 |
| Virginia Polytechnic Institute and State University | Food Science             |    0.44 |    0.38 |  80,161 |   +2.15 |         41 |
| Seton Hall University                               | Biology                  |    0.26 |    0.64 |  93,200 |   +2.12 |        175 |
| University of Nevada, Reno                          | Social Work              |    0.12 |    0.29 |  60,320 |   +2.12 |        119 |

## Institution-level systematic valuation (the more robust object)

Mean standardized residual across each institution's fields (≥3 fields; single-field institutions are NOT ranked).

**Most systematically MARKET-OVERPERFORMING institutions** (mean r>0 across their fields):

| institution                     |   n_fields |   mean_resid |   mean_G_pct |
|:--------------------------------|-----------:|-------------:|-------------:|
| Wake Forest University          |          9 |        +1.70 |         0.28 |
| Georgetown University           |         14 |        +1.57 |         0.68 |
| Dartmouth College               |         12 |        +1.56 |         0.61 |
| Stevens Institute of Technology |          9 |        +1.46 |         0.51 |
| Northeastern University         |         20 |        +1.45 |         0.43 |
| Fordham University              |         16 |        +1.40 |         0.53 |
| Vanderbilt University           |         14 |        +1.39 |         0.59 |
| George Washington University    |         20 |        +1.37 |         0.53 |
| Boston College                  |         16 |        +1.35 |         0.71 |
| Lehigh University               |         17 |        +1.28 |         0.58 |

**Most systematically ACADEMICALLY OVER-VALUED institutions** (mean r<0):

| institution                                     |   n_fields |   mean_resid |   mean_G_pct |
|:------------------------------------------------|-----------:|-------------:|-------------:|
| Jackson State University                        |          5 |        -1.18 |         0.03 |
| University of Memphis                           |         22 |        -1.06 |         0.30 |
| University of North Carolina at Greensboro, The |         25 |        -1.06 |         0.51 |
| Southern Illinois University Carbondale         |         25 |        -1.03 |         0.48 |
| University of Illinois at Chicago               |         29 |        -1.03 |         0.71 |
| University of Southern Mississippi, The         |         19 |        -1.02 |         0.13 |
| University of Alabama at Birmingham, The        |         23 |        -0.91 |         0.46 |
| University of New Orleans, The                  |         13 |        -0.86 |         0.35 |
| University of Kentucky                          |         30 |        -0.85 |         0.52 |
| University of Wisconsin - Milwaukee             |         33 |        -0.84 |         0.48 |

### What the residual actually conflates — GEOGRAPHY dominates (adversarial-review addition)

The brand column corr(mean_resid, mean_G_pct) = **+0.31** is vs the academic-HIRING brand, **not** selectivity/wealth/geography — so it is a weak read on the real confounders. A partial **geography** proxy IS derivable in-repo (PSEO institution state; matched 38% of institutions, 44% of programs):

- **R²(institution mean residual ~ state fixed effects) = 0.46** vs R²(~ brand) = 0.00.
- **R²(program residual ~ state FE) = 0.13** vs R²(~ brand) = 0.00.

**Geography/regional wages explain far more of the 'mispricing' than academic brand does** — the named extremes (Santa Clara = Silicon Valley; high-residual VA/MA/CT vs low-residual IA/MN/OH; top over-performers are selective coastal/urban privates — Georgetown, Dartmouth, Vanderbilt, BC) track regional labour markets and selectivity, exactly the selection confound. The disagreement is mostly **where you are and who you enrol**, not academic value-added.

Also note: because field prestige explains little of within-field earnings (mean Spearman(F,y)≈0.42 ⇒ ~18% of rank-variance), the standardized residual is **+0.88-correlated with the raw within-field earnings ranking** — a 'market-overperforming' program is mostly just a **high-paying one for its field**, which is the geography/selection signal again, lightly adjusted for prestige.

## Robustness — does the disagreement survive across earnings horizons?

Program-level residual rank-stability vs 4yr: **1yr +0.56, 5yr +0.64** (only ~31–41% shared rank-variance). **Moderate, not strong:** 29% of program residuals FLIP the sign of their over/under-valuation between 1yr and 4yr. The residual **DISTRIBUTION** is reasonably stable but **individual program labels are horizon-sensitive** — read the named programs as illustrative only, never as a verdict on a program.

## Selectivity netting (optional) — partial geography proxy derivable; admit-rate/SAT absent

Direct selectivity (ADM_RATE, SAT/ACT) is **not in the repo** (Scorecard FoS is field-level; IPEDS here is Completions only), so y ~ F + selectivity is not run. **But the selection confound is NOT un-probeable:** the PSEO-state geography check above already absorbs far more residual variance than brand, so the HARD CAVEAT is empirically load-bearing, not just asserted. To net selectivity fully: pull the Scorecard institution file (ADM_RATE, SAT_AVG) and re-residualize y ~ F + selectivity (still descriptive; selection-on-unobservables remains).

## Adversarial self-check

1. **Selection vs value-added (central, strengthened):** the residual is dominated by who enrols and WHERE — the PSEO-state geography check shows state fixed effects explain far more residual variance (R²≈0.46 institution-level) than academic brand (R²≈0.00), and the residual is +0.88-correlated with the raw within-field pay ranking. We claim DISAGREEMENT between two valuations, never value-added or 'underrated'; the disagreement is mostly geography/selection.
2. **Named = illustrative, distribution = robust:** institution×field Scorecard earnings are noisy (small cohorts, privacy suppression); the named tables are examples, and we require cohort ≥ 30 to surface one. The robust objects are the residual DISTRIBUTION and the institution-level aggregate, not any single program.
3. **Horizon stability** is reported above; unstable program residuals are noise.
4. **Selectivity-netting** would change which programs top the list; it is not feasible here and is flagged, not silently skipped.
5. **No causal / 'underrated' language** is used; 'over-valued' / 'over-performing' name a DESCRIPTIVE gap between two orderings, not a quality judgment.
