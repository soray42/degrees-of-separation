# TASK 2 — Multi-Horizon Earnings Robustness

Gap and CIP-2 between-discipline variance decomposition re-estimated at every consistent Scorecard FoS earnings horizon on hand: pooled completer median earnings at **1, 4, 5 years** post-completion (EARN_MDN_{1,4,5}YR). Two further columns are reported as caveated supplements (different populations): 2YR highest-credential, 3YR not-enrolled. AR (Wapman SpringRank) is fixed; only ER changes. Scope: early career (1-5 yr); no lifetime extrapolation. Run: `python scripts/21_multihorizon.py`.

## Decomposition by horizon

| horizon   | kind                                 |   n_fields |   n_reliable |   grand_gap |   tau2 |   icc |   eta2 |
|:----------|:-------------------------------------|-----------:|-------------:|------------:|-------:|------:|-------:|
| 1YR       | primary                              |         47 |           23 |       0.772 | 0.0631 | 0.496 |  0.583 |
| 4YR       | primary                              |         48 |           16 |       0.550 | 0.0231 | 0.296 |  0.497 |
| 5YR       | primary                              |         47 |           16 |       0.576 | 0.0300 | 0.328 |  0.520 |
| 2YR-HI    | supplement (highest-credential pop.) |         48 |           19 |       0.734 | 0.0335 | 0.328 |  0.390 |
| 3YR-NE    | supplement (not-enrolled pop.)       |         45 |           16 |       0.636 | 0.0346 | 0.404 |  0.592 |

**The ICC level drifts with horizon — honestly, not flat.** It is **0.50 at 1yr** and settles to **0.30-0.33 at 4-5yr**; the descriptive η² is steadier at [0.50, 0.58]. The overall (grand) gap is also higher at 1yr (0.77) than at 4-5yr (~0.56): one year out, within-field earnings are barely differentiated, so gaps are large and *more* of their variance sits between disciplines. The previously-reported headline **ICC≈0.30 is the 4-5yr figure**; at 1yr the between-discipline share is even larger, not smaller. So the direction of the headline (gap is primarily a discipline-area phenomenon) holds at every horizon — what moves is its *magnitude*, and it moves toward *more* between-discipline structure earlier.

## Cluster gap by horizon (precision-weighted), ranked by 3-horizon mean

| name                |   1YR |   4YR |   5YR |
|:--------------------|------:|------:|------:|
| Arts                | 1.316 | 0.816 | 0.991 |
| Nat. Resources      | 1.115 | 0.844 | 0.747 |
| Health              | 0.965 | 0.805 | 0.783 |
| Architecture        | 0.974 | 0.777 | 0.800 |
| Agriculture         | 0.848 | 0.574 | 0.800 |
| Physical Sci        | 0.888 | 0.685 | 0.623 |
| Social Work         | 0.919 | 0.552 | 0.716 |
| Philosophy/Religion | 0.769 | 0.657 | 0.758 |
| Education           | 0.915 | 0.451 | 0.752 |
| Biological Sci      | 0.801 | 0.699 | 0.574 |
| English             | 0.913 | 0.506 | 0.472 |
| History             | 0.762 | 0.413 | 0.430 |
| Psychology          | 0.713 | 0.429 | 0.381 |
| Business            | 0.499 | 0.476 | 0.492 |
| Social Sci          | 0.593 | 0.412 | 0.406 |
| Engineering         | 0.515 | 0.427 | 0.420 |
| Math & Stats        | 0.349 | 0.322 | 0.326 |
| Computer/Info       | 0.312 | 0.292 | 0.342 |

## Ranking & level stability across primary horizons

Cluster-mean rank correlation (Spearman):
- 1YR vs 4YR: **+0.85**
- 1YR vs 5YR: **+0.81**
- 4YR vs 5YR: **+0.82**

Field-gap rank correlation (Spearman):
- 1YR vs 4YR: **+0.74**
- 1YR vs 5YR: **+0.71**
- 4YR vs 5YR: **+0.68**

The integrated (low-gap) end — Computer/Info, Math & Stats, Social Sci, Engineering — and the decoupled (high-gap) end — Health, natural-science and arts clusters — keep their places across horizons.

## Adversarial self-check

**Strongest referee objection.** (1) The horizons are not independent — 1/4/5-yr Scorecard earnings are the same IRS-linked completers re-measured, so cross-horizon stability is partly mechanical autocorrelation, not evidence of structural robustness. (2) The cohorts also differ slightly by horizon (a 5-yr horizon reflects an earlier graduating cohort than the 1-yr), so this conflates horizon with cohort. (3) All three are early career (<=5 yr); fields with steep but late earnings growth (e.g. medicine-adjacent, law-adjacent) could re-rank at 10-15 yr, which the data cannot see.
**Does it survive?** As an *early-career ordering* claim, yes: the cluster ranking is stable (rank Spearman +0.81-+0.85) and field-gap rank correlations are high (+0.68-+0.74), so the integrated-vs-decoupled structure is not an artifact of the specific 4-yr horizon. The ICC *level* is not flat (it falls from 0.50 at 1yr to ~0.31 at 4-5yr) — but it moves toward MORE between-discipline structure earlier, so the headline direction strengthens, not weakens, at shorter horizons; we report the drift rather than averaging it away. As a *lifetime* claim, no — we scope explicitly to 1-5 yr and do not extrapolate. Autocorrelation across horizons makes this a necessary-but-weak test: instability would have falsified the headline; observed stability of ordering is consistent with it.
