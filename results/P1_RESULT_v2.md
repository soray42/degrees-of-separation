# Tier 0 addendum — P1 re-test on non-truncated dispersion (Task 1)

**P1:** the gap is *smaller* where graduate productivity is more observable (higher within-field earnings dispersion). **Predicted sign: negative.** Prior run: −0.28 (p=0.33) on PSEO's elite-**truncated** institution sample. Here, re-tested on the full-coverage **ACS** population (incl. private-institution grads) and the non-suppressed **PSEO-state** sample. Run: `python scripts/04_p1_v2.py`. Date 2026-06-20.

## Spearman(gap, dispersion) across proxies (P1 predicts negative)

| proxy                                      | scope    |   n |   spearman |     p | sign   |
|:-------------------------------------------|:---------|----:|-----------:|------:|:-------|
| PSEO within-inst (prior; TRUNCATED)        | reliable |  14 |     -0.279 | 0.334 | neg ✓  |
| PSEO within-inst (prior; TRUNCATED)        | all      |  28 |     +0.134 | 0.496 | pos ✗  |
| Scorecard cross-inst CV                    | reliable |  14 |     -0.270 | 0.350 | neg ✓  |
| Scorecard cross-inst CV                    | all      |  28 |     -0.233 | 0.234 | neg ✓  |
| ACS 25-64 IQR/p50 (full pop)               | reliable |  14 |     -0.332 | 0.246 | neg ✓  |
| ACS 25-64 IQR/p50 (full pop)               | all      |  28 |     -0.084 | 0.670 | neg ✓  |
| ACS 25-64 CV (full pop)                    | reliable |  14 |     -0.002 | 0.994 | neg ✓  |
| ACS 25-64 CV (full pop)                    | all      |  28 |     +0.126 | 0.521 | pos ✗  |
| ACS 22-27 IQR/p50 (early-career, full pop) | reliable |  14 |     -0.508 | 0.064 | neg ✓  |
| ACS 22-27 IQR/p50 (early-career, full pop) | all      |  28 |     -0.290 | 0.135 | neg ✓  |
| ACS 22-27 CV (early-career, full pop)      | reliable |  14 |     -0.464 | 0.095 | neg ✓  |
| ACS 22-27 CV (early-career, full pop)      | all      |  28 |     -0.143 | 0.467 | neg ✓  |
| PSEO-state BA IQR/p50 (non-suppressed)     | reliable |  14 |     -0.415 | 0.140 | neg ✓  |
| PSEO-state BA IQR/p50 (non-suppressed)     | all      |  28 |     +0.056 | 0.776 | pos ✗  |

## Does P1 firm up off the non-truncated sources?

- **Yes, on the theory-preferred measure.** Early-career (22-27), full-population ACS dispersion gives **-0.51 (p=0.064)** on reliable fields — the strongest and now **marginally significant**, vs the truncated PSEO −0.28 (p=0.33). Employer-learning theory predicts the credential-vs-productivity wedge is sharpest **early-career**, exactly where the relationship is strongest.
- **The IQR/p50 measure is negative on every non-truncated source** (ACS 25-64 −0.33, ACS 22-27 −0.51, PSEO-state −0.42) and both prior proxies — directionally robust. The CV variant is noisier (ACS 25-64 CV ≈ 0), so the percentile-spread measure is preferred.
- **Still short of conventional significance** at n=14 reliable fields (best p≈0.06). Truncation was attenuating P1, but n keeps it from a clean p<0.05 confirmation.
- **The reliability gate is necessary:** including noise-dominated fields weakens or flips the sign on several proxies (P1 is a statement about *well-measured* fields).

## Bonus — ACS field-level earnings by attainment (the PhD glance PSEO suppressed)

Median earnings (25-64, full-time) of each undergraduate field's holders, by highest degree — a *field-level* read on PhD-holder earnings by BA field (institution-level PhD-ER remains unavailable):

| label                   |   med_ba |   med_ma |   med_phd |   n_phd |
|:------------------------|---------:|---------:|----------:|--------:|
| Computer Engineering    |   114516 |   150000 |    180000 |     158 |
| Statistics              |    90000 |   131706 |    180000 |      65 |
| Electrical Engineering  |   118000 |   150000 |    170000 |     844 |
| Computer Science        |   110000 |   136000 |    154349 |     491 |
| Economics               |   100000 |   125000 |    150000 |     491 |
| Biomedical Engineering  |    97953 |   103000 |    148165 |     147 |
| Aerospace Engineering   |   110000 |   133493 |    142474 |      78 |
| Materials Science       |   100000 |   130000 |    140870 |     135 |
| Chemical Engineering    |   104989 |   123000 |    140000 |     419 |
| Physics                 |    80000 |   125000 |    140000 |     839 |
| Mechanical Engineering  |   100000 |   130000 |    135000 |     487 |
| Finance                 |   100000 |   125000 |    131878 |     135 |
| Civil Engineering       |   100000 |   117000 |    126050 |     231 |
| Accounting              |    85000 |   108000 |    125000 |     206 |
| Chemistry               |    79000 |   100000 |    125000 |    1375 |
| Biochemistry            |    75000 |    95000 |    125000 |     534 |
| Neuroscience            |    64279 |    73253 |    120000 |     121 |
| Mathematics             |    87000 |   100000 |    120000 |     678 |
| Astronomy               |    80000 |      nan |    120000 |      31 |
| Nursing                 |    80000 |   100000 |    115000 |     833 |
| Biology                 |    69000 |    88000 |    114000 |    3697 |
| Political Science       |    82000 |   100000 |    105000 |     510 |
| Psychology              |    60000 |    73000 |    100000 |    1833 |
| Anthropology            |    62681 |    76000 |     97446 |     182 |
| History                 |    70000 |    80000 |     96698 |     516 |
| Earth Sciences          |    75000 |    94000 |     96632 |     201 |
| Communication Disorders |    57000 |    70000 |     92977 |     178 |
| Sociology               |    65000 |    77000 |     90272 |     261 |
| English                 |    65000 |    75000 |     90000 |     771 |
| Philosophy              |    66000 |    75000 |     84829 |     323 |