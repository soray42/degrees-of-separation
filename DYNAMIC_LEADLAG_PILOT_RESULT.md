# Dynamic pilot — PSEO earnings revaluation + coarse prestige lead-lag

Two markets pricing the same human capital, over TIME. Descriptive, outcome-agnostic. PSEO earnings are REAL 2023 dollars (already inflation-adjusted). Run: `python scripts/33_dynamic_leadlag_pilot.py`.

## PART 1 — which fields are being economically REVALUED? (stands alone)

Field-level real BA earnings (5yr-after, cohort-weighted) across graduation cohorts 2001-2016, in constant 2023 dollars. Per-field linear trend (real $/yr) over fields with ≥4 cohort points (60 fields):

**Rising real earnings (revalued UP):**

| label                  |   slope_real_per_yr |   earn_first |   earn_last | span      |
|:-----------------------|--------------------:|-------------:|------------:|:----------|
| Statistics             |              +2,311 |       84,447 |     102,443 | 2007-2016 |
| Computer Science       |              +1,616 |       85,674 |     107,600 | 2001-2016 |
| Computer Engineering   |              +1,421 |      104,037 |     122,368 | 2001-2016 |
| Biomedical Engineering |                +969 |       79,406 |      92,273 | 2001-2016 |
| Finance                |                +868 |       80,486 |      91,189 | 2001-2016 |
| Economics              |                +817 |       76,829 |      86,774 | 2001-2016 |
| Mathematics            |                +790 |       69,896 |      82,065 | 2001-2016 |
| Marketing              |                +631 |       69,206 |      76,248 | 2001-2016 |
| Industrial Engineering |                +631 |       97,653 |     104,791 | 2001-2016 |
| Management             |                +567 |       68,378 |      75,108 | 2001-2016 |

**Falling real earnings (revalued DOWN):**

| label                   |   slope_real_per_yr |   earn_first |   earn_last | span      |
|:------------------------|--------------------:|-------------:|------------:|:----------|
| Teacher Ed (subjects)   |                 -68 |       56,799 |      55,100 | 2001-2016 |
| Chemical Engineering    |                 -81 |      103,985 |     101,947 | 2001-2016 |
| English                 |                 -81 |       55,120 |      52,947 | 2001-2016 |
| Music                   |                 -83 |       53,371 |      51,529 | 2001-2016 |
| Education, General      |                 -84 |       54,628 |      52,392 | 2001-2016 |
| Special Education       |                -109 |       58,456 |      56,592 | 2001-2016 |
| Environmental Sciences  |                -123 |       62,256 |      58,671 | 2001-2016 |
| Mechanical Engineering  |                -136 |       98,210 |      94,646 | 2001-2016 |
| Communication Disorders |              -1,405 |       95,936 |      74,500 | 2001-2016 |
| Pharmacy                |              -1,774 |      150,946 |     119,777 | 2001-2016 |

- 46/60 fields show rising real BA earnings, 14 falling. This is a clean novel descriptive result: the labour market is **actively repricing fields** over two decades, in constant dollars — not a static cross-section.

## PART 2 — coarse prestige lead-lag (SUGGESTIVE pilot, gated)

Two prestige periods from the ORCID-rebuilt hiring network (2011–2015 vs 2016–2020; ORCID-AR validated ρ≈0.74). **The binding constraint is the prestige side** — only 2 periods are realistically buildable and academic prestige is slow-moving — so this SIZES the real dynamic study, it is not a definitive lead-lag estimate.

**Honest pilot finding — the direction is NOT establishable with 2 noisy periods.** Within-field prestige period-to-period rank stability is **median ρ = +0.50** (+0.34–+0.58) over 10 well-covered fields — **moderate, and this is a noise-depressed LOWER bound** (per-field ORCID edges are only 300–1500 per 5-yr window, and even the public Wapman SpringRank reproduces itself only at ρ≈0.77, so true stability is higher than 0.50). Part 1 shows real earnings moving materially over the same window. Put together this is *suggestive* that prestige is the slower-moving anchor and the market is where revaluation happens — **directionally consistent with the structural model's P5** (A slow prior, E drifts) — but with only 2 noisy prestige periods a true cross-lagged lead-lag (does Δprestige FOLLOW Δearnings?) is **not identified**. This is a PILOT that DEMONSTRATES feasibility and SIZES the real study; it does NOT establish that academia lags. See the spec below.

| label                  |   n_common |   prestige_stability_rho |
|:-----------------------|-----------:|-------------------------:|
| Computer Science       |        213 |                    +0.57 |
| Biology                |        325 |                    +0.45 |
| Chemistry              |        235 |                    +0.46 |
| Physics                |        146 |                    +0.34 |
| Mathematics            |        217 |                    +0.53 |
| Psychology             |        314 |                    +0.53 |
| Economics              |        177 |                    +0.58 |
| Mechanical Engineering |        101 |                    +0.55 |
| Electrical Engineering |         44 |                    +0.47 |
| Civil Engineering      |         43 |                    +0.45 |

### Spec for the full dynamic study (what's needed to go from pilot to result)

1. **Finer prestige resolution:** rebuild ORCID SpringRank in ≥4 rolling 3-year windows (2008–10, 2011–13, …, 2020–22) per field — feasible from the cached ORCID edges but each window thins, so pool to CIP-2 clusters or the best-covered fields.
2. **Institution-level earnings panel:** PSEO single-cohort institution×field earnings are mostly disclosure-suppressed, so the institution-level cross-lagged panel is currently data-blocked; use the FIELD-level earnings series (Part 1) against field-level prestige-alignment shifts, or obtain a less-suppressed earnings panel.
3. **Cross-lagged / Granger test:** with ≥4 aligned waves, regress Δprestige_t on Δearnings_{t-1} and Δearnings_t on Δprestige_{t-1} per field (pooled) — the sign/magnitude contrast gives the lead-lag direction. With 2 waves it is not identified; we report the prestige-stability asymmetry instead.

## Adversarial self-check

1. **Part 1 is solid; Part 2 is suggestive.** The earnings revaluation (Part 1) is a clean descriptive panel. The lead-lag (Part 2) is a 2-period pilot — the prestige-stability asymmetry is directional, NOT a cross-lagged estimate; do not read it as 'academia provably lags'.
2. **Prestige periods are noisy:** field-level ORCID edges per 5-year period are 300–1500; SpringRank on them is coarse, and the field_text keyword field assignment is imperfect (scripts/15–19). The stability ρ is a lower bound on true stability (noise depresses it).
3. **Cohort vs calendar time:** PSEO cohorts are 3/5-year pooled windows; the time axis is approximate, and survivorship/coverage of the PSEO coalition changes over time.
4. **Earnings are real 2023$** (confirmed from the LEHD schema) so the trends are real, not inflation; but PSEO coalition composition drift over 2001–2019 could bias a field's series if its covered institutions change — a caveat on the levels, less on the within-field ranking.
5. **Descriptive, no causal claim**; selection unaddressed; cite Chetty/MacLeod.
