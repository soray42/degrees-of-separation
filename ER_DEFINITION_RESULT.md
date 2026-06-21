# ER definition: is earnings a thin proxy for employer reputation? (field-level robustness test)

**This is a FIELD-LEVEL robustness test, NOT a within-field gap.** Public non-wage ER data is one value per field, so we cannot recompute gap_f with a non-wage ER (that needs institution x field occupational mix -- Revelio-gated, specified at the end). We instead test whether the earnings-based gap and the integrated/decoupled typology are robust to, or an artifact of, the earnings-only ER. Seeded; run `python scripts/35_er_definition.py`.

## Non-wage ER measures (sources + crosswalks)

- **Occupational prestige** (status; survey-rated, NON-income, so non-circular): `occ_prestige_f = sum_o P(o|f) * prestige(o)`, P(o|f) from ACS PUMS 2023 recent BA (SCHL>=21, AGEP 22-27, full-time employed) x **Condon/Hughes et al. 2024 OPR** (primary), with **GSS Nakao-Treas 1989** and **GSS 2012** as robustness scales. NOT SEI/ISEI (income+education -> circular). Occupation join: ACS SOCP -> O*NET-SOC-2018 (Condon native key); Census-2010/OCCP fallback.

- **Underemployment** (vertical match): NY Fed *Labor Market for Recent College Graduates* outcomes-by-major (share in jobs not requiring a degree). Crosswalk: repo `NYFED_MAJOR_BY_KEY` + 8 added 1:1 matches.

- **Horizontal match** (optional): built for 30 fields (share of grads in a CIP-matched occupation).


Coverage: occ_prestige on **52/54** fields (weighted occupation-score coverage median 0.91); underemployment on **32** fields. Two analysis universes: **A/C** on all measured fields (N=47, ER-level tests independent of gap reliability); **B/D** on the reliable-gap set (N=20, which need trustworthy within-field gaps).

## A. Is earnings a good field-level ER proxy?

Spearman across all measured fields between the field's **earnings standing** (ACS weighted-median early-career earnings) and each non-wage ER. High -> earnings tracks the broader ER; low/divergent -> earnings is a thin proxy.

| non-wage ER | Spearman(earn_standing, ER) [95% CI] | n |
|---|---|---|
| occupational prestige (OPR, Condon 2024) | +0.67 [+0.44, +0.82] | 47 |
| occupational prestige (GSS Nakao-Treas 1989) | +0.70 [+0.49, +0.83] | 47 |
| occupational prestige (GSS 2012) | +0.67 [+0.46, +0.79] | 47 |
| -underemployment (NY Fed) | +0.70 [+0.46, +0.84] | 31 |
| -unemployment (NY Fed) | +0.09 [-0.30, +0.47] | 31 |
| horizontal match (CIP->SOC) | +0.08 [CI n/a: many ties] | 29 |

*Reading.* Earnings tracks occupational-prestige standing at Spearman +0.67 -- positive (prestigious jobs do pay more) but **well short of 1.0**, the wage-correlated-but-not-identical signature: earnings captures part of ER standing and misses part (the status dimension wages cannot see -- research/arts/clergy/teaching confer status above their pay). Underemployment is a partly-independent vertical-match dimension.

## B. Is the gap (decoupling) an earnings artifact? (H1 vs H2)

Correlate/regress the within-field **gap** on field-level non-wage ER across reliable fields.

- **H1 (earnings captures it):** high-gap *decoupled* fields ALSO have poor non-wage outcomes (high underemployment, low occ-prestige) -> decoupling is real, earnings-only ER defensible.

- **H2 (earnings overstates it):** high-gap fields have GOOD non-wage outcomes (low underemployment, high occ-prestige) -> a broader ER would show them MORE integrated; earnings-only ER overstates.


| gap vs | Spearman [95% CI] | n | H1 sign | reading |
|---|---|---|---|---|
| occ-prestige (OPR) | -0.02 [-0.57, +0.49] | 16 | negative (low status) | neither (flat) |
| underemployment | +0.15 [-0.46, +0.71] | 15 | positive (more underemployed) | neither (flat) |
| status residual (prestige net of wage) | +0.36 [-0.20, +0.74] | 16 | negative | H2 |

**Joint (standardized), overlap fields:** gap ~ 0.49 + (+0.076) z(occ_prestige) + (+0.051) z(underemp); R2=0.06, n=15.


**Verdict (B).** The within-field gap is **largely orthogonal** to field-level non-wage ER: gap vs occ-prestige -0.02 and gap vs underemployment +0.15 are both flat, so **neither H1 nor H2 holds as a general pattern** -- at the field level the decoupling is not explained (H1) by poor non-wage outcomes, nor cleanly shown (H2) to be a status illusion. The one directional signal is a weak **H2 lean on the status residual** (gap vs prestige-net-of-wage +0.36), i.e. higher-gap fields tend to have occupational status slightly above their wage-implied status -- but its CI spans zero on n=16 fields, so it is suggestive only. The earnings-artifact concern is therefore **field-specific (see C/D), not a blanket overstatement.**

## C. Divergent fields -- where 'ER = earnings' is most misleading

`divergence = z(occ-prestige rank) - z(earnings-standing rank)`. **Positive** = status far above pay (earnings UNDER-states ER); **negative** = pay far above status (earnings OVER-states ER). These are the concrete cost of the earnings-only operationalisation.


**Earnings most UNDER-states ER (high status, modest pay):**

| label        |    gap |   earn_standing |   occ_prestige_opr |   underemployment |   divergence |
|:-------------|-------:|----------------:|-------------------:|------------------:|-------------:|
| Physiology   |   1.34 |           40000 |               59.0 |             nan   |        +1.72 |
| Biochemistry |   0.58 |           42321 |               60.6 |              42.0 |        +1.68 |
| Chemistry    |   0.69 |           42000 |               59.9 |              42.8 |        +1.64 |
| Biology      |   0.65 |           42000 |               57.5 |              51.1 |        +1.06 |
| Ecology      |   0.65 |           40000 |               55.5 |             nan   |        +0.99 |
| Neuroscience | nan    |           42483 |               57.9 |             nan   |        +0.95 |


**Earnings most OVER-states ER (high pay, modest status):**

| label      |   gap |   earn_standing |   occ_prestige_opr |   underemployment |   divergence |
|:-----------|------:|----------------:|-------------------:|------------------:|-------------:|
| Marketing  |  0.48 |           55000 |               50.8 |              49.3 |        -1.97 |
| Management |  0.52 |           53000 |               51.7 |              52.6 |        -1.68 |
| Sociology  |  0.43 |           47000 |               51.6 |              52.0 |        -1.31 |
| Economics  |  0.31 |           65000 |               56.9 |              33.1 |        -1.13 |
| Agronomy   |  0.93 |           48282 |               52.4 |             nan   |        -1.09 |
| Finance    |  0.41 |           65000 |               57.3 |              27.8 |        -1.06 |
## D. Typology re-statement -- robust vs earnings-artifact decoupling

Split the **decoupled** end (top-tercile gap, reliable) by whether the non-wage ER is ALSO poor (robustly decoupled) or fine (earnings-artifact decoupled: poor on earnings, fine on non-wage ER -- a broader ER would show these MORE integrated).


| field | gap | occ-prestige | underemp | prestige pctl | (1-underemp) pctl | class |
|---|---|---|---|---|---|---|
| Communication Disorders | 1.21 | 57.8 | nan | 0.69 | nan | **earnings-artifact decoupled** |
| Nursing | 0.93 | 66.5 | 12.8 | 1.00 | 1.00 | **earnings-artifact decoupled** |
| Teacher Ed Subjects | 0.77 | nan | nan | nan | nan | **unclassified (no non-wage ER)** |
| Philosophy | 0.66 | 54.1 | 47.1 | 0.31 | 0.53 | **mixed** |
| Biology | 0.65 | 57.5 | 51.1 | 0.62 | 0.13 | **mixed** |
| Spanish | 0.62 | nan | nan | nan | nan | **unclassified (no non-wage ER)** |
| Management | 0.52 | 51.7 | 52.6 | 0.12 | 0.07 | **robustly decoupled** |

**1 robustly decoupled, 2 earnings-artifact decoupled, 4 mixed/unclassified** (of 7 decoupled fields).

## Overall verdict -- answering the reviewer

1. **Earnings is a defensible but PARTIAL ER proxy.** Field-level earnings standing tracks survey occupational-prestige standing at Spearman +0.67 (GSS scale +0.70; the two survey scales agree at +0.95) and tracks -underemployment at +0.70 -- all positive but **well short of 1.0**. Earnings carries much of ER standing and misses the status dimension.

2. **The decoupling is mostly NOT an earnings artifact in general** (B): the gap is roughly orthogonal to field-level non-wage ER, with only a weak, non-significant H2 lean on the status-residual. So the integrated/decoupled typology is **not overturned** by a field-level non-wage ER.

3. **But earnings materially mis-ranks specific fields** (C): it **under-states** ER for research/natural-science fields (Physiology, Biochemistry, Chemistry, Biology -- high occupational status, modest early pay) and **over-states** it for business fields (Marketing, Management, Sociology, Economics -- good pay, middling status).

4. **The decoupled end is heterogeneous** (D): **earnings-artifact decoupled** = Communication Disorders, Nursing (strong non-wage ER -- a broader ER would integrate them, typically licensed/compressed-pay fields); **robustly decoupled** = Management (poor on earnings AND non-wage ER); the rest mixed.

5. **Net:** earnings-only ER survives as a field-level summary of the typology, but it is a thin proxy that mis-prices identifiable fields and hides at least some earnings-artifact decoupling -- which is exactly what motivates the within-field, multi-dimensional Revelio test specified below.

## Robustness: GSS (Nakao-Treas 1989) prestige scale

Repeat A and the status-divergence with the GSS scale instead of Condon OPR, to show the status results are not an artifact of one survey prestige scale.

- A (earn vs GSS occ-prestige): Spearman +0.70 [+0.49,+0.83], n=47 (vs OPR +0.67).
- Scale agreement corr(OPR, GSS-1989) = +0.95 across fields -- the two survey scales rank fields' occupational standing near-identically, so the status findings are scale-robust.

## The institution x field test that needs Revelio (specified, NOT run)

The within-field analogue of the earnings gap, on a non-wage ER, is

```

gap^occ_f = 1 - Spearman_i( prestige_{i,f}, occ_prestige_{i,f} )

```

where `occ_prestige_{i,f} = sum_o P(o | institution i, field f) * prestige(o)` is each institution x field cell's occupational-prestige standing -- requiring **institution x field occupation flows** (Revelio resume data), which are not public. A second version weights placement by **employer prestige/selectivity** (which firms hire the grads). This is the true within-field typology-robustness test (ties to Thrust II); the present script is its field-level lower bound.

## Adversarial self-check

- **occ-prestige residual wage-correlation:** corr(occ_prestige_OPR, earn_standing) = +0.67; occupational prestige is partly wage-correlated by construction (prestigious jobs pay more). We therefore also report the **status residual** (prestige net of wage) in B -- the part wages cannot see -- which is where the H1/H2 adjudication is sharpest.

- **ACS field->occupation coverage:** weighted occupation-score coverage median 0.91; fields below 0.7 flagged in er_dimensions.csv (cov_opr).

- **NY Fed major-crosswalk losses:** underemployment covers only 32/54 fields (no clean NY Fed major for the rest, e.g. neuroscience, microbiology, biostatistics); listed as unmapped rather than forced.

- **Survey-prestige-scale sensitivity:** OPR vs GSS-1989 agree at corr +0.95 (above), so status results are not one-scale artifacts.

- **Grain caveat (load-bearing):** this is a **field-level** robustness test. It cannot recompute a within-field non-wage gap; it tests whether the earnings gap/typology survive a field-level non-wage ER. The Revelio estimand above is the within-field test proper.
