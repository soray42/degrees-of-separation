# Two-market integration map (static)

**Conceptual lens (metaphor, not literal arbitrage, not causal):** the academic prestige hierarchy and the labor market are two valuation systems pricing the same human capital (institution × field). **Integration** index = how tightly academic standing maps to pay = `c_F[f] = Spearman(F, y) = 1 − gap[f]`; **segmentation** = the gap. Descriptive and outcome-agnostic: across-institution earnings conflate value-added with student SELECTION — we describe co-movement, we do not identify causal effects (cite Chetty-Deming-Friedman: the brand's payoff is concentrated in the elite TAIL that median data cannot see; MacLeod et al. 2017). Run: `python scripts/29_two_market_integration.py`.

## 1. The integration map (reframe of the gap)

57 fields ranked from most INTEGRATED (academic standing ≈ pay) to most SEGMENTED (the two valuations diverge). *This axis is the existing gap, re-presented — disclosed.*

**Most integrated** (prestige tracks pay):

| label                    |   n |   integration | structure   |
|:-------------------------|----:|--------------:|:------------|
| Food Science             |  16 |         +0.76 | n/a         |
| Statistics               |  28 |         +0.75 | GRADED      |
| Computer Science         | 161 |         +0.71 | GRADED      |
| Computer Engineering     |  74 |         +0.70 | GRADED      |
| Economics                | 116 |         +0.69 | GRADED      |
| Political Science        | 139 |         +0.66 | GRADED      |
| Biomedical Engineering   |  48 |         +0.65 | GRADED      |
| Kinesiology/Exercise Sci |  70 |         +0.62 | GRADED      |

**Most segmented** (the two markets diverge):

| label                    |   n |   integration | structure   |
|:-------------------------|----:|--------------:|:------------|
| Agricultural Engineering |  14 |         +0.09 | n/a         |
| Nursing                  |  99 |         +0.07 | FLAT        |
| Agronomy                 |  11 |         +0.07 | n/a         |
| Microbiology             |  17 |         +0.07 | n/a         |
| Music                    |  76 |         +0.04 | FLAT        |
| Forestry                 |  17 |         -0.15 | n/a         |
| Communication Disorders  |  17 |         -0.21 | n/a         |
| Physiology               |  16 |         -0.34 | n/a         |

## 2. Integration STRUCTURE — graded, threshold, or flat? (NEW)

The same correlation hides different shapes. For each field we compare the **top-prestige-decile pay premium** against the prestige–pay correlation **among the non-top-decile** institutions: driven only by the top decile → **THRESHOLD** (winner-take-all: only the most prestigious tier earns a premium); persists in the rest → **GRADED** (academic standing is rewarded gradually); neither → **FLAT** (prestige barely maps to pay).

Among the 44 fields with ≥20 institutions: **GRADED 37, THRESHOLD 4, FLAT 3**.

Answers *'is academic standing rewarded gradually, or only at the very top?'* — and it varies by field:

| label                    |   n |   integration | structure   |   top_decile_premium |   rest_corr |
|:-------------------------|----:|--------------:|:------------|---------------------:|------------:|
| Statistics               |  28 |         +0.75 | GRADED      |                +0.42 |       +0.69 |
| Computer Science         | 161 |         +0.71 | GRADED      |                +0.93 |       +0.61 |
| Computer Engineering     |  74 |         +0.70 | GRADED      |                +0.33 |       +0.64 |
| Economics                | 116 |         +0.69 | GRADED      |                +0.63 |       +0.61 |
| Political Science        | 139 |         +0.66 | GRADED      |                +0.30 |       +0.59 |
| Biomedical Engineering   |  48 |         +0.65 | GRADED      |                +0.08 |       +0.57 |
| Kinesiology/Exercise Sci |  70 |         +0.62 | GRADED      |                +0.19 |       +0.58 |
| Mechanical Engineering   | 133 |         +0.62 | GRADED      |                +0.21 |       +0.50 |
| Mathematics              | 120 |         +0.62 | GRADED      |                +0.48 |       +0.51 |
| Finance                  |  89 |         +0.59 | GRADED      |                +0.36 |       +0.49 |
| History                  | 131 |         +0.59 | GRADED      |                +0.32 |       +0.47 |
| Human Dev & Family Sci   |  31 |         +0.58 | GRADED      |                +0.16 |       +0.49 |
| Psychology               | 173 |         +0.57 | GRADED      |                +0.22 |       +0.49 |
| Sociology                | 114 |         +0.57 | GRADED      |                +0.19 |       +0.51 |
| Public Health            |  32 |         +0.56 | GRADED      |                +0.20 |       +0.64 |
| Aerospace Engineering    |  35 |         +0.56 | GRADED      |                +0.19 |       +0.44 |
| Chemical Engineering     |  91 |         +0.55 | GRADED      |                +0.13 |       +0.47 |
| Marketing                |  81 |         +0.52 | GRADED      |                +0.28 |       +0.45 |
| Electrical Engineering   | 118 |         +0.51 | GRADED      |                +0.23 |       +0.37 |
| Accounting               |  94 |         +0.51 | GRADED      |                +0.17 |       +0.46 |
| Materials Science        |  23 |         +0.51 | GRADED      |                +0.09 |       +0.40 |
| Civil Engineering        | 105 |         +0.51 | GRADED      |                +0.10 |       +0.42 |
| Linguistics              |  26 |         +0.50 | GRADED      |                +0.28 |       +0.40 |
| English                  | 135 |         +0.49 | GRADED      |                +0.24 |       +0.38 |
| Industrial Engineering   |  39 |         +0.49 | GRADED      |                +0.13 |       +0.44 |
| Anthropology             |  75 |         +0.49 | GRADED      |                +0.21 |       +0.45 |
| Management               |  90 |         +0.48 | GRADED      |                +0.20 |       +0.43 |
| Social Work              |  68 |         +0.45 | GRADED      |                +0.09 |       +0.39 |
| Biochemistry             |  47 |         +0.42 | GRADED      |                +0.11 |       +0.35 |
| Theatre                  |  46 |         +0.38 | GRADED      |                +0.20 |       +0.29 |
| Spanish Lang & Lit       |  30 |         +0.38 | GRADED      |                +0.31 |       +0.25 |
| Biology                  | 137 |         +0.35 | GRADED      |                +0.13 |       +0.28 |
| Physics                  |  43 |         +0.35 | THRESHOLD   |                +0.37 |       +0.18 |
| Philosophy               |  41 |         +0.34 | GRADED      |                +0.07 |       +0.34 |
| Chemistry                |  76 |         +0.31 | GRADED      |                +0.08 |       +0.24 |
| Earth Sciences           |  50 |         +0.28 | GRADED      |                +0.10 |       +0.27 |
| Nutrition Sciences       |  30 |         +0.27 | THRESHOLD   |                +0.06 |       +0.17 |
| Geography                |  41 |         +0.24 | GRADED      |                +0.11 |       +0.21 |
| Teacher Ed (subjects)    |  36 |         +0.23 | GRADED      |                +0.04 |       +0.22 |
| Environmental Sciences   |  48 |         +0.23 | THRESHOLD   |                +0.06 |       +0.17 |
| Architecture             |  34 |         +0.22 | THRESHOLD   |                +0.09 |       +0.13 |
| Animal Sciences          |  35 |         +0.20 | FLAT        |                +0.01 |       +0.17 |
| Nursing                  |  99 |         +0.07 | FLAT        |                +0.01 |       +0.05 |
| Music                    |  76 |         +0.04 | FLAT        |                +0.15 |       -0.07 |

- **GRADED** fields are the genuinely integrated ones: prestige maps to pay across the whole hierarchy, not just at the apex. **THRESHOLD** fields have a winner-take-all top tier with a pooled remainder — integration is an elite-tier phenomenon there. **FLAT** fields are segmented (the gap is high and there is no top-tier rescue).
- **GRADED dominates and THRESHOLD is rare on MEDIANS — but that is exactly what Chetty-Deming-Friedman predicts.** Their winner-take-all brand premium is concentrated in the elite TAIL (top-1% earnings, elite firms/grad school) that Scorecard MEDIANS cannot see; a median-earnings view will therefore under-detect THRESHOLD shapes. So 'mostly graded on medians' is consistent with 'winner-take-all in the tail' — not a contradiction of it. The robust claim is that integration has different *shapes*, not just different *strengths*.

## 3. Integration barriers (reframe — no new channel compute)

What keeps the two markets apart? The ONE externally-identified barrier is **occupational LICENSING** (b_licensure ≈ **+0.66** on the gap, scripts/20): credentialing standardizes pay within licensed occupations, decoupling it from academic standing — a literal institutional barrier between the two valuation systems. Academic-absorption (how academia-bound a field is) was **NULL**. Beyond licensing, **most of the segmentation is unexplained** — a large residual we cannot yet attribute to a named barrier. (The channel decomposition is not re-run here; see scripts/20, 27.)

## 4. Future-work feasibility — the dynamic (lead-lag) extension (REPORT ONLY)

The exciting extension is DYNAMIC: when a field is economically revalued (earnings shift), does academic prestige ADJUST (the academy tracks the market) or LAG (autonomy)? Which market leads? **Data feasibility (not run):**
- **Scorecard FOS earnings vintages:** the repo holds **1 vintage** (Most-Recent-Cohorts). The Dept. of Education has published **annual FOS releases since ~2019 (~5–6 vintages)**, each freely downloadable — these give *calendar-time* earnings variation per field. (The 1/4/5-yr horizons inside one vintage are different *cohorts*, not calendar vintages — not a substitute.)
- **Year-varying prestige:** the public Wapman release is a single pooled 2011–2020 cross-section. Year-varying academic prestige needs the **ORCID-rebuilt hiring network** (scripts/10–19), which can yield period-specific SpringRank (e.g. 2011–2020 vs 2021–2026); the gateway already validated ORCID-AR at academia level (ρ≈0.74).
- **Minimal viable panel:** **2 Scorecard vintages × 2 Wapman/ORCID prestige periods** over the best-covered ~20 fields — a 2×2 lead-lag pilot (does Δprestige follow Δearnings or vice-versa). Cost: ~2 extra Scorecard downloads + the ORCID two-period rebuild already scoped. **Not run here** — sized, not executed.

## Adversarial self-check

1. **Integration index = 1 − gap is a REFRAME** (disclosed) — section 1 adds no new measurement. The genuinely new content is (a) the **structure classification** (GRADED/THRESHOLD/FLAT) and (b) the **feasibility sizing** of the dynamic test.
2. **'Two markets' is a metaphor**, not literal arbitrage: there is no traded asset, no enforceable law of one price; 'integration/segmentation' is a descriptive analogy for how tightly two valuation orderings co-move.
3. **Static = descriptive**: no lead-lag and no causal claim is made; section 4 only SIZES the dynamic test, it does not run or pre-judge it.
4. **Selection:** the prestige–pay co-movement reflects who enrolls as much as institutional value-added; a THRESHOLD shape, in particular, is exactly what Chetty's elite-tail story predicts on selection grounds (top-tier brands enroll the students whose tails the median still partly reflects). The structure classes describe shape, not cause.
5. **Thresholds are tunable:** the GRADED/THRESHOLD/FLAT cutoffs (|c_F|<0.2 flat; c_rest≥0.5·c_F graded; top-premium>3% threshold) are documented and reproducible; field counts may shift modestly with the cutoffs — the qualitative point (integration has different *shapes*, not just different *strengths*) is the robust claim.
