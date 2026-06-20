# Tier 0 — Go / No-Go Result

**Verdict: NO-GO** on the strict three-condition test (proposal §7) — but a *narrow*
no-go. The two **paper-defining** conditions pass decisively; the single failure is on
the **crudest, explicitly-provisional proxy** and points to a specific redesign, not to
abandoning the project. **Do not start the full ORCID+OpenAlex pipeline (Step 5) yet.**

*Run:* `python scripts/run_tier0.py` · notebook `notebooks/tier0_go_no_go.ipynb` ·
robustness `python scripts/tier0_robustness.py`. Outputs:
`results/tables/tier0_field_gaps.csv`, `results/figures/tier0_gap_vs_industry.png`.
Date: 2026-06-20.

## Data used (already-public, already-computed)

| Quantity | Source | Detail |
|---|---|---|
| AR prestige | Wapman et al. 2022, Zenodo `10.5281/zenodo.6941651` | `ranks.csv` `Rank` (SpringRank ordinal, 0 = top), `TaxonomyLevel=="Field"` |
| ER earnings | College Scorecard Field-of-Study (2026-06-10) | Bachelor's (`CREDLEV==3`) `EARN_MDN_4YR`, by `UNITID × CIP-4` |
| Integration proxy | SDR 2021 Table 12-3 (`nsf23319`) | industry share = Business-or-industry / All employed, fine field; SED 2-6 broad "Humanities and arts" for the 3 humanities (not in SEH-only SDR) |

20 fields spanning the range; 8–240 institutions matched per field (median ≈ 115).
`Gap_field = 1 − Spearman(prestige_score, earnings)` within field, across institutions.

## The three GO conditions

| # | Condition | Result | Verdict |
|---|---|---|---|
| 1 | Cross-field SD of `Gap_field` non-trivial | SD = **0.134**, range **0.246–0.715**, clear domain structure | **PASS** |
| 2 | `Gap_field` correlates **negatively** with integration proxy | Spearman = **+0.022** (p=0.93) on consistently-measured SDR fields; **−0.10** (p=0.67) full sample; OLS slope ≈ 0 | **FAIL** |
| 3 | CS in the low-gap tail, below median | CS gap **0.298**, **rank 2/20** (only Statistics lower), median 0.469 | **PASS** |

All three are required → **NO-GO**.

## What the gap actually looks like (selected, sorted low→high)

| Field | gap | industry share | | Field | gap | industry share |
|---|---|---|---|---|---|---|
| Statistics | 0.25 | 57% | | Civil Eng. | 0.49 | 53% |
| **Computer Science** | **0.30** | **63%** | | English | 0.51 | 9%* |
| Economics | 0.31 | 36% | | Biology | 0.62 | 47% |
| Political Science | 0.34 | 22% | | Physics | 0.66 | 56% |
| Mechanical Eng. | 0.38 | 65% | | Chemistry | 0.69 | 59% |
| Mathematics | 0.39 | 30% | | Earth Sciences | 0.72 | 36% |

\* humanities use the different-universe SED proxy. Full table:
`results/tables/tier0_field_gaps.csv`.

## Why the no-go is narrow (and what it means)

- **Condition 1 (variation) holds.** This was the "if false, no paper" assumption (H1).
  Gaps span 0.25–0.72 with structure: mathematics/computing + several social sciences
  are low-gap; natural sciences (physics, chemistry, biology, earth) are high-gap.
- **Condition 3 (CS headline) holds, decisively.** CS is the 2nd-lowest gap of 20 and
  far below the median, in **every** robustness specification
  (`scripts/tier0_robustness.py`): CS stays rank ≈ 2 whether ER is Bachelor's 1-yr/4-yr,
  Master's, or Doctoral earnings. The flagship prediction (CS = integrated, low-gap
  extreme) survives.
- **Condition 2 (mechanism) fails.** Among the consistently-measured SDR fine fields the
  gap does **not** decline with industry share (Spearman ≈ 0, OLS slope ≈ 0). The faint
  full-sample negative (−0.10) is an artifact of **3 humanities points** carried on a
  *different-universe* proxy (SED commitment-flow vs SDR employed-stock) pinned at one
  value. **Engineering is the decisive counterexample**: high industry employment
  (65–76%) **and** high gaps (0.38–0.49) — the opposite of the predicted direction.

The honest reading: **industry employment *share* is not a good stand-in for the H2
mechanism (task distance).** A field can place most graduates in industry yet still have
academic prestige and undergraduate earnings disagree (engineering), and a field can be
academia-centric yet have a moderate gap. The mechanism test needs the *real* variable.

## What this surfaced

1. **The proxy is wrong for H2.** Replace industry-employment share with the actual
   **Gathmann–Schönberg O\*NET task-distance** measure (academic-research vs modal-
   industry task vectors via the CIP↔SOC crosswalk). Condition 2 must be re-run on that.

> **Correction (post-review).** An earlier draft of this file called the PhD-prestige
> vs Bachelor's-earnings "level mismatch" a *confound* to be fixed. That was wrong.
> Academic reputation **is** research/PhD-level (SpringRank measures who the academy
> recognizes); employer reputation **is** about the graduate mass, i.e. undergraduates.
> So `Gap = 1 − Spearman(PhD-research prestige, bachelor's earnings)` is not two
> mismatched levels — **it is the AR–ER distinction itself** (a department's price in the
> academic market vs its price in the labor market). The gap construction is **accepted
> and unchanged.** The only residual measurement question is the *mechanical* one (could
> compressed within-field earnings variance inflate the gap?), which Tier 0.5 tests
> directly as "earnings dispersion." The Scorecard-doctoral-coverage note remains a fact,
> but it is **not** a reason to change the gap — it just rules out a PhD-earnings ER.

## Recommended next step (instead of Step 5)

Before building the 2026 faculty-hiring pipeline, do a **Tier 0.5**: construct the O\*NET
task-distance variable and a level-consistent ER, then re-test condition 2. If the gap
falls with task distance there, the project is GO with a corrected mechanism test; if not,
H2 is in genuine doubt and the framing needs rethinking. The descriptive contribution
(the field-level gap map + the CS result) is already real and robust regardless.

## Known caveats (per proposal §7, now quantified)

- Undergraduate-earnings vs PhD-prestige level mismatch — **material** (see above).
- Small institution counts in some fields (Materials 23, Physics 43, Philosophy 41).
- Earnings confounded by geography / cost-of-living (uncontrolled in Tier 0).
- Humanities proxy is cross-universe (SED vs SDR) — excluded from the honest cond-2 test.
