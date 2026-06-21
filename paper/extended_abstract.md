# Degrees of Separation: a measurable wedge between academic prestige and the labour-market pricing of fields

**Sora Yng** (+ co-authors TBD) · extended abstract (non-archival) · prepared for the NetSciX 2027 / IC2S2 2027 / ICSSI cycle and for faculty outreach

---

## The problem
Public university rankings (QS, US News *overall*) are **field-agnostic**: one "brand" stands in for the quality
of every program. If the way academic prestige maps onto labour-market pay is itself **field-specific**, a
field-agnostic ranking systematically mis-prices programs. We measure the object such a ranking hides.

## The object: the academic–employer reputation gap
For a field *f* with institutions *i*,
> **gap_f = 1 − Spearman_i( academic prestige, graduate earnings )**, within field, across institutions.

Academic prestige (AR) is the continuous **SpringRank** score of the US faculty-hiring network (Wapman et al.
2022, recomputed from the public edge subgraph so score *differences* are meaningful). Earnings (ER) are
College Scorecard field-of-study bachelor's medians, **cross-validated against Census PSEO**. The gap is a
**rank** statistic, so it ignores level differences (engineering > education) and asks only whether, *among the
schools that teach f*, the prestige and pay orderings coincide. A diagnostic battery confirms it is **real
signal, not an artifact** (only 15% of fields fall inside a perfect-agreement-plus-noise null, 65% above it;
the artifact channel explains only R²=0.20), with a per-field reliability filter.

## Result 1 — the gap is structured by discipline
A precision-weighted multilevel decomposition (`gap_f ~ 1 + (1|CIP-2)`, REML, n=48 fields) attributes
**~30% of cross-field gap variance to between-discipline clusters** (ICC = 0.30; measurement-corrected **0.45**;
expanded-universe **0.57**; one-way η² = 0.50). The gap is primarily about *which discipline a field is in*:

- **Integrated** (prestige predicts pay): Computer/Info **0.35**, Math & Statistics **0.35**, Economics-adjacent
  Social Science **0.42**, Engineering **0.43**.
- **Decoupled**: Health **0.75**, Arts **0.75**, Natural Resources **0.73**, Biological Sciences **0.68**,
  Physical Sciences **0.66**.

## Result 2 — one clean channel (licensing) + a large residual
Netting external, field-level mechanical channels (ACS PUMS): **occupational licensing raises the gap,
b = +0.66 [+0.33, +0.97]**; academic absorption is **null** (−0.02), and the two anchors are near-orthogonal so
the null is real. Together they explain only R²=0.28 — the rest is an **irreducible residual**. Adding pay-wedge
and public-sector channels raises R² only 0.27→0.29; they are **collinear** with licensing and add nothing
(standardized licensing +0.11 dominates; the others collapse to ~0; residual ranking unchanged, Spearman +1.00).
*The search for more channels finds redundancy, not explanation.*

## Result 3 — robust to a completely different earnings source
PSEO (UI wages) and Scorecard (Title-IV median) rank institution pay **near-identically (Spearman +0.94)**, and
the field-level **gap replicates across the two independent population definitions (Spearman +0.68 [+0.52,
+0.80])**. Using PSEO flows, 34% of graduates work out-of-state, and **netting destination geography barely
moves the gap (0.68 → 0.67)** — the rank gap is robust to geography even though geography drives earnings
*levels*. The design is **cross-national-ready** (any country with a faculty-hiring network + earnings-by-field).

## Result 4 — the market actively reprices fields (temporal)
In real (constant-2023) dollars over 2001–2019, **46 of 60 fields rise**: Statistics **+$2,311/yr**, Computer
Science **+$1,616/yr**, Computer Engineering +$1,421, Biomedical Engineering +$969, Finance +$868, Economics
+$817; the sharpest falls are **Pharmacy −$1,774/yr** and Communication Disorders −$1,405. The forward frontier
is the **prestige–earnings lead–lag** (does academic prestige adjust to, or lag, economic revaluation?). A
two-period prestige pilot finds prestige stability (median ρ≈0.50, a noise-depressed lower bound) against
materially moving earnings — *suggestive* that prestige is the slow anchor — but with two noisy periods the
direction **is not yet identified**; we specify the finer panel it needs.

## What we ruled out (scope conditions — a strength)
- **Not generic-brand vs. field-specific prestige.** On *medians* the academia-wide brand predicts pay at least
  as well as field-specific prestige (c_G = +0.45 ≥ c_F = +0.40; field's advantage negative in 42/57 fields).
  Consistent with Chetty–Deming–Friedman: brand effects live in the elite **tail** medians cannot see.
- **Not a distinct "shape".** Graded/threshold/flat classification is ~redundant with integration strength
  (Spearman +0.80–0.96); median data cannot resolve winner-take-all.
- **Program "mispricing" is geography/selection, not value-added.** Destination-state fixed effects explain
  R²=0.46 of the program residual vs. 0.00 for brand; the residual is +0.88-correlated with raw pay ranking.

## Framing and stance
Descriptive, **outcome-agnostic, not causal**. We read the wedge through signalling (Spence 1973) and
employer-learning theory (Altonji–Pierret 2001; MacLeod et al. 2017), and cite Chetty–Deming–Friedman for why
median data cannot see the tail. The practical implication is a **critique of field-agnostic prestige rankings**;
a CS-school demonstrator (prestige × earnings, cross-validated, with low-robustness flags) makes it concrete.

**Why this matters / why you (Giustinelli, Sinatra):** the gap connects students' *beliefs about returns* to a
measurable prestige–pay structure (decision-under-subjective-expectations angle), and is built natively on a
*science-of-science faculty-hiring network* (bibliometric-network angle). We are seeking a senior co-author to
sharpen the framing and decide the venue (descriptive CSS vs. a bolder behavioural push) and whether to add a
UK cross-national replication before submission.
