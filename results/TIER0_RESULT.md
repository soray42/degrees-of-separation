# Tier 0 — Decision: **CONDITIONAL**

A real, reliability-gated static gap map exists and the anchors reproduce — but **P1 is not
confirmed** (correct-signed, underpowered, fragile) and the **PSEO PhD-level enrichment that
v2 is built around is infeasible** (cells suppressed at the field grain; prestige-truncated).
The static paper is sound-but-modest; the top-tier case rests **entirely** on the untested
Tier-1 dynamic compression (H4/P2). **Proceed to Tier 1 only if the human judges the dynamic
bet worth the ORCID-rebuild cost; otherwise shelve.** Not a GO; not a clean SHELVE.

> Provenance: operational spec = the Tier-0 kickoff + `degrees_of_separation_proposal_v2.md`
> (the v2 file was supplied after kickoff; v1 was read for continuity). Every number
> reproduces via `scripts/01_probe.py`, `02_gap_map.py`, `03_p1.py`. Real data only;
> sources in `data/raw/SOURCES.md`. Date 2026-06-20.

## Decision criteria, scored against the evidence (proposal §7 / kickoff §7)

| GO criterion | Result | Pass? |
|---|---|---|
| ≥ ~15 fields reliable at undergrad | **14/30** reliable (signal_frac ≥ 0.5 **and** bootstrap CI excludes the null noise band) | ✗ borderline |
| PhD-level coverage exists for a usable subset | **0** released doctoral cells at the 4-digit-CIP grain (100% disclosure-suppressed); master's also 0 | ✗ **hard fail** |
| Prestige range not catastrophically truncated | **Scorecard** spans the full hierarchy (256/371, tracks Wapman across all quartiles); **PSEO** is top-truncated (108/371, **22%** of the top prestige quartile, ~18 of the top-20 elite privates absent) | ✓ for the core map (Scorecard); ✗ for PSEO |
| Anchors reproduce | **CS** gap 0.292 (reliable, low); **biology** gap 0.650 (reliable, high) — both reproduce | ✓ |
| P1 holds (primary or robustness) | correct **negative** sign on reliable fields (PSEO −0.28 p=0.33; CV −0.27 p=0.35) but **not significant**, and **fragile** (PSEO flips to +0.13 if unreliable fields wrongly included) | ✗ not confirmed |

Two GO criteria hard-fail (PhD coverage; P1), one is borderline (14 vs 15 reliable). The core
undergrad map is reliable and anchors reproduce, so this is **not** a SHELVE on the map. →
**CONDITIONAL.**

## What is established (the static foundation)

- **The gap map is real and reliability-gated.** 30 fields; 28 clear n ≥ 10; **14 pass the
  full reliability gate.** Measured on the *full* prestige hierarchy (Scorecard, not the
  truncated PSEO). `results/GAP_MAP.csv`, `figures/gap_map.png`, `figures/gap_signalfrac.png`.
- **Anchors reproduce** under the new *generic* `compute_gap` (the Tier-1 hook): CS low &
  reliable, biology high & reliable — the construct is stable across the refactor.
- **New high-gap fields surface**: nursing (0.93) and communication disorders (1.21, i.e.
  prestige *anti*-correlated with earnings) — reliable and worth interpreting later.

## What Tier 0 newly establishes as a problem (the honest bad news)

1. **PhD-level ER is unavailable at the field grain.** PSEO disclosure-suppresses *every*
   doctoral and master's institution × 4-digit-CIP earnings cell (0 released); graduate
   earnings exist only at the 2-digit-CIP grain (too coarse to separate chemistry from
   physics). **The v2 §5.3 plan — measure ER at the level where each field's talent lands,
   PhD-ER for pipeline fields — cannot be executed with public PSEO.** The noise-dominated
   fields (chemistry, earth sciences, several engineering branches) therefore **stay
   unreliable**; the prior diagnostic's recommended fix is data-blocked.
2. **PSEO truncates the prestige hierarchy.** It misses essentially the entire elite-private
   top. So PSEO cannot be used for the gap itself (only Scorecard can), and its within-
   institution dispersion (the P1 primary proxy) is measured on a public-skewed, top-
   truncated sample — weakening P1's primary test. `figures/coverage_prestige_range.png`.
3. **P1 is suggestive, not confirmed.** The theory-predicted negative sign appears on
   reliable fields on both proxies (consistent with the prior ≈ −0.41) but is not significant
   at n = 14 and flips on the primary proxy when the gate is removed. `results/P1_RESULT.md`.

## Recommendation (the human makes the call)

**CONDITIONAL — lean cautious.** The descriptive, reliability-gated map clears a *Science
Advances*-style descriptive bar on its own (Wapman-style: comprehensive, rigorous, open). But
Tier 0 did **not** deliver a clean P1 confirmation, and it *removed* the PSEO PhD enrichment
that distinguished v2 from v1. So the **only** remaining path to the Nature-tier bar is the
Tier-1 **dynamic out-of-sample compression** test (H4/P2), which:
- does **not** depend on PSEO PhD (it needs the ORCID AR rebuild + two-period undergrad ER), so it is **not** blocked by the findings above; **but**
- is an untested empirical bet that "may fail" (v2 §13), now carrying the *entire* top-tier case alone.

**Go to Tier 1 iff** the human accepts: (a) a static map + unconfirmed P1 + dead PhD
enrichment as the floor, and (b) the dynamic compression as the sole upside, worth the
multi-week ORCID rebuild. **If the bar requires either a confirmed static mechanism or the
two-level PhD richness, Tier 0 has effectively answered SHELVE** — the cheap gate did its job.

### Concrete, low-cost things that could change the call before committing Tier 1
- Re-test P1 on a field-level dispersion source that is **not** prestige-truncated (ACS
  field-of-degree earnings percentiles; SDR salary spread) to see if the negative sign
  firms up off the public-skewed PSEO sample.
- Check whether the 14→15 reliable-field count is sensitive to the (un-tuned) 0.5 threshold;
  report the curve rather than a single cut.
- Confirm Yifeng's resolved-edges dataset actually carries education→employment links with
  years + a field tag (v2 §11) — Tier 1's AR rebuild is dead on arrival without it.

## Files
`results/COVERAGE_REPORT.md` · `results/GAP_MAP.csv` · `results/P1_RESULT.md` ·
`results/figures/{coverage_prestige_range,gap_map,gap_signalfrac,p1_scatter}.png` ·
`data/interim/coverage_long.csv`. Code: `src/{load_ar,load_er,gap,predictions}.py`
(+ `src/crosswalks/`, reused `src/diagnostic.py` reliability), `scripts/01_probe.py`,
`02_gap_map.py`, `03_p1.py`.
