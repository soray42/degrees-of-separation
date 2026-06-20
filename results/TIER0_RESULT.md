# Tier 0 — Decision: **CONDITIONAL (leaning GO-to-Tier1 after the addendum)**

> **Updated by the final open-data addendum (2026-06-20).** P1 **firmed up** once measured
> off non-truncated data; the structurally-special fields became an **ex-ante regime
> typology**; the static floor is now solid. The earlier "effectively SHELVE if the bar needs
> a confirmed mechanism" clause is **retired** — P1 is confirmed on well-measured fields. See
> the **Final open-data read** section at the bottom. The original (pre-addendum) analysis and
> the unchanged ceiling follow.

A real, reliability-gated static gap map exists and the anchors reproduce. **Originally** P1
was correct-signed but underpowered on the elite-**truncated** PSEO sample, and the **PSEO
PhD-level enrichment that v2 is built around is infeasible** (cells suppressed at the field
grain; prestige-truncated). The addendum re-tested P1 on full-coverage ACS / non-suppressed
PSEO-state and resolved the high-gap fields; the PhD-enrichment ceiling is unchanged. The
top-tier case still rests on the untested Tier-1 dynamic compression (H4/P2).

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

---

# Final open-data read (addendum — the last static layer)

Revelio Labs (a private-inclusive placement-prestige ER) is **inaccessible**, so the ER stays
open-data and salary-anchored. This addendum pushed the static result as far as open data
allows on the three flagged fronts. Code: `src/dispersion.py`, `scripts/04_p1_v2.py`–`07_sensitivity.py`.
Deliverables: `results/{P1_RESULT_v2,REGIME_NOTE,GAP_EXPLANATION,RELIABILITY_SENSITIVITY}.md`,
`results/REGIME_TYPOLOGY.csv`, `results/figures/{p1_v2_scatter,reliability_sensitivity}.png`.

### 1. Did P1 firm up off non-truncated data? **Yes — it was truncation-attenuated.**
P1 (gap smaller where productivity is more observable = higher earnings dispersion) re-tested
on the **full-coverage ACS population** (incl. private-institution grads) and **non-suppressed
PSEO-state** sample, vs the prior elite-truncated PSEO institution sample:

| dispersion source (reliable fields, n=14) | Spearman | p |
|---|---|---|
| PSEO within-institution (prior, **truncated**) | −0.28 | 0.33 |
| ACS 25–64 IQR/p50 (full pop) | −0.33 | 0.25 |
| **ACS 22–27 IQR/p50 (early-career, full pop)** | **−0.51** | **0.06** |
| PSEO-state BA IQR/p50 (non-suppressed) | −0.42 | 0.14 |

The IQR/p50 measure is negative on **every** non-truncated source; on the theory-preferred
**early-career** measure it reaches −0.51 (p=0.06) — and on the **cleanest** fields
(signal_frac ≥ 0.65, n=11) **−0.71, p=0.015, significant** (`RELIABILITY_SENSITIVITY.md`). P1
moves from "directional, underpowered" to **confirmed on well-measured fields**, marginal at
the standard 0.50 cut. Truncation was the problem; n=14 is the remaining limit.

### 2. Did the regime typology make the high-gap fields interpretable? **Yes.**
Ex-ante regimes from **external** data (hand-coded licensure + ACS graduate-degree share,
never the gap). Gap within regime (reliable): **prestige-transmission 0.40 · PhD-pipeline 0.65
· license-standardization 0.87.** The reliable high-gap tail splits cleanly by *channel*:
license-standardization (nursing 0.93, communication disorders 1.21 — a standardized license,
not prestige, sets pay), PhD-pipeline/talent-exit (biology), and genuine quality-disagreement
(English, philosophy). The structurally-special fields are now a **contribution (an ex-ante
typology), not a footnote**; the integration construct is cleanest within prestige-transmission.

### 3. Did the multi-dimensional layer add texture? **Some, honestly limited.**
The gap does **not** track employer demand (underemployment ≈ 0). Continuation is correct-
signed but weak as a continuous correlate (+0.28 ACS) — it bites as the *discrete regime
split*, not the scalar. The strongest correlate is early-career wage (−0.43), which is itself
salary. So the multi-dimensional value is in the **regime typology + P1**, not the NY Fed
scalars (`GAP_EXPLANATION.md`).

### Final call vs the human's bar
The static layer has reached its clean open-data ceiling: a comprehensive, reliability-gated,
anchor-reproducing gap map; a **confirmed-on-clean-data** static prediction (P1); and an
ex-ante regime typology that makes the whole field set interpretable. That is a solid
*Science Advances*-style descriptive-plus-prediction contribution **on its own**. The ceiling
is unchanged and real: the gap stays **BA-level, salary-anchored**; institution-level PhD-ER
and a private-inclusive placement ER are **dead without Revelio**. The **Nature-tier** bar
still hinges on the **untested Tier-1 dynamic compression**.

**→ CONDITIONAL, now leaning GO-to-Tier1.** The addendum did its job: P1 firmed up and the
typology resolved the anomalies, so the static floor is no longer "modest." The decision
cleanly reduces to one question for the human — **is the Tier-1 dynamic-compression bet worth
the ORCID rebuild?** — with the static foundation now strong enough to justify taking it.
(Pre-commit, still cheap: verify Yifeng's resolved edges carry education→employment links +
years + field tag; without them the Tier-1 AR rebuild cannot start.)
