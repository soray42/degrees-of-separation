# Revealed-demand layer --- FEASIBILITY GATE (does the data exist + join?)

Gate only: verify open application/acceptance data exists at provider x field and joins to the earnings + prestige axes. **No slopes, no conclusions.** No-fabrication run. Seeded; `python scripts/45_demand_feasibility.py`.

## Per-source availability

| source | available? | grain | years | downloadable? |
|---|---|---|---|---|
| **UK UCAS** end-of-cycle (free provider-level) | **yes (free)** | provider x subject group (JACS3 / HECoS) | 2007--2021 (JACS3); 2019--2021 (HECoS) | yes, per-resource CSV/zip via `/media/<id>/download` |
| UK UCAS OFFERS / admit rates | partial-free / **PAID** | provider x subject | -- | complete offers = paid **UCAS EXACT** (NOT used); only an 18-yr-old offer subset is free |
| **US UC** freshman admission by discipline | **yes** | campus x **discipline** (15 broad + undeclared) | 2012--2023 | yes (Tableau crosstab export; download-instructions PDF) |
| US CSU (secondary) | gated here | college/major | -- | dashboard; data centre **HTTP 403** in this environment -> scrape/portal-gated |

**Applications + acceptances are FREE** at provider x subject group (UK) and campus x discipline (US UC). The funnel middle (complete OFFERS / admit rates) is paid UCAS EXACT in the UK and is not used.

## UK join test (demand x earnings x prestige)

Joining on the **normalised provider name** (UCAS provider = code + name, not UKPRN; LEO has UKPRN + name; ORCID has ROR name):

- UCAS providers (free release): **427**; LEO providers (earnings): **331**; ORCID UK providers (prestige): **551**.

- UCAS $\cap$ LEO = 184; UCAS $\cap$ ORCID = 168; LEO $\cap$ ORCID = 149.

- **Three-way (demand + earnings + prestige) = 126 providers** (exact normalised-name match; fuzzy matching + a UCAS-code$\to$UKPRN crosswalk would raise it). Sample: abertay university, aberystwyth university, accrington rossendale college, anglia ruskin university, arts university bournemouth, aston university.

- **Subject join:** UCAS subject group (HECoS / JACS3) $\to$ CAH2 (the LEO + ORCID-prestige grain) via the official HESA/ONS HECoS$\to$CAH and JACS3$\to$CAH lookups (documented; the actual UCAS subject-group resource 014/015 is free but its per-resource media id is mapped only on the dynamic page --- a manual pull completes it).


So a usable UK panel exists: **126 providers x CAH2 subjects** (intersected with the ~16 CAH2 subjects that carry a stable UK ORCID prestige axis, scripts/40) carry demand + earnings + prestige.

## US join test (demand x gap/prestige x Scorecard)

UC discipline $\to$ project CIP-2 is a clean crosswalk (15 broad disciplines $\approx$ CIP-2 clusters). The binding problem is **institutional coverage**, not the crosswalk:

- The US prestige/earnings axis spans **226 institutions** nationally (inst x field, `valuation_residuals.csv`; publics, privates, and the elite tail).

- The only free by-field US demand source (UC) covers the **9 UC undergraduate campuses** --- all present in the US panel, but only **4%** of the axis, **California public only, no privates/elites**. CSU would add ~23 more CA publics (still single-state, public, and access-gated here).

- A within-field, across-institution demand--prestige--earnings design needs broad cross-institution demand variation; UC gives at most 9 institutions per discipline and misses the entire private/elite tail. **IPEDS** admissions (all Title-IV institutions) is institution-level and has **no by-major/field breakdown**, so there is no broad US provider x field demand source.

## GATE VERDICT

- **UK: GO.** Free UCAS provider x subject-group applications + acceptances join to LEO earnings + ORCID prestige on 126 providers via normalised name; the subject crosswalk to CAH2 is documented. Steps 2--4 can proceed on the UK panel (finish the UCAS subject-resource pull + name/UKPRN crosswalk first). The funnel is demand(applications) + outcome(acceptances); admit-RATE needs paid EXACT.

- **US: NO-GO** for a panel comparable to the national gap/prestige axis. The demand-by-discipline data exists and is downloadable (UC), but its institutional coverage (~9 CA-public campuses, ~4% of the axis, no privates/elites) is far too narrow and non-comparable to the US prestige/earnings axis. It is at best a **partial, UC-system-internal** exercise (9 campuses x 15 disciplines), not a US-wide revealed-demand layer. No free broad US provider x field demand source exists.

## Adversarial self-check

- **UCAS applications vs offers (free vs EXACT).** Applications and acceptances are free at provider x subject group; the complete offers / admit-rate funnel is the paid UCAS EXACT product and is excluded. A demand(applications)+outcome(acceptances) panel is free; an admit-RATE slope is not.

- **UC discipline-not-major coarseness.** UC publishes 15 broad DISCIPLINES, not majors --- comparable to CIP-2 / CAH2 but it loses within-discipline field variation; not finer than the gap grain.

- **Provider/campus name-matching losses.** The UK join is exact normalised name (126 of 331 LEO / 427 UCAS providers); a UCAS-code$\to$UKPRN crosswalk and fuzzy matching would raise coverage. Unmatched providers are listed by the join, not forced.

- **Demand-count comparability across the prestige axis.** UCAS counts are rounded to the nearest 5 and UC counts are specific to the California applicant pool; absolute demand levels are not comparable across sources or against the (national) prestige axis --- any design must use within-provider / within-subject RANKS or shares, not raw counts.

- **Gate only.** This verifies existence + join; it does NOT estimate demand slopes or draw any conclusion about revealed demand.
