# Degrees of Separation — Roadmap (public data only)

> Working notes to myself. **Constraint (2026-09-23): public data only** — no restricted-use
> licences (B&B, NSCG/FSRDC, restricted SED, Texas ERC), no IRB-dependent surveys, no resume/profile
> data. Findings and numbers live in [`README.md`](README.md) §2–§3; this file is the plan.

---

## 0. Where things stand (after the Phase 1 re-checks, 2026-09-23)

- **Solid:** the within-field prestige–earnings coupling object, its reliability gate, and its
  Scorecard–PSEO replication; the three ruled-out alternatives; licensing as a marker of fields
  where pay is set by setting (state, employer, pay scale) — *not* wage compression
  (corr(licensure, wage CV) = +0.02).
- **Narrowed by Phase 1 (scripts 52, 55–57):**
  1. *Selectivity* absorbs most of the level of coupling (mean +0.43 → +0.14 net of SAT/admit/Pell/
     state; within-institution department slope +0.084 → +0.013). The cross-field ordering beats
     shared noise but is not separable from field-specific pricing of selectivity on public data.
  2. *Discipline structure* is borderline once field size is controlled; legacy ICCs do not clear a
     permutation null. Cross-field heterogeneity itself is real (I² = 0.69).
  3. *Brand vs field prestige*: field prestige is no better than brand after reliability correction.
  4. *Career-time coupling*: +0.031/yr on fixed PSEO cohorts, career-time part in [+0.014, +0.032]/yr,
     general across fields, loading on academia-wide brand rather than field prestige.
- **Withdrawn:** wage compression as mechanism or continuous law (within-occupation dispersion law
  null in 0/240 specifications); the "clean same-cohort" PSEO panel; "compressed fields pinned at 0";
  "field-agnostic rankings mis-price programs"; "aggregate data sees nothing, so resume data is
  necessary" (restated: public aggregates cannot identify the within-occupation channel).

**Working headline (to test, not established):** labour markets price institutional status
(selectivity/brand), not a department's academic standing; how much status pays varies strongly
across fields and grows over careers; it largely vanishes where pay is set by setting. The academy's
own field hierarchy adds little beyond brand — itself a reportable result for science of science.

---

## 1. Next steps

### Before 2026-10-02 — AEFP 2027 abstract
- [ ] 1,000-word structured abstract (background, question, data, methods, findings) on the working
      headline, verified numbers only; no mis-pricing or compression language.

### October — the selectivity question and the data refresh
- [ ] Push the selectivity test as far as public data allow: program-level composition (Scorecard FoS
      Pell / non-Pell earnings), field-specific brand and selectivity slopes in the within-institution
      design, and a horse race of department prestige vs institution selectivity vs brand. Report both
      readings (prestige coupling vs selectivity pricing) and what would separate them.
- [ ] Refresh PSEO to the August 2026 release (1,117 institutions); rebuild the fixed-cohort panel;
      add p25/p50/p75 quantile coupling.
- [ ] Opportunity Insights college-level tail outcomes (top-1% shares) × brand and field prestige,
      labelled ecological.
- [ ] PSEO Flows industry exposure at the institution × field level — the one public design that moves
      mechanism tests from ~20 fields to thousands of cells.
- [ ] Cell-level hierarchical model on all 47–57 fields; minimum detectable effects and equivalence
      bounds for every null (at n = 16, 80% power needs |ρ| ≳ 0.65).

### November — second country and the paper
- [ ] UK LEO: academia-wide prestige × LEO earnings; fixed-cohort career time; **prior-attainment
      bands as a public selectivity control** (the most direct public answer to the selectivity
      objection).
- [ ] France: InserSup earnings × lecturer (MCF) hiring data — feasibility only.
- [ ] Licensing: report strict / broad / multilevel specifications side by side; adjusted R² for state
      effects; drop-nursing-and-communication-disorders robustness; accounting as the counter-case.
- [ ] Merge the existing `refs/references.bib` (47 entries with reading notes) into the paper; add the
      college-quality, selectivity-by-major and UK LEO literatures.
- [ ] Rewrite `paper/main.tex` on the working headline; compile a PDF with Methods and a real SI;
      move the two-signal model, the CS ranking and the behavioural probe out of the main text; freeze
      one canonical specification (other branches → specification curve in the SI).

### Mid-December — gate
- Heterogeneity survives selectivity adjustment **and** the career-time rise holds → try PNAS Nexus
  first, then EPJ Data Science.
- Otherwise → EPJ Data Science directly (or Quantitative Science Studies with a "does the academic
  hierarchy predict anything outside academia" framing).

### January 2027 — submit
- [ ] Submit; post the SocArXiv preprint the same day (not before the fixes).
- [ ] Optional second paper (Q2 2027): the ORCID-rebuilt US/UK hiring networks as a data/methods note
      for Quantitative Science Studies.

---

## 2. Open decisions

- [ ] Rewrite git history? Old commit messages on the public repo state withdrawn claims; fixing files
      does not change them, and a history rewrite needs a force push (irreversible).
- [ ] Remove outreach names from `paper/COAUTHOR_NOTES.md` and `paper/extended_abstract.md` (public)?

---

## 3. Parked

- **Restricted microdata** (B&B, NSCG/FSRDC, restricted SED/HSLS, Texas ERC, LEHD): not available
  under the public-data constraint.
- **Belief survey** (named-institution vignettes): needs an IRB; parked.
- **Pay-deregulation event study** (Wisconsin Act 10 and similar, via PSEO): feasible on public data,
  but no longer central now that compression is withdrawn; could be reframed as "does deregulating
  teacher pay raise the pay-for-status gradient in education?". (Cite Biasi 2021 as *AEJ: Economic
  Policy*, not the AER.)
- **Resume/profile data (e.g. Revelio):** would add direct multi-dimensional placement (employer tier,
  occupation, seniority), the elite tail, individual trajectories and cross-national coverage; would
  not fix selectivity; salaries are modelled; coverage is thinnest in exactly the licensed fields.
  Access is usually through an institution's WRDS subscription.
