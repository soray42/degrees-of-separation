# Degrees of Separation — Roadmap (public data only)

> Working notes to myself. **Constraint (2026-09-23): public data only** — no restricted-use
> licences (B&B, NSCG/FSRDC, restricted SED, Texas ERC), no IRB-dependent surveys, no resume/profile
> data. Findings and numbers live in [`README.md`](README.md) §2–§3; this file is the plan.

---

## 0. Where things stand (after Phases 1–2, 2026-09-24)

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

**Working headline (after Phase 2):** graduate pay lines up with where a university sits in the academic
hierarchy, and more so as careers progress, but almost entirely through institution-wide status; the
department's own academic standing adds little (clearest remainder: computer science). In the UK the status
premium mostly survives a direct prior-attainment control. Fields whose graduates work in licensed or
public-sector settings are weakly coupled — a descriptive boundary condition, not an identified mechanism.

---

## 1. Next steps

### Done (2026-09-24) — Phase 2 on public data (scripts 58–64)
- [x] Selectivity pushed to the public-data limit: small department remainder, clearly positive only in CS.
- [x] PSEO August-2026 refresh: no conclusion changes; brand-over-field loading holds; p75 ≈ median.
- [x] Opportunity Insights tail: brand adds little beyond selectivity and parental income; tail ≈ median.
- [x] PSEO Flows: placement couples like pay (mostly selectivity); setting vs market attribution not identified.
- [x] UK LEO: 85–90% of coupling survives the prior-attainment control; career rise +0.026/yr; setting-priced
      subjects nearly decoupled.
- [x] Cell-level model and power: licensing moderation not separable from field type; field-level nulls
      underpowered.
- [x] Licensing battery: gradient robust in sign; nursing "little", not "none".
- [x] Literature merged and verified (152 entries); related-work draft positions against Bloem, Hu & Hurwitz
      (2024) and Britton et al. (2022).
- AEFP 2027 abstract skipped by decision.

### Now — rewrite the paper (target EPJ Data Science; PNAS Nexus variant)
- [ ] Rewrite `paper/main.tex` on the working headline (README §6); outline and claim-by-claim evidence tiers are
      in the local `PAPER_PLAN.md`.
- [ ] Freeze one canonical specification; everything else to an SI specification curve.
- [ ] Rebuild the SI from the committed scripts; switch to `paper/refs_merged.bib`.
- [ ] Compile a PDF; adversarial read by simulated referees before any submission decision.
- [ ] Optional: fix the 19-institution Wapman–PSEO name-join gap (src/crosswalks/institutions.py) and re-run
      scripts 52 and 32 on V4.14.1.

### Later
- [ ] Mid-December gate: heterogeneity survives selectivity only partly (large fields) → EPJ Data Science first.
- [ ] January 2027: submit + SocArXiv the same day.
- [ ] Optional Q2 2027: ORCID-rebuilt US/UK hiring networks as a data/methods note (Quantitative Science
      Studies).

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
