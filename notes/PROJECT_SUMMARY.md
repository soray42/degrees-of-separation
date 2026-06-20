# Project Summary — *Degrees of Separation*
### Task Distance and the Academic–Employer Reputation Gap

*One-page restatement of `degrees_of_separation_proposal.md` (v0.1). Source of truth is the proposal; this is the working digest.*

---

## Research question

**Across academic fields, how far does academic prestige travel into the labor market, and what determines the distance it travels?**

The same field of study is priced by **two** labor markets that need not agree:

- the **academic market** rewards a department through *faculty placement / peer assessment* → **Academic Reputation (AR)**;
- the **industry market** rewards its graduates through *wages and hiring demand* → **Employer Reputation (ER)**.

Both are reputation signals on the *same* field-specific human capital. The central object of study is the **AR–ER gap** at the level of academic fields, read as a **wedge between two signaling equilibria** — i.e. how integrated vs. segmented a field's academic and industry markets are. The governing mechanism is **task distance**: when academic-research tasks resemble the tasks of the jobs graduates take, both markets reward the same skills and the gap closes (computer science); when they diverge, ER falls back on credentialing and network inertia and the gap opens (law, finance, parts of economics/business).

**Headline claim (H4):** task distance predicts the gap across fields, with **CS as a near-zero-gap extreme** and credential-driven fields at the other end. US-only, entirely open data, two levels (PhD + undergraduate), panel extended to **2026**.

---

## Positioning: three literatures that never met

| Strand | What it has | What it lacks |
|---|---|---|
| **A — Faculty hiring networks** (Clauset–Arbesman–Larremore 2015; Wapman et al. 2022; Zhang et al. 2022) | Cleanest measure of academic reputation: recursive network prestige via **SpringRank**, per field | Stays *inside* academia; never connects to the labor market |
| **B — Reputation & the labor market** (MacLeod et al. 2017; Altonji–Pierret 2001; MacLeod–Urquiola 2015) | The theory (employer-learning signaling) + labor-market outcomes; reputation = ability + value-added | Single-field, single-country; never touches the hiring network |
| **C — Rankings research** (QS/THE/GEURS indicator studies) | Documents AR/ER survey indicators | Only *static correlation*; no structural academic-market model; no field-level gaps |

**The open gap (this project):** connect the faculty-hiring-network prestige object (A) to the reputation-and-labor-market signaling framework (B) **at the field level**, and add the field-heterogeneity mechanism (**task distance**) that neither strand supplies.

---

## The four-link theoretical chain

1. **Signaling — Spence (1973)** [+ Arrow 1973, Stiglitz 1975]. Employers can't observe productivity directly; they price observable signals, including institution/field prestige.
2. **Employer learning — Altonji & Pierret (2001)** [+ Farber–Gibbons 1996; Arcidiacono–Bayer–Hizmo 2010]. As employers learn true productivity, weight on credentials falls and weight on productivity rises. ⇒ **the AR–ER gap is, in part, the inverse of employer-learning speed:** a persistently large gap = a market stuck at the credential stage.
3. **Reputation decomposition — MacLeod, Riehl, Saavedra & Urquiola (2017).** Institutional reputation carries two informations: *student-body ability* and *institutional value-added*. These map to the project's two secondary axes — **admission selectivity → ability**, **education quality → value-added** — extended here across many fields plus the academic-market side.
4. **Task-specific human capital — Gathmann & Schönberg (2010)** [+ Poletaev–Robinson 2008; Yi et al. 2017]. Skills are portable to the extent jobs share tasks. ⇒ the **distance between the task content of academic research and the task content of graduates' industry jobs** governs how well the academic signal transmits to the industry market.

**Synthesis.** Both markets are two-sided matching markets (Becker 1973; Shimer–Smith 2000; Eeckhout–Kircher 2011) pricing the same field-specific human capital. Academic-market signal = faculty-hiring-network prestige; industry-market signal = employer reputation. **The gap between them is the wedge between two sorting/signaling equilibria, and task distance sets the size of the wedge.**

---

## Hypotheses

- **H1 (variation).** Significant cross-field variation in the AR–ER gap. *(If false, no paper.)*
- **H2 (central mechanism).** Academia–industry **task distance** positively predicts the gap (larger distance → larger gap).
- **H3 (employer learning).** The gap is larger where labor markets rely on credential signals and employer learning is slow.
- **H4 (headline).** **CS is a low-gap outlier** (integrated markets); credential-driven fields (law, parts of business/finance) sit at the high-gap extreme.
- **H5 (dynamics).** Fields where industry demand for academic-frontier skills rose (ML, data science) show **gap compression** 2011→2026; stable-task-distance fields show stable gaps.

---

## Unit of analysis (two levels)

**field × (institution) × year**, with a **two-level structure**:

- **Field level** — the object of interest; the gap is a field-level quantity.
- **Institution-within-field level** — the within-field ordering of institutions, and measurement noise.

A two-level random-effects specification separates a field-level gap component from institution-level noise.

**Gap construction:** `Gap_field = 1 − Spearman( SpringRank_prestige_rank , ER_rank )` within field, across institutions, where `ER_rank` comes from Scorecard earnings (undergraduate) and/or SDR salary + sector mix (PhD).

**Main regression:**
`Gap_field ~ task_distance + employer_learning_proxy + selectivity + value_added + field_structure + year_FE`
— H2 tests the `task_distance` coefficient; H4 checks CS lies in the low-gap tail.

---

## Data stack (CIP = the join key across Scorecard / SDR / O\*NET / IPEDS)

- **AR (academic signal):** ORCID+OpenAlex faculty-hiring-network → **SpringRank** per field, extended to 2026 (replicating Jiang et al. 2026 China method); **validated** against Wapman et al. 2022 Zenodo (2011–2020, 107 fields, 368 institutions, US). Supplement: OpenAlex field-normalized citations.
- **ER (behavioral):** **undergraduate** — College Scorecard Field-of-Study earnings (inst × CIP × credential, 1/4/5-yr); **PhD** — NSF NCSES **SDR** salary + sector split (academia/industry/government), with **SED** as frame.
- **Task distance (mechanism):** **O\*NET** + CIP↔SOC crosswalk → Gathmann–Schönberg-style distance between academic-research and modal-industry task vectors. Cross-checks: affiliation-switch rate, industry co-authorship share, patent density (from OpenAlex/ORCID).
- **Controls:** selectivity (IPEDS/CDS acceptance + SAT/ACT), value-added (residualized Scorecard earnings), field structure (Boyack–Klavans citation proximity; Wapman 8-domain taxonomy).

---

## Tier 0 go/no-go (run first, only already-computed public data)

Compute `Gap_field` for 15–20 fields from **Wapman Zenodo SpringRank** + **College Scorecard earnings** + **SDR public tables**; regress on the SDR industry-employment share (quick task-distance proxy); locate CS. **GO requires all three:** (1) non-trivial cross-field SD of `Gap_field`; (2) gap correlates **negatively** with the integration proxy; (3) CS in the low-gap tail, below the median. Any failure → **NO-GO**, rethink before building the pipeline.

---

*See `degrees_of_separation_proposal.md` for full text, identification/robustness (§8), and references (§10).*
