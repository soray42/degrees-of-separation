# Degrees of Separation
### The Academic–Employer Reputation Gap and the Dynamics of Labor-Market Learning

**Working research proposal — v2**

---

## 0. What changed from v1 (read this first)

v1 proposed that a single mechanism — **task distance** between academic research and industry work — drives the gap between academic and employer reputation across fields. We ran that down empirically (Tier 0 → Tier 0.5 → diagnostic). Three things came back:

1. **The gap is a real object.** A per-field reliability analysis (null-model + bootstrap + residualization) shows most fields' gaps are genuine signal, not measurement noise. CS is a real low-gap field; biology is a real, well-measured *high*-gap field. A minority of fields (chemistry, earth sciences) are noise-dominated under the current undergraduate-earnings measure and cannot yet be interpreted.
2. **Task distance is dead.** Measured independently from O\*NET and lined up against the gap, it is *wrong-signed* (Spearman ≈ −0.38) — the opposite of the theoretical prediction — even within well-measured fields. It is dropped entirely. This proposal does **not** measure occupation task vectors.
3. **The gap is mostly unexplained.** The four pre-registered correlates jointly explain ~20% of cross-field variance; the two theory-adjacent ones reversed sign. So we have a real phenomenon and, as of now, no validated explanation.

v2 responds to this honestly. The contribution is no longer a single causal mechanism. It is **(a)** a theory-disciplined characterization of the gap as a measure of *market integration* between the academic and industry labor markets, and **(b)** a headline *dynamic* test: the signaling/employer-learning model predicts which fields' gaps should **close over time**, and we validate that prediction **out-of-sample on 2020–2026**. The static cross-field picture is the foundation; the out-of-sample dynamic prediction is the payoff.

---

## 1. Summary

Academic prestige and labor-market reward are two prices for the same underlying thing: a field's human capital. The **academic market** prices a department by where it places its PhDs as faculty (prestige). The **industry market** prices it by what its graduates earn and where they are hired. These two prices need not agree. We define, for each field, the **academic–employer reputation gap** as the degree to which a department's academic prestige *fails* to predict its graduates' labor-market outcomes, measured as a rank decorrelation across institutions within the field.

Grounded in signaling theory and employer learning, we interpret the gap as a measure of **market integration**: small where the two markets see the same underlying quality, large where they are segmented. We measure it comprehensively across US fields using fully open data, with a per-field reliability framework. We then test the theory's central *dynamic* implication — that fields where the labor market has not yet learned to price a new skill should show **compressing gaps** as the market matures — by predicting compression from early-period (2011–2020) structure and validating it on a held-out **2020–2026** sample. To our knowledge this is the first comprehensive, theory-disciplined, and out-of-sample-validated measurement of academic-vs-labor-market reputation divergence across fields.

---

## 2. Research question

1. **(Descriptive, established)** Across US academic fields, how much does academic prestige diverge from labor-market reward, and how reliably can the divergence be measured?
2. **(Theoretical interpretation)** Is the gap interpretable as a wedge between two signaling equilibria — i.e., a measure of integration between the academic and industry labor markets?
3. **(Static prediction)** Does the gap track *productivity observability* — smaller where employers can directly observe and price graduate productivity, as employer-learning theory predicts?
4. **(Dynamic prediction — headline)** Do the gaps of fields where the labor market is still *learning to price a skill* compress over time, and can that compression be predicted from early-period structure and confirmed **out-of-sample on 2020–2026**?

---

## 3. Positioning: three literatures that each did half

- **Faculty-hiring networks** (Clauset/Larremore 2015; Wapman, Zhang, Clauset & Larremore 2022, *Nature*). Built the academic-market prestige hierarchy via SpringRank on faculty-hiring networks — but stayed entirely inside academia. No labor-market side.
- **Reputation → labor-market signaling** (Spence 1973; Altonji & Pierret 2001; MacLeod, Riehl, Saavedra & Urquiola 2017, *AEJ:Applied*). Formalized how reputation maps to earnings and how employers learn — but single-field / single-country, with no network and no cross-field map.
- **Rankings research** (QS, GEURS). Correlates ranking components statically — but only one side at a time, no academic-market structure, survey-based reputation we explicitly avoid.

This project sits at the intersection: it joins the academic-market prestige hierarchy (literature A) to behavioral labor-market outcomes (literature B's theory, measured at scale), producing the cross-field integration map that none of the three has built. The closest precedent in form is Wapman 2022 — a comprehensive, descriptive, network-based *Nature* paper with no causal identification — which establishes that a rigorous characterization of a labor system can clear that bar. Our advantage over Wapman is that we have an **explicit economic theory** giving the central object meaning and testable predictions, rather than an atheoretical network statistic.

---

## 4. Theoretical framework

*(This section is the one most in need of senior theory input — see §10. The model below is the target; formalizing it and deriving clean comparative statics is the load-bearing piece.)*

### 4.1 Two markets, two signals, one human capital

A field's graduates carry human capital $h$. Two markets price it:

- The **academic market** observes a research-relevant signal and prices departments by faculty-placement prestige $A$ (the SpringRank of the faculty-hiring network).
- The **industry market** observes a productivity-relevant signal and prices graduates by earnings/placement $E$.

Both $A$ and $E$ are noisy signals of the same departmental quality $q$ (student ability + value-added, after MacLeod et al.). If the two markets weighted the same components of $q$ identically, $A$ and $E$ would be rank-identical and the gap would be zero. They do not, so:

$$\text{Gap}_{\text{field}} \;=\; 1 - \rho\big(\text{rank}(A),\, \text{rank}(E)\big) \quad\text{across institutions within the field.}$$

**Interpretation (the construct):** the gap is the wedge between two signaling equilibria — *the degree to which the academic and industry labor markets for a field are segmented rather than integrated.* This is the theoretical contribution that makes the gap an economic object, not a correlation artifact.

### 4.2 Employer learning makes it dynamic (Altonji–Pierret)

Employer learning is fundamentally a model about *time*. Early in a cohort's career, the market lacks direct evidence of productivity and leans on observable proxies (the credential, the department's prestige). As careers progress and a new skill becomes legible, the market re-prices on realized productivity and the weight on the credential **declines**.

Applied across fields rather than across career-years, this yields the **compression prediction**: when a field's skill is *new* and the labor market has not yet learned to evaluate it directly, prestige and earnings can be badly misaligned (high gap). As the market matures, it learns to price the skill on its own terms, and the gap **closes**. Mature fields, already learned, should show stable gaps.

### 4.3 Derivable predictions (what we test)

- **P1 (static):** the gap is *smaller* in fields where graduate productivity is more directly observable to employers. Proxy for observability: within-field cross-individual **earnings dispersion** (PSEO percentiles) — a market that prices individual productivity spreads earnings out. *The diagnostic already shows earnings dispersion correlates with the gap in the correct direction, giving P1 initial support.*
- **P2 (dynamic, headline):** fields whose skills were *emerging* in the early period exhibit **compressing** gaps over time; mature fields do not. Tested **out-of-sample** (§7).

The strength of P1/P2 is that they are derived from a theory, not chosen to fit the data — and the diagnostic warns us that several intuitive correlates do *not* hold, so confirmation is a genuine test, not a foregone conclusion.

---

## 5. The gap construct and its measurement

### 5.1 Academic reputation $A$ (the academic market's price)

- **Historical (2011–2020):** Wapman et al. 2022 SpringRank (Zenodo 6941651), pooled-decade, all PhD-granting US institutions, 107 fields. External, peer-reviewed, *Nature*.
- **Recent (2021–2026):** rebuilt from ORCID-derived academic-mobility edges via **Yifeng Li's resolved-edges dataset** (Zenodo 19651302) + `cs2n-orcid-affiliation-resolver`, which already performs the institution-name resolution that is the hard part of an ORCID rebuild. Filter to US education→faculty edges, bin by field × period, run SpringRank (De Bacco–Larremore–Moore 2018).
- **Method validation:** rebuild $A$ via ORCID for the **2017–2020 overlap** and confirm it correlates with Wapman on that window. A high correlation licenses the 2021–2026 extension and calibrates any source-induced level shift. *(This is the same "validate on overlap, then extend" logic as the PI's prediction-market cross-sample work.)*

### 5.2 Employer reputation $E$ (the industry market's price)

Behavioral, IRS/LEHD-linked outcomes only — no employer-survey reputation (no QS/GEURS).

- **College Scorecard Field-of-Study** earnings: institution × CIP × credential, median earnings 1/4/5 yr; multiple pooled-cohort releases give a short panel.
- **PSEO (Census/LEHD)** — the enrichment: institution × CIP × **degree level** earnings at the 25th/50th/75th percentiles (1/5/10 yr) **plus industry employment flows**. PSEO is internally rich: it supplies (i) **degree-level decomposition** (undergraduate *and* PhD earnings for the same department), (ii) **earnings dispersion** (percentiles → the P1 observability proxy), and (iii) **industry destination**. ~825 institutions / ~29% of graduates as of 2024, skewed to public systems — a coverage limitation probed in Tier 0.

CIP is the join key throughout. Field-level complements (ACS field-of-degree earnings, NSF SDR for PhD sector/salary, NSCG further-degree attainment) provide field-level robustness but cannot feed the within-field-across-institution gap.

### 5.3 The two-level point and the selection problem

The "good students leave for PhD" selection is **continuous across all fields** (a 2–30% spectrum, not pipeline-vs-terminal binary). Where it is strong, undergraduate earnings undercount a department's quality because the signal-carrying graduates exit to PhD tracks — which is why undergraduate-ER fields like chemistry are currently noise-dominated. PSEO converts this hidden bias into an **observable** one: for the same department we see the undergraduate branch *and* the PhD branch (earnings and industry destination). The gap is then measured at the level appropriate to each field — undergraduate ER for terminal-degree fields, PhD-level ER for pipeline fields — and the continuation share is itself observed and can be conditioned on.

### 5.4 Reliability framework (carried over, it works)

Every field-level gap is reported with: a **null-model** signal fraction (synthetic perfect-agreement earnings + cohort-size sampling noise), a **within-field bootstrap CI**, and a **residualization** check against the measurement-artifact channel. Only fields passing the reliability filter enter the explanatory and dynamic tests. This reliability layer is itself a methodological contribution and directly addresses the low-T/N signal-to-noise problem familiar from the PI's transfer-entropy reliability work.

### 5.5 What is dropped

O\*NET task vectors and the CIP↔SOC task-distance construct. Replaced by the theory-derived observability proxy (earnings dispersion) for P1.

---

## 6. Hypotheses

- **H1 — cross-field variation (established).** The gap varies systematically across fields and is reliably measured in most. *Status: confirmed by the diagnostic.*
- **H2 — integration / observability (P1).** The gap is smaller where graduate productivity is more observable (higher within-field earnings dispersion). *Status: initial support (correct-signed in the diagnostic); to be tested cleanly with PSEO.*
- **H3 — field anchors (established).** CS is a low-gap field; biology a real, well-measured high-gap field. *Status: confirmed.*
- **H4 — dynamic compression, out-of-sample (P2, headline).** Fields with emerging skills in 2011–2020 show compressing gaps; the set predicted to compress, identified from early-period structure, is confirmed to compress on the held-out 2020–2026 sample; mature fields are stable. *Status: the central open test.*

---

## 7. Empirical strategy (phased; each phase gates the next)

### Tier 0 — static gap, the cheap gate (days)
1. **PSEO coverage probe.** Pull PSEO earnings + flows; output the institution list, the per-field PhD-cell coverage matrix, and the overlap with Wapman. *Decision point:* if too few institutions per field or PhD cells are mostly suppressed (or elite privates are largely absent so the prestige range is truncated), the project is data-limited and shelving is the honest call.
2. **Build the gap map.** $A$ = Wapman; $E$ = Scorecard + PSEO; per-field gap with the full reliability framework (signal fraction, bootstrap CI), at undergraduate level and — where PSEO allows — PhD level.
3. **Test P1.** Gap vs within-field earnings dispersion (observability), on reliability-passing fields.

*Go criterion:* the cross-field map is reliably measured and P1 holds (or holds at one level). If the map is sound but P1 fails, that is a real but weaker "frame + puzzle" result, evaluated against the PI's stated bar.

### Tier 1 — the dynamic out-of-sample test (weeks, only if Tier 0 GO)
4. **Rebuild $A$ for 2021–2026** from Yifeng's resolved ORCID edges (academic-only; the feasible part of the pipeline). Validate against Wapman on the 2017–2020 overlap.
5. **Two-period gap.** Compute the gap for the Wapman era and for 2021–2026 (a two-point comparison — no smooth yearly series needed, which is more robust to year-noise).
6. **Out-of-sample compression test (H4).** From early-period structure + theory, label the fields predicted to compress (emerging, high-gap, skill-not-yet-learned). Test on 2020–2026 whether those fields' gaps closed while mature fields' did not. Report as a genuine predictive validation, with the confound controls of §8.

### Tier 2 — causal identification (Paper 2, optional)
A natural experiment that exogenously shifts the industry-side skill signal (a MacLeod-style exit-exam / certification shock) would push "consistent with" to "causes." No clean US-field instrument is in hand; this is explicitly deferred and not required for the descriptive/predictive paper.

---

## 8. Identification and inference

- **Generated regressors.** SpringRank $A$ (and any model-fit quantity) is an *estimated* right-hand-side variable. Standard errors must account for first-step estimation (Pagan 1984 two-step; right-level clustering). Senior econometric input is load-bearing here.
- **Out-of-sample logic.** The compression set is fixed from the early period *before* seeing late-period gaps; the late period is a true hold-out. This is identification by prediction, not by within-sample fit.
- **Confounds for compression.** Field growth, cohort-composition shifts, and changing PSEO/ORCID coverage could mimic compression. Controls: field size/growth, composition adjustment, and coverage-stability checks; placebo on mature fields predicted *not* to move.
- **Reliability gating.** Only fields above the signal-fraction threshold enter H2/H4, so noise-dominated fields cannot drive results.
- **Two-source stitching.** Wapman (AARC census) and the ORCID rebuild are different sources; the overlap-window calibration controls source-induced level shifts, reported transparently.

---

## 9. Honest status of prior empirical work

| Component | Status |
|---|---|
| Gap is a real, reliably measured object | **Established** (diagnostic) |
| CS low-gap / biology high-gap anchors | **Established** |
| Reliability framework (null/bootstrap/residual) | **Built, reusable** |
| Task-distance mechanism | **Dead, dropped** |
| Cross-field gap variance explained by tested correlates | ~20%; theory-adjacent ones reversed |
| Theory-disciplined integration frame + P1 | New; P1 has initial support |
| Dynamic out-of-sample compression (H4) | The central open bet |

The intellectually honest summary: a real phenomenon, a theoretically meaningful construct, an initial static prediction with the correct sign, and a dynamic prediction that has not yet been tested. The dynamic out-of-sample test is what would lift this from "solid descriptive" to "Nature-tier," and it may fail.

---

## 10. Senior faculty guidance — where it is load-bearing

The PI's current relationships are in NLP (Glavaš, Hovy), not economics. For this project the binding need is **labor / information economics**, in three specific places:

1. **Theory formalization (§4) — the priority.** Turning the signaling + employer-learning sketch into a model from which P1 (gap ↔ observability) and P2 (compression) *provably* follow as comparative statics. The PI can draft this from Altonji–Pierret (the learning model) and Spence/MacLeod (the signaling and reputation pieces), but a theorist should validate the derivation before submission — at a top venue, "the theory predicts X" must be a derived result, not a verbal analogy. *Best fit: an information-economics theorist (signaling).*
2. **Generated-regressor and out-of-sample inference (§8).** Correct two-step standard errors, clustering level, and the identification-by-prediction argument. *Best fit: an applied microeconometrician.*
3. **Framing and target-journal positioning.** Where to pitch (behavioural-science vs economics audience) and how to frame the integration construct for each.

**Concrete Bocconi candidates** (by role, from earlier scan):
- **Pamela Giustinelli** (education economics, applied microeconometrics, human capital; *EJ* associate editor) — strongest single fit; natural first approach for the empirical-theory bridge and inference.
- **Christoph Carnehl** (information economics, signaling-adjacent) — for the model formalization in §4.
- **Thomas Le Barbanchon** (applied labor econometrics, job search) — for inference and labor-market framing.

External early readers already engaged on adjacent work — **Eric Zitzewitz** (Dartmouth; rankings/incentives) and **Lin Peng** (Baruch) — are more topically relevant to this paper than the PI's NLP supervisors and are sensible informal reviewers for the framing.

**Practical move:** complete Tier 0, then approach Giustinelli with a 2-page version + the reliability-gated gap map figure. A clean figure plus a sharp question converts far better than a cold proposal.

---

## 11. Division of labor

- **Sora (lead):** theory (with senior validation), the ER side (Scorecard + PSEO), gap construction, reliability framework, static and dynamic analysis, write-up.
- **Yifeng Li:** the ORCID / academic-reputation pipeline and the time-series infrastructure, building on his resolved academic-mobility-edges dataset (Zenodo 19651302). Owns the 2021–2026 $A$ rebuild and the overlap validation. (Verify first that the edges include education→employment links with degree/employment years and a field tag — joining OpenAlex publication fields if needed.)

---

## 12. Contribution and target journals

**Contribution.** (i) The first comprehensive, fully open, reliability-gated measurement of academic-vs-labor-market reputation divergence across US fields; (ii) a signaling/employer-learning interpretation of that divergence as a measure of market integration; (iii) a dynamic, **out-of-sample-validated** test of employer learning at the level of fields.

**Targets.**
- *Nature Human Behaviour* / *PNAS* — if the dynamic out-of-sample compression holds (the predictive-validation result is the draw).
- *Science Advances* — for the clean comprehensive descriptive + reliability characterization, Wapman-style, if the dynamics are inconclusive.
- *AEJ: Applied* / *Journal of Labor Economics* — for the economics audience, foregrounding the model and identification.

---

## 13. Risk register (honest)

- **PSEO coverage truncation** — public-system skew, missing elite privates, suppressed PhD cells. *Probed in Tier 0; gates the project.*
- **ORCID recent-faculty lag** — 2024–2026 hires under-recorded; the very recent tail is thin. *Mitigation: weight the 2021–2023 core; treat the tail cautiously.*
- **Compression may not exist** — P2 is an empirical bet; the data may not move as theory predicts.
- **P1 may fail on clean data** — leaving a "frame + puzzle," weaker than a confirmed prediction.
- **Two-source comparability** — controlled by overlap calibration, but a residual risk.

The mitigation philosophy throughout: **cheap steps gate expensive ones.** Tier 0 answers, in days, whether the project can reach the PI's bar — before any multi-week ORCID rebuild is committed.

---

## 14. Key references

- Spence (1973), Job Market Signaling, *QJE*.
- Altonji & Pierret (2001), Employer Learning and Statistical Discrimination, *QJE*.
- MacLeod, Riehl, Saavedra & Urquiola (2017), The Big Sort, *AEJ: Applied*.
- Wapman, Zhang, Clauset & Larremore (2022), Quantifying hierarchy and dynamics in US faculty hiring and retention, *Nature* 610:120–127.
- Clauset, Arbesman & Larremore (2015), Systematic inequality and hierarchy in faculty hiring networks, *Sci. Adv.*
- De Bacco, Larremore & Moore (2018), A physical model for efficient ranking in networks, *Sci. Adv.* (SpringRank).
- Jiang et al. (2026), ORCID + OpenAlex academic-placement methodology, *Humanit. Soc. Sci. Commun.* (arXiv 2401.12739).
- US Census Bureau, Post-Secondary Employment Outcomes (PSEO), LEHD.
- US Dept. of Education, College Scorecard Field-of-Study data.
- NSF NCSES, Survey of Doctorate Recipients (SDR); NCES, Baccalaureate & Beyond (B&B).
- Li, Y. (2026), ORCID-Derived Academic Mobility Edges with Resolved Cities and Organizations, Zenodo 10.5281/zenodo.19651302.
