# Degrees of Separation
### Task Distance and the Academic–Employer Reputation Gap

*Research proposal — v0.1*

---

## Summary

The same field of study is priced by two different labor markets. The **academic market** rewards a department through faculty placement and peer assessment; the **industry market** rewards its graduates through wages and hiring demand. Both prices are reputation signals attached to the same underlying field-specific human capital, yet they need not agree. Georgia Tech computer science is prestigious in both markets at once; a French *grande école* commands enormous employer prestige with comparatively thin research standing; economics trains its strongest students for an academic frontier that investment banks do not directly pay for.

This proposal treats the **gap between academic reputation (AR) and employer reputation (ER)** as the central object of study, measured at the level of academic fields. Grounded in signaling theory and the employer-learning literature, the gap is interpreted as a *wedge between two signaling equilibria* — a measure of how integrated or segmented a field's academic and industry labor markets are. The central mechanism is **task distance**: when the tasks of academic research in a field closely resemble the tasks of the jobs its graduates take, the two markets reward the same skills and the gap closes; when they diverge, the gap opens and employer reputation falls back on credentialing and network inertia.

The headline empirical claim is that **task distance predicts the AR–ER gap across fields, with computer science as a near-zero-gap extreme** (fully integrated markets) and credential-driven fields (law, parts of business, economics→finance) at the other end. The study is **US-only and built entirely on open data**, using a two-level design (PhD and undergraduate) and a panel extended to 2026 with a reproducible faculty-hiring-network pipeline.

---

## 1. Research question and motivation

**Core question.** Across academic fields, how far does academic prestige travel into the labor market, and what determines the distance it travels?

The motivating observations span the full range of the phenomenon:

- **Aligned markets (small gap).** Computer science: academic research at the frontier (machine learning, systems, algorithms) is the same skill set industry pays most for. Academic prestige and employer demand move together; the academic and industry markets are effectively one market.
- **Credential markets (large, possibly sign-flipped gap).** French *grandes écoles*: high employer reputation built on selective admission (*concours*), dense alumni networks, and historical recruiting relationships, with research standing that does not match. Economics is a softer version — academic distinction is real, but a large share of graduates enter finance and consulting, where the academic frontier is not the binding skill.
- **Periphery in both markets.** Fields where neither academic prestige nor employer demand is strong.

**Why it matters.** University rankings collapse these two reputations into a single number, obscuring the fact that the two markets can sharply disagree. For a student choosing a field, the gap is decision-relevant: in a zero-gap field, academic prestige is a reliable guide to labor-market reward; in a high-gap field, it is not. More fundamentally, the gap is a window onto the structure of two coupled markets and the rate at which information (about graduate quality) propagates from one to the other.

---

## 2. Positioning: three literatures that never met

The contribution sits precisely at the intersection of three literatures, each of which has done one half of the problem and stopped.

**Strand A — faculty hiring networks.** Clauset, Arbesman & Larremore (2015) and Wapman, Zhang, Clauset & Larremore (2022) built the modern academic-prestige object: a recursive, network-derived prestige score (via SpringRank) extracted from who hires whose PhD graduates as faculty, computed per field. Zhang et al. (2022) extended this to productivity. This strand has the cleanest available measure of academic reputation — but it stays *inside* academia and never connects to the labor market.

**Strand B — reputation and the labor market.** MacLeod, Riehl, Saavedra & Urquiola (2017) embedded *college reputation* in an employer-learning signaling model and, using a Colombian exit-exam rollout, showed that introducing a new individual skill signal reduces the return to institutional reputation. Their decomposition — reputation = student-body ability + institutional value-added — is the theoretical backbone of the present project. Altonji & Pierret (2001) supplies the employer-learning engine; MacLeod & Urquiola (2015) the reputation-formation model. This strand has the theory and the labor-market outcomes — but it is single-field, single-country, and never touches the faculty hiring network.

**Strand C — rankings research.** A large applied literature studies QS/THE/GEURS ranking indicators, including their academic-reputation and employer-reputation survey components. This strand only ever documents *static correlation* between the two reputation indicators and never models the academic-market side structurally, nor measures gaps at the field level.

**The open gap.** No work connects the faculty-hiring-network prestige object (Strand A) to the reputation-and-labor-market signaling framework (Strand B) at the level of fields. This project does exactly that, and adds the field-heterogeneity mechanism (task distance) that neither strand supplies.

---

## 3. Theoretical framework

The framework is a chain of four well-established results, assembled into a single claim about two markets.

**(i) Signaling (Spence 1973).** Employers cannot observe a graduate's productivity directly and use observable signals — including the prestige of the institution and field — to set wages. Arrow (1973) and Stiglitz (1975) give the filter/screening variants.

**(ii) Employer learning (Altonji & Pierret 2001; Farber & Gibbons 1996).** As employers accumulate information about a worker's true productivity, the weight on easily observed credentials falls and the weight on actual productivity rises. The implication for this project: **the AR–ER gap is, in part, the inverse of employer-learning speed.** A field whose gap stays large is one where the employer's signal remains stuck at the credential stage and is never replaced by learned productivity. Arcidiacono, Bayer & Hizmo (2010) refine when ability is revealed at hiring vs. learned over time.

**(iii) Reputation decomposition (MacLeod, Riehl, Saavedra & Urquiola 2017).** Institutional reputation carries two kinds of information — about the *ability of the student body* and about the *value-added of the institution*. This is exactly the project's two secondary axes: **admission selectivity** maps to student-body ability; **education quality** maps to value-added. The project's contribution is to extend this two-way decomposition across 107 fields and to add the academic-market side.

**(iv) Task-specific human capital (Gathmann & Schönberg 2010; Poletaev & Robinson 2008; Yi, Mueller & Stegmaier 2017).** Skills are portable across jobs to the extent that the jobs share tasks; workers who move to task-distant occupations suffer larger losses. This supplies the field-heterogeneity mechanism: **the distance between the task content of academic research and the task content of graduates' industry jobs governs how well the academic signal transmits to the industry market.**

**The synthesis.** Both the academic and industry labor markets are two-sided matching markets (Becker 1973; Shimer & Smith 2000; Eeckhout & Kircher 2011) that price the same underlying field-specific human capital. Each produces a reputation signal: the academic market's signal is faculty-hiring-network prestige; the industry market's signal is employer reputation. **The gap between the two signals is a wedge between two sorting/signaling equilibria, and task distance is the variable that governs the size of the wedge.** This turns the CS-vs-economics intuition into a falsifiable economic proposition.

---

## 4. Hypotheses

- **H1 (variation).** There is significant cross-field variation in the AR–ER gap. *(If false, no paper.)*
- **H2 (central mechanism).** Academia–industry task distance positively predicts the gap: larger task distance → larger gap.
- **H3 (employer learning).** The gap is larger in fields whose labor markets rely on credential signals and where employer learning about graduate productivity is slow.
- **H4 (headline).** Computer science is a low-gap outlier — academic and industry markets are integrated — while credential-driven fields (law, parts of business/finance) sit at the high-gap extreme.
- **H5 (dynamics).** Fields where industry demand for academic-frontier skills rose over the sample (machine learning, data science) show gap *compression* between 2011 and 2026; fields with stable task distance show stable gaps.

---

## 5. Data

All sources are open. The only construction cost is the faculty-hiring-network pipeline and the cross-source linking. The **CIP (Classification of Instructional Programs) code is the join key** that ties Scorecard, SDR, O\*NET, and IPEDS together.

### 5.1 Academic reputation (AR) — the academic-market signal

- **Primary (extends to 2026):** an ORCID + OpenAlex pipeline that reconstructs field-level faculty hiring networks (directed edges: PhD-granting institution → employing institution) and computes **SpringRank** prestige per field. This is the method validated by Jiang et al. / the China faculty-hiring papers (Humanities and Social Sciences Communications 2026; arXiv 2401.12739), which map 23,994 PhD holders from ORCID-as-of-2022 + OpenAlex-as-of-2023, filtering institutions to OpenAlex type `Education`. Reproducing this method gives a panel that runs to 2026 and is not confined to the US (useful for later extensions).
- **Benchmark / validation:** Wapman, Zhang, Clauset & Larremore (2022) anonymized network data on Zenodo (2011–2020, US, 107 fields, 368 institutions, SpringRank already computed). Use to **validate** the pipeline's SpringRank against a peer-reviewed gold standard on the 2011–2020 overlap before trusting the 2021–2026 extension.
- **Supplement:** OpenAlex field-normalized citations-per-institution as a second AR proxy, enabling a multi-proxy AR factor that reduces measurement error.

### 5.2 Employer reputation (ER) — behavioral, two levels

The key design choice is to use **behavioral** ER (what employers actually pay) rather than survey ER (QS/GEURS perceptions).

- **Undergraduate level:** US Department of Education **College Scorecard, Field of Study files** — institution × 4-digit CIP × credential level, median earnings at 1 / 4 / 5 years post-graduation, linked to IRS tax records and IPEDS. Free, updated through 2026.
- **PhD level:** NSF NCSES **Survey of Doctorate Recipients (SDR)** — employer **sector** (academia / industry / government), occupation, and salary by fine field of degree. The academia-vs-industry employment split is itself a direct measure of market integration and is central to H2/H4. Public-use cross-sectional files are free; finer field × sector cells available via the NCSES restricted-data license if needed. The **Survey of Earned Doctorates (SED)** provides the sampling frame and new-cohort post-graduation commitments/salaries.

### 5.3 Task distance — the mechanism variable

- **O\*NET database** (free) + the **CIP↔SOC crosswalk** (O\*NET Resource Center; NCES-based). Build per-field task/work-activity/skill vectors and compute a **Gathmann–Schönberg-style distance** between (a) the academic-research task vector for the field and (b) the modal industry-job task vector for that field's graduates.
- **Cross-checks** (all constructable from OpenAlex/ORCID): academia↔industry **affiliation-switch rate**, **industry co-authorship share**, and **patent density** per field.

### 5.4 Controls / secondary-finding axes

- **Admission selectivity:** IPEDS / Common Data Set acceptance rate + SAT/ACT mid-50% (US). Maps to MacLeod et al.'s *student-body ability*.
- **Education quality (value-added):** Scorecard earnings residualized on entry selectivity + field + year. Maps to MacLeod et al.'s *value-added*.
- **Field structure:** Boyack–Klavans–Börner citation proximity (map of science) as a continuous "how interdisciplinary/applied" covariate; Wapman's 8-domain taxonomy as the primary clustering for analysis.

### 5.5 The 2026 data extension — methodology

Concretely, to extend the academic-reputation panel beyond Wapman's 2020 cutoff:

1. Pull employment + education records from **ORCID** (researcher PIDs, affiliations with start/end years, RINGGOLD IDs).
2. Link to **OpenAlex** via DOI/RINGGOLD to recover institution IDs, types, and field labels; supplement missing affiliation data.
3. Filter producing/employing institutions to OpenAlex `type = Education`; drop `company`, `government`, etc. (industry destinations are captured separately via SDR, not in the academic network).
4. Construct, per field, a directed hiring network: edge (PhD institution → first faculty-employing institution).
5. Run **SpringRank** (De Bacco, Larremore & Moore 2018) per field to obtain prestige scores.
6. **Validate** against Wapman SpringRank on 2011–2020; report rank correlation by field.
7. Extend the panel through 2026; this becomes the time-series substrate for H5.

---

## 6. Empirical strategy

**Unit of analysis:** field × (institution) × year, with a two-level structure (field-level and institution-within-field).

**Gap construction.** For each field, the gap is the disagreement between the academic-prestige ordering of institutions and the labor-market-reward ordering of institutions:

> `Gap_field = 1 − Spearman( SpringRank_prestige_rank , ER_rank )`  within field, across institutions,

where `ER_rank` is built from Scorecard earnings (undergraduate level) and/or SDR salary + sector mix (PhD level). A two-level random-effects specification separates a field-level gap component from institution-level noise (mirroring the cross-model structure used in the author's SalienceDx benchmark).

**Main regression.**

> `Gap_field ~ task_distance + employer_learning_proxy + selectivity + value_added + field_structure + year_FE`

with H2 testing the `task_distance` coefficient and H4 checking that CS lies in the low-gap tail of the fitted relationship.

**Dynamics (H5).** With the 2011–2026 panel, estimate field-specific gap trajectories and test whether fields with rising industry demand for academic skills (ML/DS proxies) exhibit gap compression.

---

## 7. Tier 0 go/no-go (run first)

Before building any pipeline, test the most failure-prone assumptions in under a week using **only already-public, already-computed data**.

**Data:** Wapman 2022 Zenodo field-level SpringRank (no pipeline needed) + College Scorecard Field-of-Study earnings + SDR public tables (median salary and % academia/industry by field).

**Protocol:** For 15–20 fields spanning the range (CS, pure math, theoretical physics, economics, English, EE, biology, chemistry, …), compute `Gap_field = 1 − Spearman(prestige_rank, earnings_rank)` across institutions within field; regress on the SDR industry-employment share (a quick task-distance proxy); locate CS.

**Go conditions (all three required):**
1. **Variation:** the cross-field SD of `Gap_field` is non-trivial.
2. **Mechanism direction:** `Gap_field` correlates with the integration proxy in the predicted (negative) direction.
3. **Headline:** CS sits in the low-gap tail, significantly below the median.

**No-go** on any failure → stop and rethink before investing in the full pipeline.

**Known Tier-0 caveats (tolerable now, must be fixed in the full study):** undergraduate-earnings vs PhD-prestige level mismatch; small institution counts in some fields; earnings confounded by geography/cost-of-living.

---

## 8. Identification and robustness

- **Reduced-form core, theory-disciplined.** The signaling/employer-learning framework is the *lens*; the empirical core measures the gap and regresses it on task distance. This is correlational but theory-grounded — the appropriate first-paper standard.
- **Optional causal extension (Paper 2 or robustness section).** Exploit an exogenous introduction of a new individual skill signal in a field (analogue of MacLeod et al.'s exit-exam rollout — e.g., the emergence of a standardized competence signal such as competitive-programming/Kaggle ranks in CS, or the pre-doctoral institution in economics) and test whether the gap compresses. This moves H2/H3 from correlation toward causal identification.
- **Robustness:** level alignment (undergraduate vs PhD); region/cost-of-living controls; multi-proxy AR factor; alternative field clusterings (Wapman domains vs Boyack–Klavans citation clusters); placebo fields.

---

## 9. Contribution and target journals

**Contribution.** The first field-level map of the academic–employer reputation wedge, grounded in employer-learning signaling theory, with task distance as the governing mechanism and computer science as an extreme validation point (near-zero gap, fully integrated markets).

**Targets.**
- With the dynamics panel and/or the causal extension: *Nature Human Behaviour* or *PNAS* (the home of the Wapman/Clauset strand).
- Clean theory-disciplined reduced-form with strong descriptive mapping: *Science Advances*.
- For an economics audience that speaks signaling natively: *AEJ: Applied Economics* or *Journal of Labor Economics*.

---

## 10. References

### Signaling, screening, and employer learning
- Spence, A. M. (1973). Job Market Signaling. *Quarterly Journal of Economics*, 87(3), 355–374.
- Arrow, K. J. (1973). Higher Education as a Filter. *Journal of Public Economics*, 2(3), 193–216.
- Stiglitz, J. E. (1975). The Theory of "Screening," Education, and the Distribution of Income. *American Economic Review*, 65(3), 283–300.
- Altonji, J. G., & Pierret, C. R. (2001). Employer Learning and Statistical Discrimination. *Quarterly Journal of Economics*, 116(1), 313–350.
- Altonji, J. G., & Pierret, C. R. (1996/1998). Employer Learning and the Signaling Value of Education. NBER WP 5438 / in *Internal Labour Markets, Incentives and Employment*.
- Farber, H. S., & Gibbons, R. (1996). Learning and Wage Dynamics. *Quarterly Journal of Economics*, 111(4), 1007–1047.
- Arcidiacono, P., Bayer, P., & Hizmo, A. (2010). Beyond Signaling and Human Capital: Education and the Revelation of Ability. *AEJ: Applied Economics*, 2(4), 76–104.

### Reputation, college quality, and labor-market outcomes (keystone strand)
- MacLeod, W. B., Riehl, E., Saavedra, J. E., & Urquiola, M. (2017). The Big Sort: College Reputation and Labor Market Outcomes. *AEJ: Applied Economics*, 9(3), 223–261. (NBER WP 21230.)
- MacLeod, W. B., & Urquiola, M. (2015). Reputation and School Competition. *American Economic Review*, 105(11), 3471–3488.
- Machado, C., Reyes, G., & Riehl, E. (2022). Alumni Job Networks at Elite Universities and the Efficacy of Affirmative Action. IZA DP 15026.
- Sekhri, S. (2020). Prestige Matters: Wage Premium and Value Addition in Elite Colleges. *AEJ: Applied Economics*, 12(3), 207–225.
- Hussain, I., McNally, S., & Telhaj, S. (2009). University Quality and Graduate Wages in the UK. IZA/CEP.
- Black, D. A., & Smith, J. A. (2006). Estimating the Returns to College Quality with Multiple Proxies for Quality. *Journal of Labor Economics*, 24(3).
- Hastings, J. S., Neilson, C. A., & Zimmerman, S. D. (2013). Are Some Degrees Worth More than Others? Evidence from College Admission Cutoffs in Chile. NBER WP 19241.
- Chetty, R., et al. (2023). Diversifying Society's Leaders? The Determinants and Causal Effects of Admission to Highly Selective Colleges. NBER WP 31492.
- Lee, J.-W., & Lee, H. (2024/2026). Higher Education Quality, Income, and Innovation: Cross-Country Evidence.

### Task-specific human capital and skill transferability (mechanism)
- Gathmann, C., & Schönberg, U. (2010). How General Is Human Capital? A Task-Based Approach. *Journal of Labor Economics*, 28(1), 1–49.
- Poletaev, M., & Robinson, C. (2008). Human Capital Specificity: Evidence from the DOT and Displaced Worker Surveys, 1984–2000. *Journal of Labor Economics*, 26(3).
- Yi, M., Mueller, S., & Stegmaier, J. (2017). Transferability of Skills across Sectors and Heterogeneous Displacement Costs. *American Economic Review: P&P*, 107(5), 332–336.
- (Field-of-study alignment-score approach.) Does the Labour Market Value Field-of-Study-Specific Knowledge? (2024). *Economics of Education Review*.

### Faculty hiring networks and academic prestige (AR object + 2026 method)
- Wapman, K. H., Zhang, S., Clauset, A., & Larremore, D. B. (2022). Quantifying Hierarchy and Dynamics in US Faculty Hiring and Retention. *Nature*, 610, 120–127. Data: Zenodo 10.5281/zenodo.6941651.
- Clauset, A., Arbesman, S., & Larremore, D. B. (2015). Systematic Inequality and Hierarchy in Faculty Hiring Networks. *Science Advances*, 1(1), e1400005.
- Zhang, S., Wapman, K. H., Larremore, D. B., & Clauset, A. (2022). Labor Advantages Drive the Greater Productivity of Faculty at Elite Universities. *Science Advances*. Data: Zenodo 10.5281/zenodo.7126263.
- De Bacco, C., Larremore, D. B., & Moore, C. (2018). A Physical Model for Efficient Ranking in Networks (SpringRank). *Science Advances*, 4(7), eaar8260.
- **[2026 methodology — primary for the data extension]** Jiang, F., et al. (2026). Mapping University Prestige and Hierarchy in China via Faculty Hiring Networks of Internationally Active Ph.D.s. *Humanities and Social Sciences Communications*. (See also arXiv 2401.12739, "Decoding University Hierarchy and Prestige in China through Domestic Ph.D. Hiring Network.") — ORCID + OpenAlex pipeline; the reproducible method for extending the panel to 2026.
- (Interdisciplinary PhD placement.) arXiv 2503.21912 — AARC + ProQuest linkage example.
- Jones, B., et al. (2025). Tenure and Research Trajectories. *PNAS*, 122(30), e2500322122 — AARC data-access example.

### Maps of science / field clustering
- Boyack, K. W., Klavans, R., & Börner, K. (2005). Mapping the Backbone of Science. *Scientometrics*, 64(3), 351–374.
- Klavans, R., & Boyack, K. W. (2009). Toward a Consensus Map of Science. *JASIST*, 60(3), 455–476.
- Boyack, K. W., & Klavans, R. (2014). Creation of a Highly Detailed, Dynamic, Global Model and Map of Science. *JASIST*, 65(4), 670–685.

### Two-sided matching and sorting
- Becker, G. S. (1973). A Theory of Marriage: Part I. *Journal of Political Economy*, 81(4).
- Shimer, R., & Smith, L. (2000). Assortative Matching and Search. *Econometrica*, 68(2), 343–369.
- Eeckhout, J., & Kircher, P. (2011). Identifying Sorting—In Theory. *Review of Economic Studies*, 78(3), 872–906.
- Eeckhout, J., & Kircher, P. (2018). Assortative Matching with Large Firms. *Econometrica*, 86(1), 85–132.
- Abowd, J. M., Kramarz, F., & Margolis, D. N. (1999). High Wage Workers and High Wage Firms. *Econometrica*, 67(2), 251–333.

### Credentialism and elite hiring (anomaly case)
- Rivera, L. A. (2015). *Pedigree: How Elite Students Get Elite Jobs*. Princeton University Press.
- Hungerford, T., & Solon, G. (1987). Sheepskin Effects in the Returns to Education. *Review of Economics and Statistics*, 69(1), 175–177.

### Data infrastructure
- U.S. Department of Education. College Scorecard, Field of Study data. collegescorecard.ed.gov/data
- NSF NCSES. Survey of Doctorate Recipients (SDR) and Survey of Earned Doctorates (SED). ncses.nsf.gov
- O\*NET Resource Center. O\*NET database + CIP↔SOC crosswalk. onetcenter.org
- Priem, J., Piwowar, H., & Orr, R. (2022). OpenAlex: A Fully-Open Index of Scholarly Works, Authors, Venues, Institutions, and Concepts. arXiv 2205.01833.
- ORCID. orcid.org

### Rankings methodology (reputation-survey critique)
- QS World University Rankings by Subject — methodology (academic and employer reputation indicators).
- Times Higher Education / Emerging. Global Employability University Ranking and Survey (GEURS) — methodology.
- (Critical analysis.) Unpacking the Metrics: a Critical Analysis of the 2025 QS World University Rankings (2025). *Frontiers in Education*.

---

*Note: economics is not classified as "behavioral vs mechanistic"; the relevant axis is reduced-form vs structural/causal. The signaling/employer-learning framework above is the theoretical mechanism. The empirical core is reduced-form (theory-disciplined); the causal extension in §8 is optional and may form a second paper.*
