# jiang2024decoding
**Full citation:** Tian, C., Jiang, X., Huang, Y., Ma, L., & Ma, Y. (2024). Decoding University Hierarchy and Prestige in China through Domestic Ph.D. Hiring Network. arXiv preprint arXiv:2401.12739 [cs.DL]. (Preprint of jiang2026mapping, published 2026 in Humanities and Social Sciences Communications.)
**DOI / URL:** arXiv:2401.12739 — https://arxiv.org/abs/2401.12739 (PDF: https://arxiv.org/pdf/2401.12739)
**Access level:** full text (read the complete arXiv PDF, including Section 3 "Data and Methods")

## What it contributes to THIS project
This is THE primary method we replicate to extend the US Academic-Reputation (AR) panel to 2026. It is the recipe-grade source for building a faculty-hiring prestige network entirely from open data (ORCID + OpenAlex) — exactly the pipeline our project needs because Wapman et al. 2022 (our validation target) used a proprietary AARC census that ends in 2017, whereas ORCID+OpenAlex are open and current. The paper demonstrates the full chain: ORCID employment/education records -> OpenAlex linkage and institution-type filtering -> a directed PhD->faculty hiring network -> a network prestige ranking per discipline. Crucially for our gap study, the authors themselves compare their network-based prestige rank against QS Academic Reputation AND QS Employer Reputation ranks (Figure S5) — a direct conceptual precedent for our AR-vs-ER gap. One caveat the team must heed: the paper ranks with **Minimum Violation Rankings (MVR)**, not SpringRank; our project specifies SpringRank, so we substitute the ranking step while reusing the entire data-construction pipeline upstream of it.

## Specific equation / result / dataset we reuse

### THE LINKAGE RECIPE (code this exactly; ranking step swapped for SpringRank)

**(0) Data vintages and headline numbers.**
- Primary datasets: **ORCID as of 2022** and **OpenAlex as of 2023** (Section 3.1, verbatim: "Our primary datasets are sourced from ORCID (as of 2022) and OpenAlex (as of 2023)").
- Integrated cohort: **~23,994 individuals** who earned PhDs in Chinese universities and continued academic activity at Chinese institutions **between 1990 and 2020**.
- Network size: **501 academic units** at the department/school level (N = 501). Adjacency matrix is N×N.
- Faculty production grew 59 (1990) -> 1,914 (2017), slowing to 966 (2020); the 2020 dip is attributed to "incomplete data updates."
- For our US replication these counts are the order-of-magnitude target and the field-partition template; the absolute numbers will differ for the US.

**(1) ORCID record fields used.**
- Pull, for every scholar, three ORCID record sections: **employment records, education records, and publication (works) data**.
- From ORCID, the two join keys are explicitly named: **each institution's RINGGOLD ID** (on employment/education affiliations) and **each work's DOI** (on the works section). Verbatim: ORCID "provides each institution's RINGGOLD ID and each work's DOI — to link these records to OpenAlex."
- Education records supply the **doctoral (PhD-granting) institution + year of graduation**; employment records supply **subsequent academic employment** (the hiring side of the edge). (The paper does not spell out start/end-year parsing field-by-field, but the directed edge construction below implies it uses graduation year from education and the employing institution from employment.)
- Scope filter at collection time: scholars who **earned doctorates in China AND were subsequently employed by Chinese institutions** (for our project: substitute "US" — PhD from a US institution, first faculty job at a US institution).

**(2) ORCID -> OpenAlex join.**
- OpenAlex "assigns each scholar a unique author ID along with her/his corresponding PID [ORCID iD]," which is what allows the two databases to be integrated. So the person-level join is **ORCID iD <-> OpenAlex author ID**.
- The work-level / institution-level join uses **work DOI** and **RINGGOLD ID** carried over from ORCID into OpenAlex's institution IDs.
- OpenAlex is used to **supplement missing author/affiliation information** that is absent in ORCID ("supplement missing author information in ORCID with the institutional details from OpenAlex").
- From OpenAlex they also extract each scholar's **publication and citation records** (to confirm academic activity/performance) and the **field of study**, categorized per OpenAlex's **19 discipline classifications** (concept/field labels).

**(3) Institution-type FILTERING.**
- OpenAlex categorizes institutions into **eight types: education, company, healthcare, government, facility, archive, nonprofit, and other.**
- Keep only **type = "Education"**; explicitly remove other types ("We further filter the academic institutions via OpenAlex's institution type 'Education', removing other types like 'company'"). This is the keep-Education / drop-company-government step our spec calls for.

**(4) Directed hiring EDGE construction.**
- Nodes = universities/academic units. A directed edge (i, j) means: a faculty member **at university j earned their doctorate from university i** (PhD-granting institution i -> faculty-employing institution j).
- Edge weight m_ij = **the number of PhDs graduated from university i and hired by university j**.
- "First faculty job" is operationalized as the scholar's **subsequent academic employment after the PhD** (the education record gives alma mater + graduation year; the employment record at a type=Education institution gives the hiring node). The placement trajectory is the transition "from the doctoral student's alma mater to their subsequent employing institution(s)." (Note for the team: the preprint does not give an explicit rule for picking the *earliest* faculty employment when multiple employments exist — we should impose "min start-year employment at a type=Education institution after PhD year" as our concrete first-faculty-job rule.)

**(5) Ranking step + field partitioning.**
- **Ranking (paper's method — we swap this):** Minimum Violation Rankings (MVR), following Clauset, Arbesman & Larremore (2015). A "violation" is an edge (i,j) where rank of j surpasses rank of i. MVR finds the vertex permutation π minimizing violations, equivalently maximizing net downward-edge weight:
  - S_{π(M)} = Σ_{ij} M_ij × sign[π(j) − π(i)]
  - Hierarchy strength ρ = fraction of downward-pointing edges (π_i ≤ π_j), optimized over rankings. ρ = 1/2 = no hierarchy; ρ = 1 = perfect hierarchy.
  - Solved by MCMC (Metropolis–Hastings, zero temperature), 100,000 iterations; initial ordering by out-degree; sample rankings with highest ρ; each university's prestige score = its mean rank across the sampled set (low score = high prestige).
  - **Empirical fit:** Gini of faculty production = 0.78 (22% of universities produce ~78% of faculty), comparable to Clauset et al.'s North-American 0.62–0.76. Gini rose over time: 0.63 (1990–2000), 0.73 (2001–2010), 0.76 (2011–2020).
  - **OUR SUBSTITUTION:** replace MVR with **SpringRank** on the same weighted adjacency matrix M (per project spec). The upstream data pipeline (steps 1–4) is reused unchanged.
- **Field partitioning:** discipline assigned from OpenAlex's **19 discipline classifications**; the consensus ranking and all hierarchy analyses are computed **per discipline** (paper highlights Material science, Computer science, Chemistry, Biology as the four largest). This per-field partition is exactly what our project needs to compute Gap_field = 1 − Spearman(prestige_rank, ER_rank) within each field. (For our US replication, partition by CIP/field rather than OpenAlex's 19 classes if the CIP join key is needed.)

**(6) Direct precedent for our gap metric.** The authors compare their network prestige ranking against **QS Academic Reputation and QS Employer Reputation** ranks (Figure S5), explicitly noting MVR "captures information that other rankings do not." This is the conceptual seed of our AR-vs-ER gap, validating that network-prestige and employer-reputation are non-identical signals worth differencing.
