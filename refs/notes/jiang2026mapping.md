# jiang2026mapping
**Full citation:** Tian, C., Jiang, X., Huang, Y., Ma, L., & Ma, Y. (2026). Mapping university prestige and hierarchy in China via faculty hiring networks of internationally active Ph.D.s. Humanities and Social Sciences Communications, 13, Article 379.
**DOI / URL:** https://doi.org/10.1057/s41599-026-06717-y — https://www.nature.com/articles/s41599-026-06717-y (received 2024-12-12, accepted 2026-02-07)
**Access level:** abstract only for the published Nature/Palgrave HTML (redirects to idp.nature.com paywall/SSO); however the **full method is read in full via the open-access preprint** (arXiv:2401.12739, see citekey jiang2024decoding), which is the same pipeline by the same authors with the same 23,994 cohort. Recipe details below are lifted from that preprint full text; headline figures confirmed against the published abstract.

## What it contributes to THIS project
This is the peer-reviewed published version of the open-data faculty-hiring-network pipeline we replicate to extend the US AR panel to 2026 (the preprint is jiang2024decoding). It establishes that a credible institutional-prestige hierarchy can be reconstructed end-to-end from ORCID + OpenAlex alone — no proprietary census — which is precisely why our project can build a 2026-current US AR layer rather than stopping at Wapman et al. 2022's 2017 endpoint. The published title's emphasis on "internationally active Ph.D.s" reflects that the ORCID/OpenAlex universe is biased toward research-publishing (STEM-heavy) scholars; we must carry this selection caveat into our US replication, especially for credential fields (law, finance) that publish less and are under-covered. As the journal-of-record citation, this is what we cite for the method; the arXiv preprint is what we code against.

## Specific equation / result / dataset we reuse
We reuse the identical linkage recipe documented in detail in the note for jiang2024decoding. In brief, the concrete artifacts we lift:

- **Data vintages:** ORCID **as of 2022**, OpenAlex **as of 2023**; cohort **~23,994** PhD holders, **1990–2020**, **N = 501** academic units (these exact figures appear in both the published abstract and the preprint).
- **Join keys:** ORCID iD <-> OpenAlex author ID (person level); **work DOI** and institution **RINGGOLD ID** carried from ORCID into OpenAlex (work/institution level); OpenAlex supplements missing ORCID affiliations.
- **Institution-type filter:** keep OpenAlex type **"Education"** out of the eight types (education, company, healthcare, government, facility, archive, nonprofit, other); drop company/government/etc.
- **Directed edge (i,j):** PhD-granting institution i -> first faculty-employing institution j; weight m_ij = count of i-trained PhDs hired by j; N×N adjacency matrix M.
- **Field partition:** OpenAlex's **19 discipline classifications**; rankings computed per discipline.
- **Ranking step (NOTE — paper uses MVR, our spec uses SpringRank):** the paper applies **Minimum Violation Rankings** (Clauset et al. 2015), maximizing S = Σ_ij M_ij·sign[π(j)−π(i)] via MCMC (100k iterations, zero-temperature Metropolis–Hastings), prestige score = mean sampled rank. Our project substitutes **SpringRank** on the same matrix M while reusing steps above unchanged.
- **Empirical anchor:** Gini of faculty production = 0.78; per-decade Gini 0.63/0.73/0.76 — a sanity-check benchmark for inequality once we build the US network.
- **AR-vs-ER precedent:** the authors benchmark their network prestige against **QS Academic Reputation and QS Employer Reputation** ranks (supplementary Figure S5), the direct conceptual antecedent of our Gap_field = 1 − Spearman(prestige_rank, ER_rank).
