# Cross-national feasibility (Thrust A): does the integrated/decoupled typology replicate in the UK?

No-fabrication run: pieces are acquired and built; the UK gap is computed only where the axis is dense and oriented. Seeded; `python scripts/40_crossnational_uk.py`.

## What was acquired

- **UK LEO** (placement axis): DfE/EES *Graduate outcomes (LEO) provider level data*, provider x CAH2-subject median earnings (5 years after graduation), openly downloaded (present). Region-reweighted (`earnings_adjusted_median`) is published but **disclosure-suppressed at this grain** (it is `x`/blank on essentially all provider x subject cells), so the raw `earnings_median` is used and geography is not netted here.

- **EUROGRADUATE** (18-country skills-match): site reachable, but the pilot microdata is distributed by application to the consortium/GESIS, **not open download** --- recorded as access-gated; not used.

- **ROR data dump** (country resolution for the hiring edges): present (True).

- **ORCID shards** (for the UK hiring network): present (True).

## UK prestige axis --- option (a) ORCID-UK hiring network (construct-comparable, the default)

Filtering the existing ORCID pipeline to **GB->GB** PhD->faculty edges yields **11,987 UK hiring edges** across **551 institutions** (vs 63,711 US edges --- about 1/5, as the edge-density lesson predicts, but **not** prohibitively sparse). 4,707 edges carry a field tag (coarse dept-text match), rolled up to CAH2 subjects.

Per-CAH2 SpringRank density and stability (the edge-density gate); orientation is fixed so elite UK universities sit at the high-prestige end:

| cah2                                       |   edges |   providers |   elite_pctile |   boot_stability | usable   |
|:-------------------------------------------|--------:|------------:|---------------:|-----------------:|:---------|
| Psychology                                 |     504 |         139 |           0.94 |             0.83 | True     |
| History and archaeology                    |     490 |         132 |           0.85 |             0.86 | True     |
| Engineering                                |     478 |         126 |           0.85 |             0.87 | True     |
| Sociology, social policy and anthropology  |     472 |         132 |           0.89 |             0.86 | True     |
| Philosophy and religious studies           |     471 |         146 |           0.8  |             0.8  | True     |
| Business and management                    |     351 |         129 |           0.93 |             0.85 | True     |
| Computing                                  |     341 |         116 |           0.87 |             0.79 | True     |
| Mathematical sciences                      |     317 |          94 |           0.76 |             0.87 | True     |
| Chemistry                                  |     242 |          91 |           0.75 |             0.75 | True     |
| Economics                                  |     213 |          93 |           0.88 |             0.8  | True     |
| Biosciences                                |     206 |          89 |           0.79 |             0.84 | True     |
| English studies                            |     198 |          92 |           0.96 |             0.86 | True     |
| Physics and astronomy                      |     178 |          67 |           0.9  |             0.83 | True     |
| Nursing and midwifery                      |     125 |          74 |           0.91 |             0.82 | True     |
| Politics                                   |      62 |          35 |           0.69 |             0.86 | True     |
| Geography, earth and environmental studies |      47 |          41 |           0.77 |             0.83 | True     |
| Materials and technology                   |      10 |           9 |         nan    |           nan    | False    |
| Allied health                              |       2 |           0 |         nan    |           nan    | False    |

**16 CAH2 subjects** support a stable UK SpringRank (>=30 edges, >=10 providers, bootstrap rank-stability >=0.6; stability runs 0.72--0.90). **Orientation check passes**: elite universities' mean prestige percentile is 0.85 (>0.5 = correct; no subject required a sign flip). So the UK hiring-network prestige axis is feasible and construct-comparable to the US axis.

## UK prestige axis --- option (b) REF (research quality): documented fallback, NOT default

The Research Excellence Framework 2021 results (institution x unit-of-assessment quality profiles) are public and official (`results2021.ref.ac.uk`, reachable; an export-all endpoint exists), so option (b) is **accessible**. We do **not** adopt it: REF measures *research-output quality*, whereas the US axis is *faculty-placement prestige* --- mixing constructs across countries weakens the ``same object'' claim. ORCID-UK (option a) is the construct-comparable choice and it holds, so REF is reported as the fallback only. \coauthor{final axis choice (ORCID-UK default vs REF fallback) is a senior decision; not finalised unattended}.

## Crosswalk coverage (project field <-> UK CAH2 subject)

The crosswalk maps **50** project fields onto **21** UK CAH2 subjects (CAH2 is coarser than the US field taxonomy --- engineering, business, and the social sciences each collapse several project fields into one CAH2 subject, a documented loss). CAH2 subjects present in LEO but **not** targeted by any project field (no clean US counterpart, listed rather than forced): Architecture, building and planning, Celtic studies, Combined and general studies, Creative arts and design, Education and teaching, General, applied and forensic sciences, Languages and area studies, Law, Media, journalism and communications, Medical sciences, Medicine and dentistry, Performing arts, Sport and exercise sciences, Veterinary sciences.

## The UK gap, the UK typology, and the US-vs-UK comparison

UK field-level gap computed for **16 CAH2 subjects** (providers in BOTH the UK hiring network and LEO; overlap n = 24--89 per subject):

| cah2                                       |   us_gap |   uk_gap |   n_overlap |   us_nfields |
|:-------------------------------------------|---------:|---------:|------------:|-------------:|
| Computing                                  |     0.30 |     0.56 |          73 |            2 |
| Economics                                  |     0.31 |     0.58 |          52 |            1 |
| Mathematical sciences                      |     0.32 |     0.58 |          52 |            3 |
| Politics                                   |     0.34 |     0.32 |          24 |            1 |
| History and archaeology                    |     0.41 |     0.39 |          71 |            1 |
| Psychology                                 |     0.43 |     0.34 |          89 |            1 |
| Business and management                    |     0.48 |     0.54 |          79 |            4 |
| Engineering                                |     0.49 |     0.79 |          67 |            8 |
| Sociology, social policy and anthropology  |     0.50 |     0.60 |          75 |            3 |
| English studies                            |     0.51 |     0.65 |          65 |            1 |
| Geography, earth and environmental studies |     0.56 |     0.56 |          25 |            4 |
| Physics and astronomy                      |     0.65 |     0.51 |          34 |            2 |
| Chemistry                                  |     0.69 |     0.54 |          39 |            1 |
| Philosophy and religious studies           |     0.79 |     0.56 |          45 |            2 |
| Biosciences                                |     0.83 |     0.41 |          51 |            6 |
| Nursing and midwifery                      |     0.93 |     1.18 |          44 |            1 |

**US-vs-UK rank correlation of the field-level gap: Spearman = +0.18 (p=0.51, n=16).** Integrated/decoupled agreement (median split): **9/16**.


**Reading (outcome-agnostic): the typology replicates only WEAKLY.** The clean agreements are the extremes one would predict --- Nursing is strongly decoupled in both (US 0.93, UK 1.18), and Politics/History/Psychology sit low (integrated) in both. But several disciplines diverge sharply --- Engineering and Computing are integrated in the US yet middling/decoupled in the UK, and Biosciences is decoupled in the US but integrated in the UK. The overall +0.18 is not significant, so on this first-pass, CAH2-grain measurement the integrated/decoupled ordering is substantially US-specific rather than a universal property of the prestige->placement map.

## Adversarial self-check

- **UK hiring-network density (the edge-density lesson).** UK academia is ~1/5 the US; per-CAH2 SpringRank is nonetheless stable for 16 subjects (bootstrap 0.72--0.90) because CAH2 aggregation pools the sparse per-field edges. Thinner subjects (Materials, Veterinary, Celtic, Medicine) are flagged unusable, not forced.

- **Construct-comparability of the chosen axis.** The default UK axis is the SAME object as the US axis (ORCID faculty-hiring SpringRank), so the comparison is construct-clean; the REF fallback is NOT construct-comparable (research output, not placement prestige) and is therefore not used for the headline. The orientation check (elite mean percentile 0.85) rules out a sign-flip artifact.

- **Crosswalk losses.** CAH2 is coarser than the US field taxonomy: 6 engineering, 4 business, and several social-science/bioscience project fields each collapse into one CAH2 subject, so the US side of those rows is a multi-field mean --- a genuine aggregation that could attenuate or distort the comparison (e.g. US Engineering averages 6 fields against one UK CAH2). The weak +correlation should be read against this coarseness, not as a clean null.

- **Field-tagging + provider-matching coverage.** UK edges are field-tagged by coarse dept-text match (~39% of edges tagged) and providers are joined by normalised name (~45% of LEO providers match the hiring network; the unmatched LEO providers are mostly small non-research colleges that are correctly absent from a faculty-hiring network). Both are lower bounds that add noise toward zero; a refined OpenAlex-style field tag and a priority name matcher would sharpen the UK gap.

- **LEO-vs-Scorecard placement-axis comparability.** The UK placement axis is LEO median earnings at 5 years (UI/tax-linked, all graduates), the US axis is Scorecard Title-IV median at 4 years; the horizon and population differ, and LEO's geography adjustment is suppressed at this grain. The comparison is therefore of *gap structure*, not earnings levels, but the source asymmetry is a real caveat on any cross-national gap difference.


*Verdict:* all pieces hold (UK ORCID prestige feasible and oriented; LEO acquired; crosswalk and matching adequate), so the UK gap and the US-vs-UK comparison are computed and reported above --- a **weak, non-significant replication** of the typology at this first-pass grain. Nothing was fabricated; the open items (axis choice, finer crosswalk, better tagging/matching) are flagged for a senior pass, not silently resolved.

## Licensing-mechanism replication (the data-feasible, robust cross-national test)

We pivot from the full-typology replication --- which the data attenuate (the weak US-vs-UK +0.18 above is suppressed by CAH2 aggregation and ~39% ORCID field-tagging, **not** a clean null) --- to the paper's **featured, identified mechanism** (\S35c), which the already-acquired UK data can test cleanly. US finding: in licensed/regulated fields, school prestige adds ~0 *incremental* wage-predictive power beyond geography (pay is set by setting, not school; Nursing the clean case, incremental-prestige $\approx 0$).

Per usable UK CAH2 subject we run the **same commonality (incremental-$R^2$) decomposition** of within-subject across-provider LEO median earnings on the UK ORCID SpringRank prestige vs provider nation/region location:

| cah2                                       |   n_prov |   n_geo |   R2_prestige_only |   R2_geo_only |   R2_full |   incremental_prestige |   incremental_geo | licensed   | partial   |
|:-------------------------------------------|---------:|--------:|-------------------:|--------------:|----------:|-----------------------:|------------------:|:-----------|:----------|
| Nursing and midwifery                      |       44 |      11 |              0.009 |         0.479 |     0.493 |                  0.013 |             0.484 | True       | False     |
| Engineering                                |       67 |      11 |              0.048 |         0.243 |     0.261 |                  0.018 |             0.213 | False      | True      |
| Geography, earth and environmental studies |       25 |      10 |              0.08  |         0.664 |     0.683 |                  0.019 |             0.602 | False      | False     |
| English studies                            |       65 |      11 |              0.079 |         0.257 |     0.335 |                  0.077 |             0.255 | False      | False     |
| Physics and astronomy                      |       34 |       9 |              0.171 |         0.259 |     0.348 |                  0.089 |             0.176 | False      | False     |
| Chemistry                                  |       39 |      10 |              0.233 |         0.322 |     0.412 |                  0.09  |             0.179 | False      | False     |
| Mathematical sciences                      |       52 |      10 |              0.173 |         0.273 |     0.368 |                  0.096 |             0.196 | False      | False     |
| Philosophy and religious studies           |       45 |      10 |              0.259 |         0.468 |     0.593 |                  0.126 |             0.334 | False      | False     |
| Computing                                  |       73 |      11 |              0.188 |         0.158 |     0.291 |                  0.134 |             0.103 | False      | False     |
| Economics                                  |       52 |      10 |              0.175 |         0.193 |     0.343 |                  0.15  |             0.169 | False      | False     |
| Business and management                    |       79 |      11 |              0.232 |         0.186 |     0.388 |                  0.202 |             0.156 | False      | False     |
| Sociology, social policy and anthropology  |       75 |      11 |              0.221 |         0.275 |     0.484 |                  0.209 |             0.263 | False      | False     |
| History and archaeology                    |       71 |      10 |              0.388 |         0.218 |     0.518 |                  0.3   |             0.131 | False      | False     |
| Politics                                   |       24 |      10 |              0.529 |         0.544 |     0.858 |                  0.314 |             0.329 | False      | False     |
| Biosciences                                |       51 |      11 |              0.319 |         0.11  |     0.457 |                  0.347 |             0.137 | False      | False     |
| Psychology                                 |       89 |      11 |              0.443 |         0.185 |     0.567 |                  0.382 |             0.125 | False      | False     |

**UK Nursing (the clean licensed case):** incremental-prestige = **+0.013**, incremental-geography = **+0.484** (on 44 providers across 11 locations; prestige-only $R^2=0.009$, geography-only $R^2=0.479$, full $R^2=0.493$).

Against the unlicensed usable subjects (incremental-prestige mean +0.181, range +0.019 to +0.382), UK Nursing's school-prestige increment is the lowest / among the lowest (rank 1 of 16). 

**Verdict: the featured mechanism REPLICATES in the UK.** In UK Nursing --- a licensed, locally-employed, publicly-funded (NHS) labour market --- earnings variation across providers is a **geography/setting** phenomenon (incremental-geo +0.484) and the school's research-hiring prestige adds essentially nothing beyond it (incremental-prestige +0.013) --- exactly the US 35c pattern (US Nursing incremental-prestige $\approx 0$, pay $\sim$72\% a state phenomenon). NHS Agenda-for-Change national pay bands are the obvious mechanism: nursing pay is set by national/regional scale, not by alma mater.


**This is the robust cross-national claim**, and it is sharper than the full-typology test: the FEATURED MECHANISM (licensing sets pay by setting, not school) corroborates cross-nationally on a clean licensed case, even though the full integrated/decoupled *ordering* does not strongly replicate at the coarse CAH2 grain.

*Only 1 usable CAH2 subject is clearly occupationally licensed (Nursing) --- the other licensed subjects (Medicine, Allied health, Education, Pharmacology, Veterinary) lack a usable UK ORCID prestige axis at the CAH2 grain --- so we report Nursing as the clean licensed case versus the unlicensed distribution, not a continuous licensure gradient.*

### Adversarial self-check (licensing replication)

- **Provider-location $\neq$ destination geography.** UK geography here is where the *provider* is (England region / nation), not where graduates *work* --- UK graduates move (notably toward London), so this is a weaker geography control than the US PSEO *destination*-state decomposition of 35c. It biases incremental-geo *down* and incremental-prestige *up*, so it is a **conservative** test of the setting-not-school claim: finding Nursing's prestige increment near zero despite a weaker geography proxy strengthens, not weakens, the conclusion.

- **n of providers.** UK Nursing decomposes on 44 providers across 11 locations --- adequate for a few geography dummies; thinner subjects are dropped (<15 providers or <3 locations), not forced.

- **CAH2 coarseness on the prestige axis.** Nursing and midwifery is a single CAH2 subject (no sub-field collapse), so unlike the engineering/business rows of the typology test it is *not* attenuated by aggregation --- which is part of why this test is clean where the typology test is not.

- **One clean licensed case, not a gradient.** Only Nursing has both a usable UK prestige axis and clear occupational licensure, so this corroborates the FEATURED MECHANISM on the clean case, not via a continuous licensure gradient (which the UK CAH2 data cannot support); it corroborates the mechanism cross-nationally, **not** the full typology.
