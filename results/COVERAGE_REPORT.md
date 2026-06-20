# Tier 0 — Coverage Report

> **Provenance note.** Operational spec from the Tier-0 kickoff + `degrees_of_separation_proposal_v2.md`. All numbers reproduce via `python scripts/01_probe.py`. Date 2026-06-20.

AR = Wapman SpringRank (30 fields, 323 institutions). ER undergrad = Scorecard FoS; ER enrichment = PSEO.

## Matched institutions per field (AR ∩ ER), undergrad level

| field                   | label                   |   ar_inst |   ar_x_scorecard |   ar_x_pseo |
|:------------------------|:------------------------|----------:|-----------------:|------------:|
| psychology              | Psychology              |       239 |              173 |          88 |
| computer_science        | Computer Science        |       216 |              161 |          84 |
| political_science       | Political Science       |       194 |              139 |          75 |
| biology                 | Biology                 |       201 |              137 |          75 |
| english                 | English                 |       203 |              135 |          80 |
| mechanical_engineering  | Mechanical Engineering  |       175 |              133 |          65 |
| history                 | History                 |       209 |              131 |          81 |
| mathematics             | Mathematics             |       220 |              120 |          83 |
| electrical_engineering  | Electrical Engineering  |       184 |              118 |          69 |
| economics               | Economics               |       200 |              116 |          65 |
| sociology               | Sociology               |       188 |              114 |          74 |
| civil_engineering       | Civil Engineering       |       150 |              105 |          59 |
| nursing                 | Nursing                 |       143 |               99 |          59 |
| accounting              | Accounting              |       146 |               94 |          59 |
| chemical_engineering    | Chemical Engineering    |       128 |               91 |          45 |
| finance                 | Finance                 |       143 |               89 |          56 |
| chemistry               | Chemistry               |       223 |               76 |          82 |
| anthropology            | Anthropology            |       172 |               75 |          61 |
| computer_engineering    | Computer Engineering    |       147 |               74 |          50 |
| earth_sciences          | Earth Sciences          |       161 |               50 |          60 |
| biomedical_engineering  | Biomedical Engineering  |        86 |               48 |          24 |
| biochemistry            | Biochemistry            |       168 |               47 |          39 |
| physics                 | Physics                 |       214 |               43 |          73 |
| philosophy              | Philosophy              |       192 |               41 |          68 |
| aerospace_engineering   | Aerospace Engineering   |        69 |               35 |          22 |
| statistics              | Statistics              |       117 |               28 |          19 |
| materials_science       | Materials Science       |        89 |               23 |          19 |
| communication_disorders | Communication Disorders |        68 |               17 |          20 |
| astronomy               | Astronomy               |        92 |                3 |          14 |
| neuroscience            | Neuroscience            |        77 |                1 |           1 |

- Fields with **AR∩Scorecard ≥ 10**: **28/30**.
- Fields with **AR∩PSEO ≥ 10**: **29/30** (PSEO is partial / public-skewed).

## PhD-level ER coverage in PSEO (the v2 enrichment)

- Doctoral (degree_level 17) institution×4-digit-CIP cells: **18442** rows, **0 released** (0%). **Master's (07): 0 released.**
- → **PhD- and master's-level earnings are disclosure-suppressed at the field (4-digit CIP) grain.** PSEO releases graduate earnings only at the 2-digit-CIP grain (too coarse to separate e.g. chemistry from physics). **The PhD-level gap cannot be measured at the field level.**

## Prestige-range coverage (truncation check — critical)

PSEO match rate of Wapman institutions by prestige quartile (Q1 = most prestigious):

| pq        |   Scorecard % |   PSEO % |
|:----------|--------------:|---------:|
| Q1_top    |          74.2 |     21.5 |
| Q2        |          64.5 |     32.3 |
| Q3        |          71.7 |     30.4 |
| Q4_bottom |          65.6 |     32.3 |

Top-20 most prestigious Wapman institutions — present in each ER source?

| InstitutionName                         | in_sc   | in_ps   |
|:----------------------------------------|:--------|:--------|
| California Institute of Technology      | True    | False   |
| Massachusetts Institute of Technology   | True    | False   |
| Harvard University                      | True    | False   |
| Princeton University                    | True    | False   |
| Stanford University                     | True    | False   |
| University of California, Berkeley      | True    | False   |
| University of Chicago, The              | True    | False   |
| Yale University                         | True    | False   |
| Columbia University                     | False   | False   |
| University of California, Los Angeles   | True    | False   |
| Cornell University                      | True    | False   |
| University of Pennsylvania              | True    | False   |
| University of California, San Francisco | False   | False   |
| Carnegie Mellon University              | True    | False   |
| University of Michigan                  | True    | True    |
| Hofstra University                      | True    | False   |
| Brandeis University                     | True    | False   |
| Northwestern University                 | True    | False   |
| University of Wisconsin - Madison       | True    | True    |
| University of California, San Diego     | True    | False   |

- **Scorecard spans the full hierarchy** (≈69% of Wapman institutions, even coverage across quartiles).
- **PSEO is severely top-truncated**: only 29% of Wapman institutions, and just 22% of the top prestige quartile. ~18 of the top-20 (Caltech, MIT, Harvard, Princeton, Stanford, Berkeley, Yale, Columbia, Chicago, Penn, Cornell, CMU, …) are **absent**. Any PSEO-based gap is measured on a truncated hierarchy. See `figures/coverage_prestige_range.png`.

## Go / shelve read on coverage

- Undergrad map (Scorecard AR×ER): **28 fields ≥ 10**, full prestige range → **sound**.
- PhD-level enrichment (PSEO): **blocked** (cells suppressed at field grain) **and** prestige-truncated → the v2 two-level/PhD-ER plan is **not feasible** with public PSEO. The static map and P1 proceed on the undergrad level; the PhD-level rescue of noise-dominated fields (chemistry, earth sciences) is **unavailable**.
