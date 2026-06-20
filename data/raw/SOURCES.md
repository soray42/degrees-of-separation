# Data Provenance — `data/raw/`

Every raw input is logged here with **URL · version/vintage · access date · file(s) · license**.
Raw files are gitignored; this file is the reproducible manifest. Re-fetch with the scripts
in `scripts/` (or the commands below). All sources are open. First pull: **2026-06-20**.

---

## 1. Academic reputation — Wapman et al. 2022 (Tier 0 prestige source)

- **Source:** Zenodo **10.5281/zenodo.6941651** — Wapman, Zhang, Clauset & Larremore,
  *Quantifying hierarchy and dynamics in US faculty hiring and retention* (Nature 2022).
- **Download:** `https://zenodo.org/api/records/6941651/files/us-faculty-hiring-networks.zip/content`
- **Version:** v1, published 2022-07-29 · **License:** CC BY 3.0 US · **Accessed:** 2026-06-20
- **Archive:** `us-faculty-hiring-networks.zip` (3,454,885 bytes) → extracted to `wapman2022/`
  - `ranks.csv` — **used.** Columns `Rank, InstitutionId, InstitutionName, TaxonomyValue, TaxonomyLevel`.
    `Rank` = SpringRank prestige ordinal (0 = most prestigious). Filter `TaxonomyLevel=="Field"`
    (81 fields). **This is `SpringRank_prestige` for AR.**
  - `edge_lists.csv` — directed hiring network (PhD inst → employing inst), for Step-5 validation.
  - `institution-stats.csv`, `stats.csv`, `yearly-stats.csv` — aggregate stats (not used in Tier 0).
- **Coverage:** US, 2011–2020, 8 domains, ~368 PhD-granting institutions.
- *Note:* the GitHub README (`LarremoreLab/us-faculty-hiring-networks`) documents
  `PrestigeRank/ProductionRank/OrdinalPrestigeRank`, which live in `institution-stats.csv`,
  **not** in the per-institution `ranks.csv` we use; `ranks.csv` carries the ordering in `Rank`.

## 2. Employer reputation, undergraduate — College Scorecard, Field of Study

- **Source:** U.S. Department of Education, College Scorecard **Field-of-Study** files.
- **Download:** `https://ed-public-download.scorecard.network/downloads/Most-Recent-Cohorts-Field-of-Study_06102026.zip` (17 MB)
- **Version:** "Most Recent", updated **2026-06-10** · **License:** public domain (US Gov) · **Accessed:** 2026-06-20
- **File:** `Most-Recent-Cohorts-Field-of-Study.csv` (≈153 MB, 227,980 rows, 178 cols) → `scorecard_fos/`
- **Key columns:** `UNITID, INSTNM, CONTROL, CIPCODE` (4-digit, zero-padded), `CREDLEV`
  (3 = Bachelor's used; 5 = Master's, 6 = Doctoral), `EARN_MDN_4YR` / `EARN_MDN_1YR`
  (median earnings 4 / 1 yr post-completion; `'PS'` = privacy-suppressed → NaN).

## 3. Employer reputation, PhD — NSF NCSES SDR / SED

- **SDR 2021 — Table 12-3** (fine-field sector counts; **integration proxy source**):
  - `https://ncses.nsf.gov/pubs/nsf23319/assets/data-tables/tables/nsf23319-tab012-003.xlsx`
  - ~98 fine fields × {All employed, Educational institution, Business or industry, Government}
    (Number + SE). industry_share = Business / All. → `sdr2021/tab012-003.xlsx`. Accessed 2026-06-20.
  - Companion (not yet used): Table 54 salary by field × sector `…/nsf23319-tab054.xlsx`.
- **SED 2021 (NSF 23-300) data tables** (broad-field fallbacks / cross-checks):
  - Table 2-6 employment sector by broad field: `…/nsf23300/assets/data-tables/tables/nsf23300-tab002-006.xlsx`
    → `sed2021/tab002-006.xlsx` (humanities industry share for english/history/philosophy).
  - Table 6-7 median salary by broad field × sector: `…/nsf23300-tab006-007.xlsx` → `sed2021/tab006-007.xlsx`.
  - All-Excel archive: `…/nsf23300/assets/data-tables/nsf23300-data-tables-tables-excels.zip` → `sed2021/all-excels.zip`.
- **License:** public domain (US Gov). **Accessed:** 2026-06-20.

## 4. Task distance — O\*NET + CIP↔SOC crosswalk  *(Tier 0.5)*

- **O\*NET database, text format, v30.3:** `https://www.onetcenter.org/dl_files/database/db_30_3_text.zip`
  → `onet/`. Used: `Work Activities.txt` (O\*NET-SOC × 41 Generalized Work Activities, scales
  IM & LV; task vector = IM×LV per element) and `Occupation Data.txt`. License: CC BY 4.0.
- **CIP→O\*NET-SOC crosswalk (2020 CIP → O\*NET-SOC 2019, updated Jul 2024):**
  `https://www.onetcenter.org/crosswalks/cip/Education_CIP_to_ONET_SOC.xlsx` → `onet/`.
  Real header on row 3; cols CIP / CIP title / O\*NET-SOC / SOC title.
- **Accessed:** 2026-06-20. Field→teacher-SOC and field→SDR-salary crosswalks: `src/crosswalks/fields.py`.

## 5. Grad-school pull — IPEDS Completions  *(Tier 0.5)*

- **IPEDS Completions C2023 (finalized), part A:** `https://nces.ed.gov/ipeds/datacenter/data/C2023_A.zip`
  → `ipeds/C2023_a.csv`. Dict: `…/C2023_A_Dict.zip`. **Accessed:** 2026-06-20. Public domain.
- **Key columns:** `UNITID, CIPCODE` (2020 CIP), `MAJORNUM` (filter = 1), `AWLEVEL`
  (5 = Bachelor's, 17 = Doctor's research/scholarship), `CTOTALT` (grand total completions).
  grad-school pull = Σ doctoral(17) / Σ bachelor(5) by CIP-4 field.

## 6. Industry employment weights — OEWS  *(Tier 0.5 — UNAVAILABLE)*

- BLS OEWS national file (`https://www.bls.gov/oes/special-requests/oesm24nat.zip`, cols
  `OCC_CODE`, `TOT_EMP`) would employment-weight the industry task vectors. **Not obtained:**
  bls.gov returns a uniform Akamai **HTTP 403** to all automated fetches (curl/WebFetch),
  masking 200/404. Tier-0.5 task distances therefore use **equal weighting** over CIP→SOC
  destinations (documented in `notes/TIER0_5_RESULT.md`).

## 7. AR pipeline inputs — ORCID + OpenAlex  *(Step 5 / 2026 extension — not started; gated on mechanism reframe)*

- ORCID public data file (`https://orcid.org/`, CC0); OpenAlex snapshot/API
  (`https://docs.openalex.org/`, CC0).

---

### Crosswalk note
**CIP is the join key** across sources 2–4 and IPEDS. Field↔CIP↔SDR mappings are centralized
in [`src/crosswalks/fields.py`](../../src/crosswalks/fields.py) (all 20 Tier-0 `wapman_field`
labels verified present in source 1). Institution joins (Wapman names ↔ Scorecard `INSTNM`)
use the normalizer in [`src/crosswalks/institutions.py`](../../src/crosswalks/institutions.py).
