# Data Provenance — `data/raw/`

Every raw input is logged here with **URL · version/vintage · access date · file(s) · license**.
Raw files themselves are gitignored; this file is the reproducible manifest. Re-fetch with
the scripts in `scripts/`.

Access dates use ISO format. Project timezone-agnostic; "accessed 2026-06-20" = first pull.

---

## 1. Academic reputation — Wapman et al. 2022 (Tier 0 prestige source)

- **Source:** Zenodo record **10.5281/zenodo.6941651** — Wapman, Zhang, Clauset & Larremore,
  *Quantifying hierarchy and dynamics in US faculty hiring and retention* (Nature 2022).
- **URL:** https://zenodo.org/records/6941651  ·  API: https://zenodo.org/api/records/6941651
- **Coverage:** US, 2011–2020, ~107 fields / 8 domains, ~368 PhD-granting institutions.
- **Version:** _to fill on download_  ·  **Access date:** 2026-06-20
- **Files pulled:** _to fill — list each filename + the column we read as SpringRank prestige_
- **License:** _to fill from Zenodo (typically CC-BY 4.0)_

## 2. Employer reputation, undergraduate — College Scorecard, Field of Study

- **Source:** U.S. Department of Education, College Scorecard **Field-of-Study** data files
  (institution × 4-digit CIP × credential; median earnings 1/4/5 yr, IRS-linked).
- **URL:** https://collegescorecard.ed.gov/data/  ·  data dictionary + "Most-Recent-Field-of-Study" zip.
- **Version:** _to fill (release label / date)_  ·  **Access date:** 2026-06-20
- **Files pulled:** _to fill — e.g. Most-Recent-Cohorts-Field-of-Study.csv_
- **Key columns:** `UNITID`, `INSTNM`, `CIPCODE`, `CIPDESC`, `CREDLEV`, `EARN_MDN_*` (earnings).
- **License:** public domain (US Gov).

## 3. Employer reputation, PhD — NSF NCSES SDR / SED

- **Source:** NSF National Center for Science and Engineering Statistics, **Survey of Doctorate
  Recipients (SDR)** public tables — median salary + employment sector (academia/industry/gov)
  by fine field of degree; **Survey of Earned Doctorates (SED)** as the cohort frame.
- **URL:** https://ncses.nsf.gov/surveys/doctorate-recipients/  ·  https://ncses.nsf.gov/surveys/earned-doctorates/
- **Version:** _to fill (survey year / table id)_  ·  **Access date:** 2026-06-20
- **Files pulled:** _to fill — salary-by-field table + sector-by-field table_
- **License:** public domain (US Gov).

## 4. Task distance — O\*NET + CIP↔SOC crosswalk  *(Step 5 / full study)*

- **Source:** O\*NET Resource Center database (work activities, skills, abilities) + the
  CIP↔SOC crosswalk.
- **URL:** https://www.onetcenter.org/database.html  ·  https://www.onetcenter.org/crosswalks.html
- **Version:** _to fill_  ·  **Access date:** _n/a in Tier 0_
- **License:** O\*NET — CC-BY 4.0.

## 5. AR pipeline inputs — ORCID + OpenAlex  *(Step 5 / 2026 extension)*

- **ORCID:** public data file (annual dump). https://orcid.org/  ·  version _to fill_.
- **OpenAlex:** snapshot / API. https://openalex.org/  ·  https://docs.openalex.org/ · version _to fill_.
- **License:** ORCID public data CC0; OpenAlex CC0.

---

### Crosswalk note
The **CIP code** is the join key across sources 2–4 and IPEDS. Field↔CIP↔SDR-field mappings are
centralized in [`src/crosswalks/`](../../src/crosswalks/) — never redefine them ad hoc.
Institution-level joins (Wapman names ↔ Scorecard `UNITID`/`INSTNM`) use the name-normalizer in
`src/crosswalks/institutions.py`.
