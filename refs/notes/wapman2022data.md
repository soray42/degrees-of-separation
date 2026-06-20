# wapman2022data
**Full citation:** Wapman, K. H., Zhang, S., Clauset, A., & Larremore, D. B. (2022). *Data for "Quantifying hierarchy and dynamics in U.S. faculty hiring and retention"* [Data set]. Zenodo. Version 1, published 2022-07-29. License: CC BY 3.0 US.
**DOI / URL:** 10.5281/zenodo.6941651 — https://zenodo.org/records/6941651 ; mirror with file-level docs at https://github.com/LarremoreLab/us-faculty-hiring-networks ; companion site https://larremorelab.github.io/us-faculty/
**Access level:** full text (read the Zenodo API record metadata and the GitHub mirror's file/column documentation; the underlying AARC faculty census is deidentified in this release)

## What it contributes to THIS project
This is the **Tier-0 prestige dataset** — the concrete files from which we read Academic Reputation (AR) and the exact benchmark our 2026 ORCID+OpenAlex pipeline must reproduce. It supplies, per field, the institution-level SpringRank prestige we rank-correlate against behavioral Employer Reputation (ER) to form Gap_field. The deidentified hiring edge lists let us re-derive prestige ourselves (sanity check on the published ranks and on our replication SpringRank code), while the precomputed `ranks.csv` lets us skip straight to AR for Tier 0. The TaxonomyLevel/TaxonomyValue scheme is how we slice the 107 fields and 8 domains; mapping `TaxonomyValue` (AARC field names) onto CIP codes is the join step that links this AR source to Scorecard earnings, NSF SDR salaries, and O*NET task vectors. Coverage matches the paper: 2011–2020, US, 368 PhD-granting institutions.

## Specific equation / result / dataset we reuse
**Zenodo record 6941651** contains exactly **one archive**: `us-faculty-hiring-networks.zip` (3,454,885 bytes, CC BY 3.0 US, v1, 2022-07-29). The GitHub mirror (`LarremoreLab/us-faculty-hiring-networks`) documents the CSVs inside it:

1. **`edge_lists.csv`** — the directed faculty-hiring network (the input to SpringRank). Columns: `DegreeInstitutionId`, `DegreeInstitutionName`, `InstitutionId`, `InstitutionName` (edge = PhD origin → employing institution), `Total`, `Women`, `Men` (faculty counts on that edge), `TaxonomyValue`, `TaxonomyLevel`. This is what we re-run SpringRank on to validate our 2026 pipeline.

2. **`ranks.csv`** — institution prestige rankings (our AR table). Columns: `TaxonomyValue`, `TaxonomyLevel`, `InstitutionId`, `InstitutionName`, `NonAttritionEvents`, `AttritionEvents`, `ProductionRank` (ordinal by faculty production), **`PrestigeRank`** (Float, SpringRank scaled 0–1; 0 = highest prestige, 1 = lowest), `OrdinalPrestigeRank` (Integer ordinal prestige).

3. **`yearly-stats.csv`** — time series: `TaxonomyValue`, `TaxonomyLevel`, `Year`, `GiniCoefficient`, `FractionWomen`.

4. **`stats.csv`** — academia/domain/field-level summary stats (doctorate origins, attrition, gender, hierarchy/Gini, null-model comparisons).

**Taxonomy keys** that slice every file: `TaxonomyLevel ∈ {Academia, Domain, Field}` and `TaxonomyValue` (the specific subset, e.g. "Computer Science"); 107 field values nested under 8 domains. Fields are non-exclusive (~23% of faculty in >1 field).

**Prestige method behind the column:** SpringRank *without regularization* (De Bacco, Larremore & Moore 2018) applied separately per network — each directed edge → oriented spring with displacement s_i − s_j; node ranks s minimize total spring energy, solved as a sparse linear system; then scaled to 0–1.

**Tier-0 read rule (decision):** `SpringRank_prestige` for AR = the **`PrestigeRank`** column of `ranks.csv` filtered to `TaxonomyLevel == "Field"` and the target `TaxonomyValue`. For Spearman against ER we use `OrdinalPrestigeRank` (or rank `PrestigeRank` directly); note PrestigeRank is *ascending in rank but descending in prestige* (0 = best), so we invert/handle sign before correlating with ER_rank. We do NOT use `ProductionRank` for AR.
