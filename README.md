# Degrees of Separation

### The Academic–Employer Reputation gap across academic fields

A reproducible analysis of the gap between a field's **Academic Reputation (AR)** — its
position in the US faculty-hiring prestige hierarchy — and its **Employer Reputation (ER)** —
the labour market's reward for its graduates. The gap is measured per field, across institutions:

```
Gap_field = 1 − Spearman( AR_rank , ER_rank )      # within field, across institutions
```

- **AR** = SpringRank prestige from the US faculty-hiring network (Wapman et al., *Nature* 2022),
  with an independent rebuild from ORCID-derived placement edges as a robustness gateway.
- **ER** = median bachelor's earnings (US College Scorecard Field-of-Study), with ACS-PUMS,
  PSEO, and NCSES SDR/SED cross-checks.

A field where prestige and earnings agree has a small gap; a field where the academy and the
labour market disagree on who is "best" has a large one.

## Headline findings

- The gap is **real signal**, not a low-SNR artifact, in well-measured fields (null-model,
  bootstrap, residualization and restriction tests; `scripts/run_diagnostic.py`).
- Its dominant structured component is **between discipline clusters**: a precision-weighted
  multilevel decomposition puts ICC ≈ 0.30 (descriptive η² ≈ 0.50) of cross-field gap variance
  between CIP-2 parent disciplines. Credential and natural-science areas sit at the high-gap end;
  computing, mathematics, economics and integrated engineering at the low-gap end
  (`scripts/14_cluster_decomp.py`).
- An "observability" gradient (gap vs within-field earnings dispersion) is at most a **weak
  second-order modifier** once the between-cluster structure is accounted for.
- **Field taxonomy matters.** Tagging the ORCID placement population by OpenAlex research field
  does **not** raise per-field ORCID-AR vs Wapman agreement: at matched edge count research-field
  tags reproduce Wapman as well as degree text, but research field cross-cuts degree field, so the
  residual gap to the data ceiling is **coverage-bound, not tagging-noise-bound**
  (`scripts/17`–`19`).

## Repository layout

```
src/                      analysis library
  gap.py                  generic gap = 1 − Spearman(AR, ER) with reliability flags
  load_ar.py load_er.py   source-normalized AR / ER loaders
  diagnostic.py           signal-vs-artifact tests
  dispersion.py           within-field earnings dispersion (ACS / PSEO)
  genreg_bootstrap.py     generated-regressor inference for the dispersion test
  anchored_bootstrap.py   published-rank-anchored SpringRank bootstrap
  openalex_tag.py         OpenAlex field-tagging (ORCID -> 26 research fields)
  ar_pipeline/            self-contained SpringRank implementation
  crosswalks/             field <-> CIP <-> SDR / institution-name crosswalks (curated CSVs)
scripts/                  numbered pipeline stages (run_*.py + 01..19)
data/raw/SOURCES.md       full provenance manifest (URL, version, access date, license)
Makefile                  one-click reproduction (see below)
requirements.txt          pinned dependencies (also environment.yml for conda)
```

Raw data, intermediate caches, narrative result reports, exploratory notebooks and the planning
documents are **not tracked** (see `.gitignore`); committed figures and result tables make the
repo reviewable, and the pipeline regenerates every derived artifact from the documented sources.

## Reproduce

```bash
# 1. environment (Python 3.12)
python -m venv .venv && . .venv/bin/activate
make setup                      # pip install -r requirements.txt

# 2. data — fetch every source listed in data/raw/SOURCES.md into data/raw/
make check-data                 # reports which core inputs are still missing

# 3. run the public-data pipeline (Tier 0 -> cluster decomposition)
make all
```

Two stages need extra inputs and are run separately:

- `make orcid` — the ORCID placement-network gateway. Place Zenodo record **19651302**
  (`20260419.7z`) under `data/yifeng_orcid/` and extract to `data/yifeng_orcid/all/edge_aff/`.
- `make openalex` — OpenAlex research-field tagging. Create `secrets.json` (gitignored) with an
  OpenAlex API key: `{"openalex_api_key": "...", "openalex_mailto": "you@example.org"}`.

Individual stage groups: `make tier0 tier05 diagnostic gap expand granularity`. `make clean`
removes regenerated outputs.

## Data & licensing

All inputs are open; each carries its own terms, documented in
[`data/raw/SOURCES.md`](data/raw/SOURCES.md) (Wapman et al. CC BY 3.0; College Scorecard, NCSES
SDR/SED, IPEDS, Census ACS-PUMS/PSEO public domain; ORCID and OpenAlex CC0). **No datasets are
redistributed in this repository.** The code is released under the MIT License
([`LICENSE`](LICENSE)).

## Citation

If you use this code, please cite this repository and the upstream data sources listed in
`data/raw/SOURCES.md` — in particular Wapman, Zhang, Clauset & Larremore, *Quantifying hierarchy
and dynamics in US faculty hiring and retention*, Nature 610 (2022).
