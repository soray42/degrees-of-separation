# Degrees of Separation — one-click reproduction pipeline.
#
# Prerequisites:
#   1. Python env:   make setup        (or: pip install -r requirements.txt)
#   2. Raw data:     fetch every source listed in data/raw/SOURCES.md into data/raw/
#                    (run `make check-data` to see what is missing)
#   3. ORCID gateway (stages orcid/openalex): place Zenodo 19651302 `20260419.7z` at
#      data/yifeng_orcid/ and extract to data/yifeng_orcid/all/edge_aff/*.parquet
#   4. OpenAlex tagging (stage openalex): create secrets.json with an OpenAlex API key
#      {"openalex_api_key": "...", "openalex_mailto": "you@example.org"}
#
# Then:   make all          (public-data stages, tier0 -> cluster)
#         make openalex      (ORCID + OpenAlex field-tagging gateway; needs 3 & 4 above)
#
# Override the interpreter with:  make all PYTHON=.venv/bin/python

PYTHON ?= python
S      := $(PYTHON) scripts

.PHONY: all setup check-data tier0 tier05 diagnostic gap expand orcid granularity openalex figures clean

## --- environment -----------------------------------------------------------
setup:
	$(PYTHON) -m pip install -r requirements.txt

check-data:
	@$(PYTHON) -c "import pathlib,sys; \
req=['data/raw/wapman2022/ranks.csv','data/raw/wapman2022/edge_lists.csv',\
'data/raw/scorecard_fos','data/raw/acs','data/raw/onet','data/raw/ipeds']; \
miss=[r for r in req if not pathlib.Path(r).exists()]; \
print('missing raw inputs (see data/raw/SOURCES.md):') or [print('  -',m) for m in miss] if miss else print('core raw inputs present'); \
sys.exit(1 if miss else 0)"

## --- public-data analysis (the reproducible core) --------------------------
tier0:       ; $(S)/run_tier0.py
tier05:      ; $(S)/run_tier0_5.py
diagnostic:  ; $(S)/run_diagnostic.py

gap:         ## v2 gap machinery: probe, gap map, P1, regime, explanation, sensitivity
	$(S)/01_probe.py
	$(S)/02_gap_map.py
	$(S)/03_p1.py
	$(S)/04_p1_v2.py
	$(S)/05_regime.py
	$(S)/06_explanation.py
	$(S)/07_sensitivity.py

expand:      ## field expansion (30->54) + generated-regressor inference
	$(S)/08_expand.py
	$(S)/09_recalibrate.py

granularity: ## CIP-2 roll-up, multilevel meta-regression, cluster variance decomposition
	$(S)/12_license.py
	$(S)/13_granularity.py
	$(S)/14_cluster_decomp.py

## --- ORCID + OpenAlex gateway (needs the ORCID dump and an OpenAlex key) ----
orcid:       ## build ORCID PhD->faculty placement edges + validate vs Wapman
	$(S)/10_orcid_build.py
	$(S)/11_orcid_validate.py

openalex: orcid  ## OpenAlex field-tagging + per-field re-validation
	$(S)/15_openalex_pilot.py
	$(S)/16_openalex_tag_full.py
	$(S)/17_openalex_field_validate.py
	$(S)/18_oa_vs_ft_robust.py
	$(S)/19_oa_figures.py

## --- aggregate -------------------------------------------------------------
all: tier0 tier05 diagnostic gap expand granularity   ## full public-data pipeline

figures:     ## regenerate figures only (requires upstream derived data present)
	$(S)/19_oa_figures.py

clean:       ## remove regenerated outputs (keeps raw data and source)
	rm -rf outputs/figures/*.png outputs/*.csv results/figures/*.png results/*.csv \
	       data/interim/*.parquet data/interim/*.csv data/interim/openalex_cache
