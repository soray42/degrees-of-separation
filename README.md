# Degrees of Separation
### Task Distance and the Academic–Employer Reputation Gap

Research repository. The **central object** is the gap between a field's **Academic
Reputation (AR)** — faculty-hiring-network SpringRank prestige — and its **Employer
Reputation (ER)** — behavioral labor-market reward (earnings, salary, sector mix).
We measure the gap per field and test whether **academia–industry task distance**
predicts it, with **computer science as the near-zero-gap extreme**.

The full research design is in [`degrees_of_separation_proposal.md`](degrees_of_separation_proposal.md)
(source of truth). A one-page digest is in [`notes/PROJECT_SUMMARY.md`](notes/PROJECT_SUMMARY.md).

```
Gap_field = 1 − Spearman( SpringRank_prestige_rank , ER_rank )   # within field, across institutions
```

## Status / staged plan

This project is gated. **Nothing downstream is built until Tier 0 passes.**

| Step | What | State |
|---|---|---|
| 1 | Project summary + skeleton | ✅ |
| 2 | Reference library (`refs/`) | ⏳ in progress |
| 3 | Repo scaffold (this README, env, crosswalks) | ✅ |
| 4 | **Tier 0 go/no-go** (`notebooks/tier0_go_no_go.ipynb`) | ⏳ — **GATE** |
| 5 | Full AR pipeline (`src/ar_pipeline/`) | ⛔ blocked on Tier 0 = GO |

The **Tier 0 gate** (proposal §7) uses only already-public, already-computed data
(Wapman Zenodo SpringRank + College Scorecard earnings + SDR public tables) and must
clear three conditions — (1) non-trivial cross-field SD of `Gap_field`,
(2) `Gap_field` correlates **negatively** with an industry-integration proxy,
(3) CS sits in the low-gap tail — before any pipeline work begins. Result:
[`notes/TIER0_RESULT.md`](notes/TIER0_RESULT.md).

## Repository layout

```
degrees_of_separation_proposal.md   # the proposal (source of truth)
notes/
  PROJECT_SUMMARY.md                # 1-page digest
  TIER0_RESULT.md                   # GO / NO-GO verdict (Step 4)
refs/
  references.bib                    # every citation in proposal §10
  notes/<citekey>.md                # per-reference reading notes
data/
  raw/        SOURCES.md            # provenance (URL, version, access date); raw files gitignored
  interim/                          # intermediate artifacts (gitignored)
  processed/                        # analysis-ready tables (gitignored)
src/
  crosswalks/                       # CIP-centred crosswalks (the join key); reused everywhere
  ar_pipeline/                      # ORCID+OpenAlex -> hiring network -> SpringRank (Step 5)
notebooks/
  tier0_go_no_go.ipynb              # the gate
results/
  figures/  tables/                 # committed outputs
scripts/                            # standalone data-pull / build scripts
```

## Setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # or: conda env create -f environment.yml
python -m ipykernel install --user --name degrees-of-separation
```

## Data sources (all open)

| Quantity | Source | Join key |
|---|---|---|
| AR (prestige) | Wapman et al. 2022 Zenodo `10.5281/zenodo.6941651`; later ORCID+OpenAlex pipeline | field |
| ER undergrad | College Scorecard **Field of Study** earnings | **CIP** |
| ER PhD | NSF NCSES **SDR** salary + academia/industry sector split | field of degree → CIP |
| Task distance | **O\*NET** + CIP↔SOC crosswalk | **CIP** ↔ SOC |
| Controls | IPEDS / Common Data Set selectivity | IPEDS UnitID / **CIP** |

**CIP (Classification of Instructional Programs) is the join key** across Scorecard,
SDR, O\*NET and IPEDS. All crosswalks are centralized in [`src/crosswalks/`](src/crosswalks/)
and reused — do not redefine field↔CIP mappings ad hoc in notebooks.

Provenance for every downloaded file (URL, version, access date) is recorded in
[`data/raw/SOURCES.md`](data/raw/SOURCES.md).

## Reproducibility notes

- Raw data is **not** committed (see `.gitignore`); re-fetch via `scripts/` using the
  URLs in `SOURCES.md`. Curated crosswalk CSVs under `src/` **are** committed.
- Figures/tables in `results/` are committed so the repo is reviewable without re-running.
