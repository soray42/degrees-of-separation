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
| 2 | Reference library (`refs/`, 47 entries + notes) | ✅ |
| 3 | Repo scaffold (this README, env, crosswalks) | ✅ |
| 4 | **Tier 0 go/no-go** (`notebooks/tier0_go_no_go.ipynb`) | ✅ — verdict **NO-GO** (narrow) |
| 4.5 | **Tier 0.5 mechanism diagnosis** (`notebooks/tier0_5_diagnose.ipynb`) | ✅ — verdict **REFRAME** |
| 4.6 | **Diagnostic: signal vs artifact** (`scripts/run_diagnostic.py`) | ✅ — verdict **REAL SIGNAL** (filter required) |
| v2-T0 | **v2 Tier 0** — PSEO probe + generic gap map + P1 (`scripts/01–03`) | ✅ — verdict **CONDITIONAL** |
| v2-T0+ | **Field expansion + generated-regressor inference** (`scripts/08`, `outputs/`) | ✅ — n-limited; P1 survives SpringRank noise but not small-n |
| 5 / Tier 1 | ORCID AR rebuild + dynamic out-of-sample test | ⛔ **not started** — gated on the v2 Tier-0 call |

**v2 Tier 0 (see [`results/TIER0_RESULT.md`](results/TIER0_RESULT.md)) — CONDITIONAL.**
Built the spec'd generic, source-agnostic gap machinery (`src/{load_ar,load_er,gap,predictions}.py`,
`scripts/01_probe.py`/`02_gap_map.py`/`03_p1.py`; Tier-1 plugs in a new AR table via the `period`
column). The reliability-gated **undergrad gap map** (30 fields, 14 reliable, anchors CS-low /
biology-high reproduce, full prestige range via Scorecard) is sound. **But:** (1) **P1** (gap vs
earnings dispersion) is correct-signed on reliable fields (≈ −0.28) yet **not significant** and
fragile; (2) **PSEO PhD-level ER is infeasible** — institution×field doctoral cells are 100%
disclosure-suppressed and PSEO is prestige-truncated (misses the elite-private top), so v2's
two-level PhD-ER enrichment is data-blocked. The top-tier case now rests solely on the untested
**Tier-1 dynamic compression**. Proceed only if that bet is worth the ORCID-rebuild cost.

**v2 Tier-0 addendum — final open-data static layer (`scripts/04–07`, `src/dispersion.py`).**
Re-tested P1 off **non-truncated** dispersion (full-pop **ACS PUMS**; non-suppressed PSEO-state)
and added an ex-ante **regime typology** (licensure + ACS grad-share) and NY-Fed correlates.
Result: **P1 firmed up** — −0.51 (p=0.06) on early-career ACS, **−0.71 (p=0.015)** on the
cleanest fields, vs −0.28 on the old truncated PSEO (truncation was attenuating it). The
high-gap fields resolve by channel (license-standardization 0.87 · PhD-pipeline 0.65 ·
prestige-transmission 0.40). Verdict moves to **CONDITIONAL leaning GO-to-Tier1**: solid static
floor; ceiling unchanged (BA-level, salary-anchored; no PhD/private ER without Revelio); the
Nature-tier call reduces to the Tier-1 dynamic bet. See `results/TIER0_RESULT.md` (final read).

**Field expansion + generated-regressor inference (`scripts/08_expand.py` → `outputs/`).**
Expanded the universe 30→**54 fields** (Wapman→CIP+FOD1P; mapping failures logged, not hidden)
and put proper inference on P1. Two findings: (1) the expansion added only ~2 reliable fields
(16 vs 14 at the 0.50 gate) — **a real open-data ceiling** (most remaining Wapman fields are
thin/suppressed/uncrosswalkable, so **n is not the available lever**). (2) A nested
generated-regressor bootstrap (multinomial **edge resample + SpringRank re-run** — our
from-scratch SpringRank reproduces Wapman ranks at ρ≈0.72–0.84 — plus earnings/dispersion
noise): **SpringRank estimation noise alone does NOT overturn P1** (gen-reg CI excludes 0), but
adding the **small-n** field-sampling uncertainty, the full two-level CI includes 0 at the 0.50
gate (p=0.29) and is marginal at 0.65 (p=0.086). **The binding limitation is n, not the
generated regressor.** See `outputs/FIELD_EXPANSION_RESULT.md`.

**Tier 0 outcome (see [`notes/TIER0_RESULT.md`](notes/TIER0_RESULT.md)):** conditions 1
(variation) and 3 (CS is the 2nd-lowest-gap field of 20) **PASS**; condition 2 (gap falls
with the industry-share proxy) **FAILS**. ⇒ ran a Tier 0.5 mechanism diagnosis.

**Tier 0.5 outcome (see [`notes/TIER0_5_RESULT.md`](notes/TIER0_5_RESULT.md)) — REFRAME.**
Four competing explanators of the gap, real data: the original **task distance (H2)** is
**signed backwards** (−0.38) and chemistry refutes it; the gap is instead driven by an
**academic-pipeline / earnings-informativeness axis** — **grad-school pull** (+0.48, the only
significant one) and **earnings dispersion** (−0.41) place both CS (low gap) and chemistry
(high gap) correctly. The **price wedge** (price agreement) does **not** coincide with the
institution-level **rank** gap. The descriptive gap map is real; the H2 *mechanism* is not
supported. Re-specify the gap (robust to earnings informativeness) or reframe around the
pipeline mechanism **before** the Step-5 pipeline.

**Diagnostic — signal vs artifact (see [`notes/DIAGNOSTIC_RESULT.md`](notes/DIAGNOSTIC_RESULT.md)) — REAL SIGNAL.**
A null-model / bootstrap / residualization / restriction battery (real data only) shows the
gap is **not** a global low-SNR artifact: 65% of fields exceed the perfect-agreement+noise
null, the artifact channel explains only ~20% of gap variance, and high-gap **biology** is
genuine (tight CI). **But** a low-earnings-signal cluster (**chemistry, earth sciences,
low-CV engineering**) is noise-dominated → the gap needs a per-field reliability filter, and
the chemistry counterexample dissolves. Next: measure **ER at the level where each field's
talent lands** (PhD-level ER for PhD-pipeline fields), not a new pipeline yet.

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
