# Claims ledger — audit trail for `main.tex`

Every empirical claim in the draft → the exact value → the `*_RESULT.md` it comes from → the generating
script → its caveat. Values are copied verbatim from the RESULT files (verified by a parallel extraction pass
over all RESULT files). A co-author can verify nothing is overclaimed or fabricated. `[TODO: source]` marks the
(few) numbers that are not in a RESULT file and would need a script re-run or removal before submission.

Legend: **value** | source RESULT file | script | caveat.

---

## §3 Data and coverage
- Scorecard ~69% of Wapman institutions, even by quartile (**74.2 / 64.5 / 71.7 / 65.6%** top→bottom);
  PSEO ~29%, **21.5%** of top quartile, ~18 of top-20 elite privates absent | `results/COVERAGE_REPORT.md`,
  `results/TIER0_RESULT.md` | run_tier0 / 01_probe | PSEO is a selected, top-truncated, public-skewed subset.
- PSEO doctoral cells at 4-digit CIP: **18,442 rows, 0 released (0%)**; master's also 0 | `results/COVERAGE_REPORT.md`
  | 01_probe | why the analysis is bachelor's-level only.
- BA fields with ≥10 matched institutions: **Scorecard 28/30, PSEO 29/30** | `results/COVERAGE_REPORT.md` | 01_probe | —.
- From-scratch SpringRank reproduces published ranks at **ρ≈0.77** (data ceiling) | `outputs/GENREG_RECALIBRATION_RESULT.md`
  | 09_recalibrate | public aggregated edges are lossy vs the proprietary census; not a code bug.
- ORCID-rebuilt SpringRank reproduces Wapman academia ranks at **ρ=0.74** | `outputs/ORCID_OVERLAP_VALIDATION_RESULT.md`
  | 10–11 | public-Wapman benchmark itself only 0.91; method certification, not 1.0.

## §4 The gap and its discipline structure
- Diagnostic verdict **REAL SIGNAL** (filter required); artifact channel **R²=0.20**; **only 15%** of fields
  inside null band, **65%** above | `notes/DIAGNOSTIC_RESULT.md` | run_diagnostic | global-artifact ruled out,
  but a low-earnings-signal cluster (Chemistry, Earth Sci) is noise-dominated and screened.
- Bootstrap reliability: CS **0.71 [0.62,0.78]**, Statistics **0.75 [0.52,0.87]**, Biology **0.37 [0.22,0.52]
  (N=138)**; high-gap end wide (Physics 0.35 [0.01,0.62], Chemistry 0.31 [0.09,0.50]) | `notes/DIAGNOSTIC_RESULT.md`
  | run_diagnostic | reliability filter needed.
- Headline decomposition (`gap_f ~ 1+(1|CIP-2)`, REML, **n=48**): **τ²=0.0231** (SD 0.152), **ICC=0.30**,
  **η²=0.50**; **18 CIP-2 clusters**, **23** locked-grain reliable units | `outputs/CLUSTER_DECOMP_RESULT.md`,
  `outputs/GRANULARITY_RESULT.md` | 13–14 | salary-anchored, descriptive variance decomposition; not causal.
- Integrated end: Computer/Info **0.348**, Math & Stats **0.352**, Social Sci **0.421**, Engineering **0.432**;
  decoupled: Health **0.753**, Arts **0.746**, Nat. Resources **0.730**, Biological Sci **0.678**, Physical Sci
  **0.661** | `outputs/CLUSTER_DECOMP_RESULT.md` | 14 | shrunk/partial-pooled means; raw means differ (e.g. Bio
  raw 0.830); regime labels descriptive.
- Within-cluster observability gradient: pooled Spearman(gap, dispersion) **−0.44 [−0.74,+0.01]**; within-cluster
  slope **−0.527 [−1.08,+0.02]** (n=46) | `outputs/CLUSTER_DECOMP_RESULT.md`, `outputs/GRANULARITY_RESULT.md` |
  13–14 | CI spans 0; weak second-order modifier, not headline.

## §5 What explains the gap
- **b_licensure = +0.66 [+0.33,+0.97]**; b_absorption **−0.02 [−0.42,+0.46]** (null); **R²=0.28**;
  corr(anchors) **−0.21** (null not collinearity) | `outputs/MIXTURE_DECOMP_RESULT.md` | 20 | decomposition, not
  causal/IV; anchors are field aggregates (ecological).
- Multi-channel **R²=0.29 vs licensure-only 0.27**; standardized licensure **+0.11 [+0.07,+0.14]**, pay-wedge
  **−0.03**, public-sector **−0.01**; residual ranking vs licensing-only **Spearman +1.00**; residual SD
  **0.228→0.225** | `EXTERNAL_CHANNELS_RESULT.md` | 27 | compression channels collinear (|corr| up to 0.54);
  read joint R², not separate slopes; unionization is a data gap (not fabricated).
- Pay-wedge bivariate **−0.33 (p=0.022, n=48)**; survives field-size control (−0.27→−0.29); public-sector
  bivariate **+0.37 (p=0.009)** collapses to ~0 when licensing netted | `EXTERNAL_CHANNELS_RESULT.md` | 27 | —.
- **Ruled-out (a) field-vs-generic:** c_G **+0.45** ≥ c_F **+0.40**; ADV<0 in **42/57**; brand-dominant **15**
  vs field-dominant **7**; ρ_PP **+0.79**; CS divergence brand-strong **$119,643** vs field-strong **$98,970**
  | `FIELD_VS_GENERIC_RESULT.md` | 28 | medians only; brand effect in the tail (Chetty); selection.
- **Ruled-out (b) integration shape:** GRADED **30** / THRESHOLD **5** / FLAT **9**; c_rest↔c_F **+0.96**, class
  ↔c_F **+0.80** | `TWO_MARKET_INTEGRATION_RESULT.md` | 29 | structure ~redundant with strength; median data
  can't test winner-take-all (limitation, not finding).
- **Ruled-out (c) mispricing = geography/selection:** destination-state FE **R²=0.46** vs brand **0.00**;
  residual **+0.88**-correlated with raw pay ranking (prestige explains ~18%); **29%** horizon sign-flip
  | `MISPRICED_PROGRAMS_RESULT.md` | 30 | disagreement ≠ value-added; named programs illustrative only.

## §6 The two-signal model (built, NOT validated)
- One parameter vector (N=150, σ_A=0.50, σ_∞=0.30, σ_0=1.00, δ=0.70); P2 full-support effect **≈+0.64** ~
  empirical +0.66; P3 absorption null **−0.04 (p=0.13)**; P5 gap-change↔divergence **+0.71** | `MODEL.md` | 23 |
  no per-fact tuning; P1/P4 partial; ICC-vs-horizon miss (between-var rises 0.016→0.019).
- **B1 make-or-break UNCONFIRMED:** O*NET **+0.41 [+0.08,+0.68]** vs SDR **−0.61 [−0.88,−0.11]**; proxies
  negatively correlated **−0.64**; B3 corr(licensure, O*NET) **−0.03** vs SDR **+0.39** | `MODEL_VALIDATION_RESULT.md`
  | 25 | residual-as-divergence downgraded "estimated"→"unconfirmed"; model NOT validated.
- ICC refinement: discipline-varying σ0 makes between-var FALL (**0.0297→0.0246**) without breaking P2(+0.604)/
  P3(+0.029, p=0.19)/P5(+0.40) | `MODEL_ICC_REFINEMENT.md` | 26 | this is fitting (added df), not prediction.

## §7 Robustness
- n-subsample cluster ranking **Spearman +0.98** (common n=20, 18 clusters); corr(gap,n) **−0.54** |
  `outputs/HARDENING_RESULT.md` | 22 | ranking not an n artifact.
- Measurement-corrected **ICC 0.30→0.45** (within obs 0.0549, mean SE² 0.0267, corrected 0.0282) |
  `outputs/HARDENING_RESULT.md` | 22 | correction strengthens the between-share.
- AR-rank propagation: gap-CI width ratio **1.04** (~4% wider); **0/16** reliability flips |
  `outputs/HARDENING_RESULT.md` | 22 | lower bound (edge noise on public edges, not the public-vs-census gap).
- Crosswalk audit 27 missing = **12 recoverable / 3 subsumed / 12 genuine gaps**; 54→66 holds (rank +0.98,
  ICC raw 0.30→0.32, corrected 0.45→0.57) | `CROSSWALK_AUDIT_RESULT.md` | 24 | ceiling mostly genuine, not naive.
- **PSEO: level Spearman +0.94 (n=1535); gap +0.68 [+0.52,+0.80] over 57 fields**; out-of-state **34%**;
  geography origin-state **R²≈0.13**; gap netting **0.68→0.67** | `PSEO_GAP_CROSSVALIDATION_RESULT.md` | 32 |
  108-institution coalition-public overlap; elite privates under-represented.
- Multi-horizon: ICC **0.50@1yr → 0.30–0.33@4–5yr**; cluster-rank stability +0.81–0.85; field-gap +0.68–0.74 |
  `outputs/MULTIHORIZON_RESULT.md` | 21 | horizons autocorrelated (same completers); early-career only.
- OpenAlex research-field tags don't beat degree-text (mean ρ 0.515 vs 0.551; de-leaked dead heat 0.515 vs
  0.517) | `outputs/OPENALEX_TAGGING_RESULT.md` | 15–19 | research field ≠ degree field; coverage-bound residual.

## §8 Discussion — temporal + applied
- Revaluation (real $/yr): Statistics **+2,311**, CS **+1,616**, Computer Eng **+1,421**, Biomedical Eng
  **+969**, Finance **+868**, Economics **+817**; Pharmacy **−1,774**, Communication Disorders **−1,405**;
  **46/60** rising | `DYNAMIC_LEADLAG_PILOT_RESULT.md` | 33 | PSEO real 2023$; coalition composition drift caveat.
- Lead-lag pilot: prestige stability median **ρ≈0.50 (0.34–0.58)**; direction **NOT identified** (2 noisy
  periods) | `DYNAMIC_LEADLAG_PILOT_RESULT.md` | 33 | suggestive only; institution-level earnings panel
  data-blocked (PSEO single-cohort suppressed).
- CS ranking demo: CMU **$246,966 (+$48,910)**, Stanford, Berkeley, MIT, UCLA; cross-check Scorecard↔PSEO
  **+0.84 (n=29)**, Wapman↔ORCID **+0.69 (n=61)**; **15** low-robustness flags; geography R²≈0.13 |
  `CS_RANKING_RESULT.md` | 34 | descriptive, not value-added; Title-IV but cross-validated; median = where grads land.

## Numbers needing a source before submission
- None of the in-text statistics are fabricated; all map to a RESULT file above. The only `[TODO: source]`
  markers in `main.tex` are author/venue placeholders, not data. The one supporting citation flagged uncertain
  is `altonji2016analysis` in `refs.bib` (a field-of-study earnings cite; confirm or drop).
