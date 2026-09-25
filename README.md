# Degrees of Separation

### A measurable wedge between academic prestige and the labour-market pricing of fields

These are my working notes on the whole project, at a glance — what it
measures, every finding with its verified number and caveat, each experiment's design, the
engineering architecture, and what is done vs. pending. Pending/roadmap lives in
[`ROADMAP.md`](ROADMAP.md). All in-text numbers are copied verbatim from the verified audit
trail (`paper/CLAIMS_LEDGER.md`), which maps every claim → value → source `*_RESULT.md` →
generating script → caveat. Nothing here is fabricated or overclaimed.

> **The object.** For a field *f* across the institutions *i* that teach it:
> ```
> gap_f = 1 − Spearman_i( academic_prestige_{i,f} , graduate_earnings_{i,f} )
> ```
> A **rank** statistic within field: it ignores level differences (engineering > education) and
> asks only whether, among schools teaching *f*, the prestige and pay orderings coincide.
> Small gap = the academy and the labour market agree on who is "best"; large gap = they disagree.
> Its affine-invariant complement, the **coupling** `ρ_f = 1 − gap_f = Spearman(prestige, earnings)`,
> is the working scalar in the extensions.

**Stance.** Descriptive, **outcome-agnostic, not causal**. The strength is a measured, structured,
cross-validated, mechanistically-*bounded* wedge, plus explicit scope conditions (things ruled out).
Read through signalling (Spence 1973) and employer-learning theory (Altonji–Pierret 2001; MacLeod
et al. 2017); Chetty–Deming–Friedman explains why median data cannot see the elite tail.

---

## 1. The measurement backbone (what everything rests on)

Three measured pillars + one identity layer. These are the load-bearing constructs; every result
is a statement about them.

| Pillar | Construct | Source | Key engineering caveat |
|---|---|---|---|
| **AR** — academic prestige | Wapman et al.'s published **SpringRank** field rank of the US faculty-hiring network. Field-level gaps use this ordinal rank; a continuous public-edge SpringRank rebuild is used only for pooled clusters and the CS demonstrator (the public edge list omits institution pairs with a single hire, so it cannot fully reproduce the published rank) | Wapman et al. *Nature* 2022; independent ORCID rebuild as a robustness gate | From-scratch SpringRank reproduces published ranks at only **ρ≈0.77** (public aggregated edges are *lossy* vs the proprietary census — a data ceiling, **not a code bug**). ORCID-rebuilt reproduces Wapman academia at **ρ=0.74** (public-Wapman benchmark itself only 0.91 → method certification, not 1.0). |
| **ER** — graduate earnings (the *placement proxy*) | **College Scorecard** field-of-study bachelor's median earnings, **cross-validated against Census PSEO** (UI wages) | Scorecard FoS (Title-IV); PSEO Explorer | Earnings is a **validated-but-incomplete** proxy for "placement," with two characterised failure modes (see §4 pilot: occupation-pinned fields, and delayed-market fields like bio). Scorecard **same-release 1yr vs 4yr are different cohorts** — never diff them for career time (see §3). |
| **gap = real signal** | diagnostic battery (null-model, bootstrap, residualization, restriction) + per-field reliability filter | `run_diagnostic.py`, `src/diagnostic.py` | Verdict **REAL SIGNAL** but **filter required**: artifact channel explains only **R²=0.20**; only **15%** of fields inside a perfect-agreement-plus-noise null, **65%** above it. A low-earnings-signal cluster (Chemistry, Earth Sci) is noise-dominated and **screened out**. *(These shares come from the early 30-field universe; an unscripted re-computation on the 57-field universe gives 53% above / 33% inside / 14% below — values below the band mean the null's noise parameter is miscalibrated for some fields. Cochran's Q/I² in scripts/57 is the cleaner heterogeneity test.)* |

**The placement axis is currently four things (the "four pillars" the frontier relaxes):**
it is a **single scalar**, at the **institution grain**, **US-only**, and **median-level** —
and disconnected from student beliefs. The forward agenda (`paper/next_stage_agenda.tex`,
`ROADMAP.md`) is organized as relaxing these four, cheapest-access-first.

**Coverage / why bachelor's-level.** Scorecard covers ~69% of Wapman institutions, even across
quartiles (74.2 / 64.5 / 71.7 / 65.6% top→bottom); PSEO ~29% (only 21.5% of the top quartile, ~18
of the top-20 elite privates absent — a selected, top-truncated, public-skewed subset). PSEO
**doctoral cells at 4-digit CIP: 18,442 rows, 0 released (0%)**; master's also 0 → **the analysis is
bachelor's-level only**. BA fields with ≥10 matched institutions: Scorecard 28/30, PSEO 29/30.

---

## 2. Established findings (the four results)

Numbers verbatim from `CLAIMS_LEDGER.md`; each line ends with its binding caveat.

### Result 1 — cross-field heterogeneity is real; discipline structure is borderline
Precision-weighted multilevel decomposition `gap_f ~ 1 + (1|CIP-2)` (REML, **n=48**):
**τ²=0.0231** (SD 0.152), **ICC=0.30**, measurement-corrected **0.45**, expanded-universe **0.57**;
one-way **η²=0.50**; 18 CIP-2 clusters, 23 reliable locked-grain units. *(Descriptive variance
decomposition, salary-anchored, not causal.)*

**Phase 1 re-check (scripts/57, three-level REML meta-analysis on Fisher-z coupling, 57 fields):**
heterogeneity across fields is real (Cochran Q = 183.2 on 56 df, I² = 0.69), but the CIP-2 component is
borderline. It is significant unadjusted (permutation p = 0.005) but only p = 0.028–0.046 net of field
size (coupling rises with the number of institutions per field), its profile CI reaches 0, and it fails
leave-one-cluster-out (dropping Engineering gives p = 0.12). The legacy ICCs below (0.30 raw, 0.57
corrected) do not clear a size-preserving permutation null (p ≈ 0.06). Read the cluster pattern as a
descriptive map, not an established finding.

- **Integrated** (prestige predicts pay): Computer/Info **0.348**, Math & Stats **0.352**, Social Sci
  **0.421**, Engineering **0.432**.
- **Decoupled**: Health **0.753**, Arts **0.746**, Nat. Resources **0.730**, Biological Sci **0.678**,
  Physical Sci **0.661**. *(Shrunk partial-pooled means; raw means differ, e.g. Bio raw 0.830.)*
- Within-cluster "observability" gradient (gap vs within-field earnings dispersion): pooled Spearman
  **−0.44 [−0.74,+0.01]**, within-cluster slope **−0.527 [−1.08,+0.02]** (n=46). **CI spans 0 → weak
  second-order modifier, not headline.**

### Result 2 — one clean channel (licensing) + a large irreducible residual
Netting external field-level mechanical channels (ACS PUMS): **occupational licensing raises the gap,
b = +0.66 [+0.33,+0.97]**; academic absorption **null −0.02 [−0.42,+0.46]**; anchors near-orthogonal
(corr **−0.21**) so the null is real, not collinearity. Together **R²=0.28**. Adding pay-wedge +
public-sector channels moves R² only **0.27→0.29** — they are **collinear** with licensing (|corr| up
to 0.54) and add nothing (standardized licensing **+0.11 [+0.07,+0.14]** dominates; others collapse to
~0; residual ranking unchanged, Spearman **+1.00**). *The search for more channels finds redundancy,
not explanation.* (Pay-wedge bivariate −0.33 p=0.022; public-sector +0.37 p=0.009 collapses to ~0 once
licensing netted. Unionization is an acknowledged data gap, not fabricated.)

### Result 3 — robust to a completely different earnings source
PSEO (UI wages) vs Scorecard (Title-IV median) rank institution pay **near-identically (Spearman
+0.94, n=1535)**; the field-level **gap replicates across the two independent population definitions
(Spearman +0.68 [+0.52,+0.80], 57 fields)**. Via PSEO flows **34% work out-of-state**, yet **netting
destination geography barely moves the gap (0.68→0.67)** — the *rank* gap is geography-robust even
though geography drives earnings *levels* (origin-state R²≈0.13). **Design is cross-national-ready.**
*(108-institution coalition-public overlap; elite privates under-represented.)*

### Result 4 — the market actively reprices fields (temporal)
Real (constant-2023 $) 2001–2019, **46/60 fields rise**: Statistics **+$2,311/yr**, CS **+$1,616**,
Computer Eng +$1,421, Biomedical Eng +$969, Finance +$868, Economics +$817; sharpest falls Pharmacy
**−$1,774/yr**, Communication Disorders **−$1,405**. The frontier is the **prestige–earnings lead–lag**:
a two-period prestige pilot finds prestige stability (median **ρ≈0.50**, a noise-depressed lower bound)
against materially moving earnings — *suggestive* that prestige is the slow anchor — **but with two
noisy periods the direction is NOT identified**; the finer panel it needs is specified. *(Institution-
level earnings panel is data-blocked: PSEO single-cohort cells suppressed.)*

### What we ruled out (scope conditions — a strength, not a gap)
- **Not generic-brand vs field-specific.** On *medians* the academia-wide brand predicts pay at least
  as well as field-specific prestige (c_G **+0.45** ≥ c_F **+0.40**; field's advantage negative in
  **42/57** fields). Consistent with Chetty–Deming–Friedman: brand effects live in the elite **tail**
  medians can't see.
- **Not a distinct "shape".** Graded/threshold/flat is ~redundant with integration strength (Spearman
  **+0.80–0.96**); median data can't resolve winner-take-all.
- **Program "mispricing" is geography/selection, not value-added.** Destination-state FE explain
  **R²=0.46** of the program residual vs **0.00** for brand; residual **+0.88**-correlated with raw pay
  ranking (prestige explains ~18%); 29% horizon sign-flip.

### The two-signal model (§6) — built, **NOT validated**
One parameter vector (N=150, σ_A=0.50, σ_∞=0.30, σ_0=1.00, δ=0.70) reproduces P2 full-support effect
**≈+0.64** (~empirical +0.66), P3 absorption null, P5 gap-change↔divergence +0.71 — with **no per-fact
tuning**. But its key external prediction (**residual = valuation divergence**) is **UNCONFIRMED**:
O*NET proxy **+0.41 [+0.08,+0.68]** vs SDR proxy **−0.61 [−0.88,−0.11]**, the two proxies *negatively*
correlated (−0.64). Kept as an **interpretive lens, explicitly not validated** (open decision #3).

---

## 3. What the re-checks changed (pilot 47–51, Phase 1 scripts 52–57)

Every newer result was adversarially re-verified; several earlier claims did not survive. Prose
write-ups are local (`*_RESULT.md`); numbers below come from the committed scripts.

- **Selectivity absorbs most of the level of coupling (scripts/55).** Net of institution SAT, admit
  rate, Pell share, control and state, mean coupling falls from +0.43 to +0.14 (SAT sample, 46 fields).
  In a within-institution cross-department design, the department-prestige slope falls from +0.084 to
  +0.013. The cross-field *ordering* still beats a shared-noise null (Spearman +0.63 vs null +0.29,
  p = 0.006), but it is nearly collinear with how much each field pays for selectivity (true-score
  r = +0.94), and public data cannot separate the two readings (SAT-based checks lean toward
  selectivity pricing, admit-rate-based checks the other way). CS/economics vs nursing survives the
  broad adjustment (+0.33 [+0.09, +0.57]) but not the strict one (+0.11 [−0.31, +0.52]).
- **Brand vs field prestige (scripts/56).** Academia-wide brand predicts pay at least as well as
  field-specific prestige, and correcting for field prestige's lower reliability does not reverse
  this. Under the pessimistic reliability bound the difference is −0.005 [−0.043, +0.044]; the safe
  claim is "field prestige is no better than brand", not "brand is better".
- **Career-time coupling (scripts/52, rewritten).** On fixed PSEO cohorts 2001/2004/2007/2010 (balanced
  institutions, 39 fields), coupling rises +0.031/yr [+0.021, +0.038], in every cohort. A
  calendar-matched contrast bounds the career-time part to [+0.014, +0.032]/yr. The rise is general
  across fields (integrated minus others +0.008, p = 0.33) and loads on academia-wide brand
  (+0.038/yr) rather than field prestige (−0.001/yr).
- **Discipline structure is borderline (scripts/57)** — see the note under Result 1.
- **Extension pilot (47–51): no second mechanism.** Disciplinary kinship does not organise coupling
  (faculty-flow Moran's I +0.011, p = 0.96; the CIP-2 signal was the nursing/communication-disorders
  pair). State × occupation netting shows nursing's premium is its state's RN wage (raw $5k → $0,
  n = 22,518). The mechanism is the paper's own: **pay set by setting (state, employer, pay scale),
  prestige-orthogonal — not wage compression** (corr(licensure, wage CV) = +0.02). The within-occupation
  dispersion law (scripts/53) is null in 0 of 240 principled specifications.
- **Phase 2 (scripts 58–64, public data):**
  - *Department signal beyond status (58).* After selectivity, program Pell share, state and academia-wide
    prestige, only a small department-prestige premium remains; whether its average differs from zero depends
    on the specification. Computer science is the one field where it is clearly positive (pre-named, corrected
    CI [+0.007, +0.114]); engineering and business fields lean positive but do not survive multiple-testing
    correction.
  - *UK with a direct prior-attainment control (62).* In DfE LEO, 85–90% of UK prestige–earnings coupling
    survives standardising on graduates' prior-attainment bands (conservative floor 62%). What survives tracks
    intake selectivity rather than research prestige. Coupling rises over careers (+0.026/yr), and nursing,
    medicine and teaching are nearly decoupled (+0.05 vs +0.46 for other subjects).
  - *Brand, not field, over careers (59).* On the August-2026 PSEO release nothing changes: career-time slope
    +0.030/yr; the brand-over-field loading (dG − dF) holds under two-way clustered intervals. p75 coupling is
    only marginally above the median; p25 is clearly below.
  - *Elite tail (60).* In Opportunity Insights college outcomes, academia-wide prestige adds little beyond
    selectivity and parental income, and no more for the top 1% than for the median.
  - *Non-wage placement (61).* Placement into market-priced industries couples with prestige about as strongly
    as pay does, mostly through selectivity. Whether the pay gradient flattens with setting-priced or with
    market-sector exposure depends on when sector shares are measured: not identified.
  - *Setting-priced fields (63, 64).* Nursing's prestige carries little pay information beyond geography —
    "little", not "none". The cross-field licensing gradient is robust in sign, but the US tests cannot
    separate it from an engineering/CS/math/business-versus-other contrast, and broad licensure measures show no
    moderation. Treat it as a descriptive boundary condition. The four field-level nulls (scripts 50–53) are
    underpowered: none rules out a moderate effect.
  - *Prior work (literature merge).* Bloem, Hu & Hurwitz (2024; US Scorecard) and Britton et al. (2022; UK
    LEO) already report the field pattern and the weakness of within-subject rankings beyond selectivity. What is
    new here is the faculty-hiring hierarchy as the prestige axis, a reliability-gated US field map replicated on
    two earnings sources, and fixed-cohort career-time rises in coupling (US and UK) that load on
    institution-wide status.
- **Withdrawn claims:** wage compression as the mechanism or a continuous law; the "clean same-cohort"
  PSEO panel (the pooled cohort mixes cohorts across horizons); "compressed fields pinned at 0";
  "field-agnostic rankings mis-price programs" (untested and contradicted by brand ≥ field); "aggregate
  data sees nothing, so resume data is necessary" (restated: public aggregates cannot identify the
  within-occupation channel).

---

## 4. Repository & engineering architecture

```
src/                        analysis library (importable, tested units)
  gap.py                    generic gap = 1 − Spearman(AR,ER) + reliability flags
  load_ar.py load_er.py     source-normalized AR / ER loaders
  diagnostic.py             signal-vs-artifact battery (null-model, bootstrap, residualization)
  dispersion.py             within-field earnings dispersion (ACS / PSEO)
  genreg_bootstrap.py       generated-regressor inference for the dispersion test
  anchored_bootstrap.py     published-rank-anchored SpringRank bootstrap (AR-noise propagation)
  predictions.py            two-signal employer-learning model predictions
  tier0.py tier0_5.py       coverage / feasibility probes
  ar_pipeline/springrank.py self-contained SpringRank (regularized, seeded)
  crosswalks/fields.py      field ↔ CIP ↔ SDR ↔ FOD1P key matching (TIER0-30 + expansion)
  crosswalks/institutions.py institution-name normalization / join
  openalex_tag.py           OpenAlex field-tagging (ORCID → 26 research fields)
scripts/                    numbered pipeline stages 01→51 (+ run_* tier0 runners)
data/raw/SOURCES.md         full provenance manifest (URL, version, access date, license)
outputs/  results/          committed figures (*.png) + tables (*.csv); prose *.md kept LOCAL
paper/                      main.tex, methods_companion.tex, extended_abstract.md,
                            CLAIMS_LEDGER.md (audit trail), COAUTHOR_NOTES.md (decisions),
                            next_stage_agenda.tex (platform → frontier)
Makefile                    make setup / check-data / tier0 / diagnostic / gap / expand /
                            granularity / orcid / openalex / all / figures / clean
```

**Engineering conventions (important to know):**
- **Reproducible & seeded.** Every script reruns deterministically (SEED fixed; pilot scripts verified
  byte-identical across runs). `make all` regenerates every derived artifact from documented raw.
- **What's tracked vs local.** Committed = code + `outputs/`&`results/` figures/CSVs + `README.md`.
  **NOT tracked** (by `.gitignore`, deliberately): all raw & interim data (`data/**`), and **every
  prose `*.md` except this README** (`/*.md` + `!/README.md`; `results/*.md`, `outputs/*.md`). The
  narrative RESULT reports live locally + in git history — they are the analysis record, not a shipped
  artifact. **No datasets are redistributed in this repo.**
- **Data footprint** (for any data handoff): `data/raw` ≈ 3.7 GB (all open: ACS 2.8G, HSLS, PSEO,
  Scorecard, IPEDS, LEO, ONET, UCAS, ELS, Wapman…); `data/interim` ≈ 25 MB (derived analysis-ready
  tables); ORCID ≈ 5.5 GB (supplied separately). 13 scripts read `data/raw` directly, 30 read
  `data/interim` → interim-alone is **not** a sufficient handoff; ship raw+interim or fetch via `SOURCES.md`.
- **Version-pinning gotcha.** Scorecard is a *dated snapshot* (`...Field-of-Study_06102026.zip`);
  re-fetching "most recent" yields a *different vintage* → numbers drift. Share the pinned zip to match.

---

## 5. Experiment catalog (design → finding → status)

Every experiment, its script, one-line design, the finding, and status
(**●** established · **○** robustness/support · **∅** null/ruled-out · **⧗** pending).
Engineering caveats are in the finding text (not omitted).

| Script | Experiment | Design (data + method) | Finding / status |
|---|---|---|---|
| 01 | Coverage probe | Scorecard/PSEO × Wapman institution & field overlap by quartile | ○ bachelor's-only justified; PSEO top-truncated |
| 02 | Gap map | `gap_f = 1−Spearman(AR,ER)` per field, reliability-flagged | ● the core object |
| 03–07 | P1 / regime / sensitivity | is-the-gap-signal battery; regime labels; reliability sensitivity | ○ REAL SIGNAL (filter required), artifact R²=0.20 |
| 08 | Field expansion | 30 TIER0 → 54/66 field universe via crosswalk | ○ rank +0.98, ICC 0.30→0.32 raw |
| 09 | SpringRank recalibration | from-scratch SpringRank on public edges vs published | ○ ρ≈0.77 data ceiling (lossy edges, not a bug) |
| 10–11 | ORCID SpringRank rebuild | independent prestige from ORCID PhD→faculty edges | ○ ρ=0.74 vs Wapman-academia (method cert.) |
| 12 | Licensing tag | hand-coded occupational-license dummy per field | ● input to Result 2 mechanism |
| 13–14 | Granularity + cluster decomp | `gap ~ (1|CIP-2)` REML; locked-grain reliable units | ● Result 1 (ICC 0.30, η²=0.50) |
| 15–19 | OpenAlex field tagging | tag ORCID by OpenAlex research field, re-test AR agreement | ∅ research-field tags don't beat degree text; coverage-bound |
| 20 | Mixture decomposition | licensing + academic-absorption anchors net the gap | ● Result 2 (b_lic +0.66, absorption null) |
| 21 | Multi-horizon | ICC at 1yr vs 4–5yr earnings | ○ ICC 0.50@1yr → 0.30@4-5yr; early-career only |
| 22 | Hardening | n-subsample rank stability; measurement correction; AR-noise propagation | ○ rank +0.98; ICC 0.30→0.45; 0/16 flips |
| 23,25,26 | Two-signal model / validation / ICC refinement | simulate employer-learning; test residual=divergence; discipline-varying σ0 | ● built / ∅ **UNCONFIRMED** (O*NET vs SDR opposite signs) |
| 24 | Crosswalk audit | 27 missing fields → recoverable/subsumed/genuine | ○ 54→66 holds; ceiling mostly genuine |
| 27 | External channels | add pay-wedge + public-sector channels | ∅ collinear with licensing, R² 0.27→0.29, add nothing |
| 28 | Field-vs-generic prestige | field-specific vs academia-wide brand on medians | ∅ brand ≥ field (c_G +0.45 ≥ c_F +0.40); tail invisible |
| 29 | Two-market integration | graded/threshold/flat shape classification | ∅ ~redundant with strength (+0.80–0.96) |
| 30 | Mispriced programs | program residual: destination-state FE vs brand | ∅ mispricing = geography (R²=0.46) not value-added |
| 31 | Prestige autonomy | does school prestige add beyond geography | ○ prestige ≈0 incremental beyond geography |
| 32 | PSEO gap cross-validation | replicate gap on PSEO UI wages; net destination geography | ● Result 3 (level +0.94, gap +0.68, geo-robust) |
| 33 | Dynamic lead–lag pilot | real-$ revaluation 2001–2019; two-period prestige stability | ● revaluation fact / ⧗ lead-lag NOT identified |
| 34 | CS-school ranking | public-facing prestige×earnings demonstrator, cross-validated | ○ application artifact (CMU top; 15 low-robustness flags) |
| 35 / 35a–c | ER-axis **reframe** + placement-proxy identification | earnings = *revealed-placement* proxy; validate; 2 failure modes; licensing marks setting-determined pay (not compression) | ● the platform reframe (validated-but-incomplete proxy) |
| 40–41 | Cross-national UK (licensing) | UK faculty-hiring prestige × LEO earnings feasibility | ⧗ feasibility for Thrust A |
| 42–44 | Behavioural feasibility + v2 + ELS | HSLS/ELS over-credit vs gap; pre-registered battery | ∅ under-powered on public data (secondary/SI at best) |
| 45 | Demand feasibility | revealed-demand (applications vs acceptances) gate | ∅ US no-go; UK gated |
| 46 | Two-slope UK | UK make-or-break: revealed-demand paradox | ∅ paradox does NOT survive (only Engineering, non-decoupled) |
| 47 | Pilot Ph1 — coupling×kinship | ICC + ARI/NMI of ρ vs CIP-2 | ∅ ARI null; ICC = existing structure |
| 48 | Pilot Ph3-4 — netting/unification | ACS state×NAICS netting; A+B unification | ∅ industry netting misclassifies nursing; unification not confirmed |
| 49,51 | Pilot — faculty-flow + Moran | build kinship graph; Moran's I of coupling | ∅ **Idea A killed** (I=+0.011, p=0.96) |
| 50 | Pilot — occupation channel | STATE×OCCP netting; occ_hhi ↔ coupling | ● nursing $5k→$0 correction / ∅ concentration null (CS counterexample) |
| 52 | Career-time coupling (rewritten) | PSEO fixed cohorts 2001–2010, balanced; two-stage bootstrap; calendar-matched bound; brand vs field | ● +0.031/yr, general across fields; brand-loaded |
| 53 | Within-occupation dispersion law | ACS PUMS occupation wage dispersion → coupling | ∅ null (0/240 robust specifications) |
| 55 | Selectivity controls | Scorecard institution SAT/admit/Pell/state; partial coupling; within-institution FE | ⧗ level mostly selectivity; ordering not separable from selectivity pricing |
| 56 | Prestige reliability | split-half SpringRank on public edges; disattenuated c_F vs c_G | ○ field prestige no better than brand |
| 57 | Three-level ICC | REML meta-analysis, permutation nulls, field-size adjustment | ⧗ discipline component borderline |
| 58 | Selectivity deep-dive | program Pell share; prestige residualized on brand/selectivity; within-institution FE with field-specific slopes | ○ small remainder; clear only in CS |
| 59 | PSEO Aug-2026 refresh | V4.14.1 re-run of career time, brand vs field, quantiles, cross-source | ○ no Phase 1 conclusion changes |
| 60 | Elite tail (Opportunity Insights) | college top-1% shares vs brand, selectivity, parental income | ∅ brand adds little beyond selectivity; tail no stronger than median |
| 61 | PSEO Flows placement | industry shares as non-wage placement; cell-level exposure interactions | ○ placement couples like pay; setting-vs-market attribution not identified |
| 62 | UK LEO | academia-wide UK prestige; prior-attainment bands; fixed-cohort career time | ● 85–90% survives attainment control; +0.026/yr |
| 63 | Hierarchical model + power | cell-level random slopes; licensure moderators; MDE/TOST for nulls | ⧗ licensing moderation not separable from field type; nulls underpowered |
| 64 | Licensing battery | strict/broad/multilevel; adjusted and CV R²; signed partial r; selectivity | ○ gradient robust in sign; nursing 'little', not 'none' |
| 65 | Coupling dynamics | 17-cell cohort × horizon surface (PSEO V4.14.1); APC bounds; recession/COVID entry; state unemployment × status | ○ rises along both axes (+0.031/yr within cohort, +0.015/yr across windows); entry-shock tests null/opposite; curvature identified only under piecewise-linear periods |
| 66 | Employer-learning loadings | loadings of G, F, SAT, Pell by horizon (US), G vs selectivity by YAG (UK) | ∅ no status-loading decay; nearly guaranteed for programme means under rational expectations (weak evidence) |
| 67 | Graduate-school timing | ACS graduate-degree share by field; PhD production (SED) and coverage controls; y5→y10 segment | ⧗ rise survives each pre-set check, but a proportional grad-study confound cannot be excluded (power 0.17–0.31) |
| 68 | Era-matched selectivity | historical Scorecard SAT/ADM/Pell matched to entry cohorts | ○ level and within-institution slope unchanged; career slope net of entry-year selectivity 0.39 of raw, > 0 |
| 69 | Local field demand | ACS field × state earnings in the within-institution design | ∅ department slope unchanged (Δ +0.0000) |
| 72 | UK robustness | log-pay gradients; sex/POLAR4 standardisation; validity bounds for UK G | ○ survival in pay terms 73% (66% with sex/POLAR4) vs 85% on ranks |
| 73 | Earnings reliability | median sampling noise propagated into τ, I², three-level model | ○ I² 0.69 → 0.51; field ordering survives (0.95) |
| 74 | CS admission routes | direct-admit / capped / open coding of CS programmes (official pages) | ∅ no detectable difference (weak test) |
| 75 | Theory tests T0–T8 | construct audits (production, single index, location, sectors) and model predictions | ⧗ F partly production (R_DC 0.84, R_⊥ 0.60); not one index; pay location-robust; UK within-band premium rises; specified before running, not publicly registered |

---

## 6. Status & what's next

**Done.** The measurement object (within-field prestige–earnings coupling, reliability-gated) and its
cross-source replication are solid; three alternative explanations are ruled out; licensing marks fields
where pay is set by setting. The Phase 1 re-checks narrowed the claims: most of the coupling level is
institutional selectivity/brand, discipline structure is borderline, and the career-time rise is general
and brand-loaded.

**Phase 3 (2026-09-25).** Paper rewritten for EPJ Data Science with a theory spine (scripts 65–75): the two
indicators are read as revealed valuations of two audiences — the academy (net-placement order of hiring;
a valuation only under volume neutrality and vertical sorting) and employers (programme pay, selection
included). Claims are layered: *relational* (agreement of the two orderings), *construct* (stated with the
assumptions that can fail and the audits T0/T1/T7/T8), *mechanism* (not claimed: information vs content not
identified). The theory was written after most results; tests T0–T8 were specified before running but not
publicly registered.

**Working headline:** where the academy's and the market's orderings of the same institutions agree, they
agree at the institution's name, mostly in what both share with the selectivity of its intake; below the name
they agree little (about 0.8% of pay per SD of department rank, up to about 3% under the lower-bound
reliability), clearly only in computer science. Agreement grows with years since graduation in the US and UK
(a graduate-study timing confound cannot be excluded) and is weak where pay is set by schedule.

**Pending → [`ROADMAP.md`](ROADMAP.md)** (public data only).

**Open decisions** (`paper/COAUTHOR_NOTES.md`): venue (descriptive CSS vs bolder NHB push); how hard to
lean on the ranking critique; keep/cut/validate the §6 model; add UK replication before submission or cite
as next step; outreach angle (beliefs vs networks); scope of the public CS-ranking artifact.

---

## 7. Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
make setup                      # pip install -r requirements.txt (scikit-learn, networkx, statsmodels…)
make check-data                 # reports which core inputs from data/raw/SOURCES.md are still missing
make all                        # public-data pipeline: Tier 0 → gap → cluster decomposition
```
Two stages need extra inputs, run separately: `make orcid` (Zenodo record **19651302** → place under
`data/orcid/`) and `make openalex` (OpenAlex API key in gitignored `secrets.json`). Individual
groups: `make tier0 tier05 diagnostic gap expand granularity`. `make clean` removes regenerated outputs.

## 8. Data & licensing
All inputs are open, each under its own terms, documented in
[`data/raw/SOURCES.md`](data/raw/SOURCES.md) (Wapman CC BY 3.0; College Scorecard, NCSES SDR/SED, IPEDS,
Census ACS-PUMS/PSEO public domain; ORCID & OpenAlex CC0). **No datasets are redistributed here.** Code
under MIT ([`LICENSE`](LICENSE)). Cite this repo + the upstream sources, in particular Wapman, Zhang,
Clauset & Larremore, *Quantifying hierarchy and dynamics in US faculty hiring and retention*, Nature 610
(2022).
