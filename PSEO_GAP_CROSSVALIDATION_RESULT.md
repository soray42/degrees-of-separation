# PSEO gap cross-validation + flows/geography control

Bringing in **PSEO (Census LEHD)** — UI-wage-based, population-inclusive of UI-employed grads, in **real 2023 dollars** — as an INDEPENDENT earnings definition to cross-validate the gap and to make the TASK-2 geography confound an explicit, measured control. We do **not** naively merge PSEO with Scorecard (different populations + non-random coalition coverage). Descriptive, outcome-agnostic. Run: `python scripts/32_pseo_gap_crossval.py`.

## Coverage (honest caveat)

- PSEO BA earnings (5yr P50, pooled cohort, released cells) cover **502 institutions**, of which **108 overlap the Wapman prestige set** — a **SELECTED ~subset** (PSEO is a voluntary state/institution coalition, non-random; UI wages miss the federally-employed, self-employed, and out-of-state-with-no-coalition movers).
- 57 fields have enough PSEO∩Wapman institutions for a gap. Read everything below as holding on this selected overlap, not the full universe.

## CROSS-VALIDATION — does the gap survive a different earnings source? (headline)

- **Level agreement:** Spearman(y_PSEO, y_Scorecard) on the same institution×field = **+0.94** (n=1535). Two different earnings definitions (UI wages vs Title-IV median) rank institutions' pay very similarly.
- **Gap agreement (the test):** Spearman(gap_PSEO, gap_Scorecard) across 57 fields = **+0.68** [+0.52, +0.80]. The gap — and the integrated↔segmented ranking — **replicates** across two independent population definitions.

| label                     |   n_PSEO |   gap_PSEO |   n_Scorecard |   gap_Scorecard |
|:--------------------------|---------:|-----------:|--------------:|----------------:|
| Food Science              |       13 |       0.27 |            16 |            0.24 |
| Statistics                |       19 |       0.33 |            28 |            0.25 |
| Computer Science          |       84 |       0.28 |           161 |            0.29 |
| Computer Engineering      |       50 |       0.37 |            74 |            0.30 |
| Economics                 |       65 |       0.58 |           116 |            0.31 |
| Political Science         |       75 |       0.41 |           139 |            0.34 |
| Biomedical Engineering    |       24 |       0.62 |            48 |            0.35 |
| Kinesiology/Exercise Sci  |       54 |       0.55 |            70 |            0.38 |
| Mechanical Engineering    |       65 |       0.47 |           133 |            0.38 |
| Mathematics               |       83 |       0.42 |           120 |            0.38 |
| Finance                   |       56 |       0.33 |            89 |            0.41 |
| History                   |       81 |       0.62 |           131 |            0.41 |
| Human Dev & Family Sci    |       22 |       0.59 |            31 |            0.42 |
| Health/PE/Recreation      |       18 |       0.49 |            15 |            0.43 |
| Psychology                |       88 |       0.57 |           173 |            0.43 |
| Sociology                 |       74 |       0.58 |           114 |            0.43 |
| Public Health             |       22 |       0.73 |            32 |            0.44 |
| Aerospace Engineering     |       22 |       0.42 |            35 |            0.44 |
| Special Education         |       16 |       0.50 |            19 |            0.45 |
| Chemical Engineering      |       45 |       0.63 |            91 |            0.45 |
| Marketing                 |       52 |       0.33 |            81 |            0.48 |
| Electrical Engineering    |       69 |       0.44 |           118 |            0.49 |
| Accounting                |       59 |       0.25 |            94 |            0.49 |
| Materials Science         |       19 |       0.74 |            23 |            0.49 |
| Civil Engineering         |       59 |       0.48 |           105 |            0.49 |
| Linguistics               |       23 |       0.75 |            26 |            0.50 |
| English                   |       80 |       0.57 |           135 |            0.51 |
| Industrial Engineering    |       32 |       0.19 |            39 |            0.51 |
| Anthropology              |       61 |       0.63 |            75 |            0.51 |
| Management                |       60 |       0.36 |            90 |            0.52 |
| Social Work               |       44 |       0.61 |            68 |            0.55 |
| Pharmacy                  |       12 |       1.64 |             8 |            0.57 |
| Biochemistry              |       39 |       0.71 |            47 |            0.58 |
| Theatre                   |       39 |       0.89 |            46 |            0.62 |
| Spanish Lang & Lit        |       20 |       0.52 |            30 |            0.62 |
| Ecology                   |        9 |       1.59 |            11 |            0.65 |
| Biology                   |       75 |       0.63 |           137 |            0.65 |
| Physics                   |       73 |       0.94 |            43 |            0.65 |
| Philosophy                |       68 |       0.62 |            41 |            0.66 |
| Chemistry                 |       82 |       0.80 |            76 |            0.69 |
| Earth Sciences            |       60 |       0.73 |            50 |            0.72 |
| Nutrition Sciences        |       28 |       0.64 |            30 |            0.73 |
| Geography                 |       46 |       0.88 |            41 |            0.76 |
| Teacher Ed (subjects)     |       27 |       0.98 |            36 |            0.77 |
| Environmental Sciences    |       32 |       1.07 |            48 |            0.77 |
| Architecture              |       30 |       0.68 |            34 |            0.78 |
| Animal Sciences           |       24 |       0.75 |            35 |            0.80 |
| Environmental Engineering |       18 |       1.02 |            19 |            0.81 |
| Urban & Regional Planning |       12 |       0.51 |            12 |            0.87 |
| Agricultural Engineering  |       15 |       0.56 |            14 |            0.91 |
| Nursing                   |       59 |       0.97 |            99 |            0.93 |
| Agronomy                  |       12 |       1.00 |            11 |            0.93 |
| Microbiology              |       17 |       0.59 |            17 |            0.93 |
| Music                     |       64 |       1.01 |            76 |            0.96 |
| Forestry                  |       15 |       1.11 |            17 |            1.15 |
| Communication Disorders   |       20 |       0.90 |            17 |            1.21 |
| Physiology                |       10 |       1.02 |            16 |            1.34 |

## FLOWS — geography confound made explicit (addresses TASK 2)

- **Out-of-state leakage (UI undercount):** weighted across institutions, only **66%** of BA grads are employed IN the institution's state at 5yr → **34% work out-of-state**, where a single state's UI misses them. This bounds the PSEO undercount and is itself the destination-mobility signal.
- **Geography's share of WITHIN-field earnings (PSEO-measured):** state fixed effects explain **0.13** (origin state) and **0.04** (dominant destination state, institution-level — the flows are only CIP-2) of within-field cross-institution earnings; mean destination concentration HHI 0.59 (graduates cluster regionally). These are MODEST on this overlap — smaller than TASK 2's origin-state R²≈0.46 on Scorecard institution-mean residuals, because the PSEO∩Wapman set is only 108 mostly-coalition-public institutions (a narrower, more homogeneous sample) and the destination measure is coarse (institution-level, all-CIP). Honest: geography is a real but here-modest share, not the whole story.
- **Netting destination geography barely moves the gap:** mean gap_PSEO 0.68 → after residualizing earnings on destination-state FE **0.67** (n=60 fields). The prestige↔pay (mis)alignment is **robust to geography netting** — complementing TASK 2: destination geography drives institution-level earnings *levels/residuals*, but the field-level prestige↔pay *rank gap* survives it.

## Adversarial self-check

1. **PSEO is a selected ~subset** (voluntary coalition; UI-bounded). The cross-validation holds only on the 108-institution Wapman overlap, skewed to coalition-state publics — elite privates (the brand tail) are under-represented, exactly where Chetty's tail effects live.
2. **Two earnings definitions differ** (UI all-employed vs Title-IV median); the level Spearman (+0.94) shows they co-move but are NOT the same number — we cross-validate the RANK-based gap, never concatenate the dollars.
3. **Out-of-state leakage (34%)** means PSEO national earnings still net across destination states, but in-state UI files miss movers — the flows quantify this rather than assume it away.
4. **Geography control is destination-STATE FE**, not full wage levels; selection/value-added remains unfixed (a high-paying destination may reflect who moves there). Descriptive only; cite Chetty/MacLeod for causal backing, not re-estimated here.
5. **No naive PSEO+Scorecard merge:** the two are kept separate and only the gap (a within-source rank statistic) is compared across them.
