# Field Granularity — CIP-2 roll-up + multilevel meta-regression

Outcome-agnostic; ex-ante license/regime tags; AR anchored on Wapman edges (canonical SpringRank α=0.5, binary). Run: `python scripts/13_granularity.py`. Date 2026-06-20.

## Task 1 — CIP-2 hierarchy + license-aware roll-up

Parent → children (★ = fine-reliable, [L] = license):

- **CIP01 Agriculture** (4 fields, 0 reliable): Agricultural Economics, Agronomy, Animal Sciences, Food Science
- **CIP03 Nat. Resources** (2 fields, 0 reliable): Environmental Sciences, Forestry
- **CIP04 Architecture** (1 fields, 0 reliable): Architecture
- **CIP11 Computer/Info** (1 fields, 1 reliable): Computer Science★
- **CIP13 Education** (1 fields, 0 reliable): Special Education
- **CIP14 Engineering** (10 fields, 1 reliable, MIXED-LICENSE): Aerospace Engineering, Biomedical Engineering, Chemical Engineering, Civil Engineering[L], Computer Engineering, Electrical Engineering★, Environmental Engineering, Industrial Engineering, Materials Science, Mechanical Engineering
- **CIP23 English** (1 fields, 1 reliable): English★
- **CIP26 Biological Sci** (7 fields, 1 reliable): Biochemistry, Biology★, Biostatistics, Ecology, Microbiology, Neuroscience, Physiology
- **CIP27 Math & Stats** (2 fields, 1 reliable): Mathematics★, Statistics
- **CIP38 Philosophy/Religion** (2 fields, 1 reliable): Philosophy★, Religious Studies
- **CIP40 Physical Sci** (5 fields, 0 reliable): Astronomy, Atmospheric Sciences, Chemistry, Earth Sciences, Physics
- **CIP42 Psychology** (1 fields, 1 reliable): Psychology★
- **CIP44 Social Work** (1 fields, 0 reliable): Social Work
- **CIP45 Social Sci** (5 fields, 2 reliable): Anthropology, Economics★, Geography, Political Science★, Sociology
- **CIP50 Arts** (2 fields, 0 reliable): Music, Theatre
- **CIP51 Health** (4 fields, 2 reliable, MIXED-LICENSE): Communication Disorders★[L], Nursing★[L], Pharmacy, Public Health
- **CIP52 Business** (4 fields, 4 reliable, MIXED-LICENSE): Accounting★[L], Finance★, Management★, Marketing★
- **CIP54 History** (1 fields, 1 reliable): History★

### Rolled-up parent units (thin children pooled to the 2-digit parent)

| name                |   n_children |   n_inst |   gap |   signal_frac | reliable   | mixed_license   |
|:--------------------|-------------:|---------:|------:|--------------:|:-----------|:----------------|
| Agriculture         |            4 |       35 | 0.491 |          0.53 | False      | False           |
| Nat. Resources      |            2 |       52 | 0.690 |          0.27 | False      | False           |
| Engineering         |           10 |      126 | 0.409 |          0.86 | True       | True            |
| Biological Sci      |            7 |      149 | 0.667 |          0.71 | True       | False           |
| Math & Stats        |            2 |      105 | 0.403 |          0.87 | True       | False           |
| Philosophy/Religion |            2 |       36 | 0.564 |          0.68 | False      | False           |
| Physical Sci        |            5 |       86 | 0.580 |          0.61 | True       | False           |
| Social Sci          |            5 |      150 | 0.305 |          0.95 | True       | False           |
| Arts                |            2 |       89 | 0.747 |          0.59 | True       | False           |
| Health              |            4 |      104 | 1.058 |          0.83 | True       | True            |

- **Reliable units after roll-up: 23** (16 fine-reliable + 7 rolled-up reliable parents).
- **Newly represented (previously-dumped) fields:** Aerospace Engineering, Anthropology, Astronomy, Atmospheric Sciences, Biochemistry, Biomedical Engineering, Biostatistics, Chemical Engineering, Chemistry, Civil Engineering, Computer Engineering, Earth Sciences, Ecology, Environmental Engineering, Geography, Industrial Engineering, Materials Science, Mechanical Engineering, Microbiology, Music, Neuroscience, Pharmacy, Physics, Physiology, Public Health, Sociology, Statistics, Theatre.
- **Mixed-license parents (excluded from the clean observability test):** CIP14 Engineering, CIP51 Health
. Engineering (CIP14), Health (CIP51), Business (CIP52) mix license + non-license at the fine grain and so cannot be pooled cleanly for the observability test.

## Task 2 — precision-weighted multilevel meta-regression

`gap_f ~ dispersion_f + license_f + (1 | CIP2_parent)`, GLS with V = diag(SE²) + τ²·(same-parent), τ² by REML (per-field SE = bootstrap-CI half-width; floored 0.05):

| term             |   coef |    se |     lo |     hi |
|:-----------------|-------:|------:|-------:|-------:|
| intercept        | +0.545 | 0.042 | +0.462 | +0.628 |
| dispersion_slope | -0.044 | 0.021 | -0.085 | -0.004 |
| license_shift    | +0.063 | 0.058 | -0.051 | +0.178 |

- **Parent-cluster variance τ² = 0.0248** (SD 0.158 gap-units) — the discipline cluster matters: a non-trivial share of gap variance is between-CIP2.

- Shrinkage toward the parent mean is strongest for thin/noisy fields (per-parent shrinkage 0.42–0.94); high-SE fields borrow most.

- **Dispersion slope (observability, license-controlled, partially pooled) = -0.044 [-0.08, -0.00]**; **license shift = +0.063 [-0.05, +0.18]**.

## Task 3 — observability gradient, three ways (outcome-agnostic)

| estimate | n | slope/Spearman | 95% CI |
|---|---|---|---|
| fine non-license Spearman (12_license) | 13 | -0.36 | (CI spans 0) |
| rolled-up + fine reliable non-license | 18 | -0.31 | [-0.69, +0.23] |
| multilevel dispersion slope (license-controlled) | 46 | -0.044 | [-0.08, -0.00] |

### Verdict (what the numbers warrant)

- The observability channel stays **weak**. Both small-n rank views span 0 (fine non-license -0.36; rolled-up -0.31 [-0.69,+0.23]). The precision-weighted multilevel slope is small and **only marginally bounded** (-0.044 [-0.08,-0.00] — upper CI ≈ 0), driven by adding all 46 fields with 1/SE² weighting + partial pooling, not by a robust gradient. Partial pooling and roll-up improve **coverage and rigor, not the information** — they cannot create signal that isn't there.
- **The license shift also weakens under precision weighting:** +0.063 [-0.05,+0.18] (CI spans 0), vs the +0.42 unweighted OLS in `12_license`. The big shift was driven by the two **highest-SE** license fields (communication disorders, nursing); 1/SE² weighting down-weights them, and low-SE low-gap **accounting** pulls the license mean back — so license fields are **heterogeneous**, not a clean uniform up-shift.
- **The dominant structure is the discipline cluster** (τ²=0.025, SD 0.16 gap-units > either covariate). The robust finding is the **multi-dimensional decomposition** (observability / credential-standardization / talent-exit as discipline-clustered axes), not a single P1 gradient nor a single license shift. The clean fine-grained fix is **more institutions per field (Revelio)**, not aggregation.

*Note: coarser rolled-up units are more heterogeneous (within-parent dispersion of gap is larger); the integration reading is cleanest at the FINE grain in prestige-transmission fields. Partial pooling improves rigor, not the fundamental information.*

## Figures
`all_discipline_gap_map.png` (every discipline placed, CI bars, by regime) · `multilevel_coefs.png` (multilevel coefficients).
