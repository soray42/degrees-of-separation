# Research pilot (Ideas A + B) --- kinship×coupling and geography/occupation netting

Two extension ideas run as **depth-first pilots** to decide if either is worth formalizing. No-fabrication run; seeded/deterministic. Phase 1-4 executed with Sonnet 5, Phase 5 synthesis with Opus 4.8. Reproduce: `python scripts/47_phase1_coupling_kinship.py`, `scripts/48_phase3_4_netting_unification.py`, `scripts/49_faculty_flow_net.py` + `scripts/51_moran_kinship_coupling.py`, `scripts/50_occupation_channel.py`. Coupling scalar $\rho_f=\mathrm{Spearman}_i(\text{prestige}_{i,f},\text{earnings}_{i,f})=1-\text{gap}_f$.

## Verdict (one line)

**Two ideas, four independent analytic lenses, all collapse back onto the paper's established licensing / wage-compression mechanism.** No new organizing mechanism found. **NO-GO** for A and B as standalone findings; the A+B unification ("big news" target) is **not confirmed**. One reusable product survives for the paper.

## Idea A --- disciplinary kinship × coupling

Do academically-kin fields share a characteristic $\rho_f$?

| lens | method | result | read |
|---|---|---|---|
| 1. variance | ICC($\rho_f$ \| CIP-2) | 0.30 / 0.45 (broad n=57); reliable-subset 0.63+ is a d.o.f. artifact (3/15 CIP-2 groups have >1 reliable field) | real but modest = **existing** discipline structure |
| 2. hard clustering | $\rho$-clustering vs CIP-2 partition, ARI / NMI | **NULL** | CIP is a single-parent TREE, kinship is a GRAPH (Stats near both Math & CS) |
| 3. network autocorr ("elegant reframe") | Moran's $I$ of $\rho_f$ over data-driven **faculty-hiring kinship graph** (ORCID PhD→faculty cross-field flows) | $I=+0.011$, perm $p=0.96$, **robust NULL** across 7 W-constructions, 5 seeds, B=100k; CI [−0.18,+0.19] tight enough to catch a moderate effect | coupling does **not** flow over empirical kinship; reframe **fails** |
| 3b. control | CIP-2 co-membership Moran's $I$ | $+1.71$, $p=0.002$ **but** adversarial leave-pair-out: **84% of the numerator is the single nursing/comm-disorders CIP-51 license pair**; drop it → non-significant | not a taxonomy property — it's **licensing in disguise** |
| 4. Stats↔CS dual-kinship | faculty-flow network position | **FALSIFIED**: Stats exchanges faculty 9.2:1 with Mathematics over CS (config-model affinity 9.74× math vs 0.79× CS ≈ chance) | Stats is a Math satellite (partly combined Math&Stats dept artifact) |

Power ceiling: reliable n≈20 (only 14 reach the faculty-flow network). Finer CIP granularity cannot fix n (bottleneck = institutions-per-field; sub-field earnings/prestige don't exist in Scorecard). Nursing has a genuinely distinctive flow signature (most self-contained, imports from psych/socio) but it does **not** generalize — comm-disorders (lowest coupling) is flow-unremarkable.

## Idea B --- geography × industry netting of placement (ACS PUMS, field-level)

Replace raw wage with excess over a local×industry benchmark; does netting change field rankings / reproduce A's coupling structure?

| step | result | read |
|---|---|---|
| netting logic (ACS, field-level) | sound within (state×cell) | valid, but field-level only (ACS has no institution) |
| industry (NAICS-sector) netting | **MISCLASSIFIES nursing** — premium *amplifies* $5k→$10k (sector lumps RNs with low-paid health BAs) | wrong netting axis |
| **STATE×OCCP netting** | **fixes nursing**: raw $5k → **$0** (collapse +1.000, n=22,518) — its wage *is* the RN occupation wage | **real, non-circular methods correction** |
| occ_hhi (occupation concentration) ↔ $\rho_f$ | **clean NULL** (reliable −0.056 p=0.84; broad +0.056 p=0.70). CS is the counterexample: occupation-*pinned* yet **highest** coupling (0.708) | "pinning → low coupling" **falsified** |
| occ_netted_collapse ↔ $\rho_f$ | NULL (−0.074, p=0.81) | fixes nursing point, does **not** strengthen unification |
| occ_hhi vs hand-coded license dummy | Spearman **0.643, p=0.007** | occ_hhi ≈ **relabel** of the license variable |
| prestige-link version | needs (institution × field × work-geography) EARNINGS — no open non-resume source exists (PSEO has only school-state; flows file has no earnings) | **Revelio-gated → excluded per user** |

## Phase 4 --- A+B unification ("big news" target): NOT CONFIRMED

Nursing fits perfectly (occupation-pinned + netting-collapse + low coupling + license). But **CS breaks it** (occupation-pinned + high collapse + *highest* coupling); comm-disorders is flow-unremarkable; field-level unification correlation is indistinguishable from zero under both netting axes. A and B do not fuse into a *new* mechanism — both re-discover the *existing* one.

## What survives (for the paper)

1. **STATE×OCCP nursing collapse** ($5k→$0, n=22,518) — a sharper, more causal illustration of licensing than the current incremental-$R^2$ argument.
2. **CS counterexample** — the sufficient condition for decoupling is *not* "occupation pinning" but *"the field's occupation lacks prestige-sortable within-occupation wage variance"* (= licensing/compression). Sharpens the mechanism from "licensed fields" to "wage-compressed fields".
3. The **negative result itself**: kinship (graph or tree) and geography/occupational composition all fail to organize cross-field coupling — three natural competing explanations ruled out, leaving licensing as the sole survivor.

Detailed per-phase records: `outputs/PHASE1_COUPLING_RESULT.md`, `outputs/PHASE3_4_RESULT.md`, `outputs/MORAN_KINSHIP_COUPLING.md`, `outputs/OCCUPATION_CHANNEL_RESULT.md`, `outputs/PHASE5_PILOT_SYNTHESIS.md`, `outputs/ALT_INST_GEO_DATA_SCOUT.md` (kept local; `outputs/*.md` is gitignored by repo convention).
