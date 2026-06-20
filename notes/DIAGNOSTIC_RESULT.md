# Diagnostic — is the AR–ER gap real signal or a measurement artifact?

## Verdict: **REAL SIGNAL** (with a per-field reliability filter required)

The gap is **not** a global low-SNR artifact. In the majority of fields the observed gap
**exceeds** what sampling noise under perfect market agreement can produce, the artifact
channel explains only ~20% of the cross-field gap variance, and the residual keeps clear
structure. **But** the gap's reliability is **two-regime**: in a cluster of
low-earnings-signal fields (**chemistry, earth sciences, and the low-CV engineering
fields**) the gap is noise-dominated and must not be interpreted as market segmentation.
Two consequences: (i) the **chemistry "counterexample" dissolves** — it sits in the
noise regime; (ii) even restricting to well-measured fields, **task distance and the price
wedge still do not predict the gap** (wrong sign) — the H2 mechanism stays dead.

*Run:* `python scripts/run_diagnostic.py`. Engine `src/diagnostic.py`. Real data only
(Wapman ranks + Scorecard FoS incl. cohort count `EARN_COUNT_WNE_4YR`; no new data, no
pipeline). Figures: `results/figures/diagnostic_{nullmodel,bootstrap}.png`. Date 2026-06-20.

---

## The artifact mechanism being tested
`Gap = 1 − Spearman(prestige, earnings)`. Where cross-institution bachelor's earnings have
little true spread and/or small cohorts, the earnings *ranking* is sampling-noise-dominated,
so its rank-correlation with prestige is mechanically attenuated → high gap **by
construction**, even under perfect market agreement (the same low-signal precision collapse
documented in the PI's transfer-entropy reliability audit).

## Test 1 — Null-model / placebo calibration (decisive)

Synthetic earnings that **perfectly** rank-track prestige + realistic per-institution
sampling noise (median SE = 1.2533·α·earn/√cohort, α = within-school individual-earnings CV
= 0.6; 1000 draws/field). If the resulting `synth_gap` reproduces the observed gap, that
field's gap is indistinguishable from noise.

**Result:** across-field corr(observed, synth) = **0.49** (Pearson) / 0.65 (Spearman);
only **15%** of fields' observed gaps fall inside the null 95% band; **65%** sit **above**
it (excess disagreement beyond noise). See `results/figures/diagnostic_nullmodel.png` — most
fields lie **above** the diagonal (real gap), a right-hand cluster lies **on/below** it (noise).

| field | N | observed gap | null gap (mean) | signal frac | regime |
|---|---|---|---|---|---|
| Computer Science | 161 | 0.29 | 0.03 | 0.92 | **real** (gap ≫ noise) |
| Economics | 116 | 0.31 | 0.05 | 0.90 | **real** |
| Statistics | 28 | 0.25 | 0.11 | 0.78 | real |
| Mathematics | 120 | 0.38 | 0.07 | 0.87 | **real** |
| **Biology** | 138 | **0.63** | **0.15** | **0.71** | **real (high gap, well-measured!)** |
| Physics | 43 | 0.66 | 0.42 | 0.34 | real-ish (above band) |
| Psychology / Sociology / History / English / Anthropology | 75–173 | 0.41–0.51 | 0.11–0.28 | 0.51–0.77 | real |
| Electrical Eng. | 118 | 0.49 | 0.16 | 0.71 | real |
| **Chemistry** | 76 | 0.69 | **0.88** | **0.02** | **ARTIFACT** (covered/below null) |
| **Earth Sciences** | 50 | 0.72 | 0.87 | 0.02 | **ARTIFACT** |
| Mech/Civil/Chem/Materials Eng. | 23–133 | 0.38–0.49 | 0.57–0.92 | 0.02–0.18 | noise-dominated |

α-sensitivity (0.4 / 0.6 / 0.8): corr 0.63–0.65, covered 15–35%, mean null gap 0.19–0.51 vs
observed 0.48 — the "most fields exceed noise" finding is robust. (Where `null > observed`,
e.g. engineering, the α=0.6 noise is an over-estimate — those fields' real Spearman ≈ 0.5
proves more signal than the null assumes; they are *low-reliability*, not proven artifacts.)

## Test 2 — Bootstrap reliability (`diagnostic_bootstrap.png`)

Resample institutions 1000× per field. The **low-gap fields have tight CIs** clearly
positive (CS 0.71 [0.62, 0.78]; Economics 0.69 [0.58, 0.77]; Statistics 0.75 [0.52, 0.87]).
The **high-gap end is wide**: Earth Sciences 0.29 [**−0.00**, 0.57], Physics 0.35 [0.01,
0.62], Chemistry 0.31 [0.09, 0.50] — imprecise, some touching zero. **But Biology 0.37
[0.22, 0.52] (N=138) is tight and genuinely below CS** → a real, well-measured low
correlation. So *some* high-gap fields are genuine (biology) and *some* are noise (earth,
physics, chemistry). Highest-gap-5 and lowest-gap-5 CIs overlap → the extreme gaps cannot be
finely rank-ordered, only coarsely separated.

## Test 3 — Residualization

`gap ~ earn_cv + grad-school pull`: **R² = 0.20** (the artifact channel explains only a
fifth of gap variance). Residuals **retain structure**: Earth Sci +0.21, Philosophy +0.22,
Biology +0.17, Chemistry +0.17, Physics +0.15 stay high; engineering/stats/econ/CS negative.
**CS does not stand out** in the residual (z = −0.67) — its low gap is "as expected" for a
high-CV field — but per Test 1 its gap is still genuine disagreement, not noise. A
structure-laden residual with low R² is the **real-signal** signature.

## Test 4 — Restriction to well-measured (top-half CV) fields

Among the 10 highest-earnings-dispersion fields (artifact weak), re-test the mechanisms:
- Spearman(gap, **task_distance**) = **−0.38** (p=0.28) — still **wrong sign**.
- Spearman(gap, **|price_wedge|**) = **−0.47** (p=0.29) — still **wrong sign**.

Neither H2 task distance nor the price wedge predicts the gap **even where the gap is
well-measured**. The mechanism is dead independent of the artifact.

## What this means

1. **The gap carries real signal.** Most fields — including the CS poster-child and,
   importantly, the high-gap **biology** — show genuine prestige↔earnings disagreement that
   noise cannot manufacture. The descriptive AR–ER gap is a real object.
2. **Reliability is heterogeneous; filter on it.** Fields with near-zero earnings signal
   (signal_frac ≲ 0.1 and/or bootstrap CI touching 0): **chemistry, earth sciences, and the
   low-CV engineering fields** — their gaps are noise. Use a per-field reliability gate
   (bootstrap-CI width or signal fraction) before interpreting any single field's gap.
3. **The motivating chemistry counterexample dissolves.** Chemistry's high gap is among the
   *least* reliably measured (signal_frac 0.02), so it never was a clean test of task distance.
4. **The mechanism is still unexplained** where the gap is solid: task distance is reversed,
   the price wedge is reversed, even in the clean subset.

## Recommended next step

**Proceed — but measure ER at the level where each field's talent actually lands.** The
contamination is concentrated exactly where the bachelor's labor market is a thin antechamber
to a PhD (chemistry, earth sciences, engineering): there the BA-earnings ER is the *wrong*
price, so the fix is a level-appropriate ER, not abandoning the gap.

### Candidate selection-bias fixes — **LISTED, NOT BUILT** (for later)
- **(a) Condition on selection** — combine **SED baccalaureate-origin** counts with **IPEDS**
  bachelor's production to build an **institution × field PhD-feeder rate** (share of a
  school's BAs in a field who go on to a PhD). Including this lets the gap be interpreted net
  of "the signal-carrying graduates left for grad school," directly addressing the artifact's
  source rather than just flagging it.
- **(b) An uncontaminated PhD-level ER** — **ORCID + OpenAlex** PhD-placement outcomes, or
  **Revelio Labs** full-trajectory (résumé) data, to measure where a department's *PhDs* and
  high-ability graduates actually land and what they earn — the ER that matches PhD-level AR
  in the PhD-pipeline fields, replacing the noise-dominated BA-earnings proxy there.

## Caveats (real data only; provenance in `data/raw/SOURCES.md`)
- The null model's noise scale depends on α (assumed within-school individual-earnings CV);
  set to 0.6 with 0.4/0.8 sensitivity. For very-low-CV fields α=0.6 over-states noise (they
  are *low-reliability*, not confirmed artifacts) — which only strengthens "real signal" in
  the rest. Median-SE constant 1.2533 assumes ~normal individual earnings.
- Small N inflates bootstrap CIs (Materials 23, Statistics 28, Philosophy 41, Physics 43).
- Gap here uses cohort-weighted earnings on institutions with a non-missing cohort count
  (100% of earnings-present rows), so it differs by ≲0.01 from the Tier-0 unweighted gap.
