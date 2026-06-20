# Tier 0.5 — Diagnosing the real driver of the AR–ER gap

**Verdict: REFRAME.** The original mechanism (H2 *task distance*) does **not** explain the
gap — its relationship is **signed backwards**, and chemistry refutes it directly. The gap
is instead organized by an **academic-pipeline / earnings-informativeness axis**:
**grad-school pull** (the only individually significant explanator, correct sign) and
**earnings dispersion** (correct sign) together place **both** the CS poster-child **and**
the chemistry counterexample correctly. The **price wedge** (a PRICE-agreement measure)
does **not** coincide with the institution-level RANK gap. Do **not** start the
ORCID+OpenAlex pipeline (Step 5); the construct needs reframing first.

*Run:* `python scripts/run_tier0_5.py` · notebook `notebooks/tier0_5_diagnose.ipynb`.
Engine `src/tier0_5.py`. Date 2026-06-20.

## Setup (gap construction unchanged)

`Gap_field = 1 − Spearman(Wapman SpringRank prestige, Scorecard bachelor's EARN_MDN_4YR)`,
within field across institutions. **AR is research/PhD-level** (the academic market's price);
**ER is the undergraduate graduate mass** (the labor market's price). That is the AR–ER
distinction itself, *not* a level confound — the gap is accepted as coherent. Four competing
explanators of the cross-field gap, each pre-registered with a predicted direction; **none
dropped on significance**.

## Per-explanator results (Spearman vs `Gap_field`)

| Explanator | source | n | Spearman (p) | predicted | **sign** |
|---|---|---|---|---|---|
| **Task distance** (H2) | O\*NET WA IM×LV, academic-teacher vs CIP→SOC industry, GS angular dist | 20 | **−0.38** (0.10) | + | ❌ **WRONG (reversed)** |
| **Earnings dispersion** | within-field CV of inst. bachelor's `EARN_MDN_4YR` | 20 | **−0.41** (0.07) | − | ✅ OK |
| **Grad-school pull** | IPEDS 2023 PhD/Bachelor completions ratio | 20 | **+0.48** (0.03) | + | ✅ **OK (only sig.)** |
| **\|Price wedge\|** | SDR T54 (industry−academic)/mean salary | 17 | **−0.42** (0.10) | + | ❌ **WRONG (reversed)** |

## The decisive test — do CS *and* chemistry land where each mechanism predicts?

CS is the integrated low-gap poster-child (gap 0.30); chemistry is the alleged
counterexample (low academic↔industry task distance, yet gap 0.69). Rank within 20 fields
(1 = lowest value):

| Explanator | CS | Chemistry | Verdict on this pair |
|---|---|---|---|
| Task distance | 2/20 (low) | **12/20 (mid)** | ❌ task distance can't make chemistry high-gap — chem is only mid-distance |
| Earnings dispersion | 19/20 (high) | 5/20 (low) | ✅ CS dispersed→low gap; chem compressed→high gap |
| Grad-school pull | 3/20 (low) | **18/20 (high)** | ✅ CS low pull→low gap; chem high pull→high gap |
| \|Price wedge\| | 12/17 | 11/17 | ❌ doesn't separate them |

**Only earnings dispersion and grad-school pull explain CS *and* chemistry simultaneously.**
Task distance fails precisely on the case it was invoked for.

## Combined standardized OLS (n = 17 SEH fields; all four kept)

`gap ~ z(task_distance) + z(earn_cv) + z(doc_ba_ratio) + z(price_wedge_abs)`

| term | coef | p |
|---|---|---|
| const | +0.466 | 0.000 |
| task_distance | −0.048 | 0.15 |
| earn_cv | −0.059 | 0.21 |
| doc_ba_ratio | +0.024 | 0.47 |
| price_wedge_abs | +0.010 | 0.82 |

R² = 0.40, adj-R² = 0.20. No individual coefficient is significant at n=17 with four
collinear predictors, but the **signs persist** (task distance negative = wrong; grad-school
pull positive = right). The four explanators are entangled (Spearman: earn_cv–price_wedge
+0.69; earn_cv–doc_ba −0.43): they trace **one** "applied/dispersed vs academic-pipeline/
compressed" axis. The gap loads on that axis, not on task distance.

## What this means

- **The gap is driven by how *informative* bachelor's earnings are in a field.** In
  natural-science PhD-pipeline fields (chemistry, physics, biology, earth science) the
  terminal-BA labor market is **compressed** (low earnings dispersion) and most
  signal-carrying graduates **leave for PhDs** (high grad-school pull), so institution-level
  bachelor's earnings barely co-rank with departmental research prestige → a large "gap."
  In CS / economics / engineering the BA market is differentiated and graduates enter it
  directly, so earnings track prestige → small gap.
- **This is not the H2 signaling wedge.** Task distance is low in exactly the high-gap
  natural sciences (academic chemist ≈ industrial chemist), so it cannot be the cause; the
  observed sign is *opposite* to H2.
- **Rank ≠ price.** The SDR price wedge (does industry out-pay academia for a field's PhDs)
  does **not** track the institution-level rank gap — confirming they are different objects.
  That two markets can disagree on *price* yet still co-rank institutions (or vice versa) is
  itself a finding, and it means "the gap" and "the wedge" must not be conflated.

## Verdict and recommended next step

**REFRAME — do not proceed to Step 5 as specified.** Options, in order of attractiveness:

1. **Re-specify the gap to be robust to earnings informativeness.** Residualize or condition
   on within-field earnings dispersion and grad-school pull; or restrict ER to fields/levels
   where the labor-market signal is actually informative (measure ER where the field's mass
   lands — e.g. PhD-level outcomes for PhD-pipeline fields). Then re-test task distance.
2. **Reframe the question around the pipeline mechanism the data shows.** "How much a
   department's research prestige is priced by the undergraduate labor market is governed by
   whether that field's BA market is differentiated or is a thin antechamber to a PhD." This
   is a real, defensible thesis — but it is *not* the task-distance signaling story.
3. Only after (1)/(2) revive multi-proxy AR (SpringRank + OpenAlex citations/faculty), the
   SDR Table-54 signed-wedge analysis, and the 2026 ORCID+OpenAlex panel.

The descriptive result (the field-level gap map; CS = low gap, natural sciences = high gap)
is robust and publishable as description. The **mechanism** claim (H2) is not supported.

## Caveats (real data only; full provenance in `data/raw/SOURCES.md`)

- **Task vectors are equal-weighted over CIP→SOC destinations** (OEWS national employment
  weights were unobtainable: bls.gov returns a uniform Akamai 403 to automated fetches).
  Employment weighting could shift magnitudes but is very unlikely to flip the *reversed*
  sign (natural sciences are low-distance high-gap under any reasonable weighting).
- Engineering subfields share one academic teacher SOC (25-1032); they differ only on the
  industry side. Several fields have few CIP→SOC destinations (English 3, Philosophy 4,
  Physics 5) — task vectors there are noisier.
- Price wedge and combined OLS cover 17 SEH fields (humanities absent from SDR). math &
  statistics share one SDR salary row; anthropology folds into "Other social sciences."
- n = 20 fields (17 with the wedge): underpowered for multivariate significance; signs and
  the CS/chemistry placement test carry the inference, not p-values.
