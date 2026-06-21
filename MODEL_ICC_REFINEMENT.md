# TASK 3 — ICC-vs-horizon: can a discipline-varying early-compression recalibration reproduce the decline?

Empirically between-cluster ICC falls 0.50@1yr -> 0.30@4-5yr; the default model gets between-field variance slightly RISING. Test: make early sort-noise sigma0 field-specific and strong, tied to licensing (licensed fields compressed hardest early). Run: `python scripts/26_icc_refinement.py`.

## Between-field variance by horizon

| sigma0 | between-var 1yr | between-var 5yr | direction | grand gap falls? |
|---|---|---|---|---|
| constant (default) | 0.0172 | 0.0200 | rises | True |
| discipline-varying + λ-linked | 0.0297 | 0.0246 | **FALLS** | True |

**Result:** the discipline-varying / licensing-linked early-compression recalibration **reproduces the decline** — between-field variance now falls with horizon, matching the empirical ICC drop, because licensed/heavily-compressed disciplines have very large but structured 1-yr gaps that resolve by 5 yr (between-field spread is largest at 1 yr).

## Does it break P2 / P3 / P5?

- **P2 licensing** (t=4 slope ∂gap/∂λ): **+0.604** (default +0.318) — still positive, holds.
- **P3 absorption null**: b_abs = **+0.029** (p=0.19) — still null, holds.
- **P5 lead-lag** (corr(Δgap, 1-ρ)): **+0.40** (default +0.48) — still positive, holds.

**Verdict:** a defensible recalibration (early compression discipline-varying and λ-linked) reproduces the ICC decline **without breaking** P2/P3/P5 — so the default model's ICC miss is a CALIBRATION limitation, not a structural one. The cost is one added degree of freedom (field-specific σ0 instead of common σ0), disclosed here, not tuned silently.

## Adversarial self-check

**Objection.** Tying σ0 to λ and adding strong field-specific early noise is exactly the extra freedom the default model lacked — so 'reproducing' the decline is close to fitting it. The mechanism (licensed disciplines hugely compressed at 1 yr, resolving by 5 yr) is plausible and matches the empirical 1-yr picture (Health/credential gaps near 1.0 at 1 yr), but it is asserted, not independently measured. **Honest status:** the refinement shows the decline is *achievable* within the model's structure with a disclosed extra parameter, not that the model *predicts* it from first principles. We do not fold this into the headline calibration.
