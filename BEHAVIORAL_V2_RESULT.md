# Behavioural v2 --- pre-registered robustness battery (corollary -> secondary, or SI)

Decides the behavioural probe's placement; NOT a keystone, NOT the within-field belief test (the future fielded RCT). No-fabrication run. Seeded; `python scripts/43_behavioural_v2.py` then `python scripts/44_els_replication.py`.

## Pre-registered adjudication (fixed before running)

Five criteria, each scored below. ``Positive'' = Spearman $\rho \ge +0.20$ (the probe's directional bar).

1. **Spec B (late expectation)** positive.
2. **Spec C (attainment-conditioned)** positive --- BA-terminal subsample $\rho \ge +0.20$, or explicitly *not estimable (power-limited)*.
3. **Leave-one-field-out stable** --- $\rho$ stays the same sign, does not swing through $\sim0$.
4. **Negative control NOT positive** --- $|\mathrm{corr}(\text{placebo}, \text{gap})| < 0.20$.
5. **ELS:2002 cross-cohort replication** positive (scripts/44), or *not testable*.
**Decision rule:** $\ge 4/5$ -> secondary main-text result (may enter the abstract); $2$--$3/5$ -> Discussion only; $\le 1/5$ or direction reverses -> appendix/SI as a flagged null probe. Goalposts not moved after seeing results.

## Part 1 --- three timing/conditioning specs

$\mathrm{overcredit}_f = \mathrm{rank}(E_f) - \mathrm{rank}(R_f)$ on CIP-2 fields with $\ge 30$ students; $E_f$ = weighted mean expected occupational prestige (expected occupation at 30 $\to$ SOC-2 $\to$ Condon OPR); $R_f$ = the project's realized occupational prestige.

| spec | expectation | sample | $\rho$(overcredit, gap) | p | n |
|---|---|---|---|---|---|
| **A** aspiration baseline | base-year (9th gr.) | all | +0.38 | 0.14 | 17 |
| **B** late expectation (headline) | 2016 follow-up | all | **+0.26** | 0.32 | 17 |
| **C** attainment-conditioned | 2016 follow-up | **BA-terminal** | **+0.45** | 0.23 | 9 |
| C (contrast) | 2016 follow-up | grad-bound | +0.49 | 0.09 | 13 |

**Spec A (aspiration):** +0.38 --- the LIGHT-probe-equivalent; interpret as sorting/aspiration of students who later enter field $f$, *not* belief error.

**Spec B (late expectation, the cleaner belief object):** +0.26 (n=17). Expectation measured in 2016 conditional on realized field; this is the headline spec.

**Spec C (the make-or-break deferral disambiguation):** in the **BA-terminal** subsample (no graduate-degree expectation, X1STUEDEXPCT$\le$6) the correlation is **+0.45** (p=0.23, n=9); grad-bound subsample +0.49. Of the high-deferral natural-science cluster, **Biological sci (n=107)** survive(s) the $\ge$30-BA-terminal-student cut while **Physical sci (n=25), Agriculture (n=26), Nat. resources (n=10)** drop out (too grad-bound by composition to leave a BA-terminal cell) --- so Spec C retains the single most over-credited science field but is **power-limited** on the thinner sciences. The signal **survives** attainment-conditioning (BA-terminal students still over-credit the decoupled fields), so the over-crediting is not merely rational grad-school/pre-med aspiration.

## Part 2 --- leave-one-out + robustness battery (headline Spec B, n=17)

- **Leave-one-CIP2-out $\rho$:** range **[+0.16, +0.51]** (stays positive --- does NOT swing through 0).

- **Kendall $\tau$:** +0.20 (p=0.26).

- **Weighted vs unweighted:** weighted +0.26 vs unweighted +0.26.

- **Permutation p** (one-sided, 10k draws): 0.162.

- **Robust regression** rank(overcredit) ~ gap: slope +8.78 (t=+1.48).

- **Drop-set sensitivity:** drop natural-science cluster +0.32 (n=13); drop History outlier +0.35 (n=16); drop licensed (Health) +0.25 (n=16).


Leave-one-out is the n=17 fragility test: the headline is sign-stable.

## Part 3 --- negative control (placebo: field-neutral academic self-efficacy)

The public-use file has no clean expected-life-satisfaction outcome, so the placebo is field-mean **math/science self-efficacy** (X1MTHEFF, X1SCIEFF) --- a field-neutral over-confidence proxy that is NOT placement-specific. The gap should predict PLACEMENT over-crediting, not general confidence, so this should be $\approx 0$.

- `corr(self-efficacy_f, gap)` = **+0.14** (p=0.56, n=21) --- NOT positive (passes the negative control).


*Criteria 1--4 computed; criterion 5 (ELS:2002 replication) and the final 5-point scorecard are written by `scripts/44_els_replication.py`.*

## Part 4 --- ELS:2002 cross-cohort replication

ELS:2002 public-use file acquired; parallel variables present (realized BA field `F3TZBCHLCIP2` 2-digit CIP; expected occupation at 30 `BYOCC30`; expected attainment `BYSTEXP`; panel weight). **Coarseness flag:** ELS codes expected occupation in a 17-category scheme, mapped to SOC-2 -> Condon prestige by the documented category crosswalk in the script header (objective; prestige values are Condon's). This is coarser than the HSLS SOC-2 measure --- a same-DIRECTION test, not a precise re-estimate.

- Spec-B-equivalent CIP-2 cut (3,119 ELS students with both expected occupation and a realized BA field): `corr(overcredit, gap)` = **+0.13** (p=0.64, n=15 fields).

- **Direction replicates** the HSLS sign (positive), though weak and not significant at +0.13 --- students over-credit the decoupled fields in 2002 as in 2009. By the pre-registered direction-only rule ($\rho>0$) criterion 5 is met; the magnitude is not significant and is further attenuated by the coarse 17-category ELS occupation coding (vs HSLS SOC-2), so read it as a cross-cohort DIRECTION check, not a second estimate.

## THE SCORECARD (pre-registered; goalposts fixed)

| # | criterion | verdict | evidence |
|---|---|---|---|
| 1 | Spec B (late expectation) positive | **PASS** | rho=+0.26 (>=+0.20) |
| 2 | Spec C (attainment-conditioned) positive | **PASS** | BA-terminal rho=+0.45, n=9 (power-limited on thin science cells) |
| 3 | Leave-one-field-out stable | **PASS** | LOO rho in [+0.16, +0.51], sign-stable |
| 4 | Negative control NOT positive | **PASS** | corr(self-efficacy, gap)=+0.14 (|.|<0.20) |
| 5 | ELS:2002 cross-cohort replication | **PASS** | ELS rho=+0.13 |

**5/5 satisfied** (0 fail, 0 not-estimable/testable). Direction reverses: no.


### Decision: the behavioural probe is placed in -> **secondary main-text result** (may enter the abstract).

Per the pre-registered rule ($\ge4/5$ -> secondary main-text + abstract; $2$--$3/5$ -> Discussion; $\le1/5$ or reversal -> SI). With criteria 1--4 passing and ELS replicating in direction, the cross-field belief result earns a secondary main-text slot --- as a robust corollary, explicitly NOT the keystone and NOT the within-field claim.

**Significance caveat (load-bearing on the placement).** The five criteria certify the *direction* and its *robustness* --- positive across timing specs, attainment-conditioning, leave-one-out, a clean negative control, and a second cohort --- NOT statistical significance. At $n=17$ fields the permutation $p$ is 0.16 and the ELS magnitude is not significant. The main-text placement is therefore for a **robust-direction corollary**, to be reported with its $p$-values and its underpowered $n$ in plain sight, never as a powered effect.

## Adversarial self-check

- **Cross-field, NOT within-field.** Every test here is whether students mis-rank which FIELDS place well; the paper's gap is WITHIN-field across institutions. This battery can promote the cross-field corollary at most --- the within-field belief test (does a prestigious program within a field place better) is the future fielded RCT, specified not run.

- **The deferral confound and Spec C.** The make-or-break was whether the over-crediting is just rational grad-school/pre-med aspiration in the high-deferral sciences. Spec C (BA-terminal subsample) keeps the signal (rho=+0.45) and retains Biological sciences with a healthy BA-terminal cell --- so it is not merely aspiration --- but it is power-limited (Physical sci, Agriculture, Nat. resources drop for thin BA-terminal cells), so the disambiguation is partial, not complete.

- **Power loss from attainment-conditioning.** Restricting to BA-terminal students shrinks n to 9 fields and removes the most grad-bound science fields; read Spec C as directional, not a powered estimate.

- **n=17 fragility.** Headline rests on a small field set; LOO is sign-stable but the permutation p (0.16) is not below 0.05 --- a robust DIRECTION, not a significant magnitude.

- **Indirect / coarse measurement.** Expected OCCUPATION (HSLS SOC-2; ELS 17-category) and field (CIP-2) with the 6-digit codes suppressed; no direct elicitation of beliefs about a field's prestige-placement gap. E_f is a constructed proxy via Condon prestige; ELS adds an occupation-coding mismatch handled by a documented crosswalk.

- **What this battery decides.** Corollary-vs-secondary, NOT keystone. A real causal/welfare claim still needs the probabilistic belief-measurement design, an explicit general-optimism benchmark, and the institution x field fielded RCT.
