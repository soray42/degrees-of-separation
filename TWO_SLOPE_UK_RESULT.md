# Step 2 (UK) --- two-slope test: does demand follow prestige where prestige does not pay?

Within UK subject, across providers: contrast the prestige->earnings slope (placement) with the prestige->demand slope (applicant pressure). Demand = UCAS applications/acceptances (free end-of-cycle; OFFERS are paid EXACT, NOT used). Grain CAH1 (UCAS-limited); LEO earnings + ORCID prestige rolled up. No-fabrication; seeded; `python scripts/46_two_slope_uk.py`.

## The two slopes per CAH1 subject (within-subject, across providers; region FE)

$\beta^P_s$: prestige$\to$log-earnings (does a more prestigious provider pay more in subject $s$). $\beta^D_s$: prestige$\to$log(applications/acceptances) (does prestige draw application pressure). Prestige = UK ORCID SpringRank percentile within subject.

| label                          |   n |    bP |   P_lo |   P_hi |    bD |   D_lo |   D_hi | quadrant                            |
|:-------------------------------|----:|------:|-------:|-------:|------:|-------:|-------:|:------------------------------------|
| Subjects allied to medicine    |  48 | +0.11 |  +0.00 |  +0.22 | +0.36 |  -0.04 |  +0.77 | low-pay / low-demand                |
| Geography & environ            |  22 | +0.12 |  -0.14 |  +0.38 | +0.40 |  -0.04 |  +0.84 | LOW-pay / HIGH-demand (mismatch)    |
| Engineering & tech             |  63 | +0.15 |  +0.00 |  +0.29 | +0.61 |  +0.25 |  +0.96 | LOW-pay / HIGH-demand (mismatch)    |
| Language & area studies        |  60 | +0.20 |  +0.08 |  +0.33 | +0.04 |  -0.21 |  +0.29 | low-pay / low-demand                |
| Mathematical sciences          |  46 | +0.29 |  +0.09 |  +0.49 | +0.22 |  -0.08 |  +0.53 | low-pay / low-demand                |
| Biological & sport sci         |  47 | +0.30 |  +0.15 |  +0.44 | +0.25 |  -0.06 |  +0.55 | low-pay / low-demand                |
| Psychology                     |  82 | +0.30 |  +0.22 |  +0.39 | +0.40 |  +0.14 |  +0.65 | high-pay / high-demand (integrated) |
| Physical sciences              |  42 | +0.36 |  +0.17 |  +0.54 | +0.18 |  -0.22 |  +0.58 | high-pay / low-demand               |
| History, philosophy & religion |  66 | +0.37 |  +0.25 |  +0.48 | +0.09 |  -0.17 |  +0.34 | high-pay / low-demand               |
| Business & management          |  71 | +0.42 |  +0.22 |  +0.61 | +0.71 |  +0.25 |  +1.17 | high-pay / high-demand (integrated) |
| Computing                      |  65 | +0.45 |  +0.16 |  +0.74 | +0.51 |  +0.07 |  +0.96 | high-pay / high-demand (integrated) |
| Social sciences                |  77 | +0.54 |  +0.42 |  +0.66 | +0.60 |  +0.38 |  +0.82 | high-pay / high-demand (integrated) |

**corr($\beta^D_s$, $\beta^P_s$) across 12 subjects = Spearman +0.22** (p=0.48).

## The contrast --- the four quadrants

The make-or-break quadrant is **LOW $\beta^P$ / HIGH $\beta^D$**: prestige carries little wage information yet still drives demand. Populated by: **Geography & environ (bP +0.12, bD +0.40), Engineering & tech (bP +0.15, bD +0.61)**.


Integrated sanity contrast (high $\beta^P$ / high $\beta^D$ --- demand follows prestige where it also pays): Psychology, Business & management, Computing, Social sciences.

## GATE VERDICT (does Step 3 run?)

Strong-mismatch subjects (prestige pay-*irrelevant* --- $\beta^P$ CI reaches $\le 0$ --- yet demand-*relevant* --- $\beta^D$ CI $>0$): **Engineering & tech**; of these, decoupled/licensed: none.


**No convincing prestige--placement paradox in UK revealed demand (CAH1).** $\beta^P>0$ in almost every subject --- prestige predicts pay throughout; the subjects where prestige most drives demand (Social sciences, Business, Computing, Psychology) are the same ones where prestige most predicts pay (the integrated quadrant); demand and placement slopes weakly **co-move**, not diverge (Spearman +0.22, n.s.); and the median-split mismatch cell (Geography & environ, Engineering & tech) holds only imprecise slopes and is **not** the decoupled/licensed subjects the paradox predicts. **On this evidence the paradox does not survive: Steps 3--4 do NOT run, and measurement + licensing remains the paper.** This is underpowered (12 coarse CAH1 subjects, wide CIs; the free UCAS grain forces CAH1) --- a finer-grain pull could revisit, but the free-data evidence does not support a demand--placement mismatch.

## Adversarial self-check

- **Applications/acceptances, NOT offers.** Demand is UCAS applications/acceptances (free); the complete offers / admit-rate funnel is paid UCAS EXACT and is excluded --- so this is application PRESSURE, not an offer/admit rate.

- **Per-subject provider n + CAH1 coarseness.** Slopes use 10+ providers/subject; UCAS forces the coarse CAH1 grain (23 groups), so LEO earnings and ORCID prestige are rolled up from CAH2 --- within-CAH1 field variation is averaged away.

- **ORCID prestige-axis density.** Prestige is the UK hiring-network SpringRank at CAH1 (denser than the 16-subject CAH2 axis of scripts/40), oriented so elite universities sit high; thin CAH1 networks are dropped (<8 providers / <20 edges).

- **Region FE != destination geography.** Region is the PROVIDER's nation/region, not where graduates work; UK graduates move to London, so this is a weaker geography control than the US PSEO destination-state decomposition --- it under-controls the London pay premium.

- **Associational.** These are revealed-demand correlations, not causal; selectivity / option-value / capacity confounds are the domain of Step 3.
