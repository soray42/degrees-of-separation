# The revealed-placement identification layer (ER axis)

**Reframe (load-bearing).** The gap's second axis is **not** 'employer reputation' (a perception/survey construct, the QS sense) but **revealed labour-market placement** -- where graduates actually land -- which median earnings proxies. The paper's object is academic **reputation** (AR, the hiring-network SpringRank) versus revealed **placement** -- an honest asymmetry (perception vs realised outcome), not 'reputation vs reputation'. This layer establishes salary as a **defensible-but-incomplete** placement proxy: where it works (35a), where it fails via pipeline deferral (35b), and the licensing mechanism behind apparent decoupling (35c). Field-level identification; descriptive, outcome-agnostic, seeded.

## 35a. Salary as a placement proxy: where it works (horizon x source)

Field-level rank correlation between **salary standing** and each non-wage **placement** block (occupational prestige [Condon OPR]; vertical match [-NY Fed underemployment]), across Scorecard earnings horizons (1/4/5yr) and across sources (Scorecard / PSEO / ACS).

### Proxy quality by horizon and source

| source | horizon | corr(salary, occ-prestige) [95% CI] | n | corr(salary, -underemp) [95% CI] | n |
|---|---|---|---|---|---|
| Scorecard | 1YR | +0.75 [+0.56,+0.87] | 52 | +0.78 [+0.58,+0.90] | 32 |
| Scorecard | 4YR | +0.84 [+0.70,+0.91] | 52 | +0.66 [+0.38,+0.84] | 32 |
| Scorecard | 5YR | +0.83 [+0.69,+0.90] | 52 | +0.67 [+0.38,+0.83] | 32 |
| PSEO 1yr | y1 | +0.73 [+0.53,+0.85] | 52 | +0.79 [+0.59,+0.91] | 32 |
| PSEO 5yr | y5 | +0.84 [+0.70,+0.91] | 52 | +0.72 [+0.48,+0.86] | 32 |
| ACS early-career | early | +0.67 [+0.44,+0.82] | 47 | +0.70 [+0.46,+0.84] | 31 |
| NY Fed early-career | early | +0.65 [+0.32,+0.84] | 31 | +0.67 [+0.40,+0.85] | 32 |

**KEY TEST -- does the proxy improve at longer horizons?** The salary-vs-prestige proxy correlation IMPROVES from 1yr (+0.75) to 5yr (+0.83) on Scorecard. 
Early-career salary understates placement, and the understatement is concentrated by deferral: **log salary growth 1->5yr correlates +0.66 [+0.34,+0.87] (n=32) with the field's graduate-degree share** -- high-deferral fields gain the most salary rank with horizon, exactly the proxy-timing failure formalised in 35b.


## 35b. Where salary fails: pipeline deferral (a cross-field proxy-timing effect)

**Deferral classifier** (NY Fed graduate-degree share): pipeline-deferral if >= 50% pursue a graduate degree, terminal-BA if < 40%. **11 pipeline-deferral, 11 terminal-BA** (of 32 classified).

### The salary-vs-status divergence IS deferral

The field-level **status residual** (occupational prestige net of wage -- the part early-career salary cannot see) is explained by deferral:

- `corr(status_resid, grad_degree_share)` = **Pearson +0.59** [+0.26,+0.81] (Spearman +0.44), n=31 -- reproduces the observed ~+0.59.

- OLS: `status_resid = -8.1 + +0.174 * grad_degree_share`, R^2=0.35. Fields where more graduates defer to grad school carry occupational status far above their BA wage -- because BA earnings is timed before they reach the terminal occupations their degree routes them into.

### (i) This is NOT the null academic-absorption channel

Deferral (BA->grad-school) and academic absorption (PhD->academia) are **different constructs**, and the data confirm they are nearly orthogonal: `corr(grad_degree_share, absorption_acs)` = **+0.08** [-0.31,+0.48] (n=31). The absorption channel is the project's documented **null** in the gap decomposition (b_absorption = -0.02, `MIXTURE_DECOMP_RESULT.md`), and it does *not* drive the status residual (`corr(status_resid, absorption_acs)` = -0.36). So the divergence here is a deferral / proxy-timing effect, not a re-discovery of (or contradiction with) the absorption null.

### (ii) Deferral does NOT explain the within-field gap

The gap is a **within-field rank** statistic across institutions; deferral is a **field-level constant**, so it cannot generate within-field rank disagreement. Empirically `corr(gap, grad_degree_share)` = **+0.02** [-0.57,+0.59] on the reliable set (n=15; +0.16 on all). And the multi-horizon evidence (`MULTIHORIZON_RESULT.md`) shows the gap **persists and stabilises** rather than vanishing as the proxy-timing bias resolves: the between-discipline ICC moves 0.50@1yr -> 0.30-0.33@4-5yr but the cluster ranking is horizon-stable (+0.81 to +0.85) and the field gap replicates (+0.68 to +0.74). If the gap were the deferral artifact it would collapse at 5yr; it does not.

### Re-statement: natural-science 'decoupling' is substantially early-career-window bias

The high-deferral natural-science fields (Biochemistry, Biology, Chemistry, Physics) have a median graduate-degree share of **67%** -- the majority defer. Their low BA **salary standing** (which makes them look 'decoupled' on a field-level earnings axis) is therefore substantially a **measurement-timing fact**, not market undervaluation: early-career BA earnings is the wrong terminal-market proxy for a population most of which has not yet entered its terminal occupation. This is rigor-up / punch-down -- it **narrows** the earnings-mis-pricing claim for natural science to a proxy-window artifact, and leaves the within-field gap (above) untouched.


## 35c. The licensing mechanism: general claim + (a) compression vs (b) prestige-orthogonal variance

The mixture decomposition gives **b_licensure = +0.66** (licensure raises the gap). This section asks *why*, field-generally, and disambiguates two mechanisms without assuming compression.

### (i) Compression test -- does licensure shrink within-field wage variance?

`corr(licensure, within-field cross-institution wage CV)` = **+0.02** [-0.30,+0.32] (n=45). **Wage compression is NOT supported**: licensed fields do not have markedly lower within-field wage variance -- there is still real cross-institution wage spread to predict (e.g. Nursing CV 0.10 is mid-pack). So apparent decoupling in licensed fields is not 'no variance to predict'.

### (ii) Commonality decomposition -- prestige vs geography, handling their overlap

Elite institutions cluster geographically, so a separate `R^2(earnings~prestige)` and `R^2(earnings~state)` overlap; we therefore use **incremental (partial) R^2**. Per field: `incremental_prestige = R2_full - R2_geo_only` (what school prestige adds BEYOND where the school is); `incremental_geo = R2_full - R2_prestige_only`; `shared` is the geographically-confounded overlap. Decomposable fields: **n=34** (>= 15 institutions; 5 thin, 15--19 inst., flagged). **Headline (continuous gradient):**

- `corr(licensure, incremental_prestige)` = **-0.59** [-0.80,-0.30] -- **negative**: as licensure rises, school prestige adds essentially nothing beyond geography. Robust to dropping thin fields: -0.57 (n=29).

- `corr(licensure, incremental_geo)` = **+0.47** [+0.16,+0.69] -- geography adds *more* beyond prestige as licensure rises.

- `corr(licensure, shared)` = -0.20 [-0.53,+0.17] (the confounded overlap; small).


**Group means by licensure tercile (ILLUSTRATION only; the headline is the continuous gradient):**

| licensure tercile | n | licensure range | mean incremental_prestige | mean incremental_geo | mean shared |
|---|---|---|---|---|---|
| low | 12 | 0.03--0.08 | 0.218 | 0.361 | 0.130 |
| mid | 11 | 0.09--0.17 | 0.079 | 0.557 | 0.075 |
| high | 11 | 0.18--0.77 | 0.063 | 0.536 | 0.085 |

**Nursing (clean illustration):** incremental_prestige = **0.000** (prestige adds ~nothing beyond geography), incremental_geo = 0.71, on 46 institutions -- nursing pay is a state/setting phenomenon, not an alma-mater one.

### Verdict: the mechanism is **(b) prestige-orthogonal, setting-driven variance**

Licensed fields retain real within-field wage variance (no compression), but that variance is **prestige-orthogonal**: school prestige adds no *incremental* predictive power over geography, while geography adds a lot. 

**Attribution (do not over-attribute to the credential).** 'Licensure' here is the cleanest OBSERVABLE marker for a **collinear bundle** -- regulated / public-sector / locally-employed labour markets (the public-sector-share and pay-wedge channels are collinear with licensure, `EXTERNAL_CHANNELS_RESULT.md`, |corr| up to 0.54). In these fields pay is set by **setting** (state / employer / shift / local pay scale), so school prestige adds no incremental power over geography -- which mechanically produces the within-field prestige->pay rank disagreement the gap measures. This is the mechanism behind **b_licensure = +0.66**: *the marker (not necessarily the credential-as-cause)* identifies where wages depend on setting, not school. Field-general (continuous across n=34 decomposable fields), not a Nursing special case.


## Synthesis

**Salary is a defensible-but-incomplete revealed-placement proxy.** Across horizons and three independent earnings sources it tracks non-wage placement positively (Scorecard salary-vs-occupational-prestige rises +0.75->+0.83 from 1yr to 5yr; PSEO and ACS agree), so it is a real proxy -- but it fails in **two distinct, identified ways**, each a source of **apparent** decoupling rather than market undervaluation:

1. **Pipeline-deferral fields** -- a **cross-field proxy-timing** effect. Early-career BA earnings is a poor *terminal*-market proxy where graduates defer to graduate school; the field-level salary-vs-status divergence is explained by graduate-degree share (status residual ~ deferral, Pearson +0.59). This is **distinct from the project's null academic-absorption channel** (b_absorption = -0.02; absorption is PhD->academia, a different construct, near-orthogonal to deferral), and it does **not** explain the within-field gap (which is rank-based and horizon-stable). It narrows the 'earnings mis-prices natural science' claim to a measurement-timing fact.

2. **Licensed / regulated fields** -- via **(b) prestige-orthogonal, setting-driven variance (the license marks where pay is set by setting, not school)**, *not* wage compression. Licensure does not shrink within-field wage variance (CV ~ licensure +0.02, flat); rather, using a commonality (incremental-R^2) decomposition that handles the geographic clustering of elite institutions, **school prestige adds essentially nothing beyond geography as licensure rises** (corr(licensure, incremental_prestige) = -0.59; Nursing's incremental_prestige ~ 0). Here 'licensure' is the cleanest OBSERVABLE marker for a collinear bundle -- regulated / public-sector / locally-employed labour markets (the public-sector and pay-wedge channels are collinear with it, EXTERNAL_CHANNELS_RESULT.md). The mechanism behind b_licensure = +0.66: in these fields pay is set by *setting* (state / employer / shift / local pay scale), so prestige cannot predict pay and the field looks decoupled -- the marker, not necessarily the credential-as-cause.


Neither failure is evidence that the labour market undervalues these fields; both are properties of **salary as a timed, setting-sensitive proxy** for revealed placement.

## Adversarial self-check

- **Absorption vs deferral reconciliation.** The null academic-absorption channel (PhD->academia, b_absorption = -0.02) and pipeline deferral (BA->grad-school) are different constructs and near-orthogonal (corr ~ +0.08); deferral drives the status residual while absorption does not. So 35b does not contradict, re-discover, or rehabilitate the absorption null -- it identifies a separate cross-field proxy-timing effect.

- **35c mechanism stated explicitly (commonality decomposition).** The data reject (a) wage compression (CV flat in licensure, +0.02) and support (b) prestige-orthogonal/setting-driven variance via incremental R^2 (corr(licensure, incremental_prestige) = -0.59, negative), which handles the geographic clustering of elite institutions that would confound a separate-R^2 attribution. Attribution is deliberately to the **observable marker** (regulated/public-sector/locally-employed bundle), not the credential-as-cause; the licensure-tercile group means are illustration only -- the headline is the continuous gradient.

- **Rigor-up / punch-down on natural science.** 35b *narrows* rather than inflates: it converts an apparent 'market undervalues natural science' story into a proxy-window measurement fact, and leaves the within-field gap intact. We do not claim natural-science programs are well- or under-priced in terminal markets -- only that early-career BA salary cannot adjudicate it.

- **Grain caveat (load-bearing).** This is the **field-level** identification layer. It establishes where salary is and isn't a good *field-level* placement proxy; it does **not** recompute a within-field non-wage gap. The within-field analogue `gap^occ_f = 1 - Spearman_i(prestige_{i,f}, occ_prestige_{i,f})` still needs institution x field occupation flows (Revelio); its estimand is specified in `ER_DEFINITION_RESULT.md`.


*Provenance:* generated by `scripts/35a_placement_proxy_validation.py`, `scripts/35b_pipeline_deferral.py`, `scripts/35c_licensing_compression.py`, assembled by `scripts/35_assemble_er_axis.py`. Reuses the gap, ACS licensure (b_licensure), multi-horizon ICC, Condon/Hughes 2024 OPR, and NY Fed by-major; no within-field gap re-run. Interim tables are regenerated by the scripts (gitignored per repo policy).
