# TASK 1 — Mixture-Decomposition Identification of the AR-ER Gap

Net out two mechanical channels (LICENSING, PIPELINE) with public field-level anchors; isolate genuine VALUATION DIVERGENCE as the residual. **This is a decomposition (a linear projection / variance attribution), NOT a causal or IV estimate** — the anchors are not instruments and no exclusion restriction is claimed. Outcome-agnostic. Run: `python scripts/20_mixture_decomp.py`.

## Anchors (public)

- **licensure_intensity** = weighted share of a field's employed degree-holders in licensed occupations (ACS PUMS 2023, SOCP licensed-occupation flag). STRICT = near-universal licensure (health practitioners 29-1xxx, lawyers 23-1xxx, K-12 teachers 25-2xxx, architects, psychologists); BROAD adds partially-licensed (health techs, accountants/CPA, engineers/PE, counselors/social workers).
- **academic_absorption** = share of a field's DOCTORATE-holders employed as postsecondary teachers (ACS, SOCP 25-1xxx). Cross-check: SDR 2021 Table 12-3 Educational/All employed (SEH fine fields only).

**Coverage.** 47 fields have both primary anchors (of 48 non-degenerate gap fields). licensure 48, ACS absorption 47, SDR absorption 17 (SEH only).

## Channel coefficients (precision-weighted WLS, gap ~ licensure + absorption)

| spec | n | b_licensure [95% CI] | b_absorption [95% CI] | R² |
|---|---|---|---|---|
| primary (strict lic, ACS absorp) | 47 | +0.66 [+0.33, +0.97] | -0.02 [-0.42, +0.46] | 0.28 |
| broad licensure | 47 | +0.45 [+0.05, +0.72] | +0.08 [-0.34, +0.54] | 0.14 |
| SDR absorption | 17 | +0.89 [-0.00, +1.66] | -0.35 [-0.66, -0.04] | 0.39 |

Both anchors range 0-1, so a coefficient is the gap change from 0% to 100% of the channel. Primary R²=0.28 of cross-field gap variance is linearly attributable to the two mechanical channels; the rest is the residual decoupling gap.

## Cluster gap: raw vs residual (decoupling) ranking

Field residuals (precision-weighted) aggregated to CIP-2 clusters. **raw_gap** = precision-weighted cluster gap; **resid_gap** = same after netting licensing + pipeline.

| name                |   n_fields |   raw_gap |   resid_gap |   licensure |   absorption |   rank_raw |   rank_resid |   rank_shift |
|:--------------------|-----------:|----------:|------------:|------------:|-------------:|-----------:|-------------:|-------------:|
| Nat. Resources      |          2 |     0.844 |       0.382 |        0.10 |         0.18 |          1 |            1 |            0 |
| Arts                |          2 |     0.816 |       0.329 |        0.14 |         0.37 |          2 |            2 |            0 |
| Physical Sci        |          3 |     0.685 |       0.180 |        0.16 |         0.19 |          6 |            3 |            3 |
| Architecture        |          1 |     0.777 |       0.175 |        0.31 |         0.25 |          4 |            4 |            0 |
| Philosophy/Religion |          1 |     0.657 |       0.123 |        0.21 |         0.37 |          7 |            5 |            2 |
| Agriculture         |          3 |     0.574 |       0.082 |        0.14 |         0.14 |          8 |            6 |            2 |
| Social Work         |          1 |     0.552 |       0.071 |        0.13 |         0.27 |          9 |            7 |            2 |
| Biological Sci      |          5 |     0.699 |       0.070 |        0.35 |         0.15 |          5 |            8 |           -3 |
| Business            |          4 |     0.476 |       0.041 |        0.06 |         0.15 |         11 |            9 |            2 |
| Health              |          4 |     0.805 |       0.027 |        0.58 |         0.09 |          3 |           10 |           -7 |
| Engineering         |          9 |     0.420 |      -0.012 |        0.05 |         0.16 |         14 |           11 |            3 |
| English             |          1 |     0.506 |      -0.029 |        0.22 |         0.42 |         10 |           12 |           -2 |
| Social Sci          |          5 |     0.412 |      -0.085 |        0.16 |         0.33 |         16 |           13 |            3 |
| Psychology          |          1 |     0.429 |      -0.089 |        0.18 |         0.18 |         13 |           14 |           -1 |
| Computer/Info       |          1 |     0.292 |      -0.127 |        0.03 |         0.20 |         18 |           15 |            3 |
| History             |          1 |     0.413 |      -0.134 |        0.23 |         0.32 |         15 |           16 |           -1 |
| Math & Stats        |          2 |     0.322 |      -0.146 |        0.11 |         0.29 |         17 |           17 |            0 |
| Education           |          1 |     0.451 |      -0.347 |        0.61 |         0.29 |         12 |           18 |           -6 |

- Rank correlation raw vs residual cluster gap: **Spearman +0.84**.
- Residual-ranking bootstrap stability (resample fields): Spearman to point estimate **+0.92** [+0.85, +0.97].
- Positive **rank_shift** = the cluster's gap is *more* explained by the two channels (it drops in the decoupling ranking); negative = its gap *survives* netting (genuine valuation divergence, rises in the residual ranking).

## Top residual (genuine decoupling) fields

| label                   | c2name         |   gap |   residual |
|:------------------------|:---------------|------:|-----------:|
| Forestry                | Nat. Resources | 1.150 |     +0.718 |
| Physiology              | Biological Sci | 1.344 |     +0.664 |
| Agronomy                | Agriculture    | 0.927 |     +0.493 |
| Music                   | Arts           | 0.961 |     +0.459 |
| Communication Disorders | Health         | 1.208 |     +0.406 |
| Microbiology            | Biological Sci | 0.934 |     +0.347 |
| Geography               | Social Sci     | 0.762 |     +0.316 |
| Environmental Sciences  | Nat. Resources | 0.771 |     +0.302 |
| Animal Sciences         | Agriculture    | 0.804 |     +0.266 |
| Earth Sciences          | Physical Sci   | 0.715 |     +0.250 |

## Sensitivity

- STRICT vs BROAD licensure: b_licensure +0.66 vs +0.45 — engineering moves most (PE makes it 'licensed' under BROAD), so the licensing coefficient and engineering's residual are the load-bearing choice; reported both ways.
- ACS vs SDR absorption: b_absorption -0.02 vs -0.35 (SDR SEH-only, smaller n).
- Precision-weighted vs unweighted residual rank: Spearman +1.00 (field level).
- The two anchors are only weakly correlated (corr(licensure, absorption) = -0.21), so the null absorption coefficient is **not** a collinearity artifact — the academic-pipeline channel, as measured, simply does not predict the gap (and flips sign under the SDR proxy), whereas licensing is the one robust mechanical channel.

## Adversarial self-check

**Strongest referee objection.** This is a projection, not identification: licensure and absorption are themselves *consequences* of the same discipline structure that drives the gap (collider/over-control risk), so the 'residual' is not a clean 'genuine divergence' — it is whatever is orthogonal to two correlated, non-randomly-assigned proxies, and the anchors are field aggregates (ecological), measured on an ACS undergrad-field universe that differs from the Wapman PhD-field / Scorecard BA-earnings universes of the gap. The licensing coefficient also flips materially with the strict/broad choice (engineering).
**Does it survive?** As a *causal* claim, no — and we do not make one (stated up front). As a *decomposition*, partially: the licensing coefficient is CI-excludes-0 (+0.66), the absorption coefficient is CI-spans-0 (-0.02); and the residual cluster ranking re-orders the raw one (rank Spearman +0.84) stably under field resampling (stability +0.92).. The residual ranking should be read as 'gap not co-moving with licensing/pipeline at the field level', not as a structural parameter. We report coefficients, CIs and the strict/broad split openly rather than selecting a specification.
