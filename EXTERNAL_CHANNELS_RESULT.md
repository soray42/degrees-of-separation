# External-channel decomposition of the AR-ER gap (beyond licensing)

Field-level, externally-measured channels (ACS PUMS BA-holders), each conceptually distinct from the gap. Run: `python scripts/27_external_channels.py`. Outcome-agnostic.

## Headline

**Licensing remains the one dominant external channel; the additional channels are largely REDUNDANT with it.** The pay-wedge holds bivariately (-0.33) but adds almost nothing beyond licensing: multi-channel **R²=0.29** vs **licensure-only 0.27** (and 0.28 for licensure+absorption in scripts/20). In the joint model the standardized licensure coefficient (+0.11) dominates while pay-wedge (-0.03) and public-sector (-0.01) collapse toward zero — they are collinear compression mechanisms (public-sector's +0.37 bivariate vanishes once licensing is netted). The multi-channel residual cluster ranking is **identical** to the licensing-only one (Spearman +1.00) and the residual SD barely shrinks (0.228→0.225). **The search for more external channels mostly finds collinear redundancy, not new explanatory power.**

## PRIORITY — does the academic-vs-industry PAY WEDGE hold?

Built on **BA-holders** (fixes the B1 cross-level mismatch): median industry (private for-profit) minus academic/education-sector earnings, n=48 fields (vs SDR doctorate-only n=17). Median signed wedge industry−academic = $25,000.

- **Spearman(gap, signed pay_wedge) = -0.33 (p=0.022), n=48.**
- Cross-level comparison: the SDR **doctorate-only** wedge gives **-0.42** (the B1 −0.61-class number, tiny SEH sample).
- abs |wedge| corr -0.33; relative wedge corr -0.27 (signed ≈ abs because industry > academic in nearly every field).

**Verdict: the pay-wedge channel HOLDS (attenuated).** The SDR −0.61 was inflated by the small SEH doctorate sample; on the proper BA construction it is **-0.33** — same (negative) sign, ~half the magnitude. The signed industry-premium is negatively related to the gap (stronger market demand → sharper market valuation → tracks prestige → lower gap), so it survives as a **real but weaker** second channel, not a clean −0.61 effect.

**Field-size check (priority channel):** unlike the O*NET proxy, the pay-wedge is **not** a field-size artifact — the relative-wedge↔gap correlation is -0.27 and **-0.29** after partialling out log institutions-per-field (if anything slightly stronger), so the attenuation it suffers in the multi-channel model is **collinearity with licensing, not field size**.

## Channels (bivariate Spearman with gap)

| channel | Spearman(gap, ·) | p | n | reading |
|---|---|---|---|---|
| pay_wedge (industry−academic) | -0.33 | 0.022 | 48 | market premium ↓ gap |
| public_sector_share | +0.37 | 0.009 | 48 | govt pay scales → compression ↑ gap |
| licensure_strict | +0.37 | 0.009 | 48 | regulation → compression ↑ gap |
| unionization | — | — | — | **DATA GAP: no BLS/CPS union-by-occupation file in repo; not fabricated. See note.** |

## Multi-channel decomposition (precision-weighted WLS, standardized channels)

gap ~ pay_wedge_rel + pub_share + licensure_strict  (n=48)

| channel (z) | coef | 95% CI | coef w/ field-size control |
|---|---|---|---|
| pay_wedge_rel | -0.035 | [-0.092, +0.023] | -0.028 |
| pub_share | -0.012 | [-0.079, +0.055] | -0.007 |
| licensure_strict | +0.106 | [+0.069, +0.142] | +0.104 |

- **Total R² = 0.29** (vs licensure-only R²=0.27; and ~0.28 for licensure+absorption in scripts/20). The external channels jointly explain ~29% of cross-field gap variance.
- **Residual SD shrinks**: raw 0.250 → licensing-only 0.228 → all-channels 0.225.
- With **log field-size** added, the channel coefficients move to the 'w/ field-size' column above (attenuation = the size confound).

## Collinearity (compression channels)

Spearman correlation among the channels:

|                  |   pay_wedge_rel |   pub_share |   licensure_strict |
|:-----------------|----------------:|------------:|-------------------:|
| pay_wedge_rel    |           +1.00 |       -0.54 |              -0.28 |
| pub_share        |           -0.54 |       +1.00 |              +0.40 |
| licensure_strict |           -0.28 |       +0.40 |              +1.00 |

licensure / public_sector / (union) are all 'compression' mechanisms; **they are materially correlated, so individual coefficients are unstable — read the joint R² and a combined-compression reading, not the separate slopes.**

## Cluster re-ranking on the multi-channel residual

| name                |   n |   raw |   resid_lic |   resid_multi |
|:--------------------|----:|------:|------------:|--------------:|
| Nat. Resources      |   2 | 0.844 |       0.382 |         0.377 |
| Arts                |   2 | 0.816 |       0.323 |         0.279 |
| Physical Sci        |   3 | 0.685 |       0.179 |         0.210 |
| Architecture        |   1 | 0.777 |       0.174 |         0.144 |
| Philosophy/Religion |   1 | 0.657 |       0.119 |         0.100 |
| Biological Sci      |   5 | 0.699 |       0.071 |         0.098 |
| Agriculture         |   3 | 0.574 |       0.082 |         0.060 |
| Business            |   4 | 0.476 |       0.040 |         0.036 |
| Social Work         |   1 | 0.552 |       0.068 |         0.025 |
| Health              |   4 | 0.805 |       0.031 |         0.015 |
| Engineering         |  10 | 0.427 |      -0.006 |         0.009 |
| English             |   1 | 0.506 |      -0.035 |        -0.045 |
| Social Sci          |   5 | 0.412 |      -0.089 |        -0.101 |
| Psychology          |   1 | 0.429 |      -0.089 |        -0.123 |
| Computer/Info       |   1 | 0.292 |      -0.129 |        -0.124 |
| History             |   1 | 0.413 |      -0.138 |        -0.125 |
| Math & Stats        |   2 | 0.322 |      -0.149 |        -0.143 |
| Education           |   1 | 0.451 |      -0.347 |        -0.347 |

- raw vs multi-channel residual cluster ranking: Spearman **+0.85**; licensing-residual vs multi-channel: **+1.00**.
- The residual SD shrinks modestly with the extra channels (above), i.e. the external channels explain somewhat more of the gap, but a large structured residual remains.

## Adversarial self-check

- **Not-the-gap:** every channel is a field attribute measured OUTSIDE the prestige↔earnings rank correlation — sector pay levels (wedge), employment composition (public_sector), occupational regulation (licensure). None is a transform of the gap, unlike the failed 'valuation divergence' proxies. pay_wedge is an earnings *level* difference across sectors, not the prestige-earnings *rank* agreement, so it is distinct.
- **Cross-level:** the headline pay_wedge is BA-level (the gap is a BA phenomenon); the doctorate-only SDR version (-0.42) is steeper and on n=17 — the BA number (-0.33) is the trustworthy one; the SDR −0.61 was a small-sample/cross-level overstate.
- **Field-size:** controlling log institutions-per-field attenuates the coefficients (size column); report both, do not over-read the uncontrolled slopes.
- **Collinearity:** the compression channels share a mechanism; the matrix above bounds it; joint R² is the safer summary than individual slopes.
- **Ecological:** all channels are field aggregates joined to a field-level gap — associations are field-level, not individual; no within-field causal claim.

> **Unionization data gap.** No BLS/CPS union-coverage-by-occupation table is in the repo, and this round forbids new collaborator data; the union channel is left UNBUILT rather than fabricated. To add it: drop a crosswalked `data/raw/union/union_by_field.csv` (field, union_cov) from CPS/unionstats occupation union coverage via the existing field→SOC map.
