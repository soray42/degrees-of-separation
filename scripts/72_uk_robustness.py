"""
scripts/72_uk_robustness.py
=============================================================================================
UK LEO robustness: three referee problems with scripts/62's UK result, each tested by a test that
is fixed below before any of its results were computed.

Public data only (the DfE LEO provider-level file and the ORCID-derived UK hiring network already
on disk; no data file downloaded, so no new SOURCES.md entry). Descriptive, not causal. Everything
reuses scripts/62 by import (loading, prestige G, tables, cells, the frequency-weighted rank
statistics, the per-subject provider draws) and scripts/59 (the two-way cluster variance); neither
script is edited.

Availability of sex / POLAR4 at provider x subject (task b). The DfE release 'Graduate outcomes (LEO):
provider level data', tax year 2022-23 (published 26 June 2025, last updated 4 September 2025; release
page and data-guidance page checked 2026-09-24 and re-checked 2026-09-25) has two files: the dashboard underlying data (on disk, md5 checked below) and a
graduate-movement-between-regions file (counts by home / provider / current region, no earnings by
provider x subject). The underlying file carries ONE characteristic_type / characteristic_value pair
per row, so its breakdowns are marginal (all graduates; sex F/M; POLAR4 quintiles + not known; prior
attainment bands; ethnicity), never crossed. leo_structure() prints the header check, the row
counts by characteristic type (all rows and provider x subject rows) and the values the sex and POLAR4 rows
take (asserted: F / M and POLAR4_1-5 / Not known, no composite values) from the file itself.

PRE-SPECIFIED TESTS (fixed before running; anything else in the write-up is labelled exploratory)
  Common design: YAG5, cohorts 2013/14-2016/17, primary providers, cells = subject x cohort with
  >= 15 providers (scripts/62's composition sample unless stated). Subject value = mean over its
  cells; headline = unweighted mean over subjects (as scripts/62).
  x-scale for pay gradients: zG = rank of G among the 134 primary providers, standardised to SD 1
  over those providers ("per SD of G"); zS = rank of the institution-wide share of graduates with
  >= 360 tariff points among primary providers in the same cohort x YAG, standardised the same way.
  Pay gradient beta = unweighted OLS slope of log median earnings on zG (or zS) across providers
  within the cell (frequency-weighted for resampling). Sensitivity: zG / zS on the raw score scale.

  (a) Pay terms. beta_raw, beta_nat (earnings standardised with national band medians, scripts/62
      "std"), beta_wp (standardised with within-provider band gradients, scripts/62 "std_wp"); pay
      share surviving = mean(beta_std) / mean(beta_raw); same for zS; joint regression on zG and zS
      (G beyond selectivity in pay terms); within prior-attainment band (bands 1-5): slope of log band
      median vs slope of log all-graduate median on the same providers. Test: pay share minus the
      rank (Spearman) share of scripts/62, for wp, nat and within-band.
  (b) Sex and POLAR4. The provider x CAH2 LEO file publishes marginal (not crossed) breakdowns by
      sex (F, M) and POLAR4 quintile (1-5, not known). Each is standardised exactly like scripts/62's
      within-provider band standardisation (per subject x cohort x YAG, over all HEIs releasing >= 2
      group medians: log group median = provider FE + lambda * log national group median; expected =
      sum_g share_g exp(lambda log national_g); coverage >= 80%), and the three adjustments are added
      in logs (PA + sex + POLAR4). On the common sample where all are defined: rank coupling and pay
      gradient for raw, PA, sex, POLAR4, PA+sex+POLAR4 (and national-median versions). Test: share
      surviving under PA+sex+POLAR4 minus share under PA alone (rank and pay). Plus within-sex and
      within-POLAR4-quintile pay gradients vs matched all-graduate gradients.
  (c) Validity of G. Errors-in-variables bounds with validity v = corr(observed G, construct) in
      [0.74, 0.92] (grid 0.74, 0.80, 0.86, 0.92; plus sqrt of G's split-half reliability as the
      internal ceiling). Error assumed independent of earnings and of selectivity (non-differential),
      selectivity measured without error. Model M1 (primary): every within-cell correlation of G is
      attenuated by v. Model M2 (sensitivity): classical error with constant variance over the
      provider population, so a cell whose G spread is narrower has lower validity,
      v_cell^2 = 1 - (1 - v^2) / var_cell(zG). Quantities:
        - within-band partial rho(G, band earnings | institution selectivity) (scripts/62 +0.033):
          primary = pooled-correlation correction (mean r(G,y), r(G,S), r(S,y) over cells and subjects,
          then disattenuate and form the partial; meta-analytic SEM style); secondary = per-cell
          correction (inadmissible cells counted and dropped). Also the selectivity-beyond-G partial,
          the matched all-graduate partial, the composition-sample partial, the smallest admissible v,
          and the v grid 0.70-1.00 (tipping point: largest v whose two-way CI excludes 0).
        - career slope of rho(G, earnings) per year (scripts/62 +0.026), within-provider-standardised
          and within-band: corrected slope = slope / v (M1), slope / v_cell (M2).
        - Gaussian-copula (latent Pearson) version of both as a sensitivity.
  Inference standard: two-way (subject, provider) cluster variance of scripts/59 (Cameron-Gelbach-
  Miller: V_subject + V_provider - V_subject x provider), with V_subject = jackknife over subjects,
  V_provider from one provider draw shared by every subject (per-replicate seeds from
  SeedSequence.spawn), V_subject x provider from scripts/62's independent provider draw per subject.
  Normal 95% CI, p = two-sided normal, MDE (80% power, 5% two-sided) = 2.80 SE. scripts/62's
  two-stage percentile CI is reported alongside and reproduces scripts/62's own intervals.

  Reading rule (written 2026-09-24 before this version's first run; wording only, the tests above are
  unchanged): a pre-specified difference is 'detected' if its two-way 95% CI excludes 0, otherwise 'not
  detected' and reported with its MDE; a two-way CI resting on more than STABLE inadmissible replicates
  is flagged unstable and not used for a verdict. EIV bounds are reported as ranges over v in
  [0.74, 0.92], never as point claims.

  Revision 2026-09-25 (after an independent verifier's report; the tests, samples and estimators above are
  unchanged, only the Monte Carlo handling and the wording are): every two-way CI now carries scripts/59's
  Monte Carlo SE of its endpoints (tmcse), its robust (IQR) SE (tse_r, from a subject bootstrap of the
  per-subject values) and scripts/59's status x_status ('above 0' / 'below 0' / 'includes 0' / 'touches 0' =
  an endpoint within 2 MC SE of 0). Every row and grid point whose status at 1000 replicates is 'touches 0'
  is re-estimated with NEXT = 4000 further replicates of both provider draws (own SeedSequence streams; 5000
  in all), and its verdict is taken from the 5000-replicate CI; if it still touches 0 it is reported as
  borderline (Monte Carlo cannot settle the side). The unstable rule scales with the replicate count (more
  than 5% inadmissible). The narrative states a detection only through these statuses, never through a
  hard-coded side of 0. The per-cell correction at v = 0.74 is also reported with the uncorrected per-cell
  partial on the same admissible cells, because the inadmissible cells are not a random subset.

Seeded (SEED = 72); byte-identical on re-run, independent of the worker count (all draws are made in
the parent before the per-subject work is forked). Run:
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 .venv/bin/python scripts/72_uk_robustness.py [--workers N]
(default 1 worker, which keeps peak resident memory under about 0.9 GB; more workers are faster but each
forked worker adds resident memory)
Outputs: data/interim/uk_robustness_summary.csv, uk_robustness_cells.csv, uk_robustness_subjects.csv,
         uk_robustness_eiv_grid.csv, UK_ROBUSTNESS_RESULT.md (repo root, local only).
"""
from __future__ import annotations

import os
import sys
import gc
import json
import time
import zlib
import hashlib
import argparse
import warnings
import importlib.util
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s62 = _load("s62", "62_uk_leo.py")          # UK LEO engine (not edited)
s59 = _load("s59", "59_pseo_refresh.py")    # two-way cluster variance (not edited)
twoway, jack_var, Z975, x_status = s59.twoway, s59.jack_var, s59.Z975, s59.x_status
wspear, wpartial, ones, ci95 = s62.wspear, s62.wpartial, s62.ones, s62.ci95
NMIN, COHORTS, PA_BANDS, PA_ALL, W_SLOPE = s62.NMIN, s62.COHORTS, s62.PA_BANDS, s62.PA_ALL, s62.W_SLOPE

SEED = 72
NBOOT = s62.NBOOT                # 1000: the independent per-subject draws are scripts/62's own
Z80 = float(norm.ppf(0.80))
SEX = ["F", "M"]
POL = [f"POLAR4_{q}" for q in range(1, 6)] + ["POLAR4_NK"]
V_US = 0.74                      # ORCID-rebuilt US academia rank vs Wapman published ranks (scripts/11)
V_HI = 0.92                      # UK G split-half reliability (Spearman-Brown, scripts/62), used as a validity
V_GRID = [0.74, 0.80, 0.86, 0.92]
V_FINE = np.round(np.arange(0.70, 1.0001, 0.01), 2)
STABLE = 50                      # a CI is reported only if at most 50 of 1000 replicates are inadmissible
STABLE_FRAC = STABLE / NBOOT     # the same 5% rule for the 5000-replicate re-estimates
NEXT = 4000                      # further replicates of both provider draws for rows whose CI touches 0
INTERIM = ROOT / "data" / "interim"
OUT_SUM = INTERIM / "uk_robustness_summary.csv"
OUT_CELLS = INTERIM / "uk_robustness_cells.csv"
OUT_SUBJ = INTERIM / "uk_robustness_subjects.csv"
OUT_GRID = INTERIM / "uk_robustness_eiv_grid.csv"
OUT_META = INTERIM / "uk_robustness_meta.json"
OUT_MD = ROOT / "UK_ROBUSTNESS_RESULT.md"
S62_SUM = INTERIM / "uk_leo_summary.csv"
ORCID_VALID = ROOT / "outputs" / "ORCID_OVERLAP_VALIDATION_RESULT.md"


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def _trim():
    """Return freed heap to the OS before the fork (lower resident memory of the forked workers)."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


LEO_MD5 = "7396c5a128d2e84d357f47629605f107"      # printed by scripts/62 in UK_LEO_RESULT.md (Provenance); SOURCES.md has URL + date


def leo_structure() -> dict:
    """Header check and row counts by characteristic type of the provider file (one pass, one column).
    A row can carry only one characteristic (one characteristic_type column), so breakdowns are marginal."""
    import zipfile
    z = zipfile.ZipFile(s62.LEO_ZIP)
    name = [n for n in z.namelist() if n.startswith("provider_data")][0]
    cols = list(pd.read_csv(z.open(name), encoding="latin-1", nrows=0).columns)
    char_cols = [c for c in cols if c.startswith("characteristic")]
    cnt: dict[str, int] = {}
    vals: dict[str, int] = {}
    subj: dict[str, int] = {}
    for ch in pd.read_csv(z.open(name), encoding="latin-1", dtype=str,
                          usecols=["characteristic_type", "characteristic_value", "ukprn", "cah2_subject_name"],
                          chunksize=500000):
        for k, v in ch.characteristic_type.value_counts().items():
            cnt[k] = cnt.get(k, 0) + int(v)
        # the values of the sex and POLAR4 breakdowns (a crossed table would show composite values)
        m = ch.characteristic_type.isin(["sex", "POLAR4"])
        for k, v in (ch.characteristic_type[m] + ":" + ch.characteristic_value[m]).value_counts().items():
            vals[k] = vals.get(k, 0) + int(v)
        # rows at provider x subject level (a provider, a subject) by characteristic type
        ps = (ch.ukprn != "Total") & (ch.cah2_subject_name != "Total")
        for k, v in ch.characteristic_type[ps].value_counts().items():
            subj[k] = subj.get(k, 0) + int(v)
    return dict(file=name, n_cols=len(cols), char_cols=char_cols, counts=dict(sorted(cnt.items())),
                values=dict(sorted(vals.items())), prov_subject_counts=dict(sorted(subj.items())),
                members=sorted(z.namelist()))


# =============================================================================================
# frequency-weighted linear statistics (exact equivalents of OLS on the resampled providers)
# =============================================================================================
def _c(x, W):
    return x - (W * x).sum(1, keepdims=True) / W.sum(1, keepdims=True)


def wslope(x, y, W):
    dx, dy = _c(x, W), _c(y, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (W * dx * dy).sum(1) / (W * dx * dx).sum(1)


def wmoments(x1, x2, y, W) -> dict:
    """weighted covariances of (x1, x2, y) under frequency weights W (B, n)."""
    sw = W.sum(1)
    d1, d2, dy = _c(x1, W), _c(x2, W), _c(y, W)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(c11=(W * d1 * d1).sum(1) / sw, c12=(W * d1 * d2).sum(1) / sw, c22=(W * d2 * d2).sum(1) / sw,
                    c1y=(W * d1 * dy).sum(1) / sw, c2y=(W * d2 * dy).sum(1) / sw)


def solve2(c11, c12, c22, c1y, c2y, v=1.0):
    """OLS coefficients of y on (x1, x2) from covariances, with var(x1) replaced by v^2 var(x1)
    (classical error in x1 with validity v); x1 coefficient returned per SD of the true x1 (times v)."""
    a11 = v * v * c11
    with np.errstate(invalid="ignore", divide="ignore"):
        det = a11 * c22 - c12 * c12
        b1 = (c22 * c1y - c12 * c2y) / det
        b2 = (a11 * c2y - c12 * c1y) / det
        ok = det > 0
        return np.where(ok, b1 * v, np.nan), np.where(ok, b2, np.nan)


def wvar(x, W):
    d = _c(x, W)
    return (W * d * d).sum(1) / W.sum(1)


# =============================================================================================
# errors-in-variables (disattenuation) of a partial correlation
# =============================================================================================
def eiv_partial(rGy, rGS, rSy, v):
    """Partial correlations of the TRUE G with y given S, and of S with y given the true G, when the
    observed G has validity v (corr(G_obs, G_true) = v; error independent of y and S; S error-free).
    NaN where the disattenuated correlation matrix is not positive definite (v too small)."""
    rGy, rGS, rSy = (np.asarray(x, float) for x in (rGy, rGS, rSy))
    with np.errstate(invalid="ignore", divide="ignore"):
        a, b, c = rGy / v, rGS / v, rSy
        det = 1 - a * a - b * b - c * c + 2 * a * b * c
        ok = (np.abs(a) < 1) & (np.abs(b) < 1) & (det > 0)
        pG = (a - b * c) / np.sqrt((1 - b * b) * (1 - c * c))
        pS = (c - a * b) / np.sqrt((1 - a * a) * (1 - b * b))
    return np.where(ok, pG, np.nan), np.where(ok, pS, np.nan)


def lat(r):
    """Spearman -> latent Pearson under a Gaussian copula."""
    return 2 * np.sin(np.pi * np.asarray(r, float) / 6)


def v_cell(v, varG):
    """M2: validity within a cell whose zG variance is varG (population variance 1), constant error variance."""
    with np.errstate(invalid="ignore"):
        x = 1 - (1 - v * v) / np.asarray(varG, float)
    return np.where(x > 0, np.sqrt(np.clip(x, 0, None)), np.nan)


# =============================================================================================
# sex and POLAR4 standardisation: scripts/62's within-provider lambda fit with the group set as an argument
# =============================================================================================
def group_gradients(leo: pd.DataFrame, prov: pd.DataFrame, groups: list[str]) -> tuple[pd.DataFrame, pd.Series, list]:
    """scripts/62 band_gradients(mode='lambda', weighted=False) for any set of marginal groups (62 hard-codes
    the ten prior-attainment categories). Per subject x cohort x YAG, over every HEI releasing >= 2 group
    medians: log group median = provider FE + lambda * log national group median. Checked in main() to
    reproduce scripts/62 exactly for the prior-attainment groups.
    Revision (2026-09-25, after the first complete run, whose lambda summary disagreed with an independent
    within-provider-demeaning re-computation in these cells; no test or rule changed): lambda is not
    identified in a cell where the log national
    medians of the groups each provider releases never differ within a provider (the lambda column is then
    collinear with the provider FE, and lstsq returned a minimum-norm value near 1). Such cells get no
    gradient (listed in the third return value). None occurs for the prior-attainment groups, so the
    reproduction of scripts/62 is unaffected; for sex (published national F and M medians equal) and
    POLAR4 the count is printed in the write-up."""
    hei = set(prov.ukprn[prov.provider_type == "HEI"])
    d = leo[leo.ukprn.isin(hei) & (leo.subject != "Total") & leo.grp.isin(groups) &
            leo.cohort.isin(COHORTS) & (leo.earnings_median > 0)]
    d = d[["subject", "cohort", "yag", "ukprn", "grp", "earnings_median", "grads_earnings_include"]]
    nat = np.log(leo[(leo.ukprn == "Total") & (leo.subject != "Total") & leo.cohort.isin(COHORTS)]
                 .pivot_table(index=["subject", "cohort", "yag"], columns="grp", values="earnings_median",
                              aggfunc="first").reindex(columns=groups))
    rows, lam, degen = {}, {}, []
    for key, g in d.groupby(["subject", "cohort", "yag"], sort=True):
        if key not in nat.index:
            continue
        nx = nat.loc[key]
        g = g[g.grp.map(nx).notna()]
        g = g[g.groupby("ukprn").grp.transform("size") >= 2]
        if g.empty:
            continue
        provs = sorted(g.ukprn.unique())
        pi = g.ukprn.map({p: i for i, p in enumerate(provs)}).values
        y = np.log(g.earnings_median.values.astype(float))
        X = np.zeros((len(g), len(provs) + 1))
        X[np.arange(len(g)), pi] = 1.0
        X[:, -1] = g.grp.map(nx).values
        xd = X[:, -1] - (X[:, :-1] @ (X[:, :-1].T @ X[:, -1] / X[:, :-1].sum(0)))   # within-provider deviations
        if not np.any(np.abs(xd) > 1e-9):
            degen.append(key)
            continue
        lam_ = float(np.linalg.lstsq(X, y, rcond=None)[0][-1])
        lam[key] = lam_
        rows[key] = (lam_ * nx).to_dict()
    B = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=groups)
    B.index = pd.MultiIndex.from_tuples(B.index, names=["subject", "cohort", "yag"])
    L = pd.Series(lam, dtype=float)
    L.index = pd.MultiIndex.from_tuples(L.index, names=["subject", "cohort", "yag"])
    return B, L, degen


def log_expected(med: pd.DataFrame, inc: pd.DataFrame, nat: pd.DataFrame, groups: list[str],
                 grads: pd.DataFrame) -> pd.DataFrame:
    """log expected median from the provider-subject's group mix of graduates included in earnings, with
    national group medians (nat) and within-provider gradients (wp); NaN unless the groups with a value
    cover >= 80% of the earnings sample (scripts/62's rule)."""
    shares = inc.reindex(columns=groups).fillna(0.0)
    natb = nat.reindex(columns=groups).reindex(med.index.droplevel("ukprn"))
    natb.index = med.index
    have = natb.notna() & (shares > 0)
    wsum = shares.where(have, 0).sum(1)
    ex_nat = (shares.where(have, 0) * natb.fillna(0)).sum(1) / wsum
    bg = grads.reindex(columns=groups).reindex(med.index.droplevel("ukprn"))
    bg.index = med.index
    hv = bg.notna() & (shares > 0)
    ws = shares.where(hv, 0).sum(1)
    ex_wp = (shares.where(hv, 0) * np.exp(bg.fillna(0))).sum(1) / ws
    return pd.DataFrame({"nat": np.log(ex_nat).where(wsum / inc["ALL"] >= 0.8),
                         "wp": np.log(ex_wp).where(ws / inc["ALL"] >= 0.8)})


# =============================================================================================
# draws
# =============================================================================================
class SharedDraw:
    """One provider draw shared by every subject and cell (the provider cluster of the two-way variance).
    Replicate b uses its own generator from SeedSequence([SEED, crc32(tag)]).spawn(n)[b]."""
    def __init__(self, provs: list[str], tag: str, n: int = NBOOT):
        u = sorted(provs)
        self.pos = {p: i for i, p in enumerate(u)}
        ss = np.random.SeedSequence([SEED, zlib.crc32(tag.encode())]).spawn(n)
        C = np.zeros((n, len(u)), dtype=np.int32)
        for b, s in enumerate(ss):
            np.add.at(C[b], np.random.default_rng(s).integers(len(u), size=len(u)), 1)
        self.C = C

    def W(self, s, provs):
        return self.C[:, [self.pos[p] for p in provs]]


class IndepDraw:
    """Independent provider draw per subject (the subject x provider cluster), for the extension replicates:
    subject s, replicate b uses SeedSequence([SEED, crc32(tag|s)]).spawn(n)[b] (scripts/62's own 1000-replicate
    streams are kept for the first 1000, so the two-stage CIs still reproduce scripts/62)."""
    def __init__(self, universe: dict[str, set], tag: str, n: int):
        self.pos, self.C = {}, {}
        for s in sorted(universe):
            u = sorted(universe[s])
            self.pos[s] = {p: i for i, p in enumerate(u)}
            ss = np.random.SeedSequence([SEED, zlib.crc32(f"{tag}|{s}".encode())]).spawn(n)
            self.C[s] = np.stack([np.bincount(np.random.default_rng(q).integers(len(u), size=len(u)),
                                              minlength=len(u)) for q in ss]).astype(np.int32)

    def W(self, s, provs):
        return self.C[s][:, [self.pos[s][p] for p in provs]]


# =============================================================================================
# per-cell statistics (cell, getW) -> {name: (B,) array}; getW(provs) gives the weight matrix
# =============================================================================================
ZG = ZS = ZGR = ZSR = None           # set in main() before the fork


def zs_of(cohort, yag, provs, raw=False):
    m = ZSR if raw else ZS
    return np.array([m.get((cohort, yag, p), np.nan) for p in provs], float)


def st_cross(cell, getW):
    c = cell["comp"]
    r = {}
    if len(c) < NMIN:
        return r
    W = getW(list(c.ukprn))
    G, S = c.G.values, c.inst_top.values
    earn = c.earn.values.astype(float)
    y = np.log(earn)
    zg, zs = c.zG.values, c.zS.values
    zgr, zsr = c.zG_raw.values, c.zS_raw.values
    ys = {"raw": y, "nat": c.earn_std.values, "wp": c.earn_std_wp.values}
    for t, yy in ys.items():
        r[f"rho_{t}"] = wspear(G, yy, W)
        r[f"rhoS_{t}"] = wspear(S, yy, W)
        r[f"b_{t}"] = wslope(zg, yy, W)
        r[f"bS_{t}"] = wslope(zs, yy, W)
        r[f"bR_{t}"] = wslope(zgr, yy, W)
        r[f"bSR_{t}"] = wslope(zsr, yy, W)
        m = wmoments(zg, zs, yy, W)
        r[f"bGS_{t}"], r[f"bSG_{t}"] = solve2(**m)
        for k, v in m.items():
            r[f"m{t}_{k}"] = v
    for t, yy in ys.items():
        r[f"sd_{t}"] = np.sqrt(wvar(yy, W))                     # cross-provider SD of log earnings in the cell
    r["rGS"] = wspear(G, S, W)
    r["p_inst"] = wpartial(G, earn, S[:, None], W)             # scripts/62 "p_inst"
    r["varG"] = wvar(zg, W)
    # (b) common sample: every sex / POLAR4 / PA standardisation defined
    ok = (c.lex_wp_sex.notna() & c.lex_wp_pol.notna() & c.lex_nat_sex.notna() & c.lex_nat_pol.notna()).values
    if ok.sum() >= NMIN:
        Wm, Gm, zm, ym = W[:, ok], G[ok], zg[ok], y[ok]
        lw = {k: c[f"lex_wp_{k}"].values[ok] for k in ("pa", "sex", "pol")}
        ln = {k: c[f"lex_nat_{k}"].values[ok] for k in ("pa", "sex", "pol")}
        yb = {"raw": ym, "pa_wp": ym - lw["pa"], "sex_wp": ym - lw["sex"], "pol_wp": ym - lw["pol"],
              "all_wp": ym - lw["pa"] - lw["sex"] - lw["pol"], "sexpol_wp": ym - lw["sex"] - lw["pol"],
              "pa_nat": ym - ln["pa"], "all_nat": ym - ln["pa"] - ln["sex"] - ln["pol"]}
        for t, yy in yb.items():
            r[f"B_rho_{t}"] = wspear(Gm, yy, Wm)
            r[f"B_b_{t}"] = wslope(zm, yy, Wm)
    return r


def st_band(cell, getW):
    W = getW(cell["provs"])
    G, ge, ea = cell["G"], cell["gearn"], cell["earn"]
    zg = ZG.reindex(cell["provs"]).values
    r = {"within": wspear(G, ge, W), "matched": wspear(G, ea, W),
         "bw": wslope(zg, np.log(ge), W), "bm": wslope(zg, np.log(ea), W), "varG": wvar(zg, W)}
    if cell.get("sel") is not None:
        S = cell["sel"]
        zs = zs_of(cell["cohort"], cell["yag"], cell["provs"])
        rGS = wspear(G, S, W)
        r.update(rGy_w=r["within"], rGy_m=r["matched"], rGS=rGS,
                 within_sel=wspear(S, ge, W), matched_sel=wspear(S, ea, W),
                 within_p_sel=wpartial(G, ge, S[:, None], W),            # scripts/62 rows
                 within_sel_p_G=wpartial(S, ge, G[:, None], W),
                 matched_p_sel=wpartial(G, ea, S[:, None], W),
                 varG_s=r["varG"])
        for v in (V_US, V_HI):
            pg, ps = eiv_partial(r["within"], rGS, r["within_sel"], v)
            r[f"pcG_{int(round(v * 100))}"], r[f"pcS_{int(round(v * 100))}"] = pg, ps
            # the uncorrected per-cell partial on exactly the cells (and replicates) where the correction is admissible
            r[f"pobs_{int(round(v * 100))}"] = np.where(np.isfinite(pg), r["within_p_sel"], np.nan)
        for t, yy in (("w", np.log(ge)), ("m", np.log(ea))):
            m = wmoments(zg, zs, yy, W)
            r[f"bGS_{t}"], r[f"bSG_{t}"] = solve2(**m)
            for k, v in m.items():
                r[f"m{t}_{k}"] = v
    return r


def st_group(cell, getW):
    W = getW(cell["provs"])
    zg = ZG.reindex(cell["provs"]).values
    return {"within": wspear(cell["G"], cell["gearn"], W), "matched": wspear(cell["G"], cell["earn"], W),
            "bw": wslope(zg, np.log(cell["gearn"]), W), "bm": wslope(zg, np.log(cell["earn"]), W)}


def st_retest(cell, getW):
    return s62.stats_retest(cell, getW(cell["provs"]))


def st_career(cell, getW):
    W = getW(cell["provs"])
    r = s62.stats_career(cell, W)
    zg = ZG.reindex(cell["provs"]).values
    LY = cell["Y"] if cell["col"] != "earn" and cell["grp"] == "ALL" else np.log(cell["Y"])
    b = np.stack([wslope(zg, LY[:, k], W) for k in range(3)], axis=1)
    r.update(b1=b[:, 0], b3=b[:, 1], b5=b[:, 2], bslope=b @ W_SLOPE, varG=wvar(zg, W))
    return r


# =============================================================================================
# per-subject runner (forked workers); all draws are fixed before the fork
# =============================================================================================
_JOB: dict = {}


def run_subject(s: str):
    out = {}
    for name, (cells, fn) in _JOB["analyses"].items():
        rows, acc = [], {}
        for cell in cells:
            if cell["subject"] != s:
                continue
            pt = fn(cell, lambda p: ones(len(p)))
            reps = {"ib": fn(cell, lambda p: _JOB["ib"].W(s, p)), "cb": fn(cell, lambda p: _JOB["cb"].W(s, p))}
            row = {k: v for k, v in cell.items() if k in ("subject", "cohort", "yag", "grp", "col")}
            row["n"] = len(cell["provs"])
            if "comp" in cell:
                row["n_comp"] = len(cell["comp"])
            for k, v in pt.items():
                row[k] = float(v[0])
            rows.append(row)
            for sch, bs in reps.items():
                for k, v in bs.items():
                    acc.setdefault(k, {}).setdefault(sch, []).append(np.asarray(v, float))
        df = pd.DataFrame(rows)
        subj = {}
        for k in acc:
            if k not in df.columns:
                continue
            pts = df[k].dropna()
            if len(pts) == 0:
                continue
            subj[k] = (float(pts.mean()), {sch: np.nanmean(np.stack(acc[k][sch]), axis=0) for sch in acc[k]})
        out[name] = (df, subj)
    return s, out


def run_subject_ext(s: str):
    """The NEXT extension replicates for the analyses and keys in _JOB['need'] (aggregated over cells exactly
    as run_subject does: nanmean over the subject's cells, per replicate)."""
    out = {}
    for name, keys in _JOB["need"].items():
        cells, fn = _JOB["analyses"][name]
        acc: dict = {}
        for cell in cells:
            if cell["subject"] != s:
                continue
            for sch, D in (("ib", _JOB["ib2"]), ("cb", _JOB["cb2"])):
                bs = fn(cell, lambda p, D=D: D.W(s, p))
                for k in sorted(keys):
                    if k in bs:
                        acc.setdefault(k, {}).setdefault(sch, []).append(np.asarray(bs[k], float))
        out[name] = {k: {sch: np.nanmean(np.stack(v), axis=0) for sch, v in d.items()} for k, d in acc.items()}
    return s, out


# =============================================================================================
# aggregation: estimate, two-way cluster CI (scripts/59), two-stage percentile CI (scripts/62)
# =============================================================================================
ROWS: list[dict] = []
RES: dict = {}
SPEC: dict = {}                  # key -> (fn, keys, analysis, subjects, tag): how the row was computed
ANA: dict = {}                   # id(subject dict) -> analysis name (set in main)


def mean0(a):
    return np.nanmean(a, axis=0)


def _tw(fn, pts, ib, cb, est, vf, fb_tag) -> dict:
    """Two-way CI of scripts/59 from per-subject replicate stacks ib, cb (K, n); robust (IQR) SE from a subject
    bootstrap of the per-subject point values (n draws, generator rng_for(fb_tag)); scripts/59's Monte Carlo SE
    of the CI endpoints and x_status."""
    ibr, cbr = np.asarray(fn(*ib), float), np.asarray(fn(*cb), float)
    ok = np.isfinite(ibr) & np.isfinite(cbr)
    n = int(len(ok))
    out = dict(n_bad=int((~ok).sum()), n_rep=n)
    K = len(pts[0])
    fi = rng_for(fb_tag).integers(K, size=(n, K))
    fb = np.asarray(fn(*[p[fi].T for p in pts]), float)
    fb = fb[np.isfinite(fb)]
    if np.isfinite(est) and np.isfinite(vf) and ok.sum() > 10:
        tw = twoway(est, vf, cbr[ok], ibr[ok], fb=fb if len(fb) > 10 else None)
        se = tw["tse"]
        out.update(lo=tw["tlo"], hi=tw["thi"], se=se, p=float(2 * norm.sf(abs(est / se))) if se > 0 else np.nan,
                   mde=(Z975 + Z80) * se, vf=tw["tvf"], vs=tw["tvs"], vi=tw["tvi"], fallback=tw["tfallback"],
                   mcse=tw["tmcse"], se_r=tw.get("tse_r", np.nan))
        out["status"] = (x_status(out["lo"], out["hi"], out["mcse"]) if out["n_bad"] <= STABLE_FRAC * n
                         else "unstable")
    else:
        out.update(lo=np.nan, hi=np.nan, se=np.nan, p=np.nan, mde=np.nan, vf=vf, vs=np.nan, vi=np.nan, fallback=False,
                   mcse=np.nan, se_r=np.nan, status="no CI")
    return out


def stat(fn, keys, subj, subjects, tag, rng=None):
    """fn maps per-subject arrays (axis 0 = subjects) to the statistic; keys index subj[key][s] =
    (point, {'ib': (B,), 'cb': (B,)})."""
    ss = [s for s in subjects if all(s in subj[k] for k in keys)]
    pts = [np.array([subj[k][s][0] for s in ss]) for k in keys]
    ib = [np.stack([subj[k][s][1]["ib"] for s in ss]) for k in keys]
    cb = [np.stack([subj[k][s][1]["cb"] for s in ss]) for k in keys]
    est = float(fn(*pts))
    vf = jack_var(fn, *pts)
    out = dict(est=est, k=len(ss))
    out.update(_tw(fn, pts, ib, cb, est, vf, "fb|" + tag))
    out["_spec"] = (fn, tuple(keys), ANA[id(subj)], tuple(ss), tag)
    rng = rng if rng is not None else rng_for("outer|" + tag)
    K, B = ib[0].shape
    fi = rng.integers(K, size=(B, K))
    ci_ = rng.integers(B, size=(B, K))
    ts = np.asarray(fn(*[x[fi, ci_].T for x in ib]), float)
    out["ts_lo"], out["ts_hi"] = ci95(ts) if np.isfinite(ts).sum() > 10 else (np.nan, np.nan)
    out["ts_bad"] = int((~np.isfinite(ts)).sum())
    return out


def stat_ext(spec, RS: dict, RS2: dict) -> dict:
    """Re-estimate the two-way CI of a row with its first NBOOT replicates plus the NEXT extension replicates
    (same estimate, same jackknife; replicates concatenated)."""
    fn, keys, ana, ss, tag = spec
    subj, subj2 = RS[ana], RS2[ana]
    for k in keys:
        assert all(s in subj2.get(k, {}) for s in ss), (ana, k)
    pts = [np.array([subj[k][s][0] for s in ss]) for k in keys]
    ib = [np.stack([np.concatenate([subj[k][s][1]["ib"], subj2[k][s]["ib"]]) for s in ss]) for k in keys]
    cb = [np.stack([np.concatenate([subj[k][s][1]["cb"], subj2[k][s]["cb"]]) for s in ss]) for k in keys]
    est = float(fn(*pts))
    return _tw(fn, pts, ib, cb, est, jack_var(fn, *pts), "fbx|" + tag)


EXT_COLS = ("lo", "hi", "se", "p", "mde", "vs", "vi", "fallback", "mcse", "se_r", "status", "n_bad", "n_rep")


def put(key, section, quantity, spec, r: dict | None = None, est=np.nan, **kw):
    row = dict(key=key, section=section, quantity=quantity, spec=spec)
    if r is not None:
        row.update({k: r.get(k, np.nan) for k in ("est", "lo", "hi", "se", "p", "mde", "ts_lo", "ts_hi", "k",
                                                   "n_bad", "ts_bad", "vf", "vs", "vi", "fallback", "mcse", "se_r",
                                                   "status", "n_rep")})
        SPEC[key] = r["_spec"]
    else:
        row["est"] = est
    row.update(kw)
    if r is not None:
        row["ci_ok"] = bool(np.isfinite(row.get("lo", np.nan)) and row.get("n_bad", 0) <= STABLE_FRAC * row["n_rep"])
    ROWS.append(row)
    RES[key] = row
    return row


# =============================================================================================
# main
# =============================================================================================
def main(workers: int):
    global ZG, ZS, ZGR, ZSR
    t0 = time.time()
    print("loading LEO ...", flush=True)
    leo_md5 = md5(s62.LEO_ZIP)
    assert leo_md5 == LEO_MD5, leo_md5
    struct = leo_structure()
    assert struct["char_cols"] == ["characteristic_type", "characteristic_value"], struct["char_cols"]
    # marginal breakdowns only: sex takes the values F / M and POLAR4 the quintiles + not known, no composites
    assert set(struct["values"]) == {"sex:F", "sex:M"} | {f"POLAR4:POLAR4_{q}" for q in range(1, 6)} | \
        {"POLAR4:Not known"}, struct["values"]
    print("LEO structure:", struct, flush=True)
    leo, pattr = s62.load_leo()
    e = s62.load_edges()
    prov, G, deg, e = s62.build_prestige(e, pattr)
    rel = s62.prestige_reliability(e, prov)
    GB = {"wp": s62.band_gradients(leo, prov, "lambda"), "wpw": s62.band_gradients(leo, prov, "lambda", weighted=True),
          "wpfe": s62.band_gradients(leo, prov, "fe")}
    GRADS = {k: v[0] for k, v in GB.items()}
    T, med, inc, nat = s62.build_tables(leo, prov, GRADS)

    # ---- (b) inputs: sex and POLAR4 gradients; check the generalised fit reproduces scripts/62 for PA ----
    Bpa, Lpa, Dpa = group_gradients(leo, prov, PA_ALL)
    assert not Dpa
    assert Bpa.index.equals(GB["wp"][0].index) and np.allclose(Bpa.values, GB["wp"][0].reindex(columns=PA_ALL).values,
                                                                 equal_nan=True, rtol=0, atol=1e-12)
    assert np.allclose(Lpa.values, GB["wp"][1].values, rtol=0, atol=1e-12)
    Bsex, Lsex, Dsex = group_gradients(leo, prov, SEX)
    Bpol, Lpol, Dpol = group_gradients(leo, prov, POL)
    del leo
    gc.collect()
    _trim()
    LEX = {}
    for tag, groups, Bg in (("pa", PA_ALL, Bpa), ("sex", SEX, Bsex), ("pol", POL, Bpol)):
        le = log_expected(med, inc, nat, groups, Bg)
        LEX[f"lex_nat_{tag}"], LEX[f"lex_wp_{tag}"] = le["nat"], le["wp"]
    LEX = pd.DataFrame(LEX).reset_index()
    T = T.merge(LEX, on=["subject", "cohort", "yag", "ukprn"], how="left", validate="one_to_one")
    lg = np.log(T.earn)
    for col, lx in (("earn_std", "lex_nat_pa"), ("earn_std_wp", "lex_wp_pa")):   # reproduces scripts/62's columns
        a, b = T[col].values, (lg - T[lx]).values
        assert np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=1e-12), col

    # ---- x scales ----
    P = prov[prov.primary].copy()
    rG = pd.Series(rankdata(P.G.values), index=P.ukprn.values)
    ZG = (rG - rG.mean()) / rG.std(ddof=0)
    ZGR = pd.Series(((P.G - P.G.mean()) / P.G.std(ddof=0)).values, index=P.ukprn.values)
    it = T.drop_duplicates(["cohort", "yag", "ukprn"])[["cohort", "yag", "ukprn", "inst_top"]].dropna()
    zs, zsr = {}, {}
    for (c, y), g in it.groupby(["cohort", "yag"]):
        rk = rankdata(g.inst_top.values)
        z1 = (rk - rk.mean()) / rk.std()
        z2 = (g.inst_top.values - g.inst_top.mean()) / g.inst_top.std(ddof=0)
        for p, a, b in zip(g.ukprn, z1, z2):
            zs[(c, y, p)], zsr[(c, y, p)] = a, b
    ZS, ZSR = zs, zsr
    T["zG"], T["zG_raw"] = T.ukprn.map(ZG), T.ukprn.map(ZGR)
    T["zS"] = [zs.get(k, np.nan) for k in zip(T.cohort, T.yag, T.ukprn)]
    T["zS_raw"] = [zsr.get(k, np.nan) for k in zip(T.cohort, T.yag, T.ukprn)]
    # provider-level G vs selectivity (as scripts/62 row a)
    itp = T[(T.yag == 5) & T.cohort.isin(COHORTS)].groupby("ukprn").inst_top.mean()
    r_G_sel = float(pd.Series(P.G.values).corr(pd.Series(P.ukprn.map(itp).values), method="spearman"))

    # ---- cells: scripts/62's own builders; the draw universe is exactly scripts/62's analyse() universe ----
    cx = s62.cells_cross(T, 5)
    bands = s62.cells_group(T, med, PA_BANDS, 5, with_sel=True)
    sexc = s62.cells_group(T, med, SEX, 5)
    polc = s62.cells_group(T, med, [f"POLAR4_{q}" for q in range(1, 6)], 5)
    cc = s62.cells_career(T)
    ccw = s62.cells_career(T, col="earn_std_wp")
    cband = [c for b in PA_BANDS for c in s62.cells_career(T, med, grp=b)]
    univ_cells = dict(cx=cx, bands=bands, bands_ext=s62.cells_group(T, med, ["PA7", "PA8"], 5),
                      bands_y1=s62.cells_group(T, med, PA_BANDS, 1), bands_y3=s62.cells_group(T, med, PA_BANDS, 3),
                      sex=sexc, polar=polc, cc=cc, ccs=s62.cells_career(T, col="earn_std"), ccs_wp=ccw,
                      cal=s62.cells_calendar(T), cband=cband, retest=s62.cells_retest(T, med, PA_BANDS, 5))
    univ = {}
    for cl in univ_cells.values():
        for c in cl:
            univ.setdefault(c["subject"], set()).update(c["provs"])
            if "comp" in c:
                univ[c["subject"]].update(c["comp"].ukprn)
    IB = s62.Draws(univ)
    CB = SharedDraw(sorted(set().union(*univ.values())), "shared_provider_draw")
    _JOB.update(ib=IB, cb=CB, analyses=dict(cx=(cx, st_cross), bands=(bands, st_band), sex=(sexc, st_group),
                                            polar=(polc, st_group), cc=(cc, st_career), ccw=(ccw, st_career),
                                            cband=(cband, st_career), retest=(univ_cells["retest"], st_retest)))
    subjects_all = sorted(univ)
    gc.collect()
    _trim()
    print(f"setup {time.time() - t0:.0f}s; {len(subjects_all)} subjects; workers {workers}", flush=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
            outs = dict(ex.map(run_subject, subjects_all))
    else:
        outs = dict(map(run_subject, subjects_all))
    print(f"cells done {time.time() - t0:.0f}s", flush=True)
    RS = {}
    CELLS = {}
    for name in _JOB["analyses"]:
        subj, dfs = {}, []
        for s in subjects_all:
            df, sj = outs[s][name]
            dfs.append(df)
            for k, v in sj.items():
                subj.setdefault(k, {})[s] = v
        RS[name] = subj
        d = pd.concat(dfs, ignore_index=True)
        CELLS[name] = d.assign(analysis=name)
    del outs
    ANA.update({id(v): k for k, v in RS.items()})

    sx, sb, cc_, ccw_, cb_ = RS["cx"], RS["bands"], RS["cc"], RS["ccw"], RS["cband"]
    compsubs = sorted(sx["rho_nat"])
    bsubs = sorted(sb["within"])
    ssel = sorted(sb["within_sel"])
    csubs = sorted(cc_["slope"])

    def ratio(a, b):
        return mean0(a) / mean0(b)

    def diff_ratio(a, b, c, d):
        return mean0(a) / mean0(b) - mean0(c) / mean0(d)

    # ---------------- reproduction of scripts/62 (point + two-stage CI with scripts/62's own streams) -----------
    s62sum = pd.read_csv(S62_SUM)

    def s62row(q):
        r = s62sum[s62sum.quantity == q]
        assert len(r) == 1, q
        return r.iloc[0]
    REPRO = [("raw coupling, composition sample", mean0, ["rho_raw"], sx, compsubs, "raw_c"),
             ("composition-standardised earnings (national subject x band medians)", mean0, ["rho_nat"], sx, compsubs, "std"),
             ("composition-standardised earnings (within-provider band gradients)", mean0, ["rho_wp"], sx, compsubs, "std_wp"),
             ("share surviving: std / raw_c", ratio, ["rho_nat", "rho_raw"], sx, compsubs, "ratiostd"),
             ("share surviving: std_wp / raw_c", ratio, ["rho_wp", "rho_raw"], sx, compsubs, "ratiostd_wp"),
             ("partial | institution-wide share with >= 360 points", mean0, ["p_inst"], sx, compsubs, "p_inst"),
             ("coupling with institution selectivity instead of G (all graduates)", mean0, ["rhoS_raw"], sx, compsubs, "sel"),
             ("within-band coupling (bands 1-5)", mean0, ["within"], sb, bsubs, "within"),
             ("all-graduate coupling, same providers", mean0, ["matched"], sb, bsubs, "matched"),
             ("share surviving: within-band / matched", ratio, ["within", "matched"], sb, bsubs, "wratio"),
             ("within-band: partial rho(G, band earnings | institution top-band share)", mean0, ["within_p_sel"], sb, ssel, "within_p_sel"),
             ("within-band: partial rho(institution top-band share, band earnings | G)", mean0, ["within_sel_p_G"], sb, ssel, "within_sel_p_G"),
             ("matched all-graduate: partial rho(G, earnings | institution top-band share)", mean0, ["matched_p_sel"], sb, ssel, "matched_p_sel"),
             ("career slope of rho(G, earnings) per year", mean0, ["slope"], cc_, csubs, "slope"),
             ("career slope, within-provider-standardised", mean0, ["slope"], ccw_, sorted(ccw_["slope"]), "wslope"),
             ("career slope, within prior-attainment band", mean0, ["slope"], cb_, sorted(cb_["slope"]), "bslope")]
    n_repro = 0
    for q, fn, keys, sj, pool, tag62 in REPRO:
        r = stat(fn, keys, sj, pool, "r62" + tag62, rng=s62.rng_for(f"outer|{tag62}"))
        o = s62row(q)
        for mine, theirs in ((r["est"], o.est), (r["ts_lo"], o.lo), (r["ts_hi"], o.hi)):
            assert abs(mine - theirs) <= 5e-6 * max(1.0, abs(theirs)), (q, mine, theirs)
        n_repro += 1
        put("r62." + tag62, "repro", q, "scripts/62 row reproduced (estimate and two-stage CI with scripts/62's streams); "
            "two-way CI added", r)
    sb62 = {k: {s: (v[0], v[1]["ib"]) for s, v in sb[k].items()} for k in ("within", "matched")}
    sr62 = {k: {s: (v[0], v[1]["ib"]) for s, v in RS["retest"][k].items()} for k in ("r_band", "r_all")}
    cs = s62.corrected_share(sb62, sr62, "corr")
    o = s62row("share surviving, attenuation-adjusted")
    for mine, theirs in ((cs["est"], o.est), (cs["lo"], o.lo), (cs["hi"], o.hi)):
        assert abs(mine - theirs) <= 5e-6 * max(1.0, abs(theirs)), ("corr", mine, theirs)
    n_repro += 1
    put("r62.corr", "repro", "share surviving within band, attenuation-adjusted (scripts/62)",
        "(within/matched) * sqrt(r_all/r_band); two-stage CI only (scripts/62's method)", est=cs["est"], ts_lo=cs["lo"], ts_hi=cs["hi"], k=cs["k"])
    put("r62.n", "repro", "scripts/62 rows reproduced exactly", "estimate and two-stage 95% CI equal to the 6 digits of uk_leo_summary.csv",
        est=float(n_repro))

    # ---------------- (a) pay terms ----------------
    sec = "a"
    comp_spec = "YAG5; cohorts 2013-16; scripts/62 composition sample; log points per SD of the G rank"
    for t, lab in (("raw", "raw"), ("nat", "standardised, national band medians (over-corrects)"),
                   ("wp", "standardised, within-provider band gradients")):
        put(f"a.b_{t}", sec, f"pay gradient in G: {lab}", comp_spec, stat(mean0, [f"b_{t}"], sx, compsubs, f"b_{t}"))
        put(f"a.bS_{t}", sec, f"pay gradient in institution selectivity: {lab}",
            comp_spec.replace("G rank", "selectivity rank"), stat(mean0, [f"bS_{t}"], sx, compsubs, f"bS_{t}"))
        put(f"a.bGS_{t}", sec, f"pay gradient in G holding selectivity fixed: {lab}", comp_spec + "; joint OLS on zG and zS",
            stat(mean0, [f"bGS_{t}"], sx, compsubs, f"bGS_{t}"))
        put(f"a.bSG_{t}", sec, f"pay gradient in selectivity holding G fixed: {lab}",
            comp_spec.replace("G rank", "selectivity rank") + "; joint OLS on zG and zS",
            stat(mean0, [f"bSG_{t}"], sx, compsubs, f"bSG_{t}"))
        put(f"a.bR_{t}", sec, f"pay gradient in G, raw-score scale: {lab}", "as above; per SD of the SpringRank score",
            stat(mean0, [f"bR_{t}"], sx, compsubs, f"bR_{t}"))
        put(f"a.bSR_{t}", sec, f"pay gradient in selectivity, raw-share scale: {lab}", "as above; per SD of the top-band share",
            stat(mean0, [f"bSR_{t}"], sx, compsubs, f"bSR_{t}"))
        if t != "raw":
            put(f"a.rhoS_{t}", sec, f"rank coupling with selectivity: {lab}", "YAG5; composition sample",
                stat(mean0, [f"rhoS_{t}"], sx, compsubs, f"rhoS_{t}"))
    for t in ("nat", "wp"):
        put(f"a.pshare_{t}", sec, f"pay share surviving, G: {t} / raw", "ratio of subject means of pay gradients",
            stat(ratio, [f"b_{t}", "b_raw"], sx, compsubs, f"pshare_{t}"))
        put(f"a.rshare_{t}", sec, f"rank share surviving, G: {t} / raw (scripts/62)", "ratio of subject means of Spearman couplings",
            stat(ratio, [f"rho_{t}", "rho_raw"], sx, compsubs, f"rshare_{t}"))
        put(f"a.dshare_{t}", sec, f"pay share minus rank share, G: {t} [pre-specified test]", "difference of the two ratios; paired",
            stat(diff_ratio, [f"b_{t}", "b_raw", f"rho_{t}", "rho_raw"], sx, compsubs, f"dshare_{t}"))
        put(f"a.pshareS_{t}", sec, f"pay share surviving, selectivity: {t} / raw", "ratio of subject means",
            stat(ratio, [f"bS_{t}", "bS_raw"], sx, compsubs, f"pshareS_{t}"))
        put(f"a.rshareS_{t}", sec, f"rank share surviving, selectivity: {t} / raw", "ratio of subject means",
            stat(ratio, [f"rhoS_{t}", "rhoS_raw"], sx, compsubs, f"rshareS_{t}"))
        put(f"a.pshareR_{t}", sec, f"pay share surviving, G raw-score scale: {t} / raw", "ratio of subject means",
            stat(ratio, [f"bR_{t}", "bR_raw"], sx, compsubs, f"pshareR_{t}"))
    for t in ("nat", "wp"):
        put(f"x.sdratio_{t}", "x", f"[exploratory] cross-provider SD of log earnings, {t} / raw", "composition sample; ratio of subject means of within-cell SDs",
            stat(ratio, [f"sd_{t}", "sd_raw"], sx, compsubs, f"sdratio_{t}"))
    put("x.sd_raw", "x", "[exploratory] cross-provider SD of log earnings, raw", "composition sample; mean within-cell SD",
        stat(mean0, ["sd_raw"], sx, compsubs, "sd_raw"))
    bspec = "YAG5; bands 1-5; subject x cohort x band cells, providers >= 15; same providers for both"
    put("a.bw", sec, "pay gradient in G within prior-attainment band", bspec, stat(mean0, ["bw"], sb, bsubs, "bw"))
    put("a.bm", sec, "pay gradient in G, all graduates of the same providers", bspec, stat(mean0, ["bm"], sb, bsubs, "bm"))
    put("a.pshare_band", sec, "pay share surviving within band: within / matched", "ratio of subject means",
        stat(ratio, ["bw", "bm"], sb, bsubs, "pshare_band"))
    put("a.dshare_band", sec, "pay share minus rank share, within band [pre-specified test]", "difference of the two ratios; paired",
        stat(diff_ratio, ["bw", "bm", "within", "matched"], sb, bsubs, "dshare_band"))
    put("a.bGS_w", sec, "within band: pay gradient in G holding selectivity fixed", bspec + "; cells with selectivity for every provider",
        stat(mean0, ["bGS_w"], sb, ssel, "bGS_w"))
    put("a.bSG_w", sec, "within band: pay gradient in selectivity holding G fixed", bspec + "; cells with selectivity for every provider",
        stat(mean0, ["bSG_w"], sb, ssel, "bSG_w"))
    put("a.bGS_m", sec, "matched all-graduate: pay gradient in G holding selectivity fixed", "same cells",
        stat(mean0, ["bGS_m"], sb, ssel, "bGS_m"))
    put("a.bSG_m", sec, "matched all-graduate: pay gradient in selectivity holding G fixed", "same cells",
        stat(mean0, ["bSG_m"], sb, ssel, "bSG_m"))
    # exploratory: pay gradient over careers
    for nm, sj, lab in (("cc", cc_, "all graduates"), ("ccw", ccw_, "within-provider-standardised")):
        ks = sorted(sj["bslope"])
        for y in ("b1", "b3", "b5"):
            put(f"x.{nm}_{y}", "x", f"[exploratory] pay gradient in G at YAG{y[1]}: {lab}", "balanced fixed-cohort panels (scripts/62 career cells)",
                stat(mean0, [y], sj, ks, f"{nm}{y}"))
        put(f"x.{nm}_bslope", "x", f"[exploratory] career slope of the pay gradient per year: {lab}", "OLS over YAG 1/3/5; log points per SD per year",
            stat(mean0, ["bslope"], sj, ks, f"{nm}bslope"))

    # ---------------- (b) sex and POLAR4 ----------------
    sec = "b"
    L5 = {k: v[v.index.get_level_values("yag") == 5] for k, v in (("pa", Lpa), ("sex", Lsex), ("pol", Lpol))}
    for k, lab in (("pa", "prior attainment (scripts/62)"), ("sex", "sex"), ("pol", "POLAR4")):
        put(f"b.lambda_{k}", sec, f"lambda (within-provider / national gradient): {lab}",
            "median over subject x cohort cells, YAG5, cohorts 2013-16; CI columns = IQR", est=float(L5[k].median()),
            lo=float(L5[k].quantile(0.25)), hi=float(L5[k].quantile(0.75)), k=len(L5[k]), ci_kind="IQR")
    for k, D, lab in (("sex", Dsex, "sex"), ("pol", Dpol, "POLAR4")):
        put(f"b.degen_{k}", sec, f"cells where lambda is not identified: {lab}",
            "subject x cohort x YAG cells, cohorts 2013-16, all YAG (lo column = at YAG5): the log national medians of "
            "the groups each provider releases never differ within a provider; no gradient, so no standardisation there",
            est=float(len(D)), lo=float(sum(1 for key in D if key[2] == 5)))
    bsubsB = sorted(sx["B_rho_raw"])
    ncomm = [(len(c["comp"]), int((c["comp"].lex_wp_sex.notna() & c["comp"].lex_wp_pol.notna() &
                                   c["comp"].lex_nat_sex.notna() & c["comp"].lex_nat_pol.notna()).sum())) for c in cx]
    put("b.n_cells", sec, "cells in the common sample", "composition-sample cells with >= 15 providers having every standardisation",
        est=float(sum(1 for a, b in ncomm if b >= NMIN and a >= NMIN)), k=len(bsubsB))
    # revision: the common-sample mean is over the common-sample cells only (earlier: over all composition cells)
    put("b.n_prov", sec, "providers per cell: composition sample vs common sample",
        "est = mean over the composition-sample cells; lo = mean over the common-sample cells (cells with >= 15 "
        "providers having every standardisation); hi = the earlier figure, mean over all composition cells of the "
        "providers with every standardisation",
        est=float(np.mean([a for a, b in ncomm if a >= NMIN])),
        lo=float(np.mean([b for a, b in ncomm if a >= NMIN and b >= NMIN])),
        hi=float(np.mean([b for a, b in ncomm if a >= NMIN])),
        n_cells=int(sum(1 for a, b in ncomm if a >= NMIN)))
    BT = [("raw", "raw"), ("pa_wp", "prior attainment (within-provider)"), ("sex_wp", "sex (within-provider)"),
          ("pol_wp", "POLAR4 (within-provider)"), ("sexpol_wp", "sex + POLAR4 (within-provider)"),
          ("all_wp", "prior attainment + sex + POLAR4 (within-provider)"),
          ("pa_nat", "prior attainment (national medians, over-corrects)"),
          ("all_nat", "prior attainment + sex + POLAR4 (national medians, over-corrects)")]
    cspec = "YAG5; cohorts 2013-16; common sample"
    for t, lab in BT:
        put(f"b.rho_{t}", sec, f"rank coupling: {lab}", cspec, stat(mean0, [f"B_rho_{t}"], sx, bsubsB, f"Brho_{t}"))
        put(f"b.b_{t}", sec, f"pay gradient in G: {lab}", cspec + "; log points per SD of the G rank",
            stat(mean0, [f"B_b_{t}"], sx, bsubsB, f"Bb_{t}"))
        if t != "raw":
            put(f"b.rshare_{t}", sec, f"rank share surviving: {lab}", "ratio of subject means, common sample",
                stat(ratio, [f"B_rho_{t}", "B_rho_raw"], sx, bsubsB, f"Brs_{t}"))
            put(f"b.pshare_{t}", sec, f"pay share surviving: {lab}", "ratio of subject means, common sample",
                stat(ratio, [f"B_b_{t}", "B_b_raw"], sx, bsubsB, f"Bps_{t}"))
    for t in ("all_wp", "all_nat"):
        base = "pa_wp" if t == "all_wp" else "pa_nat"
        put(f"b.drshare_{t}", sec, f"rank share: {t} minus {base} [pre-specified test]", "difference of ratios; paired; common sample",
            stat(diff_ratio, [f"B_rho_{t}", "B_rho_raw", f"B_rho_{base}", "B_rho_raw"], sx, bsubsB, f"Bdr_{t}"))
        put(f"b.dpshare_{t}", sec, f"pay share: {t} minus {base} [pre-specified test]", "difference of ratios; paired; common sample",
            stat(diff_ratio, [f"B_b_{t}", "B_b_raw", f"B_b_{base}", "B_b_raw"], sx, bsubsB, f"Bdp_{t}"))
    for nm, lab in (("sex", "within sex"), ("polar", "within POLAR4 quintile")):
        sj = RS[nm]
        ks = sorted(sj["within"])
        put(f"b.{nm}_within", sec, f"{lab}: rank coupling (scripts/62)", "YAG5; group cells, providers >= 15",
            stat(mean0, ["within"], sj, ks, f"{nm}rw"))
        put(f"b.{nm}_matched", sec, f"{lab}: rank coupling, all graduates of the same providers (scripts/62)", "same providers",
            stat(mean0, ["matched"], sj, ks, f"{nm}rm"))
        put(f"b.{nm}_bw", sec, f"{lab}: pay gradient in G", "YAG5; group cells; log points per SD", stat(mean0, ["bw"], sj, ks, f"{nm}bw"))
        put(f"b.{nm}_bm", sec, f"{lab}: pay gradient, all graduates of the same providers", "same providers",
            stat(mean0, ["bm"], sj, ks, f"{nm}bm"))
        put(f"b.{nm}_pshare", sec, f"{lab}: pay share within / matched", "ratio of subject means",
            stat(ratio, ["bw", "bm"], sj, ks, f"{nm}ps"))
        put(f"b.{nm}_rshare", sec, f"{lab}: rank share within / matched", "ratio of subject means",
            stat(ratio, ["within", "matched"], sj, ks, f"{nm}rs"))

    # ---------------- (c) validity of G ----------------
    sec = "c"
    put("c.v_us", sec, "validity input, low end: US ORCID-rebuilt academia rank vs Wapman published ranks",
        "Spearman; outputs/ORCID_OVERLAP_VALIDATION_RESULT.md (scripts/11), an input, not recomputed here", est=V_US)
    put("c.rel_sb", sec, "UK G split-half reliability (Spearman-Brown), recomputed", "scripts/62 prestige_reliability, 100 random person splits",
        est=rel["split_half_sb"])
    put("c.v_ceiling", sec, "validity ceiling implied by the reliability, sqrt(reliability)", "internal consistency only", est=float(np.sqrt(rel["split_half_sb"])))
    put("c.r_G_sel", sec, "Spearman(G, institution selectivity) across primary providers", "provider-level, cohorts 2013-16 at YAG5",
        est=r_G_sel, k=int(P.ukprn.map(itp).notna().sum()))

    def pooled_pG(v, lat_=False, m2=False):
        def f(rGy, rGS, rSy, varG):
            a, b, c = mean0(rGy), mean0(rGS), mean0(rSy)
            vv = v_cell(v, mean0(varG)) if m2 else v
            if lat_:
                a, b, c, vv = lat(a), lat(b), lat(c), lat(vv)
            return eiv_partial(a, b, c, vv)[0]
        return f

    def pooled_pS(v, lat_=False, m2=False):
        def f(rGy, rGS, rSy, varG):
            a, b, c = mean0(rGy), mean0(rGS), mean0(rSy)
            vv = v_cell(v, mean0(varG)) if m2 else v
            if lat_:
                a, b, c, vv = lat(a), lat(b), lat(c), lat(vv)
            return eiv_partial(a, b, c, vv)[1]
        return f
    wkeys = ["rGy_w", "rGS", "within_sel", "varG_s"]
    mkeys = ["rGy_m", "rGS", "matched_sel", "varG_s"]
    ckeys = ["rho_raw", "rGS", "rhoS_raw", "varG"]
    for nm, lab in (("rGy_w", "mean r(G, band earnings)"), ("rGS", "mean r(G, selectivity)"), ("within_sel", "mean r(selectivity, band earnings)"),
                    ("varG_s", "mean within-cell variance of zG (population = 1)")):
        put(f"c.pool_{nm}", sec, f"within band, pooled input: {lab}", "band cells with selectivity for every provider",
            stat(mean0, [nm], sb, ssel, f"pool{nm}"))
    for nm, lab in (("rGy_m", "mean r(G, all-graduate earnings), band-cell providers"), ("matched_sel", "mean r(selectivity, all-graduate earnings), band-cell providers")):
        put(f"c.pool_{nm}", sec, f"matched, pooled input: {lab}", "band cells with selectivity for every provider",
            stat(mean0, [nm], sb, ssel, f"pool{nm}"))
    for nm, lab in (("rho_raw", "mean r(G, earnings)"), ("rGS", "mean r(G, selectivity)"), ("rhoS_raw", "mean r(selectivity, earnings)")):
        put(f"c.poolc_{nm}", sec, f"composition sample, pooled input: {lab}", "composition-sample cells",
            stat(mean0, [nm], sx, compsubs, f"poolc{nm}"))
    put("c.pool_varG_cx", sec, "composition sample, pooled input: mean within-cell variance of zG", "cross-section cells",
        stat(mean0, ["varG"], sx, compsubs, "poolvarGcx"))
    put("c.pool_varG_cc", sec, "career panels, pooled input: mean within-cell variance of zG", "balanced fixed-cohort cells",
        stat(mean0, ["varG"], cc_, csubs, "poolvarGcc"))
    # smallest admissible v at the point estimate (pooled correlations)
    vgrid = np.round(np.arange(0.50, 1.00001, 0.001), 3)
    for q, keys, pool, sj in (("within_band", wkeys, ssel, sb), ("matched", mkeys, ssel, sb), ("composition", ckeys, compsubs, sx)):
        pw = [np.mean([sj[k][s][0] for s in pool]) for k in keys]
        adm = [v for v in vgrid if np.isfinite(eiv_partial(pw[0], pw[1], pw[2], v)[0])]
        put(f"c.vmin_{q}_M1", sec, f"smallest admissible validity, {q} (M1)", "disattenuated pooled correlation matrix positive definite; 0.001 grid",
            est=float(min(adm)) if adm else np.nan)
        adm2 = [v for v in vgrid if np.isfinite(eiv_partial(pw[0], pw[1], pw[2], float(v_cell(v, pw[3])))[0])]
        put(f"c.vmin_{q}_M2", sec, f"smallest admissible population validity, {q} (M2)", "M2: v_cell from the mean within-cell zG variance",
            est=float(min(adm2)) if adm2 else np.nan)
    GRID, GSPEC = [], []
    for v in V_FINE:
        for model, m2 in (("M1", False), ("M2", True)):
            for q, keys, pool, sj in (("within_band", wkeys, ssel, sb), ("matched", mkeys, ssel, sb), ("composition", ckeys, compsubs, sx)):
                r = stat(pooled_pG(float(v), m2=m2), keys, sj, pool, f"grid{q}{model}{v:.2f}")
                GRID.append(dict(v=float(v), model=model, sample=q, est=r["est"], lo=r["lo"], hi=r["hi"], se=r["se"], p=r["p"],
                                 n_bad=r["n_bad"], ts_lo=r["ts_lo"], ts_hi=r["ts_hi"], mcse=r["mcse"], status=r["status"],
                                 n_rep=r["n_rep"]))
                GSPEC.append(r["_spec"])
    tip_pos = len(ROWS)          # the tipping-point rows are computed after the extension pass and inserted here
    for v in V_GRID + [float(np.round(np.sqrt(rel["split_half_sb"]), 3)), 1.0]:
        vt = f"{v:.2f}" if v < 1 else "1 (uncorrected)"
        for model, m2 in (("M1", False), ("M2", True)):
            if v == 1.0 and m2:
                continue
            put(f"c.pG_w_{model}_{v:.3f}", sec, f"within band: partial rho(true G, band earnings | selectivity), v = {vt} [{model}]",
                "pooled correlations (bands 1-5, YAG5)", stat(pooled_pG(v, m2=m2), wkeys, sb, ssel, f"pGw{model}{v}"))
            put(f"c.pS_w_{model}_{v:.3f}", sec, f"within band: partial rho(selectivity, band earnings | true G), v = {vt} [{model}]",
                "pooled correlations", stat(pooled_pS(v, m2=m2), wkeys, sb, ssel, f"pSw{model}{v}"))
            put(f"c.pG_m_{model}_{v:.3f}", sec, f"matched all-graduate: partial rho(true G, earnings | selectivity), v = {vt} [{model}]",
                "pooled correlations, band-cell providers", stat(pooled_pG(v, m2=m2), mkeys, sb, ssel, f"pGm{model}{v}"))
            put(f"c.pG_c_{model}_{v:.3f}", sec, f"composition sample: partial rho(true G, earnings | selectivity), v = {vt} [{model}]",
                "pooled correlations, composition-sample cells", stat(pooled_pG(v, m2=m2), ckeys, sx, compsubs, f"pGc{model}{v}"))
            put(f"c.pS_c_{model}_{v:.3f}", sec, f"composition sample: partial rho(selectivity, earnings | true G), v = {vt} [{model}]",
                "pooled correlations, composition-sample cells", stat(pooled_pS(v, m2=m2), ckeys, sx, compsubs, f"pSc{model}{v}"))
        if v < 1:
            put(f"c.pG_w_lat_{v:.3f}", sec, f"within band: latent-Pearson partial (Gaussian copula), v = {vt} [M1]",
                "pooled correlations converted r = 2 sin(pi rho / 6); v converted likewise", stat(pooled_pG(v, lat_=True), wkeys, sb, ssel, f"pGwl{v}"))
    put("c.pG_w_lat_1.000", sec, "within band: latent-Pearson partial (Gaussian copula), v = 1 (uncorrected)", "pooled correlations",
        stat(pooled_pG(1.0, lat_=True), wkeys, sb, ssel, "pGwl1"))
    for v in [1.0, 0.92, 0.86, 0.80, 0.74]:
        for q, keys, pool, sj in (("w", wkeys, ssel, sb), ("c", ckeys, compsubs, sx)):
            fS, fG = pooled_pS(v), pooled_pG(v)
            put(f"x.dSG_{q}_{v:.2f}", "x", f"[exploratory] selectivity partial minus G partial, {'within band' if q == 'w' else 'composition sample'}, v = {v:.2f} [M1]",
                "pooled correlations; paired", stat(lambda *a, fS=fS, fG=fG: fS(*a) - fG(*a), keys, sj, pool, f"dSG{q}{v}"))
    # per-cell correction (secondary)
    dband = CELLS["bands"]
    dsel = dband[dband.rGS.notna()]
    for v in (V_US, V_HI):
        k = f"pcG_{int(round(v * 100))}"
        put(f"c.pc_{k}", sec, f"within band, per-cell correction, v = {v:.2f} [M1]: mean over admissible cells",
            "cells whose disattenuated matrix is not positive definite dropped (count in n_adm)",
            stat(mean0, [k], sb, [s for s in ssel if s in sb[k]], f"pc{k}"),
            n_adm=int(dsel[k].notna().sum()), n_cells=len(dsel))
        # revision: the same mean, uncorrected, on exactly the admissible cells (the dropped cells are not random)
        ko = f"pobs_{int(round(v * 100))}"
        put(f"c.pc_{ko}", sec, f"within band, per-cell partial UNCORRECTED on the cells admissible at v = {v:.2f}",
            "mean over the same admissible cells (per replicate as well); compare with the corrected row above",
            stat(mean0, [ko], sb, [s for s in ssel if s in sb[ko]], f"pc{ko}"),
            n_adm=int(dsel[ko].notna().sum()), n_cells=len(dsel))
        adm = dsel[k].notna()
        for nm, col, lab in (("rGS", "rGS", "r(G, selectivity)"), ("rGy", "within", "r(G, band earnings)"),
                             ("pobs", "within_p_sel", "uncorrected per-cell partial")):
            put(f"x.pc{int(round(v * 100))}_{nm}", "x",
                f"[exploratory] band cells admissible vs dropped at v = {v:.2f}: mean {lab}",
                "cell-level unweighted means of point values; est = admissible cells, lo = dropped cells",
                est=float(dsel.loc[adm, col].mean()), lo=float(dsel.loc[~adm, col].mean()) if (~adm).any() else np.nan,
                n_adm=int(adm.sum()), n_cells=len(dsel))
            # the same on the scale of the headline (mean over subjects of the subject's mean over its cells)
            sk = dsel.loc[adm].groupby("subject")[col].mean()
            sd_ = dsel.loc[~adm].groupby("subject")[col].mean()
            put(f"x.pc{int(round(v * 100))}s_{nm}", "x",
                f"[exploratory] band cells admissible vs dropped at v = {v:.2f}: mean over subjects of {lab}",
                "mean over subjects of the subject mean over its cells; est = admissible cells, lo = dropped cells; "
                "k = subjects with admissible cells, k_drop = subjects with dropped cells",
                est=float(sk.mean()), lo=float(sd_.mean()) if len(sd_) else np.nan, k=int(len(sk)), k_drop=int(len(sd_)),
                n_adm=int(adm.sum()), n_cells=len(dsel))
            if nm == "pobs":       # the kept-cell subject mean is exactly the point estimate of the c.pc_pobs row
                assert abs(float(sk.mean()) - RES[f"c.pc_{ko}"]["est"]) < 1e-9
    put("c.pc_obs", sec, "within band, per-cell partial, v = 1 (scripts/62 row)", "mean over cells", stat(mean0, ["within_p_sel"], sb, ssel, "pcobs"))
    # pay-terms corrected joint regression within band (pooled moments, M1)
    for v in (V_US, V_HI, 1.0):
        def fb(c11, c12, c22, c1y, c2y, v=v):
            return solve2(mean0(c11), mean0(c12), mean0(c22), mean0(c1y), mean0(c2y), v)[0]
        mk = [f"mw_{k}" for k in ("c11", "c12", "c22", "c1y", "c2y")]
        put(f"c.bGS_w_{v:.2f}", sec, f"[exploratory] within band: pay gradient in true G holding selectivity fixed, v = {v:.2f} [M1]",
            "pooled covariances; log points per SD of true G", stat(fb, mk, sb, ssel, f"bGSw{v}"))
        mk = [f"mwp_{k}" for k in ("c11", "c12", "c22", "c1y", "c2y")]
        put(f"c.bGS_c_{v:.2f}", sec, f"[exploratory] composition sample, within-provider-standardised: pay gradient in true G holding selectivity fixed, v = {v:.2f} [M1]",
            "pooled covariances; log points per SD of true G", stat(fb, mk, sx, compsubs, f"bGScwp{v}"))
    # career slope
    for nm, sj, lab in (("cc", cc_, "all graduates"), ("ccw", ccw_, "within-provider-standardised"), ("cband", cb_, "within prior-attainment band")):
        ks = sorted(sj["slope"])
        for v in V_GRID + [1.0]:
            put(f"c.slope_{nm}_M1_{v:.2f}", sec, f"career slope per year, {lab}, v = {v:.2f} [M1]", "slope / v; balanced fixed-cohort panels",
                stat(lambda a, v=v: mean0(a) / v, ["slope"], sj, ks, f"sl{nm}{v}"))
            if v < 1:
                put(f"c.slope_{nm}_M2_{v:.2f}", sec, f"career slope per year, {lab}, v = {v:.2f} [M2]", "slope / v_cell (mean within-cell zG variance)",
                    stat(lambda a, g, v=v: mean0(a) / v_cell(v, mean0(g)), ["slope", "varG"], sj, ks, f"sl2{nm}{v}"))

                def fl(y1, y3, y5, v=v):
                    rr = [np.arcsin(np.clip(lat(mean0(x)) / lat(v), -1, 1) / 2) * 6 / np.pi for x in (y1, y3, y5)]
                    return np.tensordot(W_SLOPE, np.stack(rr), axes=1)
                put(f"c.slope_{nm}_lat_{v:.2f}", sec, f"career slope per year, {lab}, v = {v:.2f} [Gaussian copula]",
                    "each horizon's pooled coupling corrected in the latent metric, back-transformed, then OLS slope",
                    stat(fl, ["y1", "y3", "y5"], sj, ks, f"sll{nm}{v}"))
        put(f"c.pmean_{nm}", sec, f"career slope from pooled horizon means, {lab} (uncorrected; base of the latent version)", "balanced panels",
            stat(lambda y1, y3, y5: np.tensordot(W_SLOPE, np.stack([mean0(y1), mean0(y3), mean0(y5)]), axes=1),
                 ["y1", "y3", "y5"], sj, ks, f"pm{nm}"))
    # M2: the within-cell validity each population validity implies (point values, from the pooled zG variances)
    for q, sj, key, pool, lab in (("band", sb, "varG_s", ssel, "band cells with selectivity"),
                                  ("comp", sx, "varG", compsubs, "composition-sample cells"),
                                  ("career", cc_, "varG", csubs, "career panels, all graduates"),
                                  ("cband", cb_, "varG", sorted(cb_["varG"]), "career panels, within band")):
        vg = float(np.mean([sj[key][s][0] for s in pool]))
        for v in V_GRID:
            put(f"c.vcell_{q}_{v:.2f}", sec, f"M2 within-cell validity implied by v = {v:.2f}: {lab}",
                "sqrt(1 - (1 - v^2) / mean within-cell variance of zG); point value; NaN = not defined",
                est=float(v_cell(v, vg)))
    # cross-section coupling corrected (context)
    for v in (V_US, V_HI):
        put(f"c.rho_raw_{v:.2f}", sec, f"coupling rho(true G, earnings), composition sample, v = {v:.2f} [M1]", "rho / v",
            stat(lambda a, v=v: mean0(a) / v, ["rho_raw"], sx, compsubs, f"rhoraw{v}"))

    # ---------------- Monte Carlo extension: CIs with an endpoint within 2 MC SE of 0 get NEXT more replicates ------
    touch_rows = [k for k in RES if k in SPEC and RES[k]["status"] == "touches 0"]
    touch_grid = [i for i, g in enumerate(GRID) if g["status"] == "touches 0"]
    need: dict = {}
    for spec in [SPEC[k] for k in touch_rows] + [GSPEC[i] for i in touch_grid]:
        need.setdefault(spec[2], set()).update(spec[1])
    print(f"extension: {len(touch_rows)} rows and {len(touch_grid)} grid points touch 0; analyses "
          f"{ {k: sorted(v) for k, v in sorted(need.items())} }", flush=True)
    if need:
        IB2 = IndepDraw(univ, "independent_provider_draw_ext", NEXT)
        CB2 = SharedDraw(sorted(set().union(*univ.values())), "shared_provider_draw_ext", NEXT)
        _JOB.update(ib2=IB2, cb2=CB2, need={k: sorted(v) for k, v in sorted(need.items())})
        gc.collect()
        _trim()
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("fork")) as ex:
                outs2 = dict(ex.map(run_subject_ext, subjects_all))
        else:
            outs2 = dict(map(run_subject_ext, subjects_all))
        RS2: dict = {name: {} for name in need}
        for s in subjects_all:
            for name, d in outs2[s].items():
                for k, v in d.items():
                    RS2[name].setdefault(k, {})[s] = v
        del outs2
        for kk in ("ib2", "cb2", "need"):
            _JOB.pop(kk)
        del IB2, CB2
        gc.collect()

        def extend(r: dict, spec):
            x = stat_ext(spec, RS, RS2)
            r.update({f"{c}_1k": r[c] for c in ("lo", "hi", "se", "mcse", "status", "n_bad")})
            r.update({c: x.get(c, np.nan) for c in EXT_COLS})
            if "ci_ok" in r:
                r["ci_ok"] = bool(np.isfinite(r["lo"]) and r["n_bad"] <= STABLE_FRAC * r["n_rep"])
        for k in touch_rows:
            extend(RES[k], SPEC[k])
        for i in touch_grid:
            extend(GRID[i], GSPEC[i])
        del RS2
        print(f"extension done {time.time() - t0:.0f}s", flush=True)
    GRID = pd.DataFrame(GRID)
    # tipping points (pre-specified), from the grid after the extension; inserted where the grid was computed
    sec = "c"
    n0 = len(ROWS)
    for q in ("within_band", "matched", "composition"):
        for model in ("M1", "M2"):
            g = GRID[(GRID["sample"] == q) & (GRID.model == model) & GRID.lo.notna()]
            ex = g[(g.lo > 0) | (g.hi < 0)]
            put(f"c.tip_{q}_{model}", sec, f"tipping point, {q} ({model}): largest v in 0.70-1.00 whose two-way CI of the corrected partial excludes 0",
                "grid step 0.01; NaN = none [pre-specified]", est=float(ex.v.max()) if len(ex) else np.nan)
            st_ = g[g.n_bad <= STABLE_FRAC * g.n_rep]
            ex2 = st_[(st_.lo > 0) | (st_.hi < 0)]
            tv = st_.v[st_.status == "touches 0"]
            put(f"c.vrange_{q}_{model}", sec, f"v range with a stable CI excluding 0, {q} ({model})",
                f"grid step 0.01; stable = at most {STABLE} of {NBOOT} replicates inadmissible (5% of 5000 for re-estimated "
                f"points); est = min v, lo/hi = min, max; n_stable = grid points with a stable CI; n_touch = stable points "
                f"whose CI endpoint is within 2 MC SE of 0 after re-estimation", est=float(ex2.v.min()) if len(ex2) else np.nan,
                lo=float(ex2.v.min()) if len(ex2) else np.nan, hi=float(ex2.v.max()) if len(ex2) else np.nan,
                n_stable=int(len(st_)), vmin_stable=float(st_.v.min()) if len(st_) else np.nan,
                n_touch=int(len(tv)), touch_v=" ".join(f"{x:.2f}" for x in tv))
    new_rows = ROWS[n0:]
    del ROWS[n0:]
    ROWS[tip_pos:tip_pos] = new_rows

    # ---------------- write ----------------
    R = pd.DataFrame(ROWS)
    R.to_csv(OUT_SUM, index=False, float_format="%.6g")
    GRID.to_csv(OUT_GRID, index=False, float_format="%.6g")
    pd.concat(list(CELLS.values()), ignore_index=True).to_csv(OUT_CELLS, index=False, float_format="%.5g")
    per = {}
    for nm, keys in (("cx", ["rho_raw", "rho_nat", "rho_wp", "b_raw", "b_nat", "b_wp", "bS_raw", "bS_wp", "bGS_raw", "bGS_wp",
                              "B_rho_raw", "B_rho_pa_wp", "B_rho_all_wp", "B_b_raw", "B_b_pa_wp", "B_b_all_wp"]),
                     ("bands", ["within", "matched", "bw", "bm", "within_p_sel", "rGS"]), ("cc", ["slope", "bslope"])):
        for k in keys:
            per[f"{nm}_{k}"] = {s: v[0] for s, v in RS[nm][k].items()}
    per = pd.DataFrame(per)
    per.index.name = "subject"
    per.to_csv(OUT_SUBJ, float_format="%.4f")
    meta = dict(leo_md5=leo_md5, rel={k: float(v) for k, v in rel.items()}, n_prim=int(len(P)),
                subjects_all=len(subjects_all), subjects_comp=len(compsubs), subjects_band=len(bsubs),
                subjects_sel=len(ssel), subjects_career=len(csubs), subjects_common=len(bsubsB),
                lam_pa_n=int(len(L5["pa"])), lam_sex_n=int(len(L5["sex"])), lam_pol_n=int(len(L5["pol"])),
                n_repro=int(n_repro), structure=struct, nboot=NBOOT, seed=SEED, next=NEXT,
                ext_rows=touch_rows, ext_grid=[f"{GRID.v[i]:.2f} {GRID.model[i]} {GRID['sample'][i]}" for i in touch_grid],
                ext_analyses={k: sorted(v) for k, v in sorted(need.items())},
                n_cells={k: int(len(v)) for k, v in CELLS.items()},
                band_cells_sel=int(len(dsel)))
    with open(OUT_META, "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    md_only()           # the write-up is always built from the saved tables (same bytes as --md-only)
    print(R[["key", "est", "lo", "hi", "p", "ts_lo", "ts_hi", "k"]].to_string())
    print(f"done {time.time() - t0:.0f}s", flush=True)


# =============================================================================================
# write-up
# =============================================================================================
def _num(x, d=3, sign=True) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def _p(p) -> str:
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(p):
        return "—"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def write_md(R, GRID, per, meta):
    R = R.set_index("key", drop=False)
    B = int(meta["nboot"])
    NX = B + int(meta["next"])
    vc = float(np.round(np.sqrt(meta["rel"]["split_half_sb"]), 3))
    VK = [0.74, 0.80, 0.86, 0.92, vc, 1.0]

    def row(k):
        return R.loc[k]

    def nbad(r):
        return float(r["n_bad"]) if pd.notna(r["n_bad"]) else 0.0

    def nrep(r):
        return int(float(r["n_rep"])) if pd.notna(r.get("n_rep", np.nan)) else B

    def stable(r):
        return bool(np.isfinite(float(r["lo"])) and nbad(r) <= STABLE_FRAC * nrep(r))

    def status(k):
        r = row(k)
        return str(r["status"]) if stable(r) else "unstable"

    def e(k, d=3):
        return _num(row(k)["est"], d)

    def dd(k, d):
        return 4 if (d == 3 and str(row(k)["status"]) == "touches 0") else d

    def ci(k, d=3):
        r = row(k)
        d = dd(k, d)
        if nbad(r) > STABLE_FRAC * nrep(r):
            return f"(no CI: {int(nbad(r))} of {nrep(r)} replicates inadmissible)"
        return f"[{_num(r['lo'], d)}, {_num(r['hi'], d)}]" if stable(r) else "—"

    def t(k, d=3):
        return f"{e(k, dd(k, d))} {ci(k, d)}"

    def tsci(k, d=3):
        r = row(k)
        return f"[{_num(r['ts_lo'], d)}, {_num(r['ts_hi'], d)}]"

    def ts_excl(k):
        r = row(k)
        return float(r["ts_lo"]) > 0 or float(r["ts_hi"]) < 0

    def pc(x):
        return f"{100 * float(x):.0f}%"

    def tp(k):
        r = row(k)
        return f"{pc(r['est'])} [{pc(r['lo'])}, {pc(r['hi'])}]" if stable(r) else pc(r["est"])

    def lp(k):
        return f"about {100 * (np.exp(float(row(k)['est'])) - 1):.1f}%"

    def mde(k):
        return _num(row(k)["mde"], 3, sign=False)

    def est(k):
        return float(row(k)["est"])

    def hi(k):
        return float(row(k)["hi"])

    def lo(k):
        return float(row(k)["lo"])

    def pos(k):
        return status(k) == "above 0"

    def neg(k):
        return status(k) == "below 0"

    def incl0(k):
        return status(k) == "includes 0"

    def touch(k):
        return status(k) == "touches 0"

    def excl(k):
        return stable(row(k)) and (lo(k) > 0 or hi(k) < 0)

    def need(cond, msg):
        if not bool(cond):
            raise AssertionError("write-up claim no longer holds: " + msg)

    def near_end(k):
        r = row(k)
        return ("lower", float(r["lo"])) if abs(float(r["lo"])) <= abs(float(r["hi"])) else ("upper", float(r["hi"]))

    def cw(k):
        """what the two-way CI does, in words, from the Monte Carlo status"""
        s = status(k)
        if s in ("above 0", "below 0"):
            return "excludes 0"
        if s == "includes 0":
            return "includes 0"
        if s == "touches 0":
            side, x = near_end(k)
            return (f"touches 0 (its {side} limit {_num(x, 4)} is within 2 Monte Carlo SE of 0 after {nrep(row(k))} "
                    "replicates, so its side of 0 is not settled)")
        return "is unstable"

    def desc(k):
        """estimate, CI and the Monte-Carlo-aware reading"""
        s = status(k)
        if s in ("above 0", "below 0"):
            return f"{t(k)}, detected"
        if s == "includes 0":
            return f"{t(k)}, not detected"
        if s == "touches 0":
            side, x = near_end(k)
            return (f"{t(k)}, borderline: the two-way {side} limit {_num(x, 4)} is within 2 Monte Carlo SE of 0 "
                    f"(MC SE {_num(row(k)['mcse'], 4, False)}) after {nrep(row(k))} replicates")
        return t(k)

    def verdict(k):
        s = status(k)
        if s == "unstable":
            return "no verdict (CI unstable)"
        if s == "above 0":
            return "detected (positive)"
        if s == "below 0":
            return "detected (negative)"
        if s == "touches 0":
            return (f"borderline: CI {'excludes' if excl(k) else 'includes'} 0, but an endpoint is within 2 MC SE of 0 "
                    f"after {nrep(row(k))} replicates")
        return f"not detected; MDE {mde(k)}"

    def vt(k):
        return verdict(k).replace(" (positive)", ", positive").replace(" (negative)", ", negative")

    def mcs(k):
        r = row(k)
        s_ = str(r["status"])
        return s_ if nrep(r) == B else f"{s_} ({nrep(r)} reps)"

    def vk(prefix, v):
        return f"{prefix}_{v:.3f}"

    st = meta["structure"]["counts"]
    L = []
    w = L.append
    # ------------------------------------------------------------------ guards: structure and point estimates only;
    # every statement about a CI's side of 0 is worded from its Monte Carlo status (x_status), not asserted here
    need(meta["n_repro"] == 17, "17 scripts/62 rows reproduced")
    need(0.8 < est("b.lambda_sex") < 1.1 and 0.4 < est("b.lambda_pol") < 0.7 and 0.4 < est("b.lambda_pa") < 0.7,
         "lambda: sex near national size, POLAR4 and PA about half")
    need(abs(est("b.rshare_sex_wp") - 1) < 0.02 and abs(est("b.pshare_sex_wp") - 1) < 0.02, "sex alone changes almost nothing")
    need(all(est(k) < 0 and abs(est(k)) < 0.1 for k in ("b.drshare_all_wp", "b.dpshare_all_wp")), "sex+POLAR4: small, lower")
    need(est("a.pshare_wp") < est("a.rshare_wp") and est("a.pshare_nat") < est("a.rshare_nat"), "pay share below rank share")
    need(est("a.dshare_band") > 0, "within band pay share above rank share")
    need(est(vk("c.pS_c_M1", 0.80)) < est(vk("c.pG_c_M1", 0.80)), "G ahead at 0.80 too")
    need(est(vk("c.pS_c_M1", 0.92)) > est(vk("c.pG_c_M1", 0.92)) and
         est(vk("c.pS_c_M1", 0.86)) < est(vk("c.pG_c_M1", 0.86)), "composition ordering flips between 0.92 and 0.86")
    need(all(est(f"x.dSG_c_{v:.2f}") < 0 for v in (0.86, 0.80)), "composition difference negative at v <= 0.86")
    need(0.70 < est("c.vmin_composition_M1") <= 0.74, "vmin composition at the edge of 0.74")
    need(np.isnan(est("c.slope_cband_M2_0.74")) and np.isfinite(est("c.slope_cband_M2_0.80")) and
         not stable(row("c.slope_cband_M2_0.80")), "band M2 slope: point estimate but no stable CI at 0.80, undefined at 0.74")
    need(not stable(row(vk("c.pG_c_M1", 0.74))), "composition v = 0.74 unstable")
    need(abs(float(row("c.slope_cc_M1_0.74")["p"]) - float(row("c.slope_cc_M1_1.00")["p"])) < 1e-9, "M1 rescaling keeps p")
    need(all(est(f"c.slope_{nm}_M1_{v:.2f}") > 0 for nm in ("cc", "ccw", "cband") for v in V_GRID), "career slopes positive")
    need(all(float(row(k)["vs"]) > float(row(k)["vi"]) for k in ("r62.ratiostd_wp", "r62.p_inst", "r62.within_p_sel",
                                                                   "r62.matched_p_sel", "r62.slope")), "shared draw adds variance")
    need(not R.loc[R.lo.notna(), "fallback"].astype(str).isin(["True", "1", "1.0"]).any(), "no two-way fallback used")
    need(int(row("c.pc_pcG_92")["n_adm"]) == int(row("c.pc_pcG_92")["n_cells"]), "per-cell 0.92: every cell admissible")
    need(int(row("c.pc_pcG_74")["n_adm"]) < int(row("c.pc_pcG_74")["n_cells"]), "per-cell 0.74: cells dropped")
    need(float(row("x.pc74_rGS")["lo"]) > est("x.pc74_rGS") and float(row("x.pc74_rGy")["lo"]) > est("x.pc74_rGy") and
         float(row("x.pc74_pobs")["lo"]) > est("x.pc74_pobs") and float(row("x.pc74s_pobs")["lo"]) > est("x.pc74s_pobs"),
         "dropped cells: higher r(G,S), r(G,y) and uncorrected partials")
    # within band: every correction (except the selected per-cell mean at 0.74) whose CI is stable
    WB = ([vk("c.pG_w_M1", v) for v in V_GRID] + [vk("c.pG_w_M2", v) for v in V_GRID] +
          [vk("c.pG_w_lat", v) for v in V_GRID] + ["c.pc_pcG_92"])
    WBs = [k for k in WB if stable(row(k))]
    wb_status = sorted({status(k) for k in WBs})
    pw_lo, pw_hi = min(est(k) for k in WBs), max(est(k) for k in WBs)
    ub_all = max(hi(k) for k in WBs)                                   # largest upper 95% limit among them
    gw = GRID[(GRID["sample"] == "within_band") & (GRID.model == "M1")]
    gw_stat = sorted(set(gw.status))
    gm = GRID[(GRID["sample"] == "matched") & (GRID.model == "M1")]
    gm_stat = sorted(set(gm.status))
    need(all(np.isfinite(est(vk("c.pG_w_M1", v))) for v in V_GRID), "within-band M1 defined on the grid")
    m2_ok = [v for v in V_GRID if stable(row(vk("c.pG_w_M2", v)))]
    lat_gap = max(abs(est(f"c.slope_cc_lat_{v:.2f}") - est(f"c.slope_cc_M1_{v:.2f}")) for v in V_GRID)
    ub_w = max(hi(vk("c.pG_w_M1", v)) for v in VK)
    sl_all = [est(f"c.slope_cc_M1_{v:.2f}") for v in V_GRID] + [est(f"c.slope_cc_M2_{v:.2f}") for v in V_GRID]
    cc_stat = sorted({status(f"c.slope_{nm}_{m}_{v:.2f}") for nm in ("cc", "ccw") for m in ("M1", "M2") for v in V_GRID})
    band_m1_stat = sorted({status(f"c.slope_cband_M1_{v:.2f}") for v in V_GRID})
    ext_keys = [k for k in R.key if "lo_1k" in R.columns and pd.notna(R.loc[k, "lo_1k"])]
    n_ext_grid = int((GRID.n_rep > B).sum()) if "n_rep" in GRID.columns else 0
    band_ts_incl0 = not ts_excl("a.dshare_band")

    def words(ss):
        return " or ".join(f"'{x}'" for x in ss)

    def wband_low_order():
        """exploratory ordering at v <= 0.86 in both samples: every CI of the difference includes 0 or is unstable"""
        ks = [f"x.dSG_{q}_{v:.2f}" for q in ("w", "c") for v in (0.86, 0.80, 0.74)]
        return all(status(k) in ("includes 0", "unstable") for k in ks)

    # ------------------------------------------------------------------ header
    w("# UK LEO robustness — survival in pay terms, sex and POLAR4 standardisation, and error in the prestige measure G\n")
    w(f"Script: `scripts/72_uk_robustness.py` (seed {meta['seed']}; every number below is printed by the script from its own "
      "computations; two runs give byte-identical tables and write-up, see Method). Public data only: the DfE LEO "
      "provider-level file already on disk (md5 "
      f"`{meta['leo_md5']}`, asserted equal to the md5 scripts/62 printed in UK_LEO_RESULT.md; URL and access date in "
      "data/raw/SOURCES.md) and the academia-wide UK hiring-network prestige G of "
      "scripts/62 (ORCID-derived PhD→faculty edges cached by scripts/40). scripts/62 (LEO engine) and scripts/59 (two-way "
      "cluster variance, Monte Carlo SE, x_status) are imported, not edited. No file was downloaded, so data/raw/SOURCES.md "
      "needs no new entry. Descriptive, not causal. Revised on 2026-09-25 after an independent verifier's report (section "
      "Revisions). Tables: `data/interim/uk_robustness_summary.csv` (every number below with its SE, robust SE, Monte Carlo "
      "SE, status, MDE, variance components and replicate counts), `uk_robustness_eiv_grid.csv` (validity grid), "
      "`uk_robustness_cells.csv` (per cell), `uk_robustness_subjects.csv` (per subject), `uk_robustness_meta.json`.\n")

    # ------------------------------------------------------------------ answer
    need(wband_low_order(), "ordering undetermined at v <= 0.86 in both samples")
    w("## Answer\n")
    w("**Key question: does the UK result of scripts/62 hold when (a) survival of the prestige–pay coupling under the "
      "prior-attainment control is measured in pay rather than as a ratio of rank correlations, (b) sex and POLAR4 are "
      "standardised as well, and (c) the prestige measure G is allowed to contain error (validity 0.74–0.92)?** "
      f"Partly. Less survives in pay terms than in rank terms: after within-provider prior-attainment standardisation the "
      f"log-earnings gradient in G keeps {pc(est('a.pshare_wp'))} of its raw size, against {pc(est('a.rshare_wp'))} of the rank "
      f"coupling (pre-specified difference: {verdict('a.dshare_wp').split(' (')[0]}). Standardising for sex and POLAR4 as "
      f"well lowers survival by a few points more ({pc(est('b.pshare_all_wp'))} in pay terms; pre-specified difference: "
      f"{verdict('b.dpshare_all_wp').split(' (')[0]}). For all graduates the career rise survives every validity in the range "
      f"under both error models (corrected slope {_num(min(sl_all))} to {_num(max(sl_all))} per year, against "
      f"{e('c.slope_cc_M1_1.00')} uncorrected; CI status {words(cc_stat)} at every v and model, including the "
      "within-provider-standardised version); under these error models correction can only make it steeper. Within "
      f"prior-attainment band it survives under the constant-validity model (M1; CI status {words(band_m1_stat)} at every v). "
      f"Under the constant-error-variance model (M2) its CI {cw('c.slope_cband_M2_0.92')} at v = 0.92; at v = 0.86 its CI "
      f"{cw('c.slope_cband_M2_0.86')}; at 0.80 it has a point estimate ({e('c.slope_cband_M2_0.80')}) but no "
      "stable CI, and at 0.74 it is undefined. "
      "Among graduates of the same prior-attainment band, G's contribution beyond intake selectivity is not detected at "
      f"any validity from 0.70 to 1.00 (M1 grid status: {words(gw_stat)}), but the data do not show that it is small at "
      f"low validity. Corrected for validity 0.74–0.92, the point estimates of the partial lie between {_num(pw_lo)} and "
      f"{_num(pw_hi)} (every correction with a stable CI except the selected per-cell mean at 0.74, item 4), and the data "
      f"are compatible with a partial up to {_num(ub_all)}. In pay terms (exploratory) the upper 95% limit of the gradient "
      f"in true G holding selectivity fixed is {_num(hi('c.bGS_w_0.74'))} log points per SD at v = 0.74 and "
      f"{_num(hi('c.bGS_w_0.92'))} at v = 0.92, against a gradient in G alone of {e('a.bw')} among same-band graduates and "
      f"{e('a.b_raw')} for all graduates. "
      "Exploratory (no pre-specified test): the evidence for scripts/62's ordering, selectivity ahead of G, weakens under "
      "plausible error in G. Among same-band graduates, with G taken as error-free, the selectivity-minus-G difference "
      f"in partials is {desc('x.dSG_w_1.00')}. At v = 0.92 it stays at {e('x.dSG_w_0.92')}, with a CI that "
      f"{cw('x.dSG_w_0.92')} ({_num(lo('x.dSG_w_0.92'))} to {_num(hi('x.dSG_w_0.92'))}), and at v = 0.74 it falls to "
      f"{e('x.dSG_w_0.74')}. On all-graduate earnings the uncorrected difference is {desc('x.dSG_c_1.00')}. Its two-stage "
      f"CI {tsci('x.dSG_c_1.00')} {'excludes' if ts_excl('x.dSG_c_1.00') else 'includes'} 0. At v = 0.86 or below the "
      "all-graduate point estimates put G ahead, and no CI of the difference in either sample excludes 0 there, so the "
      "order is undetermined at v ≤ 0.86.\n")

    # 0. reproduction
    w(f"0. **Reproduction, and the two-way standard.** All {meta['n_repro']} scripts/62 rows used here reproduce exactly "
      "(estimate and two-stage 95% CI equal to the 6 digits of `uk_leo_summary.csv`), so the inputs are scripts/62's. "
      "Under the project's inference standard, the two-way (subject, provider) cluster variance of scripts/59, intervals "
      "are wider than scripts/62's two-stage percentile intervals: the same providers recur across subjects, and the "
      "variance under one provider draw shared by all subjects exceeds the variance under independent draws per subject "
      "in each row quoted here. "
      f"Share surviving within-provider standardisation {tp('r62.ratiostd_wp')} (two-stage [{pc(row('r62.ratiostd_wp')['ts_lo'])}, "
      f"{pc(row('r62.ratiostd_wp')['ts_hi'])}]); coupling partialling out institution selectivity {t('r62.p_inst')} "
      f"(two-stage {tsci('r62.p_inst')}); among same-band graduates, G beyond selectivity {t('r62.within_p_sel')} "
      f"(MDE {mde('r62.within_p_sel')}); on the all-graduate medians of the same providers {t('r62.matched_p_sel')}, whose "
      f"two-stage interval {tsci('r62.matched_p_sel')} {'excluded' if ts_excl('r62.matched_p_sel') else 'included'} 0. "
      f"Under the two-way standard the CI of that last row {cw('r62.matched_p_sel')}. Career slope {t('r62.slope')}/yr.\n")

    # 1. pay terms
    w("1. **(a) In pay terms less of the coupling survives than the rank ratio says.** Per SD of the G rank (over the "
      f"{meta['n_prim']} primary providers), median earnings five years after graduation are {e('a.b_raw')} log points "
      f"({lp('a.b_raw')}) higher across providers within a subject × cohort cell (composition sample, "
      f"{int(row('a.b_raw')['k'])} subjects; CI {ci('a.b_raw')}). After standardising for prior-attainment mix with "
      f"within-provider band gradients the gradient is {t('a.b_wp')} ({lp('a.b_wp')}), and with national band medians, which "
      f"over-correct, {t('a.b_nat')}. The pay share surviving is {tp('a.pshare_wp')} (within-provider) and {tp('a.pshare_nat')} "
      f"(national), against rank shares of {tp('a.rshare_wp')} and {tp('a.rshare_nat')}. Pre-specified test, pay share minus "
      f"rank share: {t('a.dshare_wp')} ({vt('a.dshare_wp')}) and {t('a.dshare_nat')} ({vt('a.dshare_nat')}). "
      + ("The rank ratio therefore reads higher than survival in pay. The standardisation narrows the pay differences between "
         "providers more than it changes their order. " if neg("a.dshare_wp") and neg("a.dshare_nat") else "")
      + "Exploratory: the cross-provider SD of log earnings in a cell falls to "
      f"{pc(est('x.sdratio_wp'))} of its raw size ({pc(est('x.sdratio_nat'))} with national medians; raw SD "
      f"{_num(est('x.sd_raw'), 3, False)}). "
      f"On the raw G score scale the pay share is {tp('a.pshareR_wp')}. In institution selectivity (rank of the "
      "institution-wide share of graduates with ≥ 360 tariff points) the gradient goes from "
      f"{e('a.bS_raw')} to {e('a.bS_wp')} log points per SD, a pay share of {tp('a.pshareS_wp')}. Holding selectivity fixed "
      f"in a joint regression, G has a pay gradient of {t('a.bGS_raw')} raw, {t('a.bGS_wp')} within-provider-standardised "
      f"and {t('a.bGS_nat')} with national medians. Selectivity holding G fixed: {t('a.bSG_raw')}, {t('a.bSG_wp')} and "
      f"{t('a.bSG_nat')}. These are descriptive gradients of provider medians, not returns.\n")
    w(f"   Among same-band graduates the comparison goes the other way. The within-band gradient is {t('a.bw')} against "
      f"{t('a.bm')} for all graduates of the same providers, a pay share of {tp('a.pshare_band')}, above the rank share of "
      f"{pc(est('r62.wratio'))}. Pre-specified test: {t('a.dshare_band')}, {vt('a.dshare_band')} under the two-way standard "
      f"(Monte Carlo SE of the endpoints {_num(row('a.dshare_band')['mcse'], 4, False)}; variance-based SE "
      f"{_num(row('a.dshare_band')['se'], 3, False)}, robust (IQR) SE {_num(row('a.dshare_band')['se_r'], 3, False)})"
      + (f"; the two-stage percentile interval {tsci('a.dshare_band')} includes 0, so it is borderline, and its lower "
         "limit is sensitive to rare influential provider draws in the ratio (see Caveats). " if band_ts_incl0 else ". ")
      + "The direction is consistent with noise in band medians: classical noise in the outcome pulls a correlation toward "
      f"0 but does not bias a regression slope; scripts/62's noise-corrected rank share was {pc(est('r62.corr'))}. Within band, G holding "
      f"selectivity fixed has a gradient of {t('a.bGS_w')} ({vt('a.bGS_w')}); selectivity holding G fixed has "
      f"{t('a.bSG_w')} ({vt('a.bSG_w')}).\n")

    # 2. sex and POLAR4
    w("2. **(b) Sex and POLAR4 are published only as marginal breakdowns. Standardising for them as well lowers survival "
      "by a few points.** The DfE provider file has one characteristic per row (rows by characteristic type: "
      + ", ".join(f"{k} {v:,}" for k, v in st.items())
      + "; at provider × subject level: "
      + ", ".join(f"{k} {v:,}" for k, v in meta["structure"]["prov_subject_counts"].items())
      + "). Sex rows take only the values "
      + " / ".join(k.split(":", 1)[1] for k in meta["structure"]["values"] if k.startswith("sex:"))
      + " and POLAR4 rows only "
      + " / ".join(k.split(":", 1)[1] for k in meta["structure"]["values"] if k.startswith("POLAR4:"))
      + ", so no row crosses two characteristics. The release's only other file, graduate movement between regions, has "
      "no provider × subject earnings. So no "
      "cross-tabulation of sex, POLAR4 and prior attainment exists at provider × subject level, and a joint control cannot "
      "be built from published tables. Each marginal breakdown was standardised like scripts/62's prior-attainment "
      "standardisation, and the three log adjustments were added. Within a provider the pay gap by sex is close to its "
      f"national size (λ = {_num(row('b.lambda_sex')['est'], 2, False)}, IQR {_num(row('b.lambda_sex')['lo'], 2, False)}–"
      f"{_num(row('b.lambda_sex')['hi'], 2, False)}); for POLAR4 it is about half (λ = {_num(row('b.lambda_pol')['est'], 2, False)}), "
      f"as for prior attainment (λ = {_num(row('b.lambda_pa')['est'], 2, False)}). In {int(est('b.degen_sex'))} sex cells "
      f"({int(float(row('b.degen_sex')['lo']))} at YAG5) the published national F and M medians are equal and in "
      f"{int(est('b.degen_pol'))} POLAR4 cells ({int(float(row('b.degen_pol')['lo']))} at YAG5) the released groups' national "
      "medians never differ within a provider, so λ is not identified there and those cells get no standardisation. "
      "On the common sample "
      f"({int(est('b.n_cells'))} cells, {int(row('b.n_cells')['k'])} subjects; a mean of {_num(row('b.n_prov')['lo'], 1, False)} "
      f"providers per common-sample cell, against {_num(est('b.n_prov'), 1, False)} per cell over the "
      f"{int(row('b.n_prov')['n_cells'])} composition-sample cells) the rank coupling goes from "
      f"{e('b.rho_raw')} raw to {e('b.rho_pa_wp')} with prior attainment standardised and {e('b.rho_all_wp')} with prior "
      f"attainment, sex and POLAR4 (shares {tp('b.rshare_pa_wp')} and {tp('b.rshare_all_wp')}). The pay gradient goes from "
      f"{e('b.b_raw')} to {e('b.b_pa_wp')} and {e('b.b_all_wp')} (shares {tp('b.pshare_pa_wp')} and {tp('b.pshare_all_wp')}). "
      f"Pre-specified test, share under all three minus share under prior attainment alone: rank {t('b.drshare_all_wp')} "
      f"({vt('b.drshare_all_wp')}), pay {t('b.dpshare_all_wp')} ({vt('b.dpshare_all_wp')}); both small. Sex alone "
      f"changes almost nothing (rank share {tp('b.rshare_sex_wp')}, pay {tp('b.pshare_sex_wp')}); POLAR4 alone gives "
      f"{tp('b.rshare_pol_wp')} and {tp('b.pshare_pol_wp')}. With national medians, which over-correct, the shares fall to "
      f"{tp('b.rshare_all_nat')} (rank) and {tp('b.pshare_all_nat')} (pay), and the test gives {t('b.drshare_all_nat')} "
      f"({vt('b.drshare_all_nat')}) and {t('b.dpshare_all_nat')} ({vt('b.dpshare_all_nat')}). "
      f"Within one sex the pay gradient is {t('b.sex_bw')}, against {t('b.sex_bm')} for all graduates of the same providers "
      f"(pay share {tp('b.sex_pshare')}). Within one POLAR4 quintile it is {t('b.polar_bw')} against {t('b.polar_bm')} "
      f"({tp('b.polar_pshare')}; rank share {tp('b.polar_rshare')}). If POLAR4 quintile and prior attainment are "
      "correlated within providers, as is likely, adding the marginal adjustments counts what they share twice, so "
      f"{pc(est('b.pshare_all_wp'))} would lean toward over-correction relative to a joint control; the published tables "
      "cannot check this.\n")

    # 3. career slope
    w("3. **(c) The career rise survives error in G and gets steeper after correction.** Model M1 assumes the same "
      "validity v in every cell; the corrected slope is then slope / v, a rescaling that leaves the p-value unchanged. "
      f"All graduates: {t('c.slope_cc_M1_1.00')}/yr uncorrected, {t('c.slope_cc_M1_0.92')} at v = 0.92 and "
      f"{t('c.slope_cc_M1_0.74')} at v = 0.74. Model M2 assumes a constant error variance, so a cell covering a narrower "
      f"range of G has lower validity. It gives {t('c.slope_cc_M2_0.92')} and {t('c.slope_cc_M2_0.74')}. The Gaussian-copula "
      f"version differs from M1 by at most {_num(lat_gap, 4, False)}. Within-provider-standardised earnings: "
      f"{e('c.slope_ccw_M1_1.00')} → {t('c.slope_ccw_M1_0.92')} → {t('c.slope_ccw_M1_0.74')} (M1); M2 at 0.74 "
      f"{t('c.slope_ccw_M2_0.74')}. Within prior-attainment band: {e('c.slope_cband_M1_1.00')} → {t('c.slope_cband_M1_0.92')} "
      f"→ {t('c.slope_cband_M1_0.74')} (M1). Under M2 the band slope at v = 0.92 is {desc('c.slope_cband_M2_0.92')}; at "
      f"v = 0.86 it is {desc('c.slope_cband_M2_0.86')}. At 0.80 its point estimate is {e('c.slope_cband_M2_0.80')} but it has no "
      f"stable CI ({int(nbad(row('c.slope_cband_M2_0.80')))} of {nrep(row('c.slope_cband_M2_0.80'))} replicates "
      "inadmissible), and at 0.74 it is not defined: the band panels span too narrow a range of G for that "
      f"error variance. For all graduates, over v in [0.74, 0.92] and both models, the corrected slope lies "
      f"between {_num(min(sl_all))} and {_num(max(sl_all))} per year. Exploratory, in pay terms: the gradient rises from "
      f"{e('x.cc_b1')} log points per SD at YAG1 to {e('x.cc_b3')} at YAG3 and {e('x.cc_b5')} at YAG5, "
      f"{t('x.cc_bslope')} per year ({t('x.ccw_bslope')} within-provider-standardised).\n")

    # 4. G beyond selectivity
    m2_txt = (f"At {', '.join(f'{v:.2f}' for v in m2_ok)} M2 gives "
              + ", ".join(t(vk('c.pG_w_M2', v)) for v in m2_ok) + ". ") if m2_ok else ""
    w("4. **(c) Among same-band graduates, G's contribution beyond selectivity is not detected at any validity, but it is "
      "not shown to be small at low validity. Exploratory: the ordering of selectivity ahead of G weakens under error in "
      "G.** Primary version: correlations are averaged over cells and subjects, then corrected and partialled. Within band "
      f"the uncorrected partial is {t(vk('c.pG_w_M1', 1.0))}; the mean of per-cell partials (scripts/62's row) is "
      f"{e('r62.within_p_sel')}. Corrected under M1 it is {t(vk('c.pG_w_M1', 0.92))} at v = 0.92 and "
      f"{t(vk('c.pG_w_M1', 0.74))} at v = 0.74 (MDE {mde(vk('c.pG_w_M1', 0.74))}). On the 0.01 grid from 0.70 to 1.00 its "
      f"CI status is {words(gw_stat)} at every point, so there is no tipping point. Per-cell correction: "
      f"{t('c.pc_pcG_92')} at 0.92 (all {int(row('c.pc_pcG_92')['n_adm'])} cells admissible) and {t('c.pc_pcG_74')} at "
      f"0.74. The 0.74 value is a mean over only the {int(row('c.pc_pcG_74')['n_adm'])} of "
      f"{int(row('c.pc_pcG_74')['n_cells'])} cells still admissible, and it is selected. The dropped cells have a higher "
      f"r(G, selectivity) (cell mean {_num(row('x.pc74_rGS')['lo'])} against {e('x.pc74_rGS')} in the kept cells), a higher "
      f"r(G, band earnings) ({_num(row('x.pc74_rGy')['lo'])} against {e('x.pc74_rGy')}) and a higher uncorrected per-cell "
      f"partial (cell mean {_num(row('x.pc74_pobs')['lo'])} against {e('x.pc74_pobs')}; mean over subjects "
      f"{_num(row('x.pc74s_pobs')['lo'])} over the {int(row('x.pc74s_pobs')['k_drop'])} subjects with dropped cells, against "
      f"{e('x.pc74s_pobs')}). On the kept cells the uncorrected per-cell partial is {t('c.pc_pobs_74')} (at 0.92, all "
      f"cells: {t('c.pc_pobs_92')}), so on the same cells the correction at 0.74 moves the mean from "
      f"{e('c.pc_pobs_74')} to {e('c.pc_pcG_74')}. The 0.74 per-cell mean is therefore not comparable with the other "
      "corrections and is left out of the range quoted in the Answer. Gaussian copula at 0.74: "
      f"{t(vk('c.pG_w_lat', 0.74))}. M2 is admissible only for v ≥ {_num(est('c.vmin_within_band_M2'), 3, False)}: band "
      f"cells span a narrow range of G (mean within-cell variance of zG {_num(est('c.pool_varG_s'), 2, False)}, against 1 "
      "over all primary providers), so under a constant error variance a population validity of 0.92 means a within-cell "
      f"validity of {_num(est('c.vcell_band_0.92'), 2, False)}, and 0.86 means {_num(est('c.vcell_band_0.86'), 2, False)}. "
      + m2_txt
      + f"In pay terms (exploratory): {e('c.bGS_w_1.00')} → {t('c.bGS_w_0.92')} → {t('c.bGS_w_0.74')} log points per SD of "
      "true G (M1). At v = 0.74 that upper limit exceeds the within-band gradient in G alone "
      f"({e('a.bw')}) and even the all-graduate one ({e('a.b_raw')}), so at the low end of the validity range the data do "
      "not rule out a sizeable G contribution beyond selectivity; only near v = 0.92 is the upper limit small "
      f"({_num(hi('c.bGS_w_0.92'))}). For every v from 0.74 to 1.00, G's contribution beyond selectivity among same-band "
      f"graduates is not detected, and the data are compatible with a partial up to {_num(ub_w)} (M1, pooled).\n")
    comp_list = "; ".join([f"uncorrected {desc('x.dSG_c_1.00')}"] +
                          [f"at {v:.2f} {t(f'x.dSG_c_{v:.2f}')}" for v in (0.92, 0.86, 0.80)])
    comp_all_incl = all(status(f"x.dSG_c_{v:.2f}") == "includes 0" for v in (0.92, 0.86, 0.80))
    w("   The comparison with selectivity is what moves (all of this paragraph is exploratory: the docstring pre-specified "
      "the partials, not their difference). Within band, selectivity beyond true G falls from "
      f"{t(vk('c.pS_w_M1', 1.0))} uncorrected to {t(vk('c.pS_w_M1', 0.92))} at 0.92; at 0.74 it is "
      f"{desc(vk('c.pS_w_M1', 0.74))}. Uncorrected, the difference (selectivity minus G) is {desc('x.dSG_w_1.00')}. At 0.92 the point estimate barely moves, {e('x.dSG_w_0.92')}, but "
      f"its CI {cw('x.dSG_w_0.92')} ({ci('x.dSG_w_0.92')}); at 0.86 it is {t('x.dSG_w_0.86')}, at 0.80 {t('x.dSG_w_0.80')} "
      f"and at 0.74 {t('x.dSG_w_0.74')}. On the all-graduate composition sample, G beyond selectivity is "
      f"{t(vk('c.pG_c_M1', 1.0))} uncorrected, {t(vk('c.pG_c_M1', 0.92))} at 0.92, {t(vk('c.pG_c_M1', 0.86))} at 0.86 and "
      f"{t(vk('c.pG_c_M1', 0.80))} at 0.80 ({int(nbad(row(vk('c.pG_c_M1', 0.80))))} of {nrep(row(vk('c.pG_c_M1', 0.80)))} "
      "replicates inadmissible). "
      f"Selectivity beyond G is {e(vk('c.pS_c_M1', 1.0))}, {e(vk('c.pS_c_M1', 0.92))}, {e(vk('c.pS_c_M1', 0.86))} and "
      f"{e(vk('c.pS_c_M1', 0.80))}. The point estimates change order between v = 0.92 and 0.86. The difference "
      f"(selectivity minus G) is {comp_list}"
      + ("; the corrected values are not detected. " if comp_all_incl else ". ")
      + f"The two-stage CI of the uncorrected difference is {tsci('x.dSG_c_1.00')}. Under M1 and its "
      f"assumptions, a validity below {_num(est('c.vmin_composition_M1'), 3, False)} is incompatible with the observed pooled "
      "correlations (the corrected correlation matrix is not positive definite). "
      f"G and selectivity correlate {e('c.poolc_rGS')} within cells, so a G with validity 0.74 would be almost collinear "
      f"with selectivity in truth. At v = 0.74 the correction is unstable ({int(nbad(row(vk('c.pG_c_M1', 0.74))))} of "
      f"{nrep(row(vk('c.pG_c_M1', 0.74)))} replicates inadmissible). On the all-graduate medians of the band-cell providers, "
      f"the pooled partial of G beyond selectivity is {t(vk('c.pG_m_M1', 1.0))} uncorrected and {t(vk('c.pG_m_M1', 0.74))} "
      f"at 0.74, with CI status {words(gm_stat)} at the M1 grid values from 0.70 to 1.00. The mean of per-cell partials on "
      f"the same cells (scripts/62's row, item 0) is {t('r62.matched_p_sel')}, whose two-way CI {cw('r62.matched_p_sel')}. "
      + ("So for all-graduate earnings of these providers, whether G adds a detectable amount beyond selectivity depends "
         "on the estimator: averaging correlations before partialling (pooled, the primary version pre-specified for (c)) "
         "or partialling within each cell and averaging (per cell, scripts/62's version). A partial of averaged "
         "correlations is not the average of per-cell partials.\n" if not excl("r62.matched_p_sel")
         else "Both estimators detect it.\n"))

    # 5. reading
    w("5. **Reading against scripts/62.** Two of scripts/62's statements hold as stated there: coupling rises over "
      "careers (and, if G is noisy, more steeply than measured), and most of the rank coupling survives the "
      "prior-attainment control. A third, that among same-band graduates G adds little beyond selectivity, is not "
      "contradicted (no correction detects a contribution) but is not established as small at low validity: at v = 0.74 "
      f"the upper 95% limit of the corrected partial is {_num(hi(vk('c.pG_w_M1', 0.74)))}, and in pay terms "
      f"{_num(hi('c.bGS_w_0.74'))} log points per SD (exploratory), against {_num(hi(vk('c.pG_w_M1', 0.92)))} and "
      f"{_num(hi('c.bGS_w_0.92'))} at v = 0.92. Two statements need rewording. (i) 'the noise-corrected and "
      "within-provider estimates at 85–90%' describe provider orderings (rank shares). In pay terms the within-provider "
      f"standardisation keeps {pc(est('a.pshare_wp'))} of the gradient, {pc(est('b.pshare_all_wp'))} with sex and POLAR4 "
      f"standardised as well (common sample), and {pc(est('a.pshare_band'))} within band. (ii) 'What survives is mostly "
      "institutional selectivity; academia-wide research prestige adds little beyond it'. Exploratory: the evidence for "
      "its first half weakens under plausible error in G. Taking G as error-free, selectivity keeps the larger partial "
      f"among same-band graduates ({verdict('x.dSG_w_1.00').split(' (')[0]}); at v = 0.92 the difference is still "
      f"{e('x.dSG_w_0.92')} but its CI {cw('x.dSG_w_0.92').split(' (')[0]}; at v ≤ 0.86 the order is undetermined in both "
      "samples, and the all-graduate point estimates put G ahead. The second half holds near v = 0.92 but not at the low "
      "end of the range. The US validity figure of 0.74 sits at the edge of what these data allow. Public aggregates "
      "cannot pin down G's validity in the UK; that would need an external UK prestige measure, the role Wapman's "
      "published ranks played for the US check.\n")

    # ------------------------------------------------------------------ pre-specified tests
    w("## Pre-specified tests\n")
    w("Fixed in the script docstring on 2026-09-24 before any result. Verdict rule (docstring): detected if the two-way 95% "
      f"CI excludes 0; otherwise not detected, with the MDE (80% power, 5% two-sided, 2.80 SE). A CI resting on more than {STABLE} "
      f"of {B} inadmissible replicates (5% of the replicates) is flagged unstable. Revision: 'status' is scripts/59's x_status "
      "from the Monte Carlo SE of the CI endpoints (MC SE); a CI with an endpoint within 2 MC SE of 0 ('touches 0') was "
      f"re-estimated with {NX} replicates of each provider draw, and one that still touches 0 is reported as borderline. "
      "Robust SE = scripts/59's IQR version of the two-way SE (subject bootstrap of the per-subject values in place of the "
      "jackknife), shown next to the variance-based SE that the verdict uses.\n")
    w("| test | estimate | two-way 95% CI | SE | robust (IQR) SE | MC SE of endpoints | status | p | MDE | two-stage 95% CI | verdict | k |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, lab in (("a.dshare_wp", "(a) pay share − rank share, within-provider standardisation"),
                   ("a.dshare_nat", "(a) pay share − rank share, national band medians"),
                   ("a.dshare_band", "(a) pay share − rank share, within prior-attainment band"),
                   ("b.drshare_all_wp", "(b) rank share: PA + sex + POLAR4 − PA alone (within-provider)"),
                   ("b.dpshare_all_wp", "(b) pay share: PA + sex + POLAR4 − PA alone (within-provider)"),
                   ("b.drshare_all_nat", "(b) rank share: PA + sex + POLAR4 − PA alone (national medians)"),
                   ("b.dpshare_all_nat", "(b) pay share: PA + sex + POLAR4 − PA alone (national medians)"),
                   (vk("c.pG_w_M1", 0.92), "(c) within-band G partial, v = 0.92 [M1]"),
                   (vk("c.pG_w_M1", 0.74), "(c) within-band G partial, v = 0.74 [M1]"),
                   ("c.slope_cc_M1_0.92", "(c) career slope, all graduates, v = 0.92 [M1]"),
                   ("c.slope_cc_M1_0.74", "(c) career slope, all graduates, v = 0.74 [M1]"),
                   ("c.slope_ccw_M1_0.74", "(c) career slope, within-provider-standardised, v = 0.74 [M1]"),
                   ("c.slope_cband_M1_0.74", "(c) career slope, within band, v = 0.74 [M1]"),
                   ("c.slope_cband_M2_0.92", "(c) career slope, within band, v = 0.92 [M2]"),
                   ("c.slope_cband_M2_0.86", "(c) career slope, within band, v = 0.86 [M2]")):
        r = row(k)
        w(f"| {lab} | {e(k)} | {ci(k)} | {_num(r['se'], 4, False)} | {_num(r['se_r'], 4, False)} | "
          f"{_num(r['mcse'], 4, False)} | {mcs(k)} | {_p(r['p'])} | {mde(k)} | {tsci(k)} | {verdict(k)} | {int(r['k'])} |")
    for q, lab in (("within_band", "within band, partial ρ(true G, band earnings | selectivity)"),
                   ("matched", "all-graduate medians of band-cell providers, same partial"),
                   ("composition", "composition sample, partial ρ(true G, earnings | selectivity)")):
        for model in ("M1", "M2"):
            rv = row(f"c.vrange_{q}_{model}")
            tip = est(f"c.tip_{q}_{model}")
            rng_txt = (f"CI excludes 0 for v {float(rv['lo']):.2f}–{float(rv['hi']):.2f}" if np.isfinite(float(rv["lo"]))
                       else "CI includes 0 at every v")
            tt = (f"; touches 0 after re-estimation at v = "
                  + ", ".join(f"{float(x):.2f}" for x in str(rv["touch_v"]).split()) if int(rv["n_touch"]) > 0 else "")
            w(f"| (c) tipping point, {lab.replace('|', chr(92) + '|')} [{model}] | {('%.2f' % tip) if np.isfinite(tip) else 'none'} "
              f"| — | — | — | — | — | — | — | — | {rng_txt} (stable CI at {int(rv['n_stable'])} of 31 grid points, from "
              f"v = {float(rv['vmin_stable']):.2f}){tt} | — |")
    w("")

    # ------------------------------------------------------------------ key numbers
    w("## Key numbers\n")
    w(f"Estimate = statistic on the observed data. CI = two-way (subject, provider) cluster 95% CI (normal, est ± 1.96 SE; "
      f"V = V_subject + V_provider − V_subject×provider; scripts/59), {B} replicates per provider draw ({NX} for rows marked "
      f"'({NX} reps)'). Status = scripts/59's x_status from the Monte Carlo SE of the endpoints ('touches 0' = an endpoint "
      "within 2 MC SE of 0). Two-stage = scripts/62's percentile CI (subjects, then providers within subject; always "
      f"{B} replicates). p = two-sided normal p from the two-way SE. MDE = 2.80 × SE. k = subjects. Shares are ratios of "
      "subject means. Sections: repro = scripts/62 rows reproduced; a = pay terms; b = sex and POLAR4; c = validity of G; "
      f"x = exploratory. 'no CI' = more than 5% of the replicates inadmissible (EIV matrix not positive definite). SE, "
      "robust SE and MC SE of every row are in the summary table.\n")
    w("| sec | quantity | sample / spec | estimate | 95% CI (two-way) | status | p | MDE | two-stage 95% CI | k |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in R.iterrows():
        k = r["key"]
        q = str(r["quantity"]).replace("|", "\\|")
        sp = str(r["spec"]).replace("|", "\\|")
        kk = "—" if pd.isna(r["k"]) else f"{int(r['k'])}"
        estv = float(r["est"])
        if k.startswith("c.vrange"):
            val = (f"{float(r['lo']):.2f}–{float(r['hi']):.2f}" if np.isfinite(float(r["lo"])) else "none")
            w(f"| {r['section']} | {q} | {sp} | {val} | — | — | — | — | — | — |")
            continue
        if k.startswith("c.tip"):
            w(f"| {r['section']} | {q} | {sp} | {('%.2f' % estv) if np.isfinite(estv) else 'none'} | — | — | — | — | — | — |")
            continue
        if k.startswith("c.vmin"):
            w(f"| {r['section']} | {q} | {sp} | {_num(estv, 3, False)} | — | — | — | — | — | — |")
            continue
        if k in ("b.n_cells", "r62.n"):
            w(f"| {r['section']} | {q} | {sp} | {int(estv)} | — | — | — | — | — | {kk} |")
            continue
        if k.startswith("b.degen"):
            w(f"| {r['section']} | {q} | {sp} | {int(estv)} | at YAG5 {int(float(r['lo']))} | — | — | — | — | — |")
            continue
        if k == "b.n_prov":
            w(f"| {r['section']} | {q} | {sp} | {estv:.1f} | common sample {float(r['lo']):.1f} (earlier figure "
              f"{float(r['hi']):.1f}) | — | — | — | — | — |")
            continue
        if k.startswith("x.pc") and "_" in k and pd.isna(r["se"]):
            w(f"| {r['section']} | {q} ({int(r['n_adm'])} of {int(r['n_cells'])} cells admissible) | {sp} | {_num(estv)} | "
              f"dropped cells {_num(r['lo'])} | — | — | — | — | {kk} |")
            continue
        if str(r.get("ci_kind", "")) == "IQR":
            w(f"| {r['section']} | {q} | {sp} | {_num(estv)} | IQR [{_num(r['lo'])}, {_num(r['hi'])}] | — | — | — | — | {kk} |")
            continue
        if pd.isna(r["se"]) and pd.isna(r["ts_lo"]):
            w(f"| {r['section']} | {q} | {sp} | {_num(estv)} | — | — | — | — | — | {kk} |")
            continue
        extra = ""
        if pd.notna(r.get("n_adm")):
            extra = f" ({int(r['n_adm'])} of {int(r['n_cells'])} cells admissible)"
        w(f"| {r['section']} | {q}{extra} | {sp} | {_num(estv)} | {ci(k)} | {mcs(k) if stable(r) else '—'} | {_p(r['p'])} | "
          f"{mde(k) if stable(r) else '—'} | {tsci(k)} | {kk} |")
    w("")

    # ------------------------------------------------------------------ method
    w("## Method\n")
    w("- **Data and cells (all from scripts/62, imported).** LEO provider × CAH2 subject × cohort × YAG median earnings, "
      f"{meta['n_prim']} primary providers with G (academia-wide UK SpringRank on the GB→GB hiring network). Cells: subject × "
      "cohort at YAG5, cohorts 2013/14–2016/17, ≥ 15 providers; composition sample = providers with every standardisation "
      "defined; band cells = subject × cohort × band (bands 1–5) with a released band and all-graduate median; career "
      "panels = fixed cohorts at YAG 1/3/5 with balanced providers. Subject value = mean over its cells; headline = "
      "unweighted mean over subjects.")
    w("- **(a) Pay terms.** zG = rank of G among the primary providers, standardised to SD 1 over them; zS = rank of the "
      "institution-wide share of graduates with ≥ 360 tariff points among primary providers in the same cohort × YAG, "
      "standardised the same way. Pay gradient = unweighted OLS slope of log median earnings on zG (or zS) across a "
      "cell's providers; joint regressions on zG and zS give each holding the other fixed. Earnings measures: raw log "
      "median; log median minus log expected median from the provider's band mix with national band medians (scripts/62 "
      "'std', over-corrects) or with within-provider band gradients (scripts/62 'std_wp'). Pay share = mean standardised "
      "gradient / mean raw gradient; rank share = the same ratio of Spearman couplings (scripts/62). The difference is "
      "computed on the same replicates (paired). Within band: gradient of log band median vs gradient of log all-graduate "
      "median on the same providers. Raw-score scale (SD of the SpringRank score, SD of the share) as sensitivity.")
    w("- **(b) Sex and POLAR4.** Per subject × cohort × YAG cell, over every HEI releasing ≥ 2 group medians: log group "
      "median = provider FE + λ × log national group median (OLS), exactly scripts/62's within-provider band fit with the "
      "group set as an argument. The script checks that the generalised fit reproduces scripts/62's prior-attainment "
      "gradients and standardised columns to 1e-12. Expected median = Σ_g share_g × exp(λ log national median_g) over the "
      "provider-subject's mix of graduates included in earnings; groups with a value must cover ≥ 80% of the earnings "
      "sample. Cells where λ is not identified (the log national medians of the groups each provider releases never "
      "differ within a provider) get no gradient; an earlier version returned lstsq's minimum-norm value (near 1) there, "
      "found by an independent re-computation and corrected before the final run (counts in Key numbers). "
      "Sex groups F, M; POLAR4 groups quintiles 1–5 and not known. The prior-attainment, sex and POLAR4 "
      "adjustments are subtracted from log earnings together (added in logs). Common sample = composition-sample "
      "providers with all adjustments defined, cells ≥ 15 providers.")
    w("- **(c) Errors in variables.** Validity v = corr(observed G, construct), grid 0.74 (US ORCID-rebuilt academia rank vs "
      f"Wapman's published ranks, scripts/11), 0.80, 0.86, 0.92 (UK split-half reliability, Spearman–Brown, recomputed "
      f"here: {_num(meta['rel']['split_half_sb'], 3, False)}), √reliability = {vc:.3f} (ceiling), and 0.70–1.00 in steps of 0.01 "
      "(Appendix A). Assumptions: error in G independent of earnings and of selectivity (non-differential), selectivity "
      "measured without error, Spearman correlations corrected like Pearson correlations (the Gaussian-copula version, "
      "r = 2 sin(πρ/6), is a sensitivity). M1: every within-cell correlation of G attenuated by v. M2: constant error "
      "variance 1 − v² on the population zG scale, so v_cell² = 1 − (1 − v²) / var_cell(zG), with var_cell the mean "
      "within-cell variance. Partial of true G given selectivity: a = r(G,y)/v, b = r(G,S)/v, c = r(S,y), "
      "(a − bc)/√((1 − b²)(1 − c²)); NaN where the corrected matrix is not positive definite. Primary = pooled "
      "correlations (subject means of cell means, then corrected); secondary = per-cell correction, dropping inadmissible "
      "cells (reported with the uncorrected per-cell partial on the same admissible cells). Pay version: OLS from pooled "
      "covariances with var(zG) replaced by v² var(zG), coefficient per SD of true G. Career slope: slope / v (M1), "
      "slope / v_cell (M2), or each horizon's pooled coupling corrected in the latent metric (copula).")
    w("- **Inference.** Two-way (subject, provider) cluster variance in the Cameron–Gelbach–Miller form, reusing scripts/59's "
      "`twoway` and `jack_var`: V_subject = jackknife over subjects of the statistic; V_provider = variance over replicates "
      "of one provider draw shared by every subject and cell (each replicate its own generator, "
      "`SeedSequence([72, crc32(tag)]).spawn(1000)`); V_subject×provider = variance under scripts/62's independent "
      "provider draw per subject (scripts/62's own streams, so the two-stage CIs reproduce scripts/62). Normal 95% CI; if "
      "V ≤ 0 the larger one-way variance would be used (it never was: `fallback` is False in every row). Replicates whose "
      "corrected matrix is inadmissible are dropped from the variance and counted (`n_bad`).")
    w("- **Monte Carlo error (revision).** Each two-way CI carries scripts/59's Monte Carlo SE of its endpoints (`mcse`, "
      "from the fourth-moment variance of the two replicate variances) and scripts/59's `x_status`. Rows and grid points "
      f"whose CI touches 0 at {B} replicates get {int(meta['next'])} further replicates of both provider draws (shared "
      "draw `SeedSequence([72, crc32('shared_provider_draw_ext')]).spawn(4000)`; independent draw per subject "
      "`SeedSequence([72, crc32('independent_provider_draw_ext|' + subject)]).spawn(4000)`), concatenated with the first "
      f"{B}; the estimate, the jackknife and the two-stage CI are unchanged. This run re-estimated {len(ext_keys)} "
      f"row{'s' if len(ext_keys) != 1 else ''} and {n_ext_grid} grid point{'s' if n_ext_grid != 1 else ''}. The robust SE "
      "(`se_r`) replaces each variance by (IQR/1.349)², with a subject bootstrap of the per-subject values (as many draws "
      "as provider replicates) for the subject part (scripts/59's `tse_r`).")
    w("- **Reproducibility.** Every random draw is made in the parent process before per-subject work is forked, so "
      "results do not depend on the number of workers. The final version was run twice with one worker (the default, "
      "which keeps resident memory within the lane's budget) and gave byte-identical tables and write-up (md5 compared). "
      "The write-up is always built from the saved tables (`--md-only` rebuilds it without recomputation). The LEO file's "
      "md5 is asserted.\n")

    # ------------------------------------------------------------------ caveats
    w("## Caveats\n")
    w("- **Validity inputs are borrowed.** 0.74 is a US figure (ORCID-rebuilt vs published academia ranks); the UK has no "
      "published hiring-network ranking to validate against. 0.92 is a reliability, not a validity: it bounds how well G "
      "tracks the full-network SpringRank, not how well that ranking tracks the prestige employers see. Validity could lie "
      "outside [0.74, 0.92]; Appendix A covers 0.70–1.00.")
    w("- **Non-differential error is an assumption.** If G's error is correlated with selectivity (for instance, small, "
      "less selective providers with few hiring edges get noisier G), the correction direction for the selectivity partial "
      "can differ. Selectivity is treated as error-free; if it is noisy too, both partials are affected.")
    w("- **Correcting Spearman correlations as if they were Pearson correlations** is an approximation; the copula version is close "
      "for the career slope and the within-band partial (Key numbers), but no exact rank-based correction exists.")
    w("- **Inadmissible replicates are dropped.** Near the smallest admissible validity some bootstrap replicates give a "
      "corrected correlation matrix that is not positive definite. They are left out of the variance, which then describes "
      "only the admissible replicates and can understate the uncertainty; CIs resting on more than 5% inadmissible "
      "replicates are not reported, and the counts are in the summary table (`n_bad`). The same selection affects the "
      "per-cell correction at low v (item 4).")
    w("- **Monte Carlo error and heavy tails.** A CI endpoint near 0 can change side under another draw of the same "
      "procedure; that is why every CI carries a Monte Carlo status and the borderline ones were re-estimated with "
      f"{NX} replicates. The shared-provider-draw replicates of ratio statistics can be heavy-tailed (a rare draw that "
      "nearly empties the denominator), which inflates the variance-based SE; the robust (IQR) SE is shown for comparison. "
      "The within-band pay-versus-rank difference is the pre-specified test most exposed to this: its lower limit is "
      "sensitive to rare influential provider draws.")
    w("- **M2 depends on the scale of zG.** Constant error variance on the rank scale is one choice. It makes band cells, "
      "which span a narrow G range, especially sensitive, so M2 is inadmissible for low v there.")
    w("- **Marginal breakdowns only.** Sex and POLAR4 adjustments are estimated separately and added. They double count "
      "what they share with each other and with prior attainment, so the combined adjustment tends to over-correct "
      "relative to a joint control. The within-provider gradients have scripts/62's caveats: they understate an attainment "
      "effect if providers admit lower-band students for strengths the bands do not record. POLAR4 is area-based, and its "
      "'not known' group collects graduates whose home area could not be classified, whatever the reason.")
    w("- **Pay gradients are across-provider gradients of medians.** They weight providers equally, use a rank-based x "
      "scale (per SD of the G rank over the 134 primary providers), and say nothing about individual returns or earnings "
      "tails. Nominal pounds within a cell (no deflation needed for within-cell slopes).")
    w("- **Two-way vs two-stage intervals.** The two-way intervals are the inference standard; they are wider than the "
      "two-stage intervals for levels and partials (the CI of scripts/62's matched all-graduate partial "
      f"{cw('r62.matched_p_sel').split(' (')[0]} under the two-way standard, while its two-stage interval "
      f"{'excluded' if ts_excl('r62.matched_p_sel') else 'included'} 0)."
      + (" For the within-band pay-versus-rank difference the two-stage interval includes 0 while the two-way interval "
         "does not, so that difference is borderline." if band_ts_incl0 and excl("a.dshare_band") else ""))
    w("- **The selectivity-versus-G ordering is exploratory.** No test of the difference between the two partials was "
      "pre-specified; the differences in items 4 and 5 are descriptive checks of scripts/62's wording.")
    w("- **Selectivity is itself a status measure** (demand for places), measured from the same LEO cohorts. 'G adds "
      "little beyond selectivity' is not 'status does not pay'. Descriptive throughout.\n")

    # ------------------------------------------------------------------ revisions
    w("## Revisions\n")
    w("Revised 2026-09-25 after an independent verifier's report (seven points). The pre-specified tests, samples and "
      "estimators are unchanged; every number below is from this run.\n")
    ext_txt = "; ".join(
        [f"`{k}` {_num(R.loc[k, 'est'])}: [{_num(R.loc[k, 'lo_1k'], 4)}, {_num(R.loc[k, 'hi_1k'], 4)}] "
         f"({R.loc[k, 'status_1k']}) → [{_num(R.loc[k, 'lo'], 4)}, {_num(R.loc[k, 'hi'], 4)}] ({R.loc[k, 'status']})"
         for k in ext_keys] +
        [f"grid v = {g.v:.2f} {g.model} {g['sample'].replace('_', ' ')} {_num(g.est)}: [{_num(g.lo_1k, 4)}, "
         f"{_num(g.hi_1k, 4)}] ({g.status_1k}) → [{_num(g.lo, 4)}, {_num(g.hi, 4)}] ({g.status})"
         for _, g in GRID[GRID.n_rep > B].iterrows()])
    w(f"1. **Monte-Carlo-dependent claims.** Two claims had rested on a two-way CI endpoint close to 0 and on "
      "hard-coded assertions of its side. Every CI now carries scripts/59's Monte Carlo SE and x_status; the "
      f"{len(ext_keys)} row{'s' if len(ext_keys) != 1 else ''} and {n_ext_grid} grid point{'s' if n_ext_grid != 1 else ''} "
      f"whose CI touched 0 at {B} replicates were re-estimated with {NX} ({B}-replicate CI → {NX}-replicate CI, status): "
      f"{ext_txt}. The two claims had said that the CIs of the uncorrected all-graduate selectivity-minus-G difference and of "
      "the within-band M2 career slope at v = 0.86 include 0; an independent re-derivation with another seed put both "
      "lower limits above 0, which is what the 'touches 0' status registers. The text words every side-of-0 statement "
      "from the status. The uncorrected all-graduate selectivity-minus-G difference is {desc('x.dSG_c_1.00')}, with two-stage "
      f"CI {tsci('x.dSG_c_1.00')}; the within-band M2 career slope at v = 0.86 is {desc('c.slope_cband_M2_0.86')}.")
    w("2. **The selectivity-ahead-of-G ordering is labelled exploratory** in the Answer and items 4–5 (the docstring "
      "pre-specified the partials, not their difference), with its point estimates: within band "
      f"{e('x.dSG_w_1.00')} uncorrected, {e('x.dSG_w_0.92')} at v = 0.92 (CI {ci('x.dSG_w_0.92')}) and "
      f"{e('x.dSG_w_0.74')} at 0.74. 'Not robust' is replaced by 'the evidence for the ordering weakens under plausible "
      "error in G, and the order is undetermined at v ≤ 0.86'.")
    w("3. **'G adds little beyond selectivity' is no longer stated as established.** It is not detected at any v in "
      f"0.70–1.00, but at v = 0.74 the corrected partial is {t(vk('c.pG_w_M1', 0.74))} (MDE {mde(vk('c.pG_w_M1', 0.74))}) "
      f"and the pay-terms gradient {t('c.bGS_w_0.74')}, whose upper limit exceeds the raw all-graduate G gradient "
      f"({e('a.b_raw')}); at v = 0.92 the pay-terms upper limit is {_num(hi('c.bGS_w_0.92'))}. Item 5 now reads 'not "
      "contradicted, but not established as small at low validity'.")
    w(f"4. **The per-cell correction at v = 0.74 is a selected mean** over {int(row('c.pc_pcG_74')['n_adm'])} of "
      f"{int(row('c.pc_pcG_74')['n_cells'])} cells; the dropped cells have higher r(G, selectivity) and higher uncorrected "
      f"partials (item 4). The uncorrected per-cell partial on the same admissible cells is now reported "
      f"({t('c.pc_pobs_74')}), and the 0.74 per-cell value is left out of the range of corrections in the Answer (now "
      f"{_num(pw_lo)} to {_num(pw_hi)}).")
    w("5. **Robust SE and Monte Carlo SE for ratio tests.** The pre-specified tests table now shows scripts/59's robust "
      "(IQR) SE and the MC SE of the endpoints next to the variance-based SE. Within-band pay-versus-rank difference: SE "
      f"{_num(row('a.dshare_band')['se'], 4, False)}, robust SE {_num(row('a.dshare_band')['se_r'], 4, False)}, MC SE "
      f"{_num(row('a.dshare_band')['mcse'], 4, False)}, status {mcs('a.dshare_band')}; the label 'borderline' is kept "
      "and the Caveats state that its lower limit is sensitive to rare influential provider draws.")
    w(f"6. **Providers per common-sample cell** is now the mean over the {int(est('b.n_cells'))} common-sample cells "
      f"({_num(row('b.n_prov')['lo'], 1, False)}); the earlier figure ({_num(row('b.n_prov')['hi'], 1, False)}) averaged "
      f"over all {int(row('b.n_prov')['n_cells'])} composition cells, including those with fewer than 15 such providers.")
    w("7. **Wording.** 'pays' is replaced by 'has a gradient of' (descriptive gradients of provider medians). The "
      "within-band M2 career slope 'cannot be estimated at 0.80 or below' is replaced by 'has a point estimate "
      f"({e('c.slope_cband_M2_0.80')}) but no stable CI at 0.80 and is undefined at 0.74'.")
    w("")
    w("Earlier revision (before the first complete write-up): λ cells where the fit is not identified get no gradient "
      "(Method, (b)).\n")

    # ------------------------------------------------------------------ appendix A: grid
    w("## Appendix A — errors-in-variables grid (partial ρ of true G with earnings given selectivity)\n")
    w("Pooled correlations; two-way 95% CI; 'n.a. (m)' = not reported, m replicates inadmissible (more than 5%) or the "
      f"point estimate itself inadmissible; '*' = re-estimated with {NX} replicates; '†' = an endpoint still within 2 MC SE "
      "of 0.\n")
    cols = [(s, m) for s in ("within_band", "matched", "composition") for m in ("M1", "M2")]
    w("| v | " + " | ".join(f"{s.replace('_', ' ')} {m}" for s, m in cols) + " |")
    w("|---|" + "---|" * len(cols))
    for v in sorted(GRID.v.unique()):
        cells = []
        for s, m in cols:
            g = GRID[(GRID.v == v) & (GRID["sample"] == s) & (GRID.model == m)].iloc[0]
            if not np.isfinite(g.est) or g.n_bad > STABLE_FRAC * g.n_rep or not np.isfinite(g.lo):
                cells.append(f"n.a. ({int(g.n_bad)})")
            else:
                mk = ("*" if g.n_rep > B else "") + ("†" if g.status == "touches 0" else "")
                cells.append(f"{_num(g.est)} [{_num(g.lo)}, {_num(g.hi)}]{mk}")
        w(f"| {v:.2f} | " + " | ".join(cells) + " |")
    w("")

    # ------------------------------------------------------------------ appendix B: per subject
    w("## Appendix B — per subject\n")
    w("Composition sample, YAG5, mean over cohorts 2013–16: ρ raw / ρ wp = rank coupling raw / within-provider-standardised; "
      "b raw / b wp / b nat = pay gradient (log points per SD of the G rank) raw / within-provider / national medians; "
      "b G|S wp = G holding selectivity fixed, within-provider-standardised; b all (common) = pay gradient standardised "
      "for prior attainment + sex + POLAR4 (common sample). Band: pay gradient within band / all graduates of the same "
      "providers. Career slope = scripts/62's slope of ρ per year; pay slope = slope of the pay gradient per year.\n")
    pc_cols = [("cx_rho_raw", "ρ raw"), ("cx_rho_wp", "ρ wp"), ("cx_b_raw", "b raw"), ("cx_b_wp", "b wp"),
               ("cx_b_nat", "b nat"), ("cx_bGS_wp", "b G\\|S wp"), ("cx_B_b_raw", "b raw (common)"),
               ("cx_B_b_all_wp", "b all (common)"), ("bands_bw", "band b"), ("bands_bm", "matched b"),
               ("cc_slope", "career slope"), ("cc_bslope", "pay slope")]
    w("| subject | " + " | ".join(c[1] for c in pc_cols) + " |")
    w("|---|" + "---|" * len(pc_cols))
    for s, r in per.sort_values("cx_b_raw", ascending=False, na_position="last").iterrows():
        w(f"| {s} | " + " | ".join(_num(r[c], 3 if not c.endswith("bslope") else 4) for c, _ in pc_cols) + " |")
    w("")

    # ------------------------------------------------------------------ provenance
    w("## Provenance\n")
    w(f"- **DfE LEO provider-level graduate outcomes** (tax year 2022-23 release, published 26 June 2025): "
      f"`data/raw/leo/leo_dashboard.zip`, md5 `{meta['leo_md5']}` (asserted equal to the md5 in UK_LEO_RESULT.md; URL in the "
      f"data/raw/SOURCES.md entry, accessed "
      f"2026-06-22), members {', '.join(meta['structure']['members'])}; `{meta['structure']['file']}` has "
      f"{meta['structure']['n_cols']} columns, and its characteristic columns are "
      f"{' and '.join(meta['structure']['char_cols'])}. Not re-downloaded.")
    w("- **Release contents check** (no download): release page "
      "`https://explore-education-statistics.service.gov.uk/find-statistics/graduate-outcomes-leo-provider-level-data` and "
      "data guidance `.../graduate-outcomes-leo-provider-level-data/2022-23/data-guidance`, re-checked 2026-09-25 (release "
      "last updated 4 September 2025). They list two "
      "files: the dashboard underlying data (the file above) and 'Graduate movement between regions' (counts by home, "
      "provider and current region; no provider × subject earnings).")
    w("- **UK hiring network / G**: as UK_LEO_RESULT.md (ORCID-derived edges, Zenodo 10.5281/zenodo.19651302, via the "
      "scripts/40 cache; ROR v2.8), recomputed by scripts/62's functions.")
    w("- **Validity input 0.74**: `outputs/ORCID_OVERLAP_VALIDATION_RESULT.md` (scripts/11), used as a constant.")
    w("- No new data were downloaded; data/raw/SOURCES.md is unchanged.")
    with open(OUT_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")


def md_only():
    """Rewrite UK_ROBUSTNESS_RESULT.md from the tables of the last full run (no recomputation)."""
    R = pd.read_csv(OUT_SUM)
    GRID = pd.read_csv(OUT_GRID)
    per = pd.read_csv(OUT_SUBJ, index_col="subject")
    with open(OUT_META) as fh:
        meta = json.load(fh)
    write_md(R, GRID, per, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)     # 1: whole run about 0.8 GB peak RSS; 2 forked workers ~1.9 GB
    ap.add_argument("--md-only", action="store_true", help="rewrite the write-up from the saved tables")
    a = ap.parse_args()
    if a.md_only:
        md_only()
    else:
        main(a.workers)
