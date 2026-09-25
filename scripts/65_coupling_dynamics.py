"""Coupling dynamics -- how within-field coupling moves across graduation cohorts, calendar time and career time.

QUESTION. rho_f = Spearman, across institutions offering bachelor's field f, of academic prestige and graduates'
median earnings. scripts/52 and scripts/59 showed that rho_f rises with years since graduation on fixed PSEO cohorts
(+0.030/yr on V4.14.1) and bounded the career-time part under a sign assumption. This script models the whole
PSEO V4.14.1 cohort x horizon surface (7 cohort windows at y1, 6 at y5, 4 at y10: 17 cells per field) and asks
(i) what the design identifies about age (career time), period (calendar window of the earnings) and cohort
(graduation window), (ii) whether entry into a weak labour market moves coupling.

PSEO DEFINITIONS (quoted from the PSEO documentation, https://lehd.ces.census.gov/data/pseo_documentation.html,
fetched 2026-09-24; the data file's grad_cohort_years column is 3 for every bachelor's window used here):
  "For the Bachelor's degree level, the graduation cohorts are three-year cohorts, e.g. 2001-2003; 2004-2006;
   2007-2009; 2010-2012; 2013-2015; 2016-2018; 2019-2021."
  "the first year post-graduation is defined as the first calendar year following their graduation year. So for a
   student who graduates in May of 2005, year one begins in January of 2006, year five in January 2010, etc."
  "Earnings are total annual earnings ... converted to 2023 dollars using the CPI-U."
  Hence window c (graduation years c..c+2) has y1 earnings in calendar years c+1..c+3, y5 in c+5..c+7 and y10 in
  c+10..c+12. The first calendar year of the earnings window is t = c + h exactly (h = 1, 5, 10).

PRE-SPECIFICATION (fixed in this docstring before any coupling estimate of this script was computed; everything not
listed here is labelled exploratory in the write-up):
  Samples. (i) FIXED institutions: for each contrast, the institutions of field f present in every cell the
      contrast uses (all 17 cells for the surface / APC model: sample "fixed17"; the three windows of a test:
      "triple" samples). (ii) ALL institutions with released earnings and a PSEO coverage share in the cell, with
      scripts/52's coverage control (partial Spearman given same-horizon coverage = PSEO earners / IPEDS graduates).
      Cells need n >= 15 institutions. F = Wapman field prestige, G = Wapman academia-wide rank (scripts/52).
  Test 1 (Great Recession entry window). D_h = rho(2007,h) - [rho(2004,h) + rho(2010,h)] / 2, Spearman(F, p50),
      on the triple-matched institutions of each field, unweighted mean over fields. The test is D at y1 (window
      2007-09, y1 earnings 2008-10); D at y5 and y10 = persistence. Inference: two-way (field, institution)
      Cameron-Gelbach-Miller cluster variance (scripts/59.twoway), 95% normal CI; institution-clustered cohort-label
      permutation (one ordering of the three windows per institution, shared by all its fields; 1000); minimum
      detectable effect at 80% power, 5% two-sided = (1.960 + 0.842) x SE. Sensitivities: Fisher z scale,
      inverse-variance-weighted mean, Spearman(G, p50), ALL-institution coverage-controlled cells, fixed17 cells.
      COVID: E = rho(2019,y1) - [2 rho(2016,y1) - rho(2013,y1)] (window 2019-21, y1 earnings 2020-22; one-sided
      neighbours, i.e. a linear extrapolation, reported as the weaker test).
  Test 2 (entry conditions at the institution level; aggregated Oreopoulos-von Wachter-Heisz design). Separately
      for y1, y5, y10: y_ifc = within-(field, window) percentile rank of p50 earnings, centred ((rank-0.5)/n - 0.5;
      the rank of log p50 equals the rank of p50) on field x window effects, institution x field effects, U_sc and
      G~_i x U_sc, where U_sc = mean annual unemployment rate (FRED <ST>UR, monthly, seasonally adjusted, averaged
      over the calendar year) of the institution's state over the three graduation years c..c+2, and G~ = the
      institution's centred percentile rank of G among the field's sample institutions. beta_h > 0: the status
      gradient is steeper for windows that entered a weaker state labour market. Sample: institutions present in
      every window of the horizon (fixed institutions; fields with >= 15). Inference: CR1 by state; wild cluster
      restricted (WCR) bootstrap-t p with Webb six-point weights (9999); 95% CI from the unrestricted wild
      bootstrap-t (symmetric); MDE at 80% power = (bootstrap |t| critical value + 0.842) x SE. The F~ x U analogue
      in its own regression; both interactions jointly and the all-institution sample with the coverage share as a
      covariate are sensitivities. Two further sensitivities: (a) G~ x field x window effects added, which absorb
      the national component of U and any common drift of the status gradient across windows, so beta is
      identified only from differences between states within a window; (b) U averaged over the y1 earnings years
      c+1..c+3 instead of the graduation years.
  Verdicts (fixed before any estimate). Test 1 (predicted sign +) and Test 2 (beta_h > 0): "supported" if the 95%
      CI lies above 0, "opposite" if below 0; otherwise "null" if the MDE (80% power, 5% two-sided) is at most the
      benchmark and "underpowered" if it exceeds it. Benchmarks: Test 1 |D| = 0.05 (about three years of the y1
      cross-cohort drift of scripts/52, +0.017/yr, i.e. one 3-year window step); Test 2 |beta| = 0.01 per
      percentage point (a 5-pp difference in entry unemployment moving the status gradient, which is close to
      Spearman(G, earnings), by 0.05). Power against the benchmark is reported with every null.
  Known confound fixed in advance: at y10 the later neighbour of the 2007 window is the 2010 window, whose y10
      earnings are 2020-2022 (COVID years); D_y10 is read with that caveat.
  Data source for U: FRED <ST>UR (BLS LAUS LASST<fips>0000000000003, re-published by FRED). If FRED's CSV endpoint
      cannot be reached, the same BLS series is read from the DBnomics mirror (an earlier vintage); the source used
      is recorded in data/raw/SOURCES.md and in the write-up. This choice depends on availability only.
  APC model. z_fj = mu + u_f + A(h) + K(c) + P(t) + e, meta-regression on Fisher z of Spearman(F, p50) with
      Bonett-Wright sampling variance (1 + r^2/2)/(n - 3) (n - 4 for the coverage partial), field random intercepts
      u_f and residual heterogeneity tau^2 (REML), GLS; two-way (field, institution) cluster variance for every
      reported contrast (variance components held at their REML values; field part by the jackknife over fields).
      t = c + h makes the linear trends of A, P, K unidentified (exact collinearity); they are reported only as the
      bound [theta_h - theta_c, theta_h] from the linear surface z = u_f + theta_h h + theta_c c, valid if the
      period and cohort slopes are both >= 0. Identified: age curvature kappa_A = A(5) - [5/9 A(1) + 4/9 A(10)],
      cohort deviations D_K(c) = K(c) - [K(c-3) + K(c+3)]/2 and period deviations D_P(k) = P(k) - [P(k-3)+P(k+3)]/2,
      under the restriction that the period effect is piecewise linear between the 3-year knots of the y1 earnings
      windows (2002, 2005, ..., 2020) -- the y5 windows start one year after a knot and are interpolated; the 2021
      window is extrapolated from the 2017-2020 segment. A free effect per earnings window is not identified in
      this design (checked by the script). M_AC (no period curvature) and M_AP (no cohort curvature) are
      sensitivities. The model-free surface curvatures (cohort-axis deviations at each horizon, age-axis curvature
      within each window) are reported as means over fields with two-way CIs.
  Exploratory (labelled so): career slope by window (2007 vs the other windows); y1->y5 segment by window; latest
      windows at y5 and y10 against the extrapolation of their two predecessors (COVID-era earnings).

REVISION NOTE (added after a first development run had printed estimates; nothing above was changed). Two
  exploratory, post hoc additions, labelled so in the write-up, and no change to Tests 1-2, their samples, weights,
  benchmarks or verdict rule: (1) placebo windows -- the Test 1 statistic D (triple-matched institutions, weights
  -1/2, 1, -1/2) at every other interior window and horizon, to show whether the Great Recession window's deviation
  is unusual among windows; these cells get their own institution draws (tags s65:pl:*) so the draws behind the
  pre-specified statistics are unchanged. The identity E = -2 x D(2016-18 window, y1) on the COVID triple is
  printed. (2) Test 2 with a G~ x linear window trend added (between the primary spec and the pre-specified G~ x
  field x window sensitivity). The development run read U from DBnomics because FRED timed out; the final run reads
  the FRED files (data/raw/laus_state_ur/<ST>UR.csv, downloaded 2026-09-24), as pre-specified.

REVISION NOTE 2 (added after an independent verification of the first full run; nothing above was changed; every
  addition below is post hoc and labelled so in the write-up). (1) Identification: none of the age, cohort or period
  curvature contrasts is estimable with a free effect per earnings window (printed for every contrast); all of them
  rest on the piecewise-linear period restriction. Restriction-free quantities are added: the cohort combinations
  that are estimable with free period effects and the lack-of-fit Wald tests of the 17 cell means (a plane in (h, c);
  the testable implications of each restricted period profile), from a saturated cell-means GLS (M_free). (2)
  Period-profile sensitivities: knots at the y5 earnings windows (2003, 2006, ..., 2021) and a step function (each
  y5 earnings window assigned to the y1-grid window that shares two of its three years). (3) A joint two-way Wald
  test of all curvature contrasts of a model (covariance = jackknife over fields + covariance of the shared-draw
  replicates - covariance of the independent-draw replicates); the curvature contrasts carry no verdict (descriptive).
  (4) Test 2: a 95% CI obtained by inverting the WCR bootstrap test (same Webb draws, closed form in the null value)
  is added; the pre-specified unrestricted bootstrap-t CI remains the basis of the verdict and the rows where the two
  disagree are listed. (5) Test 2 all-institution sensitivity: the first run used the within-(field, window) rank of
  the coverage share, not the share as pre-specified; the share is now used (pre-specified) and the rank version is
  kept, labelled a deviation (same draws as in the first run). (6) Identities of the exploratory latest-window
  extrapolations with reported second differences are checked and printed. (7) LEHD frame: Massachusetts UI records
  enter LEHD in 2010 (PSEO documentation); caveat, the linear surface on cells whose earnings years are all >= 2010,
  and the share of employed graduates working in New England by window (PSEO Flows) are added. (8) Test 1: D per
  percentage point of the entry-year unemployment contrast and the unemployment contrasts of the placebo windows.
  The md5 of the PRE-SPECIFICATION block is printed in the write-up; the timing claims above are self-reported (the
  script was not committed before the first estimating run).

REVISION NOTE 3 (second pass after REVISION NOTE 2; nothing above was changed; no estimate changes, only checks,
  wording and provenance). (1) The model-free age-axis curvature within a window is shown to equal kappa_A plus a
  period curvature over its earnings years c+1, c+5, c+10, and every estimable function that contains A(5) carries the
  same total weight on the period effects of the y5 earnings windows (design checks); the write-up lists the
  restriction-free quantities accordingly. (2) The statement that Massachusetts institutions appear only from the
  2010-12 window and in none of the fixed samples spanning 2010 is now computed and printed. (3) The Caveats no
  longer say the script is untracked in git (it was committed after the estimating runs of REVISION NOTE 2).
  (4) The md5 of the PSEO Flows file read here is printed (its download record is in FLOWS_PLACEMENT_RESULT.md,
  scripts/61); a provenance subsection for it was appended to data/raw/SOURCES.md and the write-up checks that its
  md5 is recorded there.
  (5) The agreement of every WCR p with its WCR-inverted CI (p > 0.05 exactly when the CI contains 0) and the
  contiguity of the accepted sets are counted and printed. (6) The Method's number of lack-of-fit contrasts of the
  plane is computed, not typed; the Method names the restriction-free quantities, and the key-numbers rows of the
  M_APC fits carry the period restriction in their sample label. (7) The period-only combinations that are estimable
  with free period effects (DP2005 - DP2014, DP2008 - DP2017, sums of three adjacent period deviations) are named
  and checked, and so is the rank of the listed cohort combinations.

Descriptive and not causal. Seeded; outputs byte-identical on re-run (the FRED files are read from data/raw once
downloaded; their md5s are printed in the write-up).
Run: python scripts/65_coupling_dynamics.py
Outputs: data/interim/coupling_dynamics_panel.csv, data/interim/coupling_dynamics_results.csv,
         data/interim/laus_state_ur_annual.csv, outputs/figures/coupling_dynamics.png,
         COUPLING_DYNAMICS_RESULT.md (repo root, local only).
"""
from __future__ import annotations

import sys
import zlib
import hashlib
import importlib.util
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, norm, t as tdist, chi2, f as fdist
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_ar import load_ar_wapman


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEED = 65
s59 = _load("s59", "59_pseo_refresh.py")      # two-way variance, weighted ranks, release plumbing (not edited)
s52 = s59.s52                                # fixed-cohort loaders and helpers (not edited)
s59.SEED = SEED                              # s59.rng_for / draw_valid streams of THIS script use seed 65
FIELDS66, LAB = s52.FIELDS66, s52.LAB
HZ = ["y1", "y5", "y10"]
HY = {"y1": 1, "y5": 5, "y10": 10}
W_SLOPE = s52.W_SLOPE
wrank, wcorr, twoway, fisher, jack_var, draw_valid, PERMS3 = (s59.wrank, s59.wcorr, s59.twoway, s59.fisher,
                                                              s59.jack_var, s59.draw_valid, s59.PERMS3)
corr_rows, rk, perm_p = s52.corr_rows, s52.rk, s52.perm_p
Z975 = s59.Z975
Z80 = 0.8416212335729143
MDE_K = Z975 + Z80
NBOOT = s59.NBOOT          # 1000 institution draws per kind (shared across fields; independent per field)
assert NBOOT == 1000
NPERM = 1000               # institution-clustered cohort-label permutations
NWILD = 9999               # wild cluster bootstrap replicates
WCHUNK = 250
NMIN = 15
CO = {"y1": ["2001", "2004", "2007", "2010", "2013", "2016", "2019"],
      "y5": ["2001", "2004", "2007", "2010", "2013", "2016"],
      "y10": ["2001", "2004", "2007", "2010"]}
CELLS17 = [(c, h) for h in HZ for c in CO[h]]
WINDOWS = CO["y1"]
KNOTS = [2002, 2005, 2008, 2011, 2014, 2017, 2020]
KNOTS_Y5 = [2003, 2006, 2009, 2012, 2015, 2018, 2021]     # REVISION NOTE 2: knots on the y5 earnings grid
GR, COVID = "2007", "2019"
TRIPLES = {"gr_y1": ("y1", ["2004", "2007", "2010"]), "gr_y5": ("y5", ["2004", "2007", "2010"]),
           "gr_y10": ("y10", ["2004", "2007", "2010"]), "covid_y1": ("y1", ["2013", "2016", "2019"]),
           "covid_y5": ("y5", ["2010", "2013", "2016"])}
W_MID = np.array([-0.5, 1.0, -0.5])        # value minus the mean of its two neighbours
W_EXT = np.array([1.0, -2.0, 1.0])         # last value minus linear extrapolation of the two before it
PSEO_DOC = "https://lehd.ces.census.gov/data/pseo_documentation.html"
FRED_DIR = ROOT / "data" / "raw" / "laus_state_ur"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
FRED_DATE = "2026-09-24"
STATES51 = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS",
            "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC",
            "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
S59_SLOPE_ALL = 0.030382    # scripts/59, V4.14.1 all-field fixed-cohort slope (k=40); reproduced below
OUT_PANEL = ROOT / "data" / "interim" / "coupling_dynamics_panel.csv"
OUT_RES = ROOT / "data" / "interim" / "coupling_dynamics_results.csv"
OUT_UR = ROOT / "data" / "interim" / "laus_state_ur_annual.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "coupling_dynamics.png"
OUT_MD = ROOT / "COUPLING_DYNAMICS_RESULT.md"


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def grad_years(c) -> str:
    c = int(c)
    return f"{c}-{c + 2}"


def earn_years(c, h) -> str:
    t = int(c) + HY[h]
    return f"{t}-{t + 2}"


RES: list[dict] = []


def rec(section, stat, est, lo=np.nan, hi=np.nan, se=np.nan, z=np.nan, p=np.nan, k=np.nan, n=np.nan, sample="",
        horizon="", note=""):
    RES.append(dict(section=section, stat=stat, sample=sample, horizon=horizon, estimate=est, ci_lo=lo, ci_hi=hi,
                    se=se, z=z, p_value=p, k=k, n=n, note=note))


def stage(msg):
    print(f"[65] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------
# state unemployment (FRED <ST>UR = BLS LAUS, monthly, seasonally adjusted)
# ---------------------------------------------------------------------------------------------
FIPS = {"AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11",
        "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
        "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30",
        "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
        "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
        "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56"}
DBN_URL = "https://api.db.nomics.world/v22/series/BLS/la/{sid}?observations=1&format=json"


def bls_id(st: str) -> str:
    return f"LASST{FIPS[st]}0000000000003"


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) "
                                                              "Gecko/20100101 Firefox/128.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_ur() -> tuple:
    """Monthly seasonally adjusted state unemployment rates (BLS LAUS). FRED <ST>UR files are used when all 51 are
    present; otherwise the script tries to download them (all or none); if FRED cannot be reached, the same BLS
    series are read from the DBnomics mirror. Files are read from data/raw/laus_state_ur/ once present."""
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    fp = {st: FRED_DIR / f"{st}UR.csv" for st in STATES51}
    dp = {st: FRED_DIR / f"dbnomics_{bls_id(st)}.json" for st in STATES51}
    if not all(p.exists() for p in fp.values()) and not all(p.exists() for p in dp.values()):
        try:
            got = {st: _get(FRED_URL.format(sid=f"{st}UR")) for st in STATES51}
            for st, b in got.items():
                fp[st].write_bytes(b)
        except Exception as exc:                                        # noqa: BLE001 (network)
            stage(f"FRED unreachable ({type(exc).__name__}); reading the BLS series from DBnomics")
            for st in STATES51:
                if not dp[st].exists():
                    dp[st].write_bytes(_get(DBN_URL.format(sid=bls_id(st))))
    source = "FRED" if all(p.exists() for p in fp.values()) else "DBnomics"
    rows, monthly = [], []
    for st in STATES51:
        if source == "FRED":
            p = fp[st]
            d = pd.read_csv(p)
            assert list(d.columns) == ["observation_date", f"{st}UR"], (st, d.columns)
            m = pd.DataFrame(dict(state=st, month=d.observation_date.str[:7],
                                  ur=pd.to_numeric(d[f"{st}UR"], errors="coerce")))
            ident = f"{st}UR"
        else:
            import json
            p = dp[st]
            doc = json.loads(p.read_text())["series"]["docs"][0]
            assert doc["series_code"] == bls_id(st) and doc["dimensions"]["seasonal"] == "S", st
            assert doc["dimensions"]["measure"] == "03", st
            m = pd.DataFrame(dict(state=st, month=doc["period"], ur=pd.to_numeric(pd.Series(doc["value"]),
                                                                                  errors="coerce")))
            ident = bls_id(st)
        monthly.append(m)
        rows.append(dict(state=st, series=ident, file=p.name, md5=md5(p), bytes=p.stat().st_size,
                         first=str(m.month.iloc[0]), last=str(m.month.iloc[-1]), n_obs=len(m)))
    return source, pd.DataFrame(rows), pd.concat(monthly, ignore_index=True)


def state_ur_annual(monthly: pd.DataFrame) -> pd.DataFrame:
    """Calendar-year mean of the 12 monthly seasonally adjusted rates (years with all 12 months only)."""
    d = monthly.dropna(subset=["ur"]).assign(year=lambda x: x.month.str[:4].astype(int))
    g = d.groupby(["state", "year"]).ur.agg(["mean", "size"]).reset_index()
    g = g[g["size"] == 12].rename(columns={"mean": "ur", "size": "n_months"})
    return g.sort_values(["state", "year"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def load_long() -> pd.DataFrame:
    """One row per field x institution x cohort window x horizon: p50 earnings, coverage share, F, G, state."""
    ar = load_ar_wapman(fields=FIELDS66)
    gen = s52._s28.load_generic()
    gen = gen.assign(G=-gen["g_rank"].astype(float))[["inst_key", "G"]]
    s59.set_release("new")
    ins = pd.read_csv(s59.NEW_INST, dtype=str)
    ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
    st_of = dict(zip(ins.institution, ins.institution_state))
    parts = []
    for h in HZ:
        er = s52.load_fixed(h, CO[h])
        x = er.merge(ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"}),
                     on=["field", "inst_key"], how="inner").merge(gen, on="inst_key", how="left")
        x["horizon"] = h
        x["state"] = x["institution_id"].map(st_of)
        parts.append(x[["field", "inst_key", "institution_id", "grad_cohort", "horizon", "earnings", "coverage",
                        "P", "G", "state"]])
    L = pd.concat(parts, ignore_index=True)
    assert L.G.notna().all() and L.state.notna().all()
    assert (L.groupby("inst_key").state.nunique() == 1).all()
    return L.sort_values(["field", "horizon", "grad_cohort", "inst_key"]).reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# samples and cells
# ---------------------------------------------------------------------------------------------
CELL: list[dict] = []      # cell j: sample, field, cohort, horizon, df
J: dict = {}               # (sample, field, cohort, horizon) -> j


def add_cell(sample, fld, c, h, df):
    J[(sample, fld, c, h)] = len(CELL)
    CELL.append(dict(sample=sample, field=fld, cohort=c, horizon=h, df=df.sort_values("inst_key").reset_index(drop=True)))


def build_samples(L: pd.DataFrame):
    by = {k: g for k, g in L.groupby(["field", "grad_cohort", "horizon"], sort=True)}
    fields = sorted(L.field.unique())

    def insts(f, c, h):
        g = by.get((f, c, h))
        return set() if g is None else set(g.inst_key)

    def matched(sample, f, cells):
        common = set.intersection(*[insts(f, c, h) for c, h in cells])
        if len(common) >= NMIN:
            for c, h in cells:
                g = by[(f, c, h)]
                add_cell(sample, f, c, h, g[g.inst_key.isin(common)])

    for f in fields:
        matched("fixed17", f, CELLS17)
        for name, (h, cs) in TRIPLES.items():
            matched(name, f, [(c, h) for c in cs])
        for c in ["2001", "2004", "2007", "2010"]:                         # scripts/52 fixed-cohort panel
            matched("career", f, [(c, h) for h in HZ])
        for c in CO["y5"]:                                                 # y1 -> y5 segment by window
            matched("seg15", f, [(c, "y1"), (c, "y5")])
        for c, h in CELLS17:                                               # all institutions, coverage control
            g = by.get((f, c, h))
            if g is None:
                continue
            g = g[np.isfinite(g.coverage.astype(float))]
            if len(g) >= NMIN:
                add_cell("all", f, c, h, g)


def cell_eval(P, G, E, C, W):
    rP, rG, rE = wrank(P, W), wrank(G, W), wrank(E, W)
    rhoF, rhoG = wcorr(rP, rE, W), wcorr(rG, rE, W)
    out = dict(rhoF=rhoF, rhoG=rhoG)
    if C is not None:
        rC = wrank(C, W)
        a, b, g = wcorr(rP, rC, W), wcorr(rE, rC, W), wcorr(rG, rC, W)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["parF"] = (rhoF - a * b) / np.sqrt((1 - a ** 2) * (1 - b ** 2))
            out["parG"] = (rhoG - g * b) / np.sqrt((1 - g ** 2) * (1 - b ** 2))
    return out


OBS, SB, IB = [], [], []     # per cell: observed dict, shared-draw replicates, independent-draw replicates
ARR = []                     # per cell: (P, G, E, C, ix)
UNIV: list = []


def prepare_arrays():
    global UNIV
    UNIV = sorted(set().union(*[set(c["df"].inst_key) for c in CELL]))
    iid = {k: i for i, k in enumerate(UNIV)}
    for c in CELL:
        g = c["df"]
        C = g.coverage.to_numpy(float) if c["sample"] == "all" else None
        ARR.append((g.P.to_numpy(float), g.G.to_numpy(float), g.earnings.to_numpy(float), C,
                    np.array([iid[k] for k in g.inst_key])))


def ev(W, js=None):
    out = []
    for j in (range(len(ARR)) if js is None else js):
        P, G, E, C, ix = ARR[j]
        out.append(cell_eval(P, G, E, C, W[:, ix]))
    return out


def observed_and_boot():
    NI = len(UNIV)
    obs = ev(np.ones((1, NI)))
    for j, o in enumerate(obs):
        OBS.append({k: float(v[0]) for k, v in o.items()})
    # spot-check against scipy (unit weights = plain Spearman / partial Spearman)
    for j in range(0, len(CELL), max(1, len(CELL) // 25)):
        P, G, E, C, _ = ARR[j]
        assert np.isclose(OBS[j]["rhoF"], spearmanr(P, E)[0], atol=1e-10)
        assert np.isclose(OBS[j]["rhoG"], spearmanr(G, E)[0], atol=1e-10)
    stage(f"shared institution draw over {NI} institutions, {len(CELL)} cells")
    _, res, red_s = draw_valid(NI, "s65:shared", ev)
    SB.extend(res)
    stage("independent institution draw per field")
    by_f = {}
    for j, c in enumerate(CELL):
        by_f.setdefault(c["field"], []).append(j)
    tmp = [None] * len(CELL)
    red_i = 0
    for f in sorted(by_f):
        js = by_f[f]
        _, rf_, rd = draw_valid(NI, f"s65:ind:{f}", lambda Wf, js=js: ev(Wf, js))
        red_i += rd
        for j, r_ in zip(js, rf_):
            tmp[j] = r_
    IB.extend(tmp)
    return dict(NI=NI, redrawn_shared=red_s, redrawn_ind=red_i, n_cells=len(CELL))


# exploratory placebo windows (REVISION NOTE): triples for every interior window not already a Test 1 sample
PLACEBO = {"pl_y1_2004": ("y1", ["2001", "2004", "2007"]), "pl_y1_2010": ("y1", ["2007", "2010", "2013"]),
           "pl_y1_2013": ("y1", ["2010", "2013", "2016"]), "pl_y5_2004": ("y5", ["2001", "2004", "2007"]),
           "pl_y5_2010": ("y5", ["2007", "2010", "2013"]), "pl_y10_2004": ("y10", ["2001", "2004", "2007"])}
# interior window -> sample holding its triple (the 2016 y1 and 2013 y5 triples are the COVID samples)
PL_OF = {("y1", "2004"): "pl_y1_2004", ("y1", "2007"): "gr_y1", ("y1", "2010"): "pl_y1_2010",
         ("y1", "2013"): "pl_y1_2013", ("y1", "2016"): "covid_y1", ("y5", "2004"): "pl_y5_2004",
         ("y5", "2007"): "gr_y5", ("y5", "2010"): "pl_y5_2010", ("y5", "2013"): "covid_y5",
         ("y10", "2004"): "pl_y10_2004", ("y10", "2007"): "gr_y10"}


def placebo_pass(L: pd.DataFrame) -> dict:
    """Append the placebo triples after the main draws and give them their own institution universe and draws, so
    the main (pre-specified) draws are untouched. Statistics must not mix placebo and main cells."""
    by = {k: g for k, g in L.groupby(["field", "grad_cohort", "horizon"], sort=True)}
    j0 = len(CELL)
    for f in sorted(L.field.unique()):
        for name, (h, cs) in PLACEBO.items():
            got = [by.get((f, c, h)) for c in cs]
            if any(g is None for g in got):
                continue
            common = set.intersection(*[set(g.inst_key) for g in got])
            if len(common) >= NMIN:
                for c, g in zip(cs, got):
                    add_cell(name, f, c, h, g[g.inst_key.isin(common)])
    js = list(range(j0, len(CELL)))
    univ = sorted(set().union(*[set(CELL[j]["df"].inst_key) for j in js]))
    iid = {k: i for i, k in enumerate(univ)}
    for j in js:
        g = CELL[j]["df"]
        ARR.append((g.P.to_numpy(float), g.G.to_numpy(float), g.earnings.to_numpy(float), None,
                    np.array([iid[k] for k in g.inst_key])))
    NI = len(univ)
    obs = ev(np.ones((1, NI)), js)
    for o in obs:
        OBS.append({k: float(v[0]) for k, v in o.items()})
    _, res, red_s = draw_valid(NI, "s65:pl:shared", lambda W: ev(W, js))
    SB.extend(res)
    by_f = {}
    for j in js:
        by_f.setdefault(CELL[j]["field"], []).append(j)
    tmp = {}
    red_i = 0
    for f in sorted(by_f):
        jf = by_f[f]
        _, rf_, rd = draw_valid(NI, f"s65:pl:ind:{f}", lambda Wf, jf=jf: ev(Wf, jf))
        red_i += rd
        tmp.update(dict(zip(jf, rf_)))
    IB.extend(tmp[j] for j in js)
    assert len(OBS) == len(SB) == len(IB) == len(ARR) == len(CELL)
    return dict(NI=NI, n_cells=len(js), redrawn_shared=red_s, redrawn_ind=red_i)


def bw_var(r, n, k=0):
    """Bonett-Wright (2000) sampling variance of Fisher z of a Spearman correlation (k partialled variables)."""
    return (1 + r ** 2 / 2) / (n - 3 - k)


# ---------------------------------------------------------------------------------------------
# statistics that are linear combinations of cells within field, averaged over fields (two-way inference)
# ---------------------------------------------------------------------------------------------
def fields_with(sample, cells):
    fs = sorted({c["field"] for c in CELL if c["sample"] == sample})
    return [f for f in fs if all((sample, f, c, h) in J for c, h in cells)]


def lin_stat(sample, terms, stat="rhoF", tf=None, fields=None, weights=None, tag=""):
    """terms: [((c, h), w)]; per-field value sum_w w * tf(stat of cell); unweighted (or fixed-weight) mean over
    fields. Returns the scripts/59 two-way dict plus est, k, mde, p (normal, from the two-way z)."""
    cells = [ch for ch, _ in terms]
    fs = fields_with(sample, cells) if fields is None else fields
    f_ = (lambda x: x) if tf is None else tf

    def val(src, f):
        return sum(w * f_(np.asarray(src[J[(sample, f, c, h)]][stat])) for (c, h), w in terms)

    o = np.array([float(val(OBS, f)) for f in fs])
    sb = np.stack([val(SB, f) for f in fs])
    ib = np.stack([val(IB, f) for f in fs])
    K = len(fs)
    if weights is None:
        est, cb, ibm = o.mean(), sb.mean(0), ib.mean(0)
        vf = o.var(ddof=1) / K
        fi = rng_for("fb:" + tag).integers(K, size=(NBOOT, K))
        fb = o[fi].mean(1)
    else:
        w = np.array([weights[f] for f in fs])
        est, cb, ibm = (w @ o) / w.sum(), (w @ sb) / w.sum(), (w @ ib) / w.sum()
        vf = jack_var(lambda a, b: (a * b).sum() / b.sum(), o, w)
        fi = rng_for("fb:" + tag).integers(K, size=(NBOOT, K))
        fb = (o[fi] * w[fi]).sum(1) / w[fi].sum(1)
    r = twoway(float(est), vf, cb, ibm, fb)
    r.update(est=float(est), k=K, mde=MDE_K * r["tse"], p=float(2 * norm.sf(abs(r["tz"]))), fields=fs, per_field=o)
    return r


def perm_test(sample, h, cs, w, stat="rhoF", tag=""):
    """Institution-clustered cohort-label permutation: every institution's earnings ranks in the three windows are
    permuted by one of the 6 orderings, shared by all fields in which it appears. Null: the three windows are
    exchangeable within institution."""
    fs = fields_with(sample, [(c, h) for c in cs])
    S = rng_for("perm:" + tag).integers(6, size=(NPERM, len(UNIV)))
    obs, null = [], []
    for f in fs:
        js = [J[(sample, f, c, h)] for c in cs]
        keys = [tuple(CELL[j]["df"].inst_key) for j in js]
        assert keys[0] == keys[1] == keys[2]
        P, G, _, _, ix = ARR[js[0]]
        X = P if stat == "rhoF" else G
        E = np.stack([ARR[j][2] for j in js], -1)
        n = len(X)
        Rk = np.stack([rankdata(E[:, k]) for k in range(3)], -1)
        Rp = Rk[np.arange(n)[None, :, None], PERMS3[S[:, ix]]]
        rX = rankdata(X)[None, :]
        null.append(np.stack([corr_rows(rX, rk(Rp[:, :, k])) for k in range(3)], -1) @ w)
        obs.append(np.array([OBS[j][stat] for j in js]) @ w)
    null = np.mean(null, 0)
    o = float(np.mean(obs))
    return perm_p(null, o), o


# ---------------------------------------------------------------------------------------------
# APC meta-regression
# ---------------------------------------------------------------------------------------------
def pw_weights(t: int, knots=None) -> np.ndarray:
    """Period weights on the knots (default KNOTS): a knot -> one-hot; between two knots -> linear interpolation;
    beyond the last knot (or before the first) -> linear extrapolation from the last (first) segment."""
    knots = KNOTS if knots is None else knots
    w = np.zeros(len(knots))
    if t in knots:
        w[knots.index(t)] = 1.0
        return w
    if t < knots[0]:                                      # only reached with KNOTS_Y5 (t = 2002 < 2003)
        f = (knots[0] - t) / 3.0
        w[0], w[1] = 1 + f, -f
        return w
    j = max(i for i, k in enumerate(knots) if k < t)
    f = (t - knots[j]) / 3.0
    if j + 1 < len(knots):
        w[j], w[j + 1] = 1 - f, f
    else:
        w[j], w[j - 1] = 1 + f, -f
    return w


def step_weights(t: int, knots=None) -> np.ndarray:
    """Step-function period profile (REVISION NOTE 2): one-hot on the knot window [k, k+2] that shares the most
    calendar years with the earnings window [t, t+2] (ties to the earlier knot; the y5 windows share two years with
    the y1-grid window that starts one year earlier)."""
    knots = KNOTS if knots is None else knots
    ov = [len(set(range(t, t + 3)) & set(range(k, k + 3))) for k in knots]
    w = np.zeros(len(knots))
    w[int(np.argmax(ov))] = 1.0
    return w


AGES = [1, 5, 10]
KCOH = [int(c) for c in WINDOWS]


def full_row(c: int, h: int, knots=None, step=False) -> np.ndarray:
    a = [float(h == x) for x in AGES]
    k = [float(c == x) for x in KCOH]
    pw = step_weights(c + h, knots) if step else pw_weights(c + h, knots)
    return np.array([1.0] + a + k + list(pw))


NFULL = 1 + len(AGES) + len(KCOH) + len(KNOTS)
assert len(KNOTS_Y5) == len(KNOTS)


def ix_A(a):
    return 1 + AGES.index(a)


def ix_K(c):
    return 1 + len(AGES) + KCOH.index(c)


def ix_P(k, knots=None):
    return 1 + len(AGES) + len(KCOH) + (KNOTS if knots is None else knots).index(k)


def contrast_vec(kind, x=None, knots=None) -> np.ndarray:
    v = np.zeros(NFULL)
    if kind == "kA":
        v[ix_A(5)], v[ix_A(1)], v[ix_A(10)] = 1.0, -5 / 9, -4 / 9
    elif kind == "kAs":       # step profile: y5 earnings sit in the block one year earlier -> age metric (1, 4, 10)
        v[ix_A(5)], v[ix_A(1)], v[ix_A(10)] = 1.0, -2 / 3, -1 / 3
    elif kind == "DK":
        v[ix_K(x)], v[ix_K(x - 3)], v[ix_K(x + 3)] = 1.0, -0.5, -0.5
    elif kind == "DP":
        v[ix_P(x, knots)], v[ix_P(x - 3, knots)], v[ix_P(x + 3, knots)] = 1.0, -0.5, -0.5
    return v


MODELS = {
    # kept columns of the full design (the rest are normalisations); M_APC also pins the first two period knots to 0,
    # which fixes the unidentified linear trend (identified contrasts do not depend on this choice). Positions are the
    # same for every knot set (7 knots).
    "M_APC": [0, ix_A(5), ix_A(10)] + [ix_K(c) for c in KCOH[1:]] + [ix_P(k) for k in KNOTS[2:]],
    "M_AC": [0, ix_A(5), ix_A(10)] + [ix_K(c) for c in KCOH[1:]],
    "M_AP": [0, ix_A(5), ix_A(10)] + [ix_P(k) for k in KNOTS[1:]],
}


def contrasts_for(model, knots=None, step=False):
    """Curvature contrasts of a model. Under the step profile the age curvature on (1, 5, 10) is not estimable
    (the y5 earnings windows are assigned to the block one year earlier, so the age metric is effectively 1, 4, 10);
    the age curvature on that metric (kAs) is reported instead."""
    knots = KNOTS if knots is None else knots
    ka = ("kAs", None) if step else ("kA", None)
    return {"M_APC": [ka] + [("DK", c) for c in KCOH[1:-1]] + [("DP", k) for k in knots[1:-1]],
            "M_AC": [ka] + [("DK", c) for c in KCOH[1:-1]],
            "M_AP": [ka] + [("DP", k) for k in knots[1:-1]]}[model]


CONTRASTS = {m: contrasts_for(m) for m in MODELS}
# restriction-free cohort combinations (estimable with a free effect per earnings window; REVISION NOTE 2)
DKCOMBO = {"DK2004-DK2013": {2004: 1.0, 2013: -1.0}, "DK2007-DK2016": {2007: 1.0, 2016: -1.0},
           "DK2004+DK2007+DK2010": {2004: 1.0, 2007: 1.0, 2010: 1.0},
           "DK2007+DK2010+DK2013": {2007: 1.0, 2010: 1.0, 2013: 1.0},
           "DK2010+DK2013+DK2016": {2010: 1.0, 2013: 1.0, 2016: 1.0}}


def dk_combo_vec(combo: dict) -> np.ndarray:
    return sum(w * contrast_vec("DK", c) for c, w in combo.items())


def estimable(c: np.ndarray, Xf: np.ndarray) -> bool:
    a = np.linalg.lstsq(Xf.T, c, rcond=None)[0]
    return bool(np.abs(Xf.T @ a - c).max() < 1e-8)


def left_null(X: np.ndarray, tol=1e-9) -> np.ndarray:
    """Orthonormal basis (rows) of {a : X' a = 0}: cell-mean contrasts that vanish under the model with design X."""
    U_, s, _ = np.linalg.svd(X, full_matrices=True)
    r = int((s > tol * s[0]).sum())
    return U_[:, r:].T


def cell_design(knots=None, step=False) -> np.ndarray:
    return np.array([full_row(int(c), HY[h], knots, step) for c, h in CELLS17])


def design_checks() -> dict:
    """Rank of the 17-cell design with a free effect per earnings window vs the restricted period profiles; which
    contrasts are estimable without the restriction; restriction-free cohort combinations as cell-mean weights; the
    testable implications (left null spaces) of each restricted model and of the plane in (h, c)."""
    per = sorted({int(c) + HY[h] for c, h in CELLS17})
    nobs = {t: sum(int(c) + HY[h] == t for c, h in CELLS17) for t in per}
    nAK = 1 + len(AGES) + len(KCOH)
    Xfree = np.array([[1.0] + [float(HY[h] == a) for a in AGES] + [float(int(c) == k) for k in KCOH]
                      + [float(int(c) + HY[h] == t) for t in per] for c, h in CELLS17])
    Xpw = cell_design()
    for c, h in CELLS17:          # the interpolated period coordinate reproduces t = c + h exactly
        assert abs(pw_weights(int(c) + HY[h]) @ np.array(KNOTS, float) - (int(c) + HY[h])) < 1e-9
        assert abs(pw_weights(int(c) + HY[h], KNOTS_Y5) @ np.array(KNOTS_Y5, float) - (int(c) + HY[h])) < 1e-9

    def free_vec(kind, x):
        if kind != "DP":
            return np.concatenate([contrast_vec(kind, x)[:nAK], np.zeros(len(per))])
        v = np.zeros(nAK + len(per))      # free-period analogue on the y1 earnings grid
        v[nAK + per.index(x)], v[nAK + per.index(x - 3)], v[nAK + per.index(x + 3)] = 1.0, -0.5, -0.5
        return v

    names = [f"{kind}{'' if x is None else x}" for kind, x in CONTRASTS["M_APC"]]
    FV = {n: free_vec(kind, x) for n, (kind, x) in zip(names, CONTRASTS["M_APC"])}
    free_est = {n: estimable(v, Xfree) for n, v in FV.items()}
    assert not any(free_est.values())
    # dimension of the estimable part of span{all curvature contrasts} and of span{D_K}
    Rw = np.linalg.svd(Xfree, full_matrices=False)[2][:np.linalg.matrix_rank(Xfree)]
    Pr = Rw.T @ Rw

    def est_dim(M):
        s_ = np.linalg.svd((np.eye(len(Pr)) - Pr) @ M, compute_uv=False)
        return M.shape[1] - int((s_ > 1e-9).sum())

    dim_all = est_dim(np.column_stack(list(FV.values())))
    # the model-free cohort-axis deviations at y1 and y10 are DK(c) + DP(c + h) in free-period form: all estimable;
    # together with the cohort-only combinations they span the estimable part; the age curvature enters none of it
    devs = [FV[f"DK{c}"] + FV[f"DP{c + 1}"] for c in KCOH[1:-1]] + [FV[f"DK{c}"] + FV[f"DP{c + 10}"] for c in (2004, 2007)]
    assert all(estimable(v, Xfree) for v in devs)
    n_dev, rank_dev = len(devs), int(np.linalg.matrix_rank(np.column_stack(devs)))
    Mall = np.column_stack(list(FV.values()))
    s_, V_ = np.linalg.svd((np.eye(len(Pr)) - Pr) @ Mall)[1:]
    nullc = V_[int((s_ > 1e-9).sum()):]                    # coefficient vectors of estimable combinations
    ka_in = bool(np.abs(nullc[:, names.index("kA")]).max() > 1e-9)
    span_dev_dk = int(np.linalg.matrix_rank(np.column_stack(
        devs + [np.concatenate([dk_combo_vec(c)[:nAK], np.zeros(len(per))]) for c in DKCOMBO.values()])))
    assert span_dev_dk == dim_all
    dim_dk = est_dim(np.column_stack([FV[n] for n in names if n.startswith("DK")]))
    dim_ka_dp = est_dim(np.column_stack([FV[n] for n in names if not n.startswith("DK")]))
    rank_all = int(np.linalg.matrix_rank(Mall))
    # REVISION NOTE 3: the period-only combinations estimable with free period effects (the analogues of DKCOMBO);
    # they span the estimable part among the age and period contrasts
    dpc = [{2005: 1.0, 2014: -1.0}, {2008: 1.0, 2017: -1.0}, {2005: 1.0, 2008: 1.0, 2011: 1.0},
           {2008: 1.0, 2011: 1.0, 2014: 1.0}, {2011: 1.0, 2014: 1.0, 2017: 1.0}]
    dpv = [sum(w_ * FV[f"DP{t_}"] for t_, w_ in cmb.items()) for cmb in dpc]
    assert all(estimable(v, Xfree) for v in dpv)
    assert int(np.linalg.matrix_rank(np.column_stack(dpv))) == dim_ka_dp
    dkv = [np.concatenate([dk_combo_vec(c)[:nAK], np.zeros(len(per))]) for c in DKCOMBO.values()]
    assert int(np.linalg.matrix_rank(np.column_stack(dkv))) == dim_dk
    # REVISION NOTE 3: the model-free age-axis curvature within window c (Answer 6), z(c,y5) - 5/9 z(c,y1) -
    # 4/9 z(c,y10), is a cell-mean combination (estimable); in parameters it is kA plus the period curvature
    # P(c+5) - 5/9 P(c+1) - 4/9 P(c+10). A(5) enters only the y5 cells, whose earnings windows are all single-cell, so
    # in every row of the design (hence in every estimable function) the A(5) coefficient equals the total coefficient
    # on the y5 earnings windows' period effects: kA cannot be separated from them.
    y5per = sorted({int(c) + HY["y5"] for c in CO["y5"]})
    assert all(nobs[t] == 1 for t in y5per)
    ia5 = 1 + AGES.index(5)
    assert all(abs(r[ia5] - sum(r[nAK + per.index(t)] for t in y5per)) < 1e-12 for r in Xfree)
    # equivalently: A(5) + d together with P(t) - d for every y5 earnings window t leaves every cell mean unchanged
    d_null = np.zeros(Xfree.shape[1])
    d_null[ia5] = 1.0
    for t in y5per:
        d_null[nAK + per.index(t)] = -1.0
    assert np.abs(Xfree @ d_null).max() < 1e-12
    assert abs(np.concatenate([contrast_vec("kA")[:nAK], np.zeros(len(per))]) @ d_null - 1.0) < 1e-12
    agecurv = {}
    for c in CO["y10"]:
        rw = {h: Xfree[CELLS17.index((c, h))] for h in HZ}
        v = rw["y5"] - 5 / 9 * rw["y1"] - 4 / 9 * rw["y10"]
        assert estimable(v, Xfree)
        assert np.allclose(v[:nAK], contrast_vec("kA")[:nAK], atol=1e-12)
        pv = {per[i]: float(x) for i, x in enumerate(v[nAK:]) if abs(x) > 1e-12}
        assert set(pv) == {int(c) + 1, int(c) + 5, int(c) + 10}
        assert abs(pv[int(c) + 5] - 1.0) < 1e-12 and abs(pv[int(c) + 1] + 5 / 9) < 1e-12
        agecurv[c] = pv
    combo_w = {}
    for name, cmb in DKCOMBO.items():
        v = np.concatenate([dk_combo_vec(cmb)[:nAK], np.zeros(len(per))])
        assert estimable(v, Xfree), name
        a = np.linalg.lstsq(Xfree.T, v, rcond=None)[0]         # unique: Xfree has full row rank 17
        assert np.abs(Xfree.T @ a - v).max() < 1e-10
        combo_w[name] = a
        assert estimable(dk_combo_vec(cmb), Xpw)
    pw_est = {f"{k}{'' if x is None else x}": estimable(contrast_vec(k, x), Xpw) for k, x in CONTRASTS["M_APC"]}
    assert all(pw_est.values())
    lin = np.zeros(NFULL); lin[ix_A(10)], lin[ix_A(1)] = 1.0, -1.0
    assert not estimable(lin, Xpw)
    alt = {}
    for key, kn, st in (("k5", KNOTS_Y5, False), ("step", KNOTS, True)):
        Xa = cell_design(kn, st)
        assert all(estimable(contrast_vec(k, x, kn), Xa) for k, x in contrasts_for("M_APC", kn, st))
        if st:
            assert estimable(contrast_vec("kAs"), Xa)
        assert estimable(contrast_vec("kA"), Xa) != st            # kA estimable under k5, not under step
        assert not estimable(lin, Xa)
        alt[key] = dict(cols=Xa.shape[1], rank=int(np.linalg.matrix_rank(Xa)), N=left_null(Xa))
    Xlin = np.array([[1.0, HY[h], int(c)] for c, h in CELLS17])
    return dict(n_periods=len(per), periods=per, n_single=sum(v == 1 for v in nobs.values()), nobs=nobs,
                free_cols=Xfree.shape[1], free_rank=int(np.linalg.matrix_rank(Xfree)),
                pw_cols=Xpw.shape[1], pw_rank=int(np.linalg.matrix_rank(Xpw)), free_est=free_est, pw_est=pw_est,
                dim_all=dim_all, n_all=len(FV), dim_dk=dim_dk, dim_ka_dp=dim_ka_dp, combo_w=combo_w,
                n_dev=n_dev, rank_dev=rank_dev, ka_in=ka_in, rank_all=rank_all, agecurv=agecurv, y5per=y5per,
                dp_combos=[" + ".join(f"DP{t_}" for t_ in c_) if len(c_) == 3 else f"DP{min(c_)} - DP{max(c_)}"
                           for c_ in dpc],
                N_pw=left_null(Xpw), N_lin=left_null(Xlin), alt=alt)


class Meta:
    """z = X b + u_f + e; var(u_f) = s2, var(e) = v + t2 (REML); GLS operator L with the variance components fixed."""

    def __init__(self, z, v, X, fid):
        self.z, self.v, self.X, self.fid = z, v, X, np.asarray(fid)
        self.groups = [np.flatnonzero(self.fid == f) for f in sorted(set(self.fid))]
        self.fields = sorted(set(self.fid))
        assert np.linalg.matrix_rank(X) == X.shape[1], "design not of full column rank"
        self.s2, self.t2 = self._reml()
        self.L = self._operator(np.arange(len(z)))
        self.b = self.L @ z
        self._bj = None

    def _parts(self, s2, t2, rows=None):
        X, z, v = self.X, self.z, self.v
        p = X.shape[1]
        XtVX, XtVz, zVz, logdet = np.zeros((p, p)), np.zeros(p), 0.0, 0.0
        for idx in self.groups:
            if rows is not None:
                idx = idx[np.isin(idx, rows)]
                if not len(idx):
                    continue
            d = v[idx] + t2
            di = 1 / d
            a = s2 / (1 + s2 * di.sum())
            Xf, zf = X[idx], z[idx]
            VX = di[:, None] * Xf - a * np.outer(di, di @ Xf)
            Vz = di * zf - a * di * (di @ zf)
            XtVX += Xf.T @ VX
            XtVz += Xf.T @ Vz
            zVz += zf @ Vz
            logdet += np.log(d).sum() + np.log(1 + s2 * di.sum())
        return XtVX, XtVz, zVz, logdet

    def _reml(self):
        def nll(lp):
            s2, t2 = np.exp(lp)
            A, bz, zz, ld = self._parts(s2, t2)
            beta = np.linalg.solve(A, bz)
            return 0.5 * (ld + np.linalg.slogdet(A)[1] + zz - beta @ bz)
        grid = [(a, b) for a in np.linspace(-12, 0, 7) for b in np.linspace(-12, 0, 7)]
        start = min(grid, key=lambda g: nll(np.array(g)))
        r = minimize(nll, np.array(start), method="L-BFGS-B", bounds=[(-18, 2), (-18, 2)])
        s2, t2 = np.exp(r.x)
        return float(s2), float(t2)

    def _operator(self, rows):
        X, v, s2, t2 = self.X, self.v, self.s2, self.t2
        p = X.shape[1]
        A = np.zeros((p, p))
        VXfull = np.zeros((len(self.z), p))
        rows_set = np.zeros(len(self.z), bool)
        rows_set[rows] = True
        for idx in self.groups:
            idx = idx[rows_set[idx]]
            if not len(idx):
                continue
            d = v[idx] + t2
            di = 1 / d
            a = s2 / (1 + s2 * di.sum())
            Xf = X[idx]
            VX = di[:, None] * Xf - a * np.outer(di, di @ Xf)
            A += Xf.T @ VX
            VXfull[idx] = VX
        return np.linalg.solve(A, VXfull.T)                       # (p, N); zero columns outside rows

    def bjack(self) -> np.ndarray:
        """(K, p) delete-one-field GLS coefficients (variance components fixed); computed once."""
        if self._bj is None:
            bs = []
            for idx in self.groups:
                keep = np.setdiff1d(np.arange(len(self.z)), idx)
                bs.append(self._operator(keep) @ self.z)
            self._bj = np.array(bs)
        return self._bj

    def jack(self, cvec):
        """delete-one-field jackknife variance of cvec' b (variance components fixed)."""
        th = np.array([float(cvec @ b) for b in self.bjack()])
        K = len(th)
        return float((K - 1) / K * ((th - th.mean()) ** 2).sum())

    def jack_cov(self, C):
        th = self.bjack() @ C.T                                    # (K, q)
        K = th.shape[0]
        d = th - th.mean(0)
        return (K - 1) / K * d.T @ d


def _wald(th: np.ndarray, V: np.ndarray, K: int) -> dict:
    ev, Q = np.linalg.eigh(V)
    contrib = (Q.T @ th) ** 2 / ev
    W = float(contrib.sum())
    q = len(th)
    F = W * (K - q) / (q * (K - 1)) if K > q else np.nan
    return dict(W=W, p=float(chi2.sf(W, q)), F=F, pF=float(fdist.sf(F, q, K - q)) if K > q else np.nan,
                dom=float(contrib.max() / W), cond=float(ev.max() / ev.min()))


def wald_twoway(m: Meta, C: np.ndarray, ysb: np.ndarray, yib: np.ndarray) -> dict:
    """Joint Wald test of C b = 0 (REVISION NOTE 2). (i) Two-way (CGM) covariance, the matrix form of
    scripts/59.twoway: V = V_jack + Cov(shared-draw replicates) - Cov(independent-draw replicates); in directions
    where V has a non-positive eigenvalue the larger one-way variance is used (the scalar fallback of scripts/59,
    applied eigen-direction by eigen-direction). (ii) Conservative covariance V_jack + Cov(shared), which drops the
    subtracted term, is positive definite and bounds the CGM covariance from above. Each with a chi2(q) p-value and a
    Hotelling-type F(q, K - q) version that treats the covariance as estimated from K fields; dom = share of W carried
    by the single largest eigen-direction (a diagnostic of ill-conditioning)."""
    th = C @ m.b
    cb, ib = C @ (m.L @ ysb), C @ (m.L @ yib)
    q, K = C.shape[0], len(m.fields)
    Vj = m.jack_cov(C)
    Vs, Vi = np.atleast_2d(np.cov(cb)), np.atleast_2d(np.cov(ib))
    V = Vj + Vs - Vi
    ev, Q = np.linalg.eigh(V)
    nfb = int((ev <= 0).sum())
    if nfb:
        ev = np.array([e if e > 0 else max(u @ Vj @ u, u @ Vs @ u) for e, u in zip(ev, Q.T)])
        V = (Q * ev) @ Q.T
    two = _wald(th, V, K)
    con = _wald(th, Vj + Vs, K)
    return dict(W=two["W"], q=q, p=two["p"], F=two["F"], df2=K - q, pF=two["pF"], dom=two["dom"],
                cond=two["cond"], mineig=float(np.linalg.eigvalsh(Vj + Vs - Vi).min()), nfb=nfb, K=K,
                Wc=con["W"], pc=con["p"], pcF=con["pF"], domc=con["dom"])


def meta_fit(sample, stat, model, scale="z", tag="", knots=None, step=False, tmin=None, extra=None, lof=None,
             label=None):
    """Fit one meta-regression on the cells of `sample`; returns dict of contrasts with two-way inference.
    knots/step: period profile (default: piecewise linear on KNOTS). tmin: keep cells whose first earnings year is
    >= tmin. extra: {name: full-parameter contrast} estimated but not part of the joint Wald. M_free: saturated
    cell-means design (17 cells); its contrasts are cell-weight vectors (extra) and lof = {name: rows of cell-mean
    contrasts} gives lack-of-fit Wald tests."""
    js = [j for j, c in enumerate(CELL) if c["sample"] == sample and np.isfinite(OBS[j].get(stat, np.nan))
          and (tmin is None or int(c["cohort"]) + HY[c["horizon"]] >= tmin)]
    r = np.array([OBS[j][stat] for j in js])
    n = np.array([len(CELL[j]["df"]) for j in js])
    kpar = 1 if stat.startswith("par") else 0
    vz = bw_var(r, n, kpar)
    if scale == "z":
        y, v = fisher(r), vz
        ysb = fisher(np.stack([SB[j][stat] for j in js]))           # (N, B)
        yib = fisher(np.stack([IB[j][stat] for j in js]))
    else:
        y, v = r, (1 - r ** 2) ** 2 * vz
        ysb = np.stack([SB[j][stat] for j in js])
        yib = np.stack([IB[j][stat] for j in js])
    cc = [int(CELL[j]["cohort"]) for j in js]
    hh = [HY[CELL[j]["horizon"]] for j in js]
    fid = [CELL[j]["field"] for j in js]
    joint = []
    if model == "M_lin":
        X = np.column_stack([np.ones(len(js)), np.array(hh, float), np.array(cc, float) - 2001.0])
        cons = {"theta_h": np.array([0, 1.0, 0]), "theta_c": np.array([0, 0, 1.0]),
                "theta_h_minus_theta_c": np.array([0, 1.0, -1.0])}
    elif model == "M_free":
        cix = {(int(c), HY[h]): i for i, (c, h) in enumerate(CELLS17)}
        X = np.zeros((len(js), len(CELLS17)))
        X[np.arange(len(js)), [cix[(c, h)] for c, h in zip(cc, hh)]] = 1.0
        cons = dict(extra or {})
    else:
        Xf = np.array([full_row(c, h, knots, step) for c, h in zip(cc, hh)])
        keep = MODELS[model]
        X = Xf[:, keep]
        cons = {}
        for kind, x in contrasts_for(model, knots, step):
            cv = contrast_vec(kind, x, knots)
            assert estimable(cv, Xf), (model, kind, x)
            cons[f"{kind}{'' if x is None else x}"] = cv[keep]
        joint = list(cons)
        for name, cv in (extra or {}).items():
            assert estimable(cv, Xf), (model, name)
            cons[name] = cv[keep]
    m = Meta(y, v, X, fid)
    out = {}
    K = len(m.fields)
    lab = f"{sample}:{stat}:{scale}" if label is None else label
    for name, cv in cons.items():
        est = float(cv @ m.b)
        cb = cv @ (m.L @ ysb)
        ib = cv @ (m.L @ yib)
        vf = m.jack(cv)
        t = twoway(est, vf, cb, ib)
        t.update(est=est, k=K, n_cells=len(js), mde=MDE_K * t["tse"], p=float(2 * norm.sf(abs(t["tz"]))))
        out[name] = t
        rec("apc", f"{model}:{name}", est, t["tlo"], t["thi"], t["tse"], t["tz"], t["p"], K, len(js),
            sample=lab, note=f"REML s2_field={m.s2:.3g}, tau2={m.t2:.3g}; two-way CI")
    walds = {}
    if joint:
        walds["curvature"] = wald_twoway(m, np.array([cons[k] for k in joint]), ysb, yib)
    for name, Cc in (lof or {}).items():
        walds[name] = wald_twoway(m, Cc, ysb, yib)
    for name, wd in walds.items():
        rec("apc_wald", f"{model}:{name}", wd["W"], k=K, n=len(js), p=wd["p"], z=wd["F"], sample=lab,
            note=f"joint two-way (CGM) Wald, q={wd['q']}; chi2 p={wd['p']:.4g}; F({wd['q']},{wd['df2']})="
                 f"{wd['F']:.4g}, p={wd['pF']:.4g}; min eigenvalue of the CGM matrix={wd['mineig']:.3g} "
                 f"(fallback in {wd['nfb']} directions); largest single direction carries {wd['dom']:.0%} of W; "
                 f"estimate column = Wald statistic, z column = F")
        rec("apc_wald_conservative", f"{model}:{name}", wd["Wc"], k=K, n=len(js), p=wd["pc"], sample=lab,
            note=f"joint Wald with V_jack + Cov(shared) (positive definite, >= CGM), q={wd['q']}; chi2 "
                 f"p={wd['pc']:.4g}; F version p={wd['pcF']:.4g}; largest single direction carries {wd['domc']:.0%}")
    out["_meta"] = dict(s2=m.s2, t2=m.t2, K=K, N=len(js), vmed=float(np.median(v)), wald=walds, joint=joint)
    return out


# ---------------------------------------------------------------------------------------------
# Test 2: institution-level entry-conditions regression with wild cluster bootstrap
# ---------------------------------------------------------------------------------------------
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def pct_rank(x):
    x = np.asarray(x, float)
    return (rankdata(x) - 0.5) / len(x) - 0.5


def t2_sample(L, Ua, h, balanced=True, cov="share") -> pd.DataFrame:
    """cov (all-institution sample only): "share" = the coverage share itself (pre-specified); "rank" = its
    within-(field, window) centred percentile rank (used in the first run; a deviation, REVISION NOTE 2)."""
    X = L[L.horizon == h][["field", "inst_key", "grad_cohort", "earnings", "coverage", "P", "G", "state"]].copy()
    if balanced:
        nwin = X.groupby(["field", "inst_key"]).grad_cohort.transform("nunique")
        X = X[nwin == len(CO[h])]
        nf = X.groupby("field").inst_key.transform("nunique")
        X = X[nf >= NMIN].copy()
    else:
        X = X[np.isfinite(X.coverage.astype(float))]
        n = X.groupby(["field", "grad_cohort"]).inst_key.transform("size")
        X = X[n >= NMIN].copy()
        if cov == "rank":
            X["COV"] = X.groupby(["field", "grad_cohort"]).coverage.transform(pct_rank)
        else:
            X["COV"] = X.coverage.astype(float)
    X["y"] = X.groupby(["field", "grad_cohort"]).earnings.transform(pct_rank)
    if not balanced:     # institutions seen once in a field are absorbed by the institution x field effect
        m = X.groupby(["field", "inst_key"]).grad_cohort.transform("size")
        X = X[m >= 2].copy()
    u = X[["field", "inst_key", "P", "G"]].drop_duplicates(["field", "inst_key"])
    u["Gt"] = u.groupby("field").G.transform(pct_rank)
    u["Ft"] = u.groupby("field").P.transform(pct_rank)
    X = X.merge(u[["field", "inst_key", "Gt", "Ft"]], on=["field", "inst_key"], how="left")
    X["U"] = [Ua[(s, int(c))] for s, c in zip(X.state, X.grad_cohort)]
    X["GU"], X["FU"] = X.Gt * X.U, X.Ft * X.U
    return X.sort_values(["field", "grad_cohort", "inst_key"]).reset_index(drop=True)


def fwl_blocks(X: pd.DataFrame, gx: bool = False):
    """Per field: orthonormal basis of the absorbed effects (window, institution; with gx also G~ x window).
    Returns the blocks and the number of absorbed columns not nested in states (institution x field effects are
    nested in the state clusters and are not counted in the CR1 small-sample factor)."""
    blocks, nonnested = [], 0
    for f, g in X.groupby("field", sort=True):
        rows = g.index.to_numpy()
        Dw = pd.get_dummies(g.grad_cohort).to_numpy(float)
        Di = pd.get_dummies(g.inst_key).to_numpy(float)
        cols = [Dw, Di] + ([Dw * g.Gt.to_numpy(float)[:, None]] if gx else [])
        D = np.column_stack(cols)
        U_, s, _ = np.linalg.svd(D, full_matrices=False)
        Q = U_[:, s > 1e-9 * s[0]]
        blocks.append((rows, Q))
        nonnested += Q.shape[1] - Di.shape[1]
    return blocks, nonnested


def mproj(Y, blocks):
    Y = np.array(Y, float, copy=True)
    one = Y.ndim == 1
    if one:
        Y = Y[None, :]
    for rows, Q in blocks:
        Y[:, rows] -= (Y[:, rows] @ Q) @ Q.T
    return Y[0] if one else Y


def wild_reg(X: pd.DataFrame, regs: list, tests: list, tag: str, gx: bool = False, bench: float = np.nan) -> dict:
    blocks, nonnested = fwl_blocks(X, gx)
    N = len(X)
    yt = mproj(X.y.to_numpy(float), blocks)
    Xt = np.column_stack([mproj(X[c].to_numpy(float), blocks) for c in regs])
    k = Xt.shape[1]
    XtXi = np.linalg.inv(Xt.T @ Xt)
    beta = XtXi @ Xt.T @ yt
    e = yt - Xt @ beta
    states = sorted(X.state.unique())
    sidx = X.state.map({s: i for i, s in enumerate(states)}).to_numpy()
    G = len(states)
    OH = np.zeros((N, G)); OH[np.arange(N), sidx] = 1.0
    scale = G / (G - 1) * (N - 1) / (N - k - nonnested)

    def crv(R):                                  # R (b, N) residuals -> (b, k, k) CR1 variances
        S = np.stack([(R * Xt[:, i]) @ OH for i in range(k)], -1)      # (b, G, k)
        meat = np.einsum("bgk,bgl->bkl", S, S)
        return scale * XtXi @ meat @ XtXi

    V = crv(e[None, :])[0]
    se = np.sqrt(np.diag(V))
    out = dict(N=N, G=G, fields=X.field.nunique(), insts=X[["field", "inst_key"]].drop_duplicates().shape[0],
               n_inst=X.inst_key.nunique(), windows=X.grad_cohort.nunique(), states=states, nonnested=nonnested,
               sd_U=float(X.U.std()), sd_Ut=float(Xt[:, regs.index("U")].std()) if "U" in regs else np.nan)
    for c in regs:
        i = regs.index(c)
        out[c] = dict(b=float(beta[i]), se=float(se[i]), t=float(beta[i] / se[i]),
                      p_cr1=float(2 * tdist.sf(abs(beta[i] / se[i]), G - 1)), sd_xt=float(Xt[:, i].std()))
    fit_u = Xt @ beta
    for c in tests:
        i = regs.index(c)
        rg = [q for q in range(k) if q != i]
        Xr = Xt[:, rg]
        br = np.linalg.lstsq(Xr, yt, rcond=None)[0]
        fit_r = Xr @ br
        er = yt - fit_r
        Wr = WEBB[rng_for(f"wild:{tag}:{c}").integers(6, size=(NWILD, G))]
        Wu = WEBB[rng_for(f"wildu:{tag}:{c}").integers(6, size=(NWILD, G))]

        def boot(Wb, base, res):
            Ys = base[None, :] + mproj(Wb[:, sidx] * res[None, :], blocks)
            bs = Ys @ Xt @ XtXi                                       # (b, k)
            Vs = crv(Ys - bs @ Xt.T)
            return bs[:, i], np.sqrt(Vs[:, i, i])

        # WCR test inversion (REVISION NOTE 2): under H0 beta_i = b0 the restricted bootstrap sample is
        # Y0 + b0 Y1 with Y0 = the b0 = 0 sample and Y1 = mx - M(w * mx) (mx = x_i residualised on the other
        # regressors), so b*_i = B0 + b0 B1 and the CR1 variance is a0 + b0 a1 + b0^2 a2 per replicate (same draws Wr).
        mx = Xt[:, i] - Xr @ np.linalg.lstsq(Xr, Xt[:, i], rcond=None)[0]
        qi = XtXi[i] @ Xt.T
        inv = {k_: [] for k_ in ("B0", "B1", "a0", "a1", "a2")}
        tr, tu, bu = [], [], []
        for s0 in range(0, NWILD, WCHUNK):
            b_r, s_r = boot(Wr[s0:s0 + WCHUNK], fit_r, er)            # restricted (H0: coefficient = 0)
            tr.append(b_r / s_r)
            b_u, s_u = boot(Wu[s0:s0 + WCHUNK], fit_u, e)             # unrestricted, for the CI
            tu.append((b_u - beta[i]) / s_u)
            bu.append(b_u)
            Wb = Wr[s0:s0 + WCHUNK][:, sidx]
            Y0 = fit_r[None, :] + mproj(Wb * er[None, :], blocks)
            Y1 = mx[None, :] - mproj(Wb * mx[None, :], blocks)
            R0 = Y0 - (Y0 @ Xt @ XtXi) @ Xt.T
            R1 = Y1 - (Y1 @ Xt @ XtXi) @ Xt.T
            g0, g1 = (R0 * qi) @ OH, (R1 * qi) @ OH
            inv["B0"].append(Y0 @ qi); inv["B1"].append(Y1 @ qi)
            inv["a0"].append(scale * (g0 ** 2).sum(1)); inv["a1"].append(2 * scale * (g0 * g1).sum(1))
            inv["a2"].append(scale * (g1 ** 2).sum(1))
        tr, tu = np.concatenate(tr), np.concatenate(tu)
        t0 = beta[i] / se[i]
        p = float((1 + np.sum(np.abs(tr) >= abs(t0) - 1e-12)) / (1 + NWILD))
        q = float(np.quantile(np.abs(tu), 0.95))
        pw = (float(norm.cdf(abs(bench) / se[i] - q) + norm.cdf(-abs(bench) / se[i] - q))
              if np.isfinite(bench) else np.nan)
        inv = {k_: np.concatenate(v_) for k_, v_ in inv.items()}
        ci_inv = wcr_invert(inv, float(beta[i]), float(se[i]))
        assert abs(ci_inv["p0"] - p) <= 2.0 / (1 + NWILD), (tag, c, ci_inv["p0"], p)
        out[c].update(p_wcr=p, q95=q, lo=float(beta[i] - q * se[i]), hi=float(beta[i] + q * se[i]),
                      mde=float((q + Z80) * se[i]), sd_boot=float(np.std(np.concatenate(bu), ddof=1)),
                      power_bench=pw, bench=bench, lo_inv=ci_inv["lo"], hi_inv=ci_inv["hi"],
                      inv_connected=ci_inv["connected"], p_inv0=ci_inv["p0"],
                      q95_wcr=float(np.quantile(np.abs(tr), 0.95)))
    return out


def wcr_pvals(inv: dict, beta: float, se: float, b0: np.ndarray) -> np.ndarray:
    """WCR bootstrap p-values of H0: beta_i = b0 for an array of b0 (same draws for every b0)."""
    b0 = np.atleast_1d(np.asarray(b0, float))
    out = np.empty(len(b0))
    for s0 in range(0, len(b0), 50):
        bb = b0[s0:s0 + 50][:, None]
        num = inv["B0"][None, :] + bb * (inv["B1"][None, :] - 1.0)
        den = np.sqrt(np.maximum(inv["a0"][None, :] + bb * inv["a1"][None, :] + bb ** 2 * inv["a2"][None, :], 1e-300))
        t0 = np.abs(beta - bb) / se
        out[s0:s0 + 50] = (1 + (np.abs(num / den) >= t0 - 1e-12).sum(1)) / (1 + len(inv["B0"]))
    return out


def wcr_invert(inv: dict, beta: float, se: float, alpha: float = 0.05) -> dict:
    """95% confidence set {b0 : WCR p(b0) > alpha}: grid over beta +/- 12 SE (extended to 60 SE if the edge is
    accepted), then bisection at the outermost accepted grid points. Reports the hull and whether the accepted grid
    points are contiguous."""
    p0 = float(wcr_pvals(inv, beta, se, [0.0])[0])
    for span in (12.0, 60.0):
        g = beta + se * np.linspace(-span, span, 1201)
        acc = wcr_pvals(inv, beta, se, g) > alpha
        if not (acc[0] or acc[-1]):
            break
    idx = np.flatnonzero(acc)
    connected = bool(len(idx) and (idx[-1] - idx[0] + 1 == len(idx)))

    def bis(a_in, b_out):
        for _ in range(40):
            mid = 0.5 * (a_in + b_out)
            if wcr_pvals(inv, beta, se, [mid])[0] > alpha:
                a_in = mid
            else:
                b_out = mid
        return 0.5 * (a_in + b_out)

    lo = -np.inf if acc[0] else bis(g[idx[0]], g[idx[0] - 1])
    hi = np.inf if acc[-1] else bis(g[idx[-1]], g[idx[-1] + 1])
    return dict(lo=float(lo), hi=float(hi), connected=connected, p0=p0)


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------
def fm(v, d=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def ci_s(r, d=3, lo="tlo", hi="thi"):
    return f"[{fm(r[lo], d)}, {fm(r[hi], d)}]"


def side(lo, hi):
    return "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0"


def fp(p):
    if not np.isfinite(p):
        return "—"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def fu(v, d=3):
    return "—" if v is None or not np.isfinite(v) else f"{v:.{d}f}"


def power(bench, se, crit=Z975):
    """Power of a two-sided test with critical value crit against a true effect of size |bench|."""
    if not (np.isfinite(bench) and np.isfinite(se) and se > 0):
        return np.nan
    return float(norm.cdf(abs(bench) / se - crit) + norm.cdf(-abs(bench) / se - crit))


def verdict(lo, hi, mde, bench) -> str:
    if lo > 0:
        return "supported"
    if hi < 0:
        return "opposite"
    return "null" if mde <= bench else "underpowered"


BENCH1 = 0.05       # Test 1 benchmark (coupling units), fixed in the docstring
BENCH2 = 0.01       # Test 2 benchmark (status-gradient units per percentage point), fixed in the docstring


# ---------------------------------------------------------------------------------------------
# PSEO Flows (V4.14.1): share of employed graduates working outside the institution's state
# ---------------------------------------------------------------------------------------------
PSEOF_NEW = ROOT / "data" / "raw" / "pseo_flows_2026q2" / "pseof_all.csv.gz"


NE_COHORTS = ["2001", "2004", "2007", "2010", "2013", "2016"]


def flows_outstate(id_sets: dict, ne_sets: dict | None = None) -> tuple:
    """Institution level, all CIP, bachelor's, pooled cohorts (0000), national geography row, all industries:
    1 - employment-weighted mean of (graduates employed in the institution's state / graduates employed), as
    scripts/32 computes it (there at y5 on V4.13.0). Same pass (REVISION NOTE 2): share of y1 employed graduates
    working in census division 1 (New England, which contains Massachusetts) by graduation window, from the
    window-specific national and division rows (institutions with both rows released, status 1)."""
    cols = ["degree_level", "inst_level", "cip_level", "geo_level", "geography", "ind_level", "institution",
            "grad_cohort"]
    for h in HZ:
        cols += [f"{h}_grads_emp", f"{h}_grads_emp_instate", f"status_{h}_grads_emp",
                 f"status_{h}_grads_emp_instate"]
    parts, neparts = [], []
    for ch in pd.read_csv(PSEOF_NEW, dtype=str, usecols=cols, chunksize=400_000):
        base = ((ch.degree_level == "05") & (ch.inst_level == "I") & (ch.cip_level == "A") &
                (ch.ind_level == "A"))
        b = ch[base & (ch.grad_cohort == "0000") & (ch.geo_level == "N")]
        parts.append(b.drop(columns=["degree_level", "inst_level", "cip_level", "geo_level", "geography",
                                     "ind_level", "grad_cohort"]))
        nb = ch[base & ch.grad_cohort.isin(NE_COHORTS) & (((ch.geo_level == "N") & (ch.geography == "00")) |
                                                          ((ch.geo_level == "D") & (ch.geography == "1")))]
        neparts.append(nb[["institution", "grad_cohort", "geo_level", "y1_grads_emp", "status_y1_grads_emp"]])
    d = pd.concat(parts, ignore_index=True)
    assert not d.institution.duplicated().any()
    ne = pd.concat(neparts, ignore_index=True)
    ne = ne[ne.status_y1_grads_emp == "1"]
    ne = ne.pivot_table(index=["institution", "grad_cohort"], columns="geo_level", values="y1_grads_emp",
                        aggfunc="first").dropna().astype(float).reset_index()
    NE = {}
    for name, ids in (ne_sets or {}).items():
        x = ne[ne.institution.isin(ids)]
        for c in NE_COHORTS:
            xc = x[x.grad_cohort == c]
            NE[(name, c)] = dict(share=float(xc.D.sum() / xc.N.sum()) if len(xc) else np.nan, n_inst=len(xc),
                                 n_ids=len(ids))
    out = {}
    for h in HZ:
        ok = (d[f"status_{h}_grads_emp"] == "1") & (d[f"status_{h}_grads_emp_instate"] == "1")
        x = d[ok]
        g = pd.to_numeric(x[f"{h}_grads_emp"]).to_numpy(float)
        gi = pd.to_numeric(x[f"{h}_grads_emp_instate"]).to_numpy(float)
        for name, ids in id_sets.items():
            sel = (g > 0) & (x.institution.isin(ids).to_numpy() if ids is not None else True)
            sh = np.clip(gi[sel] / g[sel], 0, 1)
            out[(name, h)] = dict(out_share=float(1 - np.average(sh, weights=g[sel])), n_inst=int(sel.sum()),
                                  grads=float(g[sel].sum()))
    return out, NE


# ---------------------------------------------------------------------------------------------
# panel output
# ---------------------------------------------------------------------------------------------
def panel_frame() -> pd.DataFrame:
    rows = []
    for j, c in enumerate(CELL):
        o, n = OBS[j], len(c["df"])
        r = dict(sample=c["sample"], field=c["field"], field_label=LAB.get(c["field"], c["field"]),
                 grad_window=c["cohort"], grad_years=grad_years(c["cohort"]), horizon=c["horizon"],
                 earn_years=earn_years(c["cohort"], c["horizon"]), n_inst=n,
                 rho_F=o["rhoF"], rho_G=o["rhoG"], z_F=float(fisher(o["rhoF"])), z_G=float(fisher(o["rhoG"])),
                 var_z_F_bw=bw_var(o["rhoF"], n), var_z_G_bw=bw_var(o["rhoG"], n),
                 var_z_F_boot=float(np.var(fisher(np.asarray(IB[j]["rhoF"])), ddof=1)),
                 var_z_G_boot=float(np.var(fisher(np.asarray(IB[j]["rhoG"])), ddof=1)),
                 mean_coverage=float(np.nanmean(c["df"].coverage.astype(float))))
        if "parF" in o:
            r.update(par_F=o["parF"], par_G=o["parG"], z_par_F=float(fisher(o["parF"])),
                     var_z_par_F_bw=bw_var(o["parF"], n, 1),
                     var_z_par_F_boot=float(np.var(fisher(np.asarray(IB[j]["parF"])), ddof=1)))
        rows.append(r)
    d = pd.DataFrame(rows)
    d["_h"] = d.horizon.map(HY)
    return d.sort_values(["sample", "field", "_h", "grad_window"]).drop(columns="_h").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
T: dict = {}          # linear cell statistics (tests, sensitivities, surface, exploratory)


def lstat(name, sample, terms, stat="rhoF", tf=None, ivw=False, perm=None, fields=None, section="", label="",
          horizon="", bench=BENCH1):
    """Mean over fields of sum_k w_k tf(stat of cell k); two-way CI; optional institution-clustered cohort-label
    permutation (perm = (h, cohorts, weights)); stored in T[name] and recorded."""
    cells = [ch for ch, _ in terms]
    fs = fields_with(sample, cells) if fields is None else fields
    kpar = 1 if stat.startswith("par") else 0
    weights = None
    if ivw:
        weights = {f: 1.0 / sum(w ** 2 * bw_var(OBS[J[(sample, f, c, h)]][stat],
                                                len(CELL[J[(sample, f, c, h)]]["df"]), kpar)
                                for (c, h), w in terms) for f in fs}
    r = lin_stat(sample, terms, stat, tf, fields=fs, weights=weights, tag=name)
    ns = [len(CELL[J[(sample, f, c, h)]]["df"]) for f in fs for c, h in cells]
    r.update(n_min=min(ns), n_max=max(ns), n_med=float(np.median(ns)), n_pos=int((r["per_field"] > 0).sum()),
             n_inst=len(set().union(*[set(CELL[J[(sample, f, c, h)]]["df"].inst_key) for f in fs
                                      for c, h in cells])),
             p_perm=np.nan, power=power(bench, r["tse"]), bench=bench, sample=sample, stat=stat, label=label,
             horizon=horizon, section=section)
    r["verdict"] = verdict(r["tlo"], r["thi"], r["mde"], bench)
    if perm is not None:
        assert tf is None and weights is None
        h, cs, w = perm
        r["p_perm"], o = perm_test(sample, h, cs, np.asarray(w, float), stat, tag=name)
        assert np.isclose(o, r["est"], atol=1e-12)
    T[name] = r
    rec(section, name, r["est"], r["tlo"], r["thi"], r["tse"], r["tz"],
        r["p_perm"] if perm is not None else r["p"], r["k"], r["n_inst"], sample=label or sample, horizon=horizon,
        note=f"two-way CI; MDE80={r['mde']:.4f}; power vs {bench}={r['power']:.3f}; fields>0 {r['n_pos']}/{r['k']}; "
             f"cell n {r['n_min']}-{r['n_max']}" + ("; p = institution-clustered cohort-label permutation"
                                                    if perm is not None else "; p = two-way normal"))
    return r


def main():
    stage("state unemployment rates")
    src, ur_meta, monthly = fetch_ur()
    ua = state_ur_annual(monthly)
    ua.to_csv(OUT_UR, index=False, float_format="%.10g")
    UA = {(s, int(y)): float(u) for s, y, u in zip(ua.state, ua.year, ua.ur)}
    Ua = {(s, int(c)): float(np.mean([UA[(s, y)] for y in range(int(c), int(c) + 3)]))
          for s in STATES51 for c in WINDOWS}
    Ua1 = {(s, int(c)): float(np.mean([UA[(s, y)] for y in range(int(c) + 1, int(c) + 4)]))
           for s in STATES51 for c in WINDOWS}
    stage(f"U source {src}; {len(ur_meta)} series, annual {ua.year.min()}-{ua.year.max()}")

    stage("PSEO V4.14.1 long table")
    L = load_long()
    ins = pd.read_csv(s59.NEW_INST, dtype=str)
    ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
    for st, fp_ in zip(ins.institution_state, ins.statefips):
        if st in FIPS:
            assert FIPS[st] == fp_, (st, fp_)
    assert set(L.state) <= set(STATES51)
    build_samples(L)
    prepare_arrays()
    binfo = observed_and_boot()
    dchk = design_checks()
    stage(f"{binfo['n_cells']} cells, {binfo['NI']} institutions; redrawn {binfo['redrawn_shared']} / "
          f"{binfo['redrawn_ind']}")
    pinfo = placebo_pass(L)
    binfo["placebo"] = pinfo
    stage(f"placebo triples: {pinfo['n_cells']} cells, {pinfo['NI']} institutions; redrawn "
          f"{pinfo['redrawn_shared']} / {pinfo['redrawn_ind']}")

    # reproduction of scripts/59 (V4.14.1 all-field fixed-cohort career slope)
    ref = pd.read_csv(ROOT / "data" / "interim" / "pseo_refresh.csv")
    ref = ref[(ref.section == "career_time") & (ref.variant == "new") & (ref.stat == "slope_all")].iloc[0]
    fs_car = sorted({c["field"] for c in CELL if c["sample"] == "career"})
    sl = [np.mean([np.array([OBS[J[("career", f, c, h)]]["rhoF"] for h in HZ]) @ W_SLOPE
                   for c in ["2001", "2004", "2007", "2010"] if ("career", f, c, "y1") in J]) for f in fs_car]
    repro = dict(slope=float(np.mean(sl)), k=len(fs_car), ref=float(ref.estimate), ref_k=int(ref.k))
    assert abs(repro["slope"] - repro["ref"]) < 5e-7 and repro["k"] == repro["ref_k"], repro
    rec("reproduction", "career slope, all fields (scripts/59 V4.14.1)", repro["slope"], k=repro["k"],
        sample="career: scripts/52 balanced fixed-cohort panel", note=f"scripts/59 value {repro['ref']:.6f}")

    panel = panel_frame()
    panel.to_csv(OUT_PANEL, index=False, float_format="%.10g")
    stage(f"panel written: {len(panel)} rows")

    # ---- Test 1 ---------------------------------------------------------------------------------
    GRW, CVW = ["2004", "2007", "2010"], ["2013", "2016", "2019"]
    t1spec = {"gr_y1": ("y1", GRW, W_MID), "gr_y5": ("y5", GRW, W_MID), "gr_y10": ("y10", GRW, W_MID),
              "covid_y1": ("y1", CVW, W_EXT)}
    LAB_TRIPLE = "fixed institutions: triple-matched (present in the three windows)"
    for base, (h, cs, w) in t1spec.items():
        terms = [((c, h), float(x)) for c, x in zip(cs, w)]
        lstat(f"T1:{base}", base, terms, perm=(h, cs, w), section="test1", horizon=h,
              label=f"{LAB_TRIPLE}; Spearman(F, p50)")
        lstat(f"T1:{base}:z", base, terms, tf=fisher, section="test1_sens", horizon=h,
              label=f"{LAB_TRIPLE}; Fisher z of Spearman(F, p50)")
        lstat(f"T1:{base}:ivw", base, terms, tf=fisher, ivw=True, section="test1_sens", horizon=h,
              label=f"{LAB_TRIPLE}; Fisher z, inverse-variance-weighted mean over fields")
        lstat(f"T1:{base}:G", base, terms, stat="rhoG", perm=(h, cs, w), section="test1_sens", horizon=h,
              label=f"{LAB_TRIPLE}; Spearman(G, p50)")
        lstat(f"T1:{base}:allcov", "all", terms, stat="parF", section="test1_sens", horizon=h,
              label="all institutions with coverage; partial Spearman(F, p50 | coverage)")
        lstat(f"T1:{base}:fixed17", "fixed17", terms, section="test1_sens", horizon=h,
              label="fixed institutions: present in all 17 cells; Spearman(F, p50)")
    stage("test 1 done")

    # ---- APC meta-regression ---------------------------------------------------------------------
    A = {}
    for key, (samp, stat, model, scale) in {
            "f17:F:APC": ("fixed17", "rhoF", "M_APC", "z"), "f17:F:AC": ("fixed17", "rhoF", "M_AC", "z"),
            "f17:F:AP": ("fixed17", "rhoF", "M_AP", "z"), "f17:G:APC": ("fixed17", "rhoG", "M_APC", "z"),
            "f17:F:APC:r": ("fixed17", "rhoF", "M_APC", "r"), "all:parF:APC": ("all", "parF", "M_APC", "z"),
            "f17:F:lin:r": ("fixed17", "rhoF", "M_lin", "r"), "f17:F:lin:z": ("fixed17", "rhoF", "M_lin", "z"),
            "f17:G:lin:r": ("fixed17", "rhoG", "M_lin", "r"), "all:parF:lin:r": ("all", "parF", "M_lin", "r")
            }.items():
        A[key] = meta_fit(samp, stat, model, scale=scale)
    # REVISION NOTE 2 (post hoc): restricted estimates of the restriction-free cohort combinations, alternative
    # period profiles, saturated cell-means fits (restriction-free combinations and lack-of-fit tests), and the
    # linear surface on cells whose earnings years are all >= 2010 (complete LEHD frame)
    xdk = {n: dk_combo_vec(c) for n, c in DKCOMBO.items()}
    LOF = {"plane": dchk["N_lin"], "pw": dchk["N_pw"], "k5": dchk["alt"]["k5"]["N"], "step": dchk["alt"]["step"]["N"]}
    for key, kw in {
            "f17:F:APC:x": dict(sample="fixed17", stat="rhoF", model="M_APC", extra=xdk,
                                label="fixed17:rhoF:z:restriction-free combinations under pw"),
            "f17:F:APC:k5": dict(sample="fixed17", stat="rhoF", model="M_APC", knots=KNOTS_Y5, extra=xdk,
                                 label="fixed17:rhoF:z:knots at y5 windows"),
            "f17:F:APC:step": dict(sample="fixed17", stat="rhoF", model="M_APC", step=True, extra=xdk,
                                   label="fixed17:rhoF:z:step-function period"),
            "f17:F:free": dict(sample="fixed17", stat="rhoF", model="M_free", extra=dchk["combo_w"], lof=LOF,
                               label="fixed17:rhoF:z:saturated cell means"),
            "f17:G:free": dict(sample="fixed17", stat="rhoG", model="M_free", extra=dchk["combo_w"], lof=LOF,
                               label="fixed17:rhoG:z:saturated cell means"),
            "all:parF:free": dict(sample="all", stat="parF", model="M_free", extra=dchk["combo_w"], lof=LOF,
                                  label="all:parF:z:saturated cell means"),
            "f17:F:lin:r:t2010": dict(sample="fixed17", stat="rhoF", model="M_lin", scale="r", tmin=2010,
                                      label="fixed17:rhoF:r:cells with earnings years >= 2010"),
            "all:parF:lin:r:t2010": dict(sample="all", stat="parF", model="M_lin", scale="r", tmin=2010,
                                         label="all:parF:r:cells with earnings years >= 2010")}.items():
        A[key] = meta_fit(**kw)
    stage("APC fits done")

    # ---- model-free surface: mean per cell, cohort-axis deviations, age-axis curvature ---------------
    SF = {}
    for samp, stat, lab in (("fixed17", "rhoF", "fixed institutions: present in all 17 cells; Spearman(F, p50)"),
                            ("all", "parF", "all institutions with coverage; partial Spearman(F, p50 | coverage)")):
        fs17 = fields_with(samp, CELLS17)
        SF[samp] = dict(fields=fs17, label=lab)
        for c, h in CELLS17:
            lstat(f"cell:{samp}:{c}:{h}", samp, [((c, h), 1.0)], stat, fields=fs17, section="surface_mean",
                  horizon=h, label=f"{lab}; fields with all 17 cells")
        for h in HZ:
            for i in range(1, len(CO[h]) - 1):
                cs = CO[h][i - 1:i + 2]
                lstat(f"D:{samp}:{h}:{cs[1]}", samp, [((c, h), float(x)) for c, x in zip(cs, W_MID)], stat,
                      fields=fs17, section="surface_cohort_dev", horizon=h, label=f"{lab}; fields with all 17 cells")
        for c in CO["y10"]:
            lstat(f"kA:{samp}:{c}", samp, [((c, "y1"), -5 / 9), ((c, "y5"), 1.0), ((c, "y10"), -4 / 9)], stat,
                  fields=fs17, section="surface_age_curv", horizon="y1/y5/y10",
                  label=f"{lab}; fields with all 17 cells")
    stage("surface done")

    # ---- exploratory -----------------------------------------------------------------------------
    CW = ["2001", "2004", "2007", "2010"]
    for c in CW:
        lstat(f"E:slope:{c}", "career", [((c, h), float(W_SLOPE[k])) for k, h in enumerate(HZ)],
              section="exploratory", horizon="y1/y5/y10",
              label="fixed institutions within window (balanced y1/y5/y10); OLS slope per year")
    fs4 = fields_with("career", [(c, h) for c in CW for h in HZ])
    lstat("E:slope:2007-others", "career",
          [((c, h), float(W_SLOPE[k]) * (1.0 if c == "2007" else -1 / 3)) for c in CW for k, h in enumerate(HZ)],
          fields=fs4, section="exploratory", horizon="y1/y5/y10",
          label="fields with all four windows; slope(2007) - mean slope(2001, 2004, 2010)")
    lstat("E:slope:2007-neighbours", "career",
          [((c, h), float(W_SLOPE[k]) * {"2004": -0.5, "2007": 1.0, "2010": -0.5}[c])
           for c in ["2004", "2007", "2010"] for k, h in enumerate(HZ)],
          section="exploratory", horizon="y1/y5/y10",
          label="fields with windows 2004/2007/2010; slope(2007) - mean slope(2004, 2010)")
    for c in CO["y5"]:
        lstat(f"E:seg:{c}", "seg15", [((c, "y1"), -0.25), ((c, "y5"), 0.25)], section="exploratory",
              horizon="y1->y5", label="fixed institutions within window (y1 and y5); (rho_y5 - rho_y1)/4 per year")
    lstat("E:seg:2007-neighbours", "seg15",
          [((c, h), wt * m) for c, m in (("2004", -0.5), ("2007", 1.0), ("2010", -0.5))
           for h, wt in (("y1", -0.25), ("y5", 0.25))],
          section="exploratory", horizon="y1->y5", label="segment slope(2007) - mean of 2004 and 2010")
    lstat("E:covid_y5", "covid_y5", [((c, "y5"), float(x)) for c, x in zip(["2010", "2013", "2016"], W_EXT)],
          perm=("y5", ["2010", "2013", "2016"], W_EXT), section="exploratory", horizon="y5",
          label="fixed institutions: triple-matched; rho(2016) - [2 rho(2013) - rho(2010)]")
    lstat("E:covid_y10", "gr_y10", [((c, "y10"), float(x)) for c, x in zip(GRW, W_EXT)],
          perm=("y10", GRW, W_EXT), section="exploratory", horizon="y10",
          label="fixed institutions: triple-matched; rho(2010) - [2 rho(2007) - rho(2004)]")
    # placebo windows (post hoc, REVISION NOTE): the Test 1 statistic at every interior window and horizon
    for (h, c), samp in PL_OF.items():
        i = CO[h].index(c)
        cs = CO[h][i - 1:i + 2]
        name = f"PL:{h}:{c}"
        if samp.startswith("gr_"):
            T[name] = T[f"T1:{samp}"]
            continue
        lstat(name, samp, [((x, h), float(wt)) for x, wt in zip(cs, W_MID)], section="exploratory_placebo",
              horizon=h, label=f"fixed institutions: triple-matched ({', '.join(grad_years(x) for x in cs)}); "
                               f"rho(c) - [rho(c-3) + rho(c+3)]/2")
    # identity: the COVID extrapolation statistic equals -2 x the 2016-18 deviation on the same triple
    assert np.isclose(T["T1:covid_y1"]["est"], -2 * T["PL:y1:2016"]["est"], atol=1e-12)
    assert np.allclose(T["T1:covid_y1"]["per_field"], -2 * T["PL:y1:2016"]["per_field"], atol=1e-12)
    # identities (REVISION NOTE 2): the exploratory latest-window extrapolations are -2 x reported second differences
    # on the same institutions: y10 2010-12 = -2 x D_y10 (Test 1 persistence), y5 2016-18 = -2 x D(2013-15, y5)
    for e_, d_ in (("E:covid_y10", "T1:gr_y10"), ("E:covid_y5", "PL:y5:2013")):
        assert T[e_]["fields"] == T[d_]["fields"]
        assert np.allclose(T[e_]["per_field"], -2 * T[d_]["per_field"], atol=1e-12), (e_, d_)
        assert np.isclose(T[e_]["est"], -2 * T[d_]["est"], atol=1e-12)
    for h in HZ:
        oth = [T[f"PL:{h}:{c}"]["est"] for (hh, c) in PL_OF if hh == h and c != GR]
        g = T[f"PL:{h}:{GR}"]["est"]
        rk_abs = 1 + sum(abs(x) > abs(g) for x in oth)
        rec("exploratory_placebo", f"placebo summary {h}", float(np.mean(oth)), float(np.min(oth)), float(np.max(oth)),
            se=float(np.std(oth, ddof=1)) if len(oth) > 1 else np.nan, k=len(oth), horizon=h,
            sample="interior windows other than 2007-09 (triple-matched)",
            note=f"estimate = mean of the other windows' D, lo/hi = min/max, se = their SD; D(2007-09) = {g:.4f}; "
                 f"|D(2007-09)| ranks {rk_abs} of {len(oth) + 1} by size")
    stage("exploratory done")

    # ---- Test 2 ----------------------------------------------------------------------------------
    T2, T2S, t2_ids = {}, {}, set()
    for h in HZ:
        X = t2_sample(L, Ua, h, True)
        X1 = t2_sample(L, Ua1, h, True)
        Xa = t2_sample(L, Ua, h, False)
        t2_ids |= set(L[(L.horizon == h) & L.inst_key.isin(set(X.inst_key))].institution_id)
        T2S[h] = X
        T2[(h, "G")] = wild_reg(X, ["U", "GU"], ["GU"], f"{h}:G", bench=BENCH2)
        T2[(h, "F")] = wild_reg(X, ["U", "FU"], ["FU"], f"{h}:F", bench=BENCH2)
        T2[(h, "GF")] = wild_reg(X, ["U", "GU", "FU"], ["GU", "FU"], f"{h}:GF", bench=BENCH2)
        T2[(h, "Gx")] = wild_reg(X, ["U", "GU"], ["GU"], f"{h}:Gx", gx=True, bench=BENCH2)
        T2[(h, "Gentry")] = wild_reg(X1, ["U", "GU"], ["GU"], f"{h}:Gentry", bench=BENCH2)
        # all institutions + coverage share (pre-specified; REVISION NOTE 2) and + coverage rank (the first run's
        # deviation, same draws as then)
        Xa_rk = t2_sample(L, Ua, h, False, cov="rank")
        T2[(h, "Gall")] = wild_reg(Xa, ["U", "GU", "COV"], ["GU"], f"{h}:Gallsh", bench=BENCH2)
        T2[(h, "Gallrk")] = wild_reg(Xa_rk, ["U", "GU", "COV"], ["GU"], f"{h}:Gall", bench=BENCH2)
        T2[(h, "Gall")]["cov_q"] = np.quantile(Xa.COV, [0.0, 0.5, 0.99, 1.0]).tolist()
        # exploratory (post hoc, REVISION NOTE): G~ x linear window trend (window steps of 3 years) added
        Xt_ = X.assign(GT=X.Gt * (X.grad_cohort.astype(int) - 2001) / 3.0)
        T2[(h, "Gtrend")] = wild_reg(Xt_, ["U", "GU", "GT"], ["GU"], f"{h}:Gtrend", bench=BENCH2)
        stage(f"test 2 {h} done (N={T2[(h, 'G')]['N']}, states={T2[(h, 'G')]['G']})")
    T2LAB = {"G": "primary: U + G~xU", "F": "F~ x U analogue: U + F~xU", "GF": "joint: U + G~xU + F~xU",
             "Gx": "G~ x field x window effects added (within-window cross-state variation only)",
             "Gentry": "U over the y1 earnings years c+1..c+3",
             "Gall": "all institutions (unbalanced) + coverage share (pre-specified)",
             "Gallrk": "all institutions (unbalanced) + coverage rank (deviation used in the first run)",
             "Gtrend": "exploratory (post hoc): G~ x linear window trend added"}
    for (h, spec), r in T2.items():
        sec2 = "test2" if spec != "Gtrend" else "test2_exploratory"
        for c in ("GU", "FU"):
            if c in r and "p_wcr" in r[c]:
                x = r[c]
                x["verdict"] = verdict(x["lo"], x["hi"], x["mde"], BENCH2)
                x["verdict_inv"] = verdict(x["lo_inv"], x["hi_inv"], x["mde"], BENCH2)
                rec(sec2, f"{c}:{spec}", x["b"], x["lo"], x["hi"], x["se"], x["t"], x["p_wcr"], r["fields"],
                    r["N"], sample=T2LAB[spec], horizon=h,
                    note=f"CR1 by state ({r['G']} states); WCR Webb p; CI = unrestricted wild bootstrap-t "
                         f"(|t|95={x['q95']:.3f}); MDE80={x['mde']:.4f}; power vs {BENCH2}={x['power_bench']:.3f}; "
                         f"CR1 p={x['p_cr1']:.4f}")
                rec(sec2 + "_wcr_inverted_ci", f"{c}:{spec}", x["b"], x["lo_inv"], x["hi_inv"], x["se"], x["t"],
                    x["p_wcr"], r["fields"], r["N"], sample=T2LAB[spec], horizon=h,
                    note=f"95% CI = {{b0: WCR p(b0) > 0.05}} (test inversion, same Webb draws; post hoc, REVISION "
                         f"NOTE 2); accepted grid contiguous={x['inv_connected']}; WCR |t|95={x['q95_wcr']:.3f}")
        rec(sec2, f"U:{spec}", r["U"]["b"], se=r["U"]["se"], z=r["U"]["t"], p=r["U"]["p_cr1"], k=r["fields"],
            n=r["N"], sample=T2LAB[spec], horizon=h, note="U main effect, CR1 by state, t(G-1) p")
    # unemployment at entry by window: 51 jurisdictions and the primary y1 sample
    Utab = []
    for c in WINDOWS:
        v = np.array([Ua[(s, int(c))] for s in STATES51])
        v1 = np.array([Ua1[(s, int(c))] for s in STATES51])
        Xs = T2S["y1"][T2S["y1"].grad_cohort == c]
        Utab.append(dict(window=c, grad_years=grad_years(c), mean51=v.mean(), sd51=v.std(ddof=1), min51=v.min(),
                         max51=v.max(), mean51_entry=v1.mean(), mean_y1_rows=float(Xs.U.mean()),
                         sd_y1_rows=float(Xs.U.std(ddof=1))))
        rec("unemployment", f"U window {c}", v.mean(), v.min(), v.max(), se=v.std(ddof=1), n=51,
            sample="51 jurisdictions, unweighted", note=f"graduation years {grad_years(c)}; "
            f"entry-years mean {v1.mean():.3f}")
    Utab = pd.DataFrame(Utab)
    stage("flows: out-of-state employment")
    ids_of = dict(zip(L.inst_key, L.institution_id))
    ne_sets = {s_: {ids_of[k] for j, c_ in enumerate(CELL) if c_["sample"] == s_ for k in c_["df"].inst_key}
               for s_ in ("gr_y1", "fixed17")}
    OS, NE = flows_outstate({"all PSEO institutions": None, "Test 2 sample institutions": t2_ids}, ne_sets)
    for (name, h), x in OS.items():
        rec("flows", f"out-of-state share, {name}", x["out_share"], k=np.nan, n=x["n_inst"], horizon=h,
            sample="PSEO Flows V4.14.1, institution level, all CIP, pooled cohorts",
            note=f"employed graduates {x['grads']:.0f}")
    for (name, c), x in NE.items():
        rec("flows_newengland", f"y1 New England share, window {c}", x["share"], n=x["n_inst"], horizon="y1",
            sample=f"PSEO Flows V4.14.1, institution level, all CIP; institutions of sample {name}",
            note=f"division 1 / national y1 employed graduates; {x['n_inst']} of {x['n_ids']} institutions with "
                 f"both rows released")

    # REVISION NOTE 3: Massachusetts institutions (LEHD frame from 2010) by window and in each sample; md5 of the
    # PSEO Flows files read above
    ma_keys = set(L.loc[L.state == "MA", "inst_key"])
    MA = dict(n=len(ma_keys), win={h: sorted(L.loc[(L.state == "MA") & (L.horizon == h), "grad_cohort"].unique())
                                   for h in HZ},
              samp={s_: len(ma_keys & set().union(*[set(c_["df"].inst_key) for c_ in CELL if c_["sample"] == s_]))
                    for s_ in ("fixed17", "gr_y1", "gr_y5", "gr_y10", "covid_y1", "covid_y5", "career", "all")},
              t2={h: len(ma_keys & set(T2S[h].inst_key)) for h in HZ},
              nh={h: int(L[(L.state == "MA") & (L.horizon == h)].inst_key.nunique()) for h in HZ})
    for h in HZ:
        rec("design", f"Massachusetts institutions with released cells, {h}", float(MA["nh"][h]), horizon=h,
            note="windows: " + ", ".join(grad_years(c) for c in MA["win"][h]))
        rec("design", f"Massachusetts institutions in the Test 2 balanced sample, {h}", float(MA["t2"][h]), horizon=h)
    for s_, n_ in MA["samp"].items():
        rec("design", f"Massachusetts institutions in sample {s_}", float(n_))
    FLM = {p_.name: dict(md5=md5(p_), bytes=p_.stat().st_size)
           for p_ in (PSEOF_NEW, PSEOF_NEW.parent / "version_pseo.txt")}
    src_txt_ = (ROOT / "data" / "raw" / "SOURCES.md").read_text()
    for n_, x_ in FLM.items():
        x_["in_sources"] = x_["md5"] in src_txt_
        rec("design", f"md5 of data/raw/pseo_flows_2026q2/{n_}", float(x_["in_sources"]), n=x_["bytes"],
            note=f"{x_['md5']}; estimate = 1 if this md5 is recorded in data/raw/SOURCES.md")

    # design checks and joint tests into the results file (REVISION NOTE 2)
    for n_, e_ in dchk["free_est"].items():
        rec("design", f"estimable with free period effects: {n_}", float(e_), note="1 = estimable, 0 = not")
    for n_, a_ in dchk["combo_w"].items():
        rec("design", f"cell weights of {n_}", np.nan,
            note="; ".join(f"{grad_years(c)} {h}: {w_:+.4f}" for (c, h), w_ in zip(CELLS17, a_) if abs(w_) > 1e-12))
    pre = __doc__.split("PRE-SPECIFICATION")[1].split("REVISION NOTE (")[0]
    PREMD5 = hashlib.md5(pre.encode()).hexdigest()
    rec("design", "md5 of the PRE-SPECIFICATION block", np.nan, note=PREMD5)
    pd.DataFrame(RES).to_csv(OUT_RES, index=False, float_format="%.10g")
    make_figure(SF)
    write_md(src, ur_meta, repro, binfo, dchk, A, SF, T2, T2S, T2LAB, Utab, OS, NE, PREMD5, MA, FLM)
    stage("done")


# ---------------------------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------------------------
COLH = {"y1": "#2a78d6", "y5": "#eb6834", "y10": "#1baf7a"}


def make_figure(SF):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True)
    off = {"y1": -0.35, "y5": 0.0, "y10": 0.35}
    top = max(T[f"cell:{s}:{c}:{h}"]["thi"] for s in ("fixed17", "all") for c, h in CELLS17)
    bot = min(T[f"cell:{s}:{c}:{h}"]["tlo"] for s in ("fixed17", "all") for c, h in CELLS17)
    for ax, samp, ttl in zip(axes, ["fixed17", "all"],
                             ["(a) fixed institutions (present in all 17 cells)",
                              "(b) all institutions, partial on coverage"]):
        fs = SF[samp]["fields"]
        for c, lab in (("2007", "Great Recession\nwindow: y1 earnings\n2008-10"),
                       ("2019", "COVID window:\ny1 earnings\n2020-22")):
            ax.axvspan(int(c) - 1.4, int(c) + 1.4, color="#bdbdbd", alpha=0.25, lw=0)
            ax.text(int(c), 0.985, lab, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2,
                    color="#555555", linespacing=1.1)
        for h in HZ:
            rs = [T[f"cell:{samp}:{c}:{h}"] for c in CO[h]]
            x = np.array([int(c) for c in CO[h]], float) + off[h]
            y = np.array([r["est"] for r in rs])
            lo, hi = np.array([r["tlo"] for r in rs]), np.array([r["thi"] for r in rs])
            ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="-o", color=COLH[h], ms=4.5, lw=1.5, capsize=2.5,
                        label=f"{h}: earnings in calendar years c+{HY[h]}..c+{HY[h] + 2}")
        ax.axhline(0, color="#333333", lw=0.6)
        ax.set_xticks([int(c) for c in WINDOWS])
        ax.set_xticklabels([grad_years(c) for c in WINDOWS], fontsize=8)
        ax.set_xlabel("graduation window c (PSEO 3-year bachelor's cohort)")
        ax.set_title(f"{ttl}; k={len(fs)} fields", fontsize=10)
        ax.grid(axis="y", color="#e5e5e5", lw=0.6)
        ax.set_xlim(1999.3, 2020.7)
        ax.set_ylim(min(bot - 0.03, -0.05), top + 0.17)
    axes[0].set_ylabel("mean over fields of Spearman(F, p50)\n(panel b: partial given coverage)")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.legend(hd, lb, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(0.01, 0.005, "Error bars: 95% two-way (field, institution) cluster CIs. PSEO V4.14.1; F = Wapman "
             "field prestige; each panel holds the same fields in every cell. Descriptive.", fontsize=7.5,
             color="#555555")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# write-up
# ---------------------------------------------------------------------------------------------
VT = {"supported": "supported (95% CI above 0)", "opposite": "opposite sign (95% CI below 0)",
      "null": "null (95% CI includes 0; MDE at or below the benchmark)",
      "underpowered": "not detected and underpowered (95% CI includes 0; MDE above the benchmark)"}


def ec(r, d=3):
    return f"{fm(r['est'], d)} {ci_s(r, d)}"


def tw_row(q, samp, r, p=None, d=3, kn=None):
    pp = fp(r["p_perm"]) + " (perm.)" if p == "perm" else (fp(r["p"]) + " (two-way z)" if p == "z" else "—")
    return (f"| {q} | {samp} | {fm(r['est'], d)} | {ci_s(r, d)} | {pp} | {fu(r['mde'], d)} | "
            f"{kn if kn is not None else r['k']} |")


def t2_line(r, c="GU", d=4):
    x = r[c]
    return (f"{fm(x['b'], d)} [{fm(x['lo'], d)}, {fm(x['hi'], d)}] (WCR p={fp(x['p_wcr'])}; CI by WCR inversion "
            f"[{fm(x['lo_inv'], d)}, {fm(x['hi_inv'], d)}]; CR1 SE {x['se']:.4f}; MDE {x['mde']:.4f}; power vs "
            f"{BENCH2}: {x['power_bench']:.2f})")


T2SPECS = ("G", "F", "GF", "Gx", "Gentry", "Gall", "Gallrk", "Gtrend")


def t2_disagree(T2) -> list:
    """Rows where the WCR p (< 0.05 or not) and the pre-specified unrestricted bootstrap-t CI (excludes 0 or not)
    disagree."""
    out = []
    for h in HZ:
        for spec in T2SPECS:
            r = T2[(h, spec)]
            for c in ("GU", "FU"):
                if c in r and "p_wcr" in r[c]:
                    x = r[c]
                    if (x["p_wcr"] <= 0.05) != (x["lo"] > 0 or x["hi"] < 0):
                        out.append((h, spec, c, x))
    return out


def write_md(src, ur_meta, repro, binfo, dchk, A, SF, T2, T2S, T2LAB, Utab, OS, NE, PREMD5, MA, FLM):
    L = []
    w = L.append
    g1, g5, g10, cv = T["T1:gr_y1"], T["T1:gr_y5"], T["T1:gr_y10"], T["T1:covid_y1"]
    lin, linz, ling, lina = A["f17:F:lin:r"], A["f17:F:lin:z"], A["f17:G:lin:r"], A["all:parF:lin:r"]
    apc, apca = A["f17:F:APC"], A["all:parF:APC"]
    n_panel = sum(1 for _ in CELL)
    k17, ka = len(SF["fixed17"]["fields"]), len(SF["all"]["fields"])
    th, tc, tm = lin["theta_h"], lin["theta_c"], lin["theta_h_minus_theta_c"]
    curv_names = [k for k in apc if not k.startswith("_")]
    sig = [k for k in curv_names if apc[k]["tlo"] > 0 or apc[k]["thi"] < 0]
    sig_a = [k for k in curv_names if k in apca and (apca[k]["tlo"] > 0 or apca[k]["thi"] < 0)]
    src_txt = ("FRED `<ST>UR` (BLS LAUS, monthly, seasonally adjusted)" if src == "FRED" else
               "the BLS LAUS series `LASST<fips>0000000000003` (monthly, seasonally adjusted; the series FRED "
               "re-publishes as `<ST>UR`), read from the DBnomics mirror because FRED's CSV endpoint could not be "
               "reached from this machine on the download date")

    def cname(k):
        if k in DKCOMBO:
            return f"cohort combination {k} (estimable with free period effects)"
        if k == "kAs":
            return "age curvature on the step-profile age metric (1, 4, 10): A(5) - [2/3 A(1) + 1/3 A(10)]"
        if k == "kA":
            return "age curvature kappa_A = A(5) - [5/9 A(1) + 4/9 A(10)]"
        if k.startswith("DK"):
            c = int(k[2:])
            return f"cohort deviation D_K({grad_years(c)}) vs {grad_years(c - 3)} and {grad_years(c + 3)}"
        k0 = int(k[2:])
        return f"period deviation D_P({k0}-{k0 + 2}) vs {k0 - 3}-{k0 - 1} and {k0 + 3}-{k0 + 5}"

    w("# Coupling dynamics — graduation window, calendar time and career time (PSEO V4.14.1)\n")
    w(f"Script: `scripts/65_coupling_dynamics.py` (seeded; outputs byte-identical on re-run). It imports the "
      f"fixed-cohort engine of `scripts/52` and the two-way variance, weighted-rank and release plumbing of "
      f"`scripts/59` (neither edited) and reads PSEO V4.14.1 (`data/raw/pseo_2026q2/`). Descriptive, not causal. "
      f"Cell panel: `data/interim/coupling_dynamics_panel.csv` ({n_panel} field x window x horizon cells over all "
      f"samples); every number: `data/interim/coupling_dynamics_results.csv`; figure: "
      f"`outputs/figures/coupling_dynamics.png`; annual state unemployment: `data/interim/laus_state_ur_annual.csv` "
      f"(source: {src_txt}). Tests 1 and 2, their samples, sensitivities, benchmarks and verdict rule are written "
      f"in the script's docstring (PRE-SPECIFICATION block, md5 `{PREMD5}`); everything else is labelled "
      f"exploratory. The claim that this block was fixed before any estimate was computed is self-reported: the "
      f"script was not committed (or hashed) before the first estimating run, so the repository cannot confirm the "
      f"timing. Two exploratory analyses were added after a development run had printed estimates (placebo windows "
      f"and a G~ x window-trend version of Test 2; REVISION NOTE); further post hoc additions and corrections made "
      f"after an independent verification are listed in REVISION NOTES 2 and 3 and in the Revisions section at the end. "
      f"The pre-specified tests, samples and verdict rule are unchanged.\n")

    # ---------------- answer
    w("## Answer\n")
    ut = Utab.set_index("window")

    def u_pair(col, c):
        a, b, g = ut.loc[str(int(c) - 3), col], ut.loc[str(int(c) + 3), col], ut.loc[c, col]
        return g, (a + b) / 2, a, b

    def dU(col, c):
        g_, m_, _, _ = u_pair(col, c)
        return g_ - m_

    gg, gm, ga, gb = u_pair("mean51", GR)          # graduation-year unemployment: GR window, neighbour mean
    eg, em, ea, eb = u_pair("mean51_entry", GR)    # y1 earnings-year unemployment

    def say(v, pos, neg, nul):
        return pos if v == "supported" else neg if v == "opposite" else nul

    t2v = {h: T2[(h, "G")]["GU"]["verdict"] for h in HZ}

    def cshort(k):
        if k in DKCOMBO:
            return k
        if k == "kAs":
            return "age curvature (step metric 1, 4, 10)"
        if k == "kA":
            return "age curvature"
        if k.startswith("DK"):
            return f"cohort {grad_years(int(k[2:]))}"
        return f"period {int(k[2:])}-{int(k[2:]) + 2}"

    # placebo windows (exploratory): the Test 1 statistic at the other interior windows
    PLS = {}
    for h in HZ:
        oth = [(c, T[f"PL:{h}:{c}"]) for (hh, c) in PL_OF if hh == h and c != GR]
        gd = T[f"PL:{h}:{GR}"]["est"]
        big = max(oth, key=lambda cr: abs(cr[1]["est"]))
        PLS[h] = dict(oth=oth, rank=1 + sum(abs(r["est"]) > abs(gd) for _, r in oth), n=len(oth) + 1,
                      next=abs(big[1]["est"]), next_c=big[0],
                      lo=min(r["est"] for _, r in oth), hi=max(r["est"] for _, r in oth),
                      nsig=sum(r["tlo"] > 0 or r["thi"] < 0 for _, r in oth))
    p1 = PLS["y1"]
    pl16 = T["PL:y1:2016"]

    def t2_read(h):
        """Sentence reading the primary Test 2 estimate against the within-window sensitivity (pre-specified) and
        the linear-trend spec (exploratory)."""
        x, xg, xt = T2[(h, "G")]["GU"], T2[(h, "Gx")]["GU"], T2[(h, "Gtrend")]["GU"]
        return (f"at {h} the primary estimate {fm(x['b'], 4)} [{fm(x['lo'], 4)}, {fm(x['hi'], 4)}] becomes "
                f"{fm(xg['b'], 4)} [{fm(xg['lo'], 4)}, {fm(xg['hi'], 4)}] (CI {side(xg['lo'], xg['hi'])}) when G~ x "
                f"field x window effects are added (pre-specified sensitivity: only differences between states "
                f"within a window identify beta) and {fm(xt['b'], 4)} [{fm(xt['lo'], 4)}, {fm(xt['hi'], 4)}] (CI "
                f"{side(xt['lo'], xt['hi'])}) with a G~ x linear window trend instead (exploratory)")

    t2sig = [h for h in HZ if t2v[h] in ("supported", "opposite")]
    # revision-2 quantities used in several places
    free, apk5, apst, apx = A["f17:F:free"], A["f17:F:APC:k5"], A["f17:F:APC:step"], A["f17:F:APC:x"]
    wcur = apc["_meta"]["wald"]["curvature"]
    wpl, wpw = free["_meta"]["wald"]["plane"], free["_meta"]["wald"]["pw"]
    wk5, wst = free["_meta"]["wald"]["k5"], free["_meta"]["wald"]["step"]
    dku = [k for k in curv_names if k.startswith("DK")]
    mv = max(dku, key=lambda k: abs(apk5[k]["est"] - apc[k]["est"]))
    mv_s = max(dku, key=lambda k: abs(apst[k]["est"] - apc[k]["est"]))
    dUg, dUe = gg - gm, eg - em
    ny1sig = sum(T[f"D:fixed17:y1:{c}"]["tlo"] > 0 or T[f"D:fixed17:y1:{c}"]["thi"] < 0 for c in CO["y1"][1:-1])

    def peq(x):
        return "p < 0.001" if x < 0.001 else f"p = {x:.3f}"

    def wtxt(wd):
        return (f"two-way Wald chi2({wd['q']}) = {wd['W']:.1f}, {peq(wd['p'])}, F({wd['q']}, {wd['df2']}) version "
                f"{peq(wd['pF'])}, largest eigen-direction {wd['dom']:.0%} of the statistic"
                + (f", CGM matrix not positive definite (fallback in {wd['nfb']} direction(s))" if wd["nfb"] else "")
                + f"; conservative covariance chi2({wd['q']}) = {wd['Wc']:.1f}, {peq(wd['pc'])}, F version "
                f"{peq(wd['pcF'])}")

    def wverdict(wd):
        ps = [wd["p"], wd["pF"], wd["pc"], wd["pcF"]]
        return ("rejects under every version" if max(ps) < 0.05 else
                "does not reject under any version" if min(ps) >= 0.05 else "is not robust across versions")

    def plac_txt(p_):
        return (f"|D(2007-09)| is the largest of the {p_['n']} interior y1 windows (next largest "
                f"{fu(p_['next'])}, window {grad_years(p_['next_c'])}), but with {p_['n']} windows the placebo rank p "
                f"cannot fall below 1/{p_['n']}: rank p = {p_['rank']}/{p_['n']} = {p_['rank'] / p_['n']:.2f}"
                if p_["rank"] == 1 else
                f"|D(2007-09)| ranks {p_['rank']} of {p_['n']} interior y1 windows by size (placebo rank p = "
                f"{p_['rank']}/{p_['n']} = {p_['rank'] / p_['n']:.2f})")

    g1_qual = (f"{g1['verdict']} by the pre-set rule; qualifier: the two-way CI does not contain window-to-window "
               f"variation, and {plac_txt(p1)}; the entry-unemployment contrast behind D is {dUg:+.2f} percentage "
               f"points by graduation years (the definition Test 2 uses) and {dUe:+.2f} by y1 earnings years")
    w(f"**Key question: how does within-field coupling move across graduation windows, calendar time and career "
      f"time, and is it higher for graduates who entered a weak labour market?** "
      f"Coupling rises along both time axes. On the 17-cell surface of fixed institutions (k={lin['theta_h']['k']} "
      f"fields) the slope within a graduation window (per year since graduation) is {ec(th, 4)}/yr and the slope "
      f"across windows at a fixed horizon is {ec(tc, 4)}/yr. The two cannot be split into age, period and cohort "
      f"trends (t = c + h); if the period and cohort trends are both >= 0, the career-time slope lies in "
      f"[{fm(tm['est'], 4)}, {fm(th['est'], 4)}]/yr"
      + ("" if tc["est"] >= 0 else " (the cross-window slope is negative, which contradicts the sign assumption; "
                                   "the bound does not apply)")
      + f". Around these trends coupling moves up and down from window to window: at y1, {ny1sig} of "
      f"{len(CO['y1']) - 2} model-free cohort-axis deviations on the fixed-institution surface have CIs that exclude "
      f"0 (Answer 6), while a joint test that the 17 cell means lie on a plane in (years since graduation, window) "
      f"{wverdict(wpl)} ({wtxt(wpl)}; restriction-free, post hoc; Answer 5e). "
      f"Pre-specified Test 1: the window that entered the labour market in the Great Recession (graduated 2007-09, "
      f"y1 earnings 2008-10) "
      + say(g1["verdict"], "has higher", "has lower", "does not have detectably different")
      + f" y1 coupling than the mean of its two neighbouring windows (D = {ec(g1)}, k={g1['k']} fields; verdict: "
      f"{g1_qual}; the pre-specified prediction was D > 0). The same statistic at the other interior y1 windows "
      f"(placebo, exploratory) ranges from {fm(p1['lo'])} to {fm(p1['hi'])} ({p1['nsig']} of {len(p1['oth'])} CIs "
      f"exclude 0)."
      + (" So the Great Recession window's y1 coupling departs from its neighbours by about as much as other windows "
         "depart from theirs, and D says little about recession entry as such."
         if p1["next"] >= 0.75 * abs(g1["est"]) else "")
      + f" At y5 D = {ec(g5)} ({g5['verdict']}) and at y10 D = {ec(g10)} ({g10['verdict']}). For the COVID "
      f"window (2019-21, y1 earnings 2020-22), which has neighbours on one side only, the deviation from the "
      f"linear extrapolation is E = {ec(cv)} ({cv['verdict']}; weaker test). E is identically -2 times the "
      f"deviation of the 2016-18 window from the mean of 2013-15 and 2019-21 on the same institutions "
      f"({ec(pl16)}), so a low 2019-21 window and a high 2016-18 window cannot be told apart. "
      f"Pre-specified Test 2 (verdict per horizon): the interaction of the academia-wide rank G with the "
      f"unemployment rate of the institution's state in the graduation years is {fm(T2[('y1', 'G')]['GU']['b'], 4)} "
      f"per percentage point at y1 ({t2v['y1']}), {fm(T2[('y5', 'G')]['GU']['b'], 4)} at y5 ({t2v['y5']}) and "
      f"{fm(T2[('y10', 'G')]['GU']['b'], 4)} at y10 ({t2v['y10']})"
      + ("" if not t2sig else "; " + "; ".join(t2_read(h) for h in t2sig))
      + f". The age, cohort and period curvature contrasts of the APC meta-regression are not identified by the "
      f"design: with a free effect per earnings window none of the {len(curv_names)} is estimable (Answer 5), and "
      f"they are estimated only under the assumption that the period effect is piecewise linear between knots. "
      f"Under the default knots {len(sig)} of {len(curv_names)} marginal two-way CIs exclude 0"
      + (f" ({', '.join(cshort(k) for k in sig)})" if sig else "")
      + f" (the joint test {wverdict(wcur)}: {wtxt(wcur)}; descriptive, no verdict), and the values move with the "
      f"knots "
      f"(knots on the y5 earnings windows: {cshort(mv)} {fm(apc[mv]['est'])} -> {fm(apk5[mv]['est'])}).\n")

    sens = [("Fisher z", "z"), ("inverse-variance-weighted, Fisher z", "ivw"), ("Spearman(G, p50)", "G"),
            ("all institutions, partial on coverage", "allcov"), ("institutions present in all 17 cells", "fixed17")]

    def senstxt(base):
        return "; ".join(f"{lab} {ec(T[f'T1:{base}:{k}'])} (k={T[f'T1:{base}:{k}']['k']})" for lab, k in sens)

    w(f"1. **Test 1 — Great Recession entry window at y1 (pre-specified).** Fixed institutions: in each field the "
      f"institutions with released earnings in all three windows (k={g1['k']} fields, {g1['n_inst']} "
      f"institutions, cell n {g1['n_min']}-{g1['n_max']}). D = rho(2007-09 window, y1) - [rho(2004-06, y1) + "
      f"rho(2010-12, y1)]/2 = {fm(g1['est'])}, two-way 95% CI {ci_s(g1)} (z={g1['tz']:.2f}); "
      f"institution-clustered cohort-label permutation p={fp(g1['p_perm'])}; D > 0 in {g1['n_pos']}/{g1['k']} "
      f"fields. Minimum detectable effect (80% power) {fu(g1['mde'])}; power against the benchmark |D| = "
      f"{BENCH1}: {g1['power']:.2f}. Verdict: **{VT[g1['verdict']]}** by the pre-set rule -- qualifier on the "
      f"verdict: the two-way CI covers field and institution sampling, not window-to-window variation; "
      f"{plac_txt(p1)}; and the entry-unemployment contrast behind D is small and definition-dependent: relative to "
      f"the mean of its two neighbours the window's state unemployment is {dUg:+.2f} percentage points in the "
      f"graduation years (the definition Test 2 uses) and {dUe:+.2f} in the y1 earnings years (Caveats). Per "
      f"percentage point of the y1-earnings-year contrast D is {fm(g1['est'] / dUe)} "
      f"[{fm(min(g1['tlo'] / dUe, g1['thi'] / dUe))}, {fm(max(g1['tlo'] / dUe, g1['thi'] / dUe))}] (post hoc "
      f"scaling by a constant); by graduation years the "
      f"contrast has the other sign, so no per-point reading is stable. Sensitivities: {senstxt('gr_y1')}. The "
      f"APC meta-regression splits the y1 deviation into a cohort part D_K(2007-09) = {ec(apc['DK2007'])} and a "
      f"period part D_P(2008-10) = {ec(apc['DP2008'])} (Fisher z; fixed institutions); this split exists only under "
      f"the piecewise-linear period restriction (with a step-function period profile D_K(2007-09) = "
      f"{ec(apst['DK2007'])} and D_P(2008-10) = {ec(apst['DP2008'])}; with knots on the y5 earnings windows "
      f"D_K(2007-09) = {ec(apk5['DK2007'])}); with a free effect per earnings window neither part is estimable "
      f"(Answer 5). Placebo (exploratory, post hoc): the same statistic on its own triple-matched institutions at "
      f"the other interior y1 windows, with each window's unemployment contrast to its neighbours (graduation years "
      f"/ y1 earnings years, 51-jurisdiction means), is "
      + "; ".join(f"{grad_years(c)} {ec(r)} (k={r['k']}; dU {dU('mean51', c):+.2f} / {dU('mean51_entry', c):+.2f})"
                  for c, r in p1["oth"])
      + f"; Great Recession window {fm(g1['est'])} (dU {dUg:+.2f} / {dUe:+.2f}). Answer 7 has y5 and y10.")
    w(f"2. **Persistence (pre-specified).** Same contrast at y5 (earnings 2012-14): D = {fm(g5['est'])} "
      f"{ci_s(g5)}, permutation p={fp(g5['p_perm'])}, k={g5['k']}, MDE {fu(g5['mde'])}, verdict "
      f"**{g5['verdict']}**; at y10 (earnings 2017-19): D = {fm(g10['est'])} {ci_s(g10)}, permutation "
      f"p={fp(g10['p_perm'])}, k={g10['k']}, MDE {fu(g10['mde'])}, verdict **{g10['verdict']}**. At y10 the later "
      f"neighbour (2010-12 window) has its y10 earnings in 2020-22, so D_y10 also contains any COVID-period "
      f"change of that neighbour (caveat fixed in advance). Sensitivities y5: {senstxt('gr_y5')}. "
      f"Sensitivities y10: {senstxt('gr_y10')}.")
    w(f"3. **COVID window (pre-specified, weaker).** E = rho(2019-21, y1) - [2 rho(2016-18, y1) - rho(2013-15, y1)] "
      f"= {fm(cv['est'])} {ci_s(cv)} (k={cv['k']}, {cv['n_inst']} institutions; permutation p={fp(cv['p_perm'])}; "
      f"MDE {fu(cv['mde'])}; power vs {BENCH1}: {cv['power']:.2f}); verdict **{cv['verdict']}**. It is a linear "
      f"extrapolation from two earlier windows, so any curvature of the cross-window trend enters E, and its SE is "
      f"larger than that of a two-sided deviation (weights 1, -2, 1). On the same institutions E = -2 x D(2016-18), "
      f"where D(2016-18) = rho(2016-18) - [rho(2013-15) + rho(2019-21)]/2 = {ec(pl16)} (identity checked field by "
      f"field in the script): E < 0 can come from a low 2019-21 window or from a high 2016-18 window. "
      f"Sensitivities: {senstxt('covid_y1')}.")
    def frag(h):
        if t2v[h] not in ("supported", "opposite"):
            return ""
        xg, xt = T2[(h, "Gx")]["GU"], T2[(h, "Gtrend")]["GU"]
        return (f" by the pre-set rule, and fragile: with G~ x field x window effects (pre-specified sensitivity) the "
                f"estimate is {fm(xg['b'], 4)} [{fm(xg['lo'], 4)}, {fm(xg['hi'], 4)}] with MDE {xg['mde']:.3f}, "
                f"{xg['mde'] / BENCH2:.1f} times the benchmark, and with a G~ x linear window trend (post hoc) "
                f"{fm(xt['b'], 4)} [{fm(xt['lo'], 4)}, {fm(xt['hi'], 4)}]")

    t2txt = "; ".join(f"{h}: {t2_line(T2[(h, 'G')])}, {T2[(h, 'G')]['G']} states, {T2[(h, 'G')]['fields']} fields, "
                      f"N={T2[(h, 'G')]['N']} -> **{t2v[h]}**{frag(h)}" for h in HZ)
    dis = t2_disagree(T2)
    dis_txt = ("; ".join(f"{h}, {T2LAB[sp]}, {'G~' if c == 'GU' else 'F~'} x U: WCR p = {fp(x['p_wcr'])} but the "
                         f"unrestricted bootstrap-t CI [{fm(x['lo'], 4)}, {fm(x['hi'], 4)}] "
                         f"{side(x['lo'], x['hi'])} (CI by WCR inversion [{fm(x['lo_inv'], 4)}, {fm(x['hi_inv'], 4)}]; "
                         f"95% quantile of |t*| {x['q95']:.2f} unrestricted vs {x['q95_wcr']:.2f} restricted)"
                         for h, sp, c, x in dis) if dis else "none")
    f2txt = "; ".join(f"{h}: {t2_line(T2[(h, 'F')], 'FU')}" for h in HZ)
    w(f"4. **Test 2 — entry conditions at the institution level (pre-specified).** Rank-normalised p50 earnings on "
      f"field x window and institution x field effects, the institution-state unemployment rate averaged over the "
      f"graduation years (U) and G~ x U (G~ = centred percentile rank of the academia-wide rank within field; the "
      f"slope of the outcome on G~ is close to Spearman(G, earnings), so beta is roughly the change in that "
      f"coupling per percentage point of U). Fixed institutions (present in every window of the horizon). beta_h: "
      f"{t2txt}. F~ x U analogue: {f2txt}. Within-window cross-state variation only (G~ x field x window effects "
      f"added; pre-specified sensitivity): " + "; ".join(f"{h} {t2_line(T2[(h, 'Gx')])}" for h in HZ)
      + ". U over the y1 earnings years c+1..c+3 (pre-specified sensitivity): "
      + "; ".join(f"{h} {t2_line(T2[(h, 'Gentry')])}" for h in HZ)
      + ". All institutions with the coverage share as covariate (pre-specified sensitivity): "
      + "; ".join(f"{h} {t2_line(T2[(h, 'Gall')])}" for h in HZ)
      + "; the same with the within-(field, window) rank of coverage instead (the first run's deviation): "
      + "; ".join(f"{h} {t2_line(T2[(h, 'Gallrk')])}" for h in HZ)
      + ". G~ x linear window trend added (exploratory, post hoc): "
      + "; ".join(f"{h} {t2_line(T2[(h, 'Gtrend')])}" for h in HZ)
      + f". With {T2[('y1', 'G')]['G']} state clusters the unrestricted wild bootstrap-t distribution is much wider "
      f"than the restricted one, so the pre-specified CI and the WCR p can disagree; the verdict uses the "
      f"pre-specified CI, and the CI obtained by inverting the WCR test (post hoc; it agrees with the WCR p by "
      f"construction) is shown next to it. Rows where the WCR p and the pre-specified CI disagree: {dis_txt}. "
      f"Benchmark |beta| = {BENCH2} per percentage point. Full table in Appendix D.")
    l10, l10a = A["f17:F:lin:r:t2010"], A["all:parF:lin:r:t2010"]
    freeG, freeA = A["f17:G:free"], A["all:parF:free"]
    k5n = [k for k in apk5 if not k.startswith("_") and k.startswith("DP")]

    def dkline(fit):
        return "; ".join(f"{cshort(k)} {fm(fit[k]['est'])}" for k in ["kA", "kAs"] + dku if k in fit)

    w(f"5. **What the APC model identifies and what it does not (Fisher z, fixed institutions, k={apc['_meta']['K']} "
      f"fields x 17 cells).** (a) *Linear trends.* The linear age, period and cohort trends are not identified: "
      f"t = c + h is an exact linear function of the other two indices (checked: design rank {dchk['pw_rank']} of "
      f"{dchk['pw_cols']} columns; the age-1-vs-10 contrast is not estimable). A linear surface z ~ theta_h h + "
      f"theta_c c gives theta_h = a + b and theta_c = b + g for true age, period and cohort slopes a, b, g; on the "
      f"correlation scale theta_h = {ec(th, 4)}/yr and theta_c = {ec(tc, 4)}/yr, so if b >= 0 and g >= 0 the age "
      f"(career-time) slope a lies in [theta_h - theta_c, theta_h] = [{fm(tm['est'], 4)}, {fm(th['est'], 4)}]/yr "
      f"(lower end CI {ci_s(tm, 4)}). Coverage-controlled all-institution surface (k={apca['_meta']['K']}): theta_h "
      f"{ec(lina['theta_h'], 4)}, theta_c {ec(lina['theta_c'], 4)}, bound "
      f"[{fm(lina['theta_h_minus_theta_c']['est'], 4)}, {fm(lina['theta_h']['est'], 4)}]. Spearman(G, p50), fixed "
      f"institutions: theta_h {ec(ling['theta_h'], 4)}, theta_c {ec(ling['theta_c'], 4)}. Cells whose earnings years "
      f"are all >= 2010, i.e. after Massachusetts entered the LEHD frame (post hoc; {l10['theta_h']['n_cells'] // l10['theta_h']['k']} "
      f"of 17 cells): fixed institutions theta_h {ec(l10['theta_h'], 4)}, theta_c {ec(l10['theta_c'], 4)}, bound "
      f"[{fm(l10['theta_h_minus_theta_c']['est'], 4)}, {fm(l10['theta_h']['est'], 4)}]; coverage-controlled "
      f"all-institution surface theta_h {ec(l10a['theta_h'], 4)}, theta_c {ec(l10a['theta_c'], 4)}. "
      f"(b) *Curvature is not identified by the design either.* With a free effect per earnings window "
      f"({dchk['n_periods']} windows, {dchk['n_single']} of them in one cell only) the design has "
      f"{dchk['free_cols']} columns of rank {dchk['free_rank']}, i.e. it is saturated in the 17 cell means, and "
      f"none of the {len(curv_names)} curvature contrasts is estimable (checked one by one: "
      + ", ".join(f"{cshort(k)} {'estimable' if dchk['free_est'][k] else 'not estimable'}" for k in curv_names)
      + f"; the period contrasts in their free-period form on the y1 earnings grid). Of the "
      f"{dchk['rank_all']}-dimensional span of these contrasts a part of dimension {dchk['dim_all']} is estimable "
      f"without a restriction. It is spanned by the {dchk['n_dev']} model-free cohort-axis deviations at y1 and y10 "
      f"of Answer 6 (rank {dchk['rank_dev']}; each is a cohort plus a period second difference) together with the "
      f"cohort-only combinations DK2004 - DK2013, DK2007 - DK2016 and sums of three adjacent cohort deviations (for "
      f"example DK2004 + DK2007 + DK2010; {dchk['dim_dk']} dimensions among the cohort contrasts alone, "
      f"{dchk['dim_ka_dp']} among the age and period contrasts alone, spanned by the period-only combinations "
      f"{', '.join(dchk['dp_combos'][:2])} and the sums of three adjacent period deviations such as "
      f"{dchk['dp_combos'][2]}, checked; post hoc, REVISION NOTE 3); the age curvature enters "
      + ("some" if dchk["ka_in"] else "none") + " of these estimable combinations. The model-free age-axis curvature "
      f"within a window (Answer 6) is estimable, but under the additive model it equals kappa_A plus the period "
      f"curvature P(c+5) - [5/9 P(c+1) + 4/9 P(c+10)] over the window's own earnings years (checked for the "
      f"{len(dchk['agecurv'])} windows): A(5) enters only the y5 cells, and each of their earnings windows "
      f"({', '.join(str(t) for t in dchk['y5per'])}) occurs in that one cell only, so adding any constant to A(5) "
      f"and subtracting it from the period effects of these {len(dchk['y5per'])} windows leaves every cell mean "
      f"unchanged (checked). In every estimable function the weight on A(5) therefore equals the total weight on "
      f"those period effects, and kappa_A, which has weight 1 on A(5) and none on them, cannot be separated from "
      f"the period curvature (post hoc, REVISION NOTE 3). Every age, cohort and period curvature contrast is "
      f"therefore "
      f"identified only under the functional-form assumption that the period effect is piecewise linear between "
      f"knots at the y1 earnings windows ({', '.join(str(k) for k in KNOTS)}; design rank {dchk['pw_rank']} of "
      f"{dchk['pw_cols']}). (c) *Under that assumption (descriptive, no verdict):* age curvature kappa_A = "
      f"{ec(apc['kA'])}; "
      + "; ".join(f"{cname(k)} {ec(apc[k])}" for k in curv_names if k != "kA")
      + f". {len(sig)} of {len(curv_names)} marginal CIs exclude 0 (at 5% about {0.05 * len(curv_names):.1f} would "
      f"by chance if all were 0); the joint test {wverdict(wcur)}: {wtxt(wcur)}. The two-way (CGM) covariance of "
      f"{wcur['q']} contrasts is the jackknife over {wcur['K']} fields plus the covariance of the shared-draw "
      f"replicates minus that of the independent-draw replicates; estimated this way it is "
      + ("not positive definite " if wcur["nfb"] else "")
      + f"(minimum eigenvalue {wcur['mineig']:.2g}"
      + (", replaced by the one-way fallback" if wcur["nfb"] else "")
      + f") and its small eigenvalues rest on the difference of two noisy bootstrap covariances, so the statistic "
      f"depends on that subtraction; the conservative covariance (jackknife plus shared-draw only, positive definite "
      f"and never smaller than the CGM one) is the safer reading. The CIs and the Wald tests do "
      f"not contain window-to-window variation, and adjacent second differences are negatively correlated "
      f"(Caveats). On the coverage-controlled "
      f"all-institution surface {len(sig_a)} of {len(curv_names)} marginal CIs exclude 0"
      + (f" ({', '.join(cshort(k) for k in sig_a)})" if sig_a else "")
      + f"; jointly {wtxt(apca['_meta']['wald']['curvature'])}. (d) *Sensitivity to the restriction (post hoc).* "
      f"Knots moved to the y5 earnings windows ({', '.join(str(k) for k in KNOTS_Y5)}; the 2002 window "
      f"extrapolated; rank {dchk['alt']['k5']['rank']} of {dchk['alt']['k5']['cols']}): {dkline(apk5)}; period "
      + "; ".join(f"D_P({int(k[2:])}-{int(k[2:]) + 2}) {fm(apk5[k]['est'])}" for k in k5n)
      + f" (joint {wtxt(apk5['_meta']['wald']['curvature'])}). Step-function period profile (each y5 earnings "
      f"window assigned to the y1-grid window sharing two of its three years; rank {dchk['alt']['step']['rank']} of "
      f"{dchk['alt']['step']['cols']}; here the age curvature on 1, 5, 10 is not estimable at all, because the "
      f"profile places the y5 earnings one year earlier, and only the curvature on the metric 1, 4, 10 is): "
      f"{dkline(apst)}; period "
      + "; ".join(f"D_P({int(k[2:])}-{int(k[2:]) + 2}) {fm(apst[k]['est'])}" for k in apst
                  if not k.startswith("_") and k.startswith("DP"))
      + f" (joint {wtxt(apst['_meta']['wald']['curvature'])}). Default knots for comparison: {dkline(apc)}. "
      f"(e) *Restriction-free quantities (post hoc).* From a saturated cell-means GLS (same field random intercepts "
      f"and two-way inference): "
      + "; ".join(f"{n} = {ec(free[n])} (under the default restriction {fm(apx[n]['est'])})" for n in DKCOMBO)
      + f". Lack of fit of the 17 cell means (Wald against the saturated fit): the test of a plane in (h, c) "
      f"{wverdict(wpl)} ({wtxt(wpl)}); the test of the testable implications of the piecewise-linear profile with "
      f"the default knots {wverdict(wpw)} ({wtxt(wpw)}), with knots on the y5 windows {wverdict(wk5)} "
      f"({wtxt(wk5)}), and of the step function {wverdict(wst)} ({wtxt(wst)})"
      + ("; so the data cannot choose between these period profiles, although they give different curvature values "
         "(d)" if max(x["p"] for x in (wpw, wk5, wst)) >= 0.05 and min(x["p"] for x in (wpw, wk5, wst)) >= 0.05
         else "")
      + ". Spearman(G, p50): plane "
      f"{wtxt(freeG['_meta']['wald']['plane'])}; all institutions, partial on coverage: plane "
      f"{wtxt(freeA['_meta']['wald']['plane'])}. These cohort combinations, the period-only combinations named in (b) "
      f"(each a sum or difference of the model-free deviations of Answer 6 and the cohort combinations; not "
      f"estimated separately), the lack-of-fit tests and the model-free surface quantities of Answer 6 (linear "
      f"combinations of cell means) are the only restriction-free quantities of the APC analysis. Under an additive "
      f"model each cohort-axis deviation of Answer 6 equals a cohort plus a "
      f"period second difference and each within-window age-axis curvature equals kappa_A plus a period curvature; "
      f"the data cannot separate the two parts of either.")
    sf = SF["fixed17"]
    dl = "; ".join(f"{h} {grad_years(CO[h][i])}: {ec(T[f'D:fixed17:{h}:{CO[h][i]}'])}"
                   for h in HZ for i in range(1, len(CO[h]) - 1))
    kl = "; ".join(f"{grad_years(c)}: {ec(T[f'kA:fixed17:{c}'])}" for c in CO["y10"])
    w(f"6. **Model-free surface (fixed institutions, k={k17} fields; correlation scale).** Cohort-axis deviation "
      f"of each interior window from the mean of its two neighbours at the same horizon: {dl}. Under an additive "
      f"APC model each of these equals a cohort second difference plus a period second difference. Age curvature "
      f"within window, rho(y5) - [5/9 rho(y1) + 4/9 rho(y10)]: {kl}. Under an additive APC model each of these "
      f"equals the age curvature kappa_A plus the period curvature P(c+5) - [5/9 P(c+1) + 4/9 P(c+10)] over the "
      f"window's own earnings years; the two parts cannot be separated (Answer 5b). Mean coupling per cell with "
      f"CIs: Appendix A and the figure.")
    el = "; ".join(f"{grad_years(c)} {ec(T[f'E:slope:{c}'], 4)} (k={T[f'E:slope:{c}']['k']})"
                   for c in ["2001", "2004", "2007", "2010"])
    sg = "; ".join(f"{grad_years(c)} {ec(T[f'E:seg:{c}'], 4)} (k={T[f'E:seg:{c}']['k']})" for c in CO["y5"])
    w(f"7. **Exploratory (not pre-specified).** Career slope per year (OLS over y1/y5/y10) by window, fixed "
      f"institutions within window: {el}. 2007-09 minus the mean of the other three windows "
      f"{ec(T['E:slope:2007-others'], 4)} (k={T['E:slope:2007-others']['k']}); minus the mean of its neighbours "
      f"{ec(T['E:slope:2007-neighbours'], 4)}. y1->y5 segment slope by window: {sg}; 2007-09 minus neighbours "
      f"{ec(T['E:seg:2007-neighbours'], 4)}. Latest windows against the extrapolation of their two predecessors: "
      f"y5 2016-18 (earnings 2021-23) {ec(T['E:covid_y5'])} (k={T['E:covid_y5']['k']}, permutation "
      f"p={fp(T['E:covid_y5']['p_perm'])}); y10 2010-12 (earnings 2020-22) {ec(T['E:covid_y10'])} "
      f"(k={T['E:covid_y10']['k']}, permutation p={fp(T['E:covid_y10']['p_perm'])}). Neither is separate evidence "
      f"about a COVID dip: on the same institutions the y5 number is identically -2 x the placebo deviation "
      f"D(2013-15, y5) = {ec(T['PL:y5:2013'])}, and the y10 number is identically -2 x the pre-specified persistence "
      f"statistic D_y10 = {ec(g10)} (Answer 2) (both identities checked field by field). A value of either sign can "
      f"come from the latest window or from the middle one; at y10 a low 2007-09 window and a high 2010-12 window "
      f"(earnings 2020-22) give the same number. In the APC model (fixed "
      f"institutions) the period deviation D_P(2017-19) = {ec(apc['DP2017'])}: the 2017-19 earnings period lies "
      f"{'above' if apc['DP2017']['est'] > 0 else 'below'} the line through 2014-16 and 2020-22, which is the same "
      f"number as 2020-22 lying {'below' if apc['DP2017']['est'] > 0 else 'above'} the extrapolation from 2014-16 "
      f"and 2017-19 by twice that amount; it depends on the piecewise-linear period restriction. Placebo windows (post hoc; the "
      f"Test 1 statistic on triple-matched institutions at every interior window): "
      + "; ".join(f"{h}: " + ", ".join(f"{grad_years(c)} {ec(T[f'PL:{h}:{c}'])}" for (hh, c) in PL_OF if hh == h)
                  + f" (|D(2007-09)| ranks {PLS[h]['rank']} of {PLS[h]['n']})" for h in HZ) + ".")
    w(f"8. **Reproduction.** The scripts/52 balanced fixed-cohort panel rebuilt here gives the all-field career "
      f"slope {repro['slope']:.6f}/yr (k={repro['k']}), equal to scripts/59's V4.14.1 value "
      f"{repro['ref']:.6f} (k={repro['ref_k']}).\n")

    # ---------------- key numbers
    w("## Key numbers\n")
    w("Two-way CI = estimate +/- 1.96 SE with SE^2 = V_field + V_institution - V_field x institution "
      "(Cameron-Gelbach-Miller; scripts/59), from the between-field variance (jackknife over fields for the "
      "meta-regression) and two institution bootstrap draws (1000 each: one shared by all fields, one independent "
      "per field). perm. = institution-clustered cohort-label permutation (1000). Test 2: CI = unrestricted wild "
      "cluster bootstrap-t by state, p = wild cluster restricted bootstrap (Webb weights, 9999). MDE = minimum "
      "detectable effect at 80% power, 5% two-sided. Sample labels say whether institutions are held fixed or "
      "coverage is controlled.\n")
    w("| quantity | sample / spec | estimate | 95% CI | p | MDE (80%) | k fields |")
    w("|---|---|---|---|---|---|---|")
    for base, lab in (("gr_y1", "Test 1: D, GR window, y1"), ("gr_y5", "Test 1 persistence: D, y5"),
                      ("gr_y10", "Test 1 persistence: D, y10"), ("covid_y1", "Test 1 COVID: E (extrapolation), y1")):
        r = T[f"T1:{base}"]
        w(tw_row(lab, r["label"], r, "perm"))
        w(tw_row(lab + " (two-way z p)", r["label"], r, "z"))
        for sl_, k in sens:
            rs = T[f"T1:{base}:{k}"]
            w(tw_row(f"{lab}, {sl_}", rs["label"], rs, "perm" if np.isfinite(rs["p_perm"]) else "z"))
    w(f"| Test 1: D per percentage point of the window's y1-earnings-year unemployment contrast (post hoc scaling "
      f"by the constant {dUe:+.2f}) | {g1['label']} | {fm(g1['est'] / dUe)} | "
      f"[{fm(min(g1['tlo'] / dUe, g1['thi'] / dUe))}, {fm(max(g1['tlo'] / dUe, g1['thi'] / dUe))}] | — | — | "
      f"{g1['k']} |")
    for c in [c for (hh, c) in PL_OF if hh == "y1"]:
        w(f"| unemployment contrast of window {grad_years(c)} vs the mean of its neighbours, graduation years / y1 "
          f"earnings years (percentage points) | 51 jurisdictions, unweighted (Appendix C) | {dU('mean51', c):+.2f} / "
          f"{dU('mean51_entry', c):+.2f} | — | — | — | — |")
    SAMPLAB = {"Gall": "coverage share as covariate", "Gallrk": "coverage rank as covariate"}
    for h in HZ:
        for spec in T2SPECS:
            r = T2[(h, spec)]
            for c in ("GU", "FU"):
                if c in r and "p_wcr" in r[c]:
                    x = r[c]
                    w(f"| Test 2: {'G~' if c == 'GU' else 'F~'} x U, {h} | {T2LAB[spec]}; "
                      f"{SAMPLAB.get(spec, 'fixed institutions')}; {r['G']} states, "
                      f"N={r['N']} | {fm(x['b'], 4)} | [{fm(x['lo'], 4)}, {fm(x['hi'], 4)}] | {fp(x['p_wcr'])} (WCR) "
                      f"| {x['mde']:.4f} | {r['fields']} |")
                    w(f"| Test 2: {'G~' if c == 'GU' else 'F~'} x U, {h}, CI by WCR inversion (post hoc) | "
                      f"{T2LAB[spec]}; {SAMPLAB.get(spec, 'fixed institutions')}; {r['G']} states, N={r['N']} | "
                      f"{fm(x['b'], 4)} | [{fm(x['lo_inv'], 4)}, {fm(x['hi_inv'], 4)}] | {fp(x['p_wcr'])} (WCR) | "
                      f"— | {r['fields']} |")
    for key, lab in (("f17:F:lin:r", "fixed institutions (17 cells), Spearman(F), correlation scale"),
                     ("f17:F:lin:z", "fixed institutions (17 cells), Spearman(F), Fisher z"),
                     ("f17:G:lin:r", "fixed institutions (17 cells), Spearman(G), correlation scale"),
                     ("all:parF:lin:r", "all institutions, partial on coverage, correlation scale"),
                     ("f17:F:lin:r:t2010", "fixed institutions, cells with earnings years >= 2010 (post hoc), "
                                           "Spearman(F), correlation scale"),
                     ("all:parF:lin:r:t2010", "all institutions, partial on coverage, cells with earnings years >= "
                                              "2010 (post hoc), correlation scale")):
        for c, cl in (("theta_h", "linear surface: theta_h (within-window, per yr since graduation)"),
                      ("theta_c", "linear surface: theta_c (cross-window at fixed horizon, per yr)"),
                      ("theta_h_minus_theta_c", "linear surface: theta_h - theta_c (lower end of career bound)")):
            r = A[key][c]
            w(tw_row(cl, lab, r, "z", 4))
    for key, lab in (("f17:F:APC", "APC M_APC (piecewise-linear period, default knots; curvature identified only "
                                   "under this restriction), fixed institutions, Fisher z of Spearman(F)"),
                     ("f17:F:AC", "M_AC (no period effect), fixed institutions"),
                     ("f17:F:AP", "M_AP (no cohort effect), fixed institutions"),
                     ("f17:G:APC", "M_APC (piecewise-linear period, default knots), fixed institutions, "
                                   "Spearman(G)"),
                     ("f17:F:APC:r", "M_APC (piecewise-linear period, default knots), fixed institutions, "
                                     "correlation scale"),
                     ("all:parF:APC", "M_APC (piecewise-linear period, default knots), all institutions, partial on "
                                      "coverage"),
                     ("f17:F:APC:x", "M_APC (default knots), fixed institutions, Fisher z (post hoc)"),
                     ("f17:F:APC:k5", "M_APC with knots on the y5 earnings windows, fixed institutions, Fisher z "
                                      "(post hoc)"),
                     ("f17:F:APC:step", "M_APC with a step-function period profile, fixed institutions, Fisher z "
                                        "(post hoc)"),
                     ("f17:F:free", "saturated cell means (no period restriction), fixed institutions, Fisher z "
                                    "(post hoc)"),
                     ("f17:G:free", "saturated cell means, fixed institutions, Spearman(G), Fisher z (post hoc)"),
                     ("all:parF:free", "saturated cell means, all institutions, partial on coverage, Fisher z "
                                       "(post hoc)")):
        for c in [k for k in A[key] if not k.startswith("_") and (key != "f17:F:APC:x" or k in DKCOMBO)]:
            w(tw_row(cname(c), lab, A[key][c], "z"))
        for wn, wd in A[key]["_meta"]["wald"].items():
            wl = {"curvature": f"joint Wald: all {wd['q']} curvature contrasts = 0",
                  "plane": "lack of fit: cell means on a plane in (h, c)",
                  "pw": "lack of fit: piecewise-linear period, default knots (testable part)",
                  "k5": "lack of fit: piecewise-linear period, knots on y5 windows (testable part)",
                  "step": "lack of fit: step-function period (testable part)"}[wn]
            w(f"| {wl}, two-way (CGM) covariance (post hoc; estimate = chi2({wd['q']}) statistic) | {lab} | "
              f"{wd['W']:.2f} | — | {fp(wd['p'])} (chi2); {fp(wd['pF'])} (F({wd['q']}, {wd['df2']})) | — | "
              f"{wd['K']} |")
            w(f"| {wl}, conservative covariance (post hoc; estimate = chi2({wd['q']}) statistic) | {lab} | "
              f"{wd['Wc']:.2f} | — | {fp(wd['pc'])} (chi2); {fp(wd['pcF'])} (F({wd['q']}, {wd['df2']})) | — | "
              f"{wd['K']} |")
    for samp in ("fixed17", "all"):
        for h in HZ:
            for i in range(1, len(CO[h]) - 1):
                r = T[f"D:{samp}:{h}:{CO[h][i]}"]
                w(tw_row(f"surface: cohort-axis deviation, {h}, window {grad_years(CO[h][i])}", r["label"], r, "z"))
        for c in CO["y10"]:
            r = T[f"kA:{samp}:{c}"]
            w(tw_row(f"surface: age curvature, window {grad_years(c)}", r["label"], r, "z"))
    for name in [k for k in T if k.startswith("E:")]:
        r = T[name]
        lab = {"E:slope:2007-others": "exploratory: career slope 2007-09 minus other windows",
               "E:slope:2007-neighbours": "exploratory: career slope 2007-09 minus neighbours",
               "E:seg:2007-neighbours": "exploratory: y1->y5 segment 2007-09 minus neighbours",
               "E:covid_y5": "exploratory: y5 2016-18 vs extrapolation",
               "E:covid_y10": "exploratory: y10 2010-12 vs extrapolation"}.get(name)
        if lab is None:
            kind, c = name.split(":")[1:]
            lab = (f"exploratory: career slope, window {grad_years(c)}" if kind == "slope"
                   else f"exploratory: y1->y5 segment slope, window {grad_years(c)}")
        w(tw_row(lab, r["label"], r, "perm" if np.isfinite(r["p_perm"]) else "z", 4))
    for (h, c), samp in PL_OF.items():
        if samp.startswith("gr_"):
            continue
        r = T[f"PL:{h}:{c}"]
        w(tw_row(f"exploratory placebo: D at window {grad_years(c)}, {h}", r["label"], r, "z"))
    for (name, h), x in OS.items():
        w(f"| share of employed graduates working outside the institution's state, {h} | {name}; PSEO Flows V4.14.1, "
          f"institution level, all CIP, pooled cohorts | {x['out_share']:.3f} | — | — | — | {x['n_inst']} inst. |")
    for (name, c), x in NE.items():
        w(f"| share of y1 employed graduates working in New England (division 1, contains MA), window "
          f"{grad_years(c)} (post hoc) | institutions of the "
          f"{ {'gr_y1': 'Test 1 y1 (triple-matched)', 'fixed17': 'fixed-17'}.get(name, name)} sample; PSEO Flows "
          f"V4.14.1, institution level, all CIP; "
          f"{x['n_inst']} of {x['n_ids']} institutions with both rows released | {fu(x['share'], 4)} | — | — | — | "
          f"{x['n_inst']} inst. |")
    w(f"| reproduction: all-field career slope | scripts/52 balanced fixed-cohort panel | {repro['slope']:+.6f} | — "
      f"| — | — | {repro['k']} |\n")

    # ---------------- method
    w("## Method\n")
    w("- **Definitions (PSEO documentation, https://lehd.ces.census.gov/data/pseo_documentation.html, fetched "
      "2026-09-24).** \"For the Bachelor's degree level, the graduation cohorts are three-year cohorts, e.g. "
      "2001-2003; 2004-2006; 2007-2009; 2010-2012; 2013-2015; 2016-2018; 2019-2021.\" \"For all post-secondary "
      "graduates, the first year post-graduation is defined as the first calendar year following their graduation "
      "year. So for a student who graduates in May of 2005, year one begins in January of 2006, year five in "
      "January 2010, etc.\" \"Earnings are total annual earnings for attached workers from all jobs, converted to "
      "2023 dollars using the CPI-U.\" Window c (graduation years c..c+2) therefore has y1 earnings in calendar "
      "years c+1..c+3, y5 in c+5..c+7 and y10 in c+10..c+12; t = c + h is the first earnings year.")
    w(f"- **Cells.** `load_er_pseo('undergrad', h, fields=FIELDS66, grad_cohort=[...])` on V4.14.1 via scripts/59's "
      f"release switch; scripts/52's coverage = PSEO earners / IPEDS graduates over the same CIP-4 rows. Windows: "
      f"y1 {', '.join(CO['y1'])}; y5 {', '.join(CO['y5'])}; y10 {', '.join(CO['y10'])} (17 cells per field). F = "
      f"-(Wapman field rank), G = -(Wapman academia-wide rank), both fixed over time (DYNAMIC_LEADLAG_PILOT: "
      f"prestige is slow-moving). Coupling = Spearman(F, p50) across institutions; cells need n >= {NMIN}. "
      f"Samples: `fixed17` (institutions present in all 17 cells of the field; {k17} fields); triple-matched "
      f"samples for each test (institutions present in the three windows of the contrast); `career` (scripts/52's "
      f"balanced y1/y5/y10 panel per window); `seg15` (y1 and y5 of one window); `all` (every institution with "
      f"released earnings and a coverage share; coupling as partial Spearman given same-horizon coverage, as in "
      f"scripts/52; {ka} fields have all 17 cells). Sampling variance of Fisher z: Bonett-Wright (1 + r^2/2)/(n-3) "
      f"(n-4 for the partial) and the institution-bootstrap variance (both in the panel file).")
    w(f"- **Inference for means over fields.** Unweighted mean over fields of a within-field linear combination of "
      f"cells. Two-way (field, institution) cluster variance as in scripts/59 ({NBOOT} institution multinomial draws "
      f"shared by all cells, {NBOOT} independent per field; {binfo['NI']} institutions; degenerate replicates "
      f"redrawn: {binfo['redrawn_shared']} shared, {binfo['redrawn_ind']} field-level). Permutation: every "
      f"institution's within-window earnings ranks are permuted across the three windows by one of the six "
      f"orderings, shared by all fields in which it appears ({NPERM}); null = the three windows are exchangeable "
      f"within institution. A field-label permutation is not used: the statistics are means over fields of "
      f"within-field contrasts and there is no second group of fields to exchange labels with.")
    w(f"- **APC meta-regression.** z_fj = mu + u_f + A(h) + K(c) + P(t) + e on the fixed-institution cells, GLS "
      f"with Bonett-Wright variances plus residual heterogeneity tau^2 and field random intercepts u_f (REML; "
      f"fixed-17 Spearman(F): s2_field={apc['_meta']['s2']:.3g}, tau2={apc['_meta']['t2']:.3g}, median sampling "
      f"variance {apc['_meta']['vmed']:.3g}). Period effect piecewise linear in the first earnings year with knots at "
      f"the y1 earnings windows {', '.join(str(k) for k in KNOTS)}; the y5 earnings windows (2006, 2009, ..., 2021) "
      f"are interpolated (2021 extrapolated from 2017-2020). With a free effect per earnings window the design has "
      f"{dchk['free_cols']} columns of rank {dchk['free_rank']}, i.e. it is saturated in the 17 cell means "
      f"({dchk['n_single']} of {dchk['n_periods']} earnings windows occur in one cell only), and none of the "
      f"curvature contrasts is estimable ({sum(not v_ for v_ in dchk['free_est'].values())} of "
      f"{len(dchk['free_est'])} checked: age curvature, the five cohort and the five period second differences); "
      f"under the piecewise-linear restriction all of them are estimable (checked). Every curvature contrast is "
      f"therefore identified by the assumed functional form of the period profile, not by the design. The only "
      f"restriction-free quantities are linear combinations of the 17 cell means: the model-free surface quantities "
      f"of Answer 6 (each cohort-axis deviation is a cohort plus a period second difference, each within-window "
      f"age-axis curvature is kappa_A plus a period curvature), the cohort-only combinations of Answer 5(e), the "
      f"period-only combinations of Answer 5(b) and the lack-of-fit tests (REVISION NOTES 2 and 3). Post hoc "
      f"sensitivities (REVISION NOTE 2): knots on the y5 earnings windows ({', '.join(str(k) for k in KNOTS_Y5)}) and "
      f"a step-function period profile, both of rank {dchk['alt']['k5']['rank']}; a saturated cell-means fit "
      f"(M_free: one mean per cell, same field random intercepts and inference) gives the restriction-free cohort "
      f"combinations and the lack-of-fit tests (the contrasts of cell means that vanish under a model: "
      f"{dchk['N_lin'].shape[0]} for a plane in (h, c), {dchk['N_pw'].shape[0]} for the piecewise-linear profile). "
      f"Normalisations A(1) = K(2001) = "
      f"P(first knot) = P(second knot) = 0 fix levels and the unidentified linear trend; reported contrasts do not "
      f"depend on them. Two-way CI for each contrast c'b: V_field = delete-one-field jackknife (variance components "
      f"fixed), V_institution and V_field x institution = the GLS operator applied to the two bootstrap draws. Joint "
      f"Wald tests (post hoc) use the matrix version, V = V_jack + Cov(shared-draw replicates) - Cov(independent-draw "
      f"replicates), with the scripts/59 fallback (larger one-way variance) in any eigen-direction with a "
      f"non-positive eigenvalue, and a conservative version V_jack + Cov(shared-draw replicates), each with a "
      f"chi2(q) reference and a Hotelling-type F(q, K - q) version. The linear surface (M_lin) "
      f"replaces A, K, P by theta_h h + theta_c c; with t = c + h and linear true effects a h + b t + g c, theta_h = "
      f"a + b and theta_c = b + g exactly.")
    w(f"- **Test 2.** Separately by horizon: y = within-(field, window) centred percentile rank of p50 (equal to the "
      f"rank of log p50); absorbed effects field x window and institution x field (Frisch-Waugh within field); "
      f"regressors U_sc (mean of the calendar-year means of the monthly seasonally adjusted rate over graduation "
      f"years c..c+2 of the institution's state) and G~ x U. Clusters = states; CR1 with the small-sample factor "
      f"counting absorbed effects not nested in states; WCR bootstrap-t p and unrestricted bootstrap-t CI with Webb "
      f"six-point weights ({NWILD} each; the CI is the pre-specified basis of the verdict). Post hoc: the 95% CI "
      f"obtained by inverting the WCR test, {{b0 : p(b0) > 0.05}} with the same draws; under H0 beta = b0 the "
      f"restricted bootstrap sample is linear in b0, so p(b0) is exact in closed form, evaluated on 1201 grid points "
      f"over beta +/- 12 SE and refined by bisection. The all-institution sensitivity adds the coverage share "
      f"(pre-specified; the first run used its within-(field, window) rank, kept as a labelled deviation). MDE = "
      f"(unrestricted bootstrap |t| 95% quantile + 0.842) x SE; power against "
      f"|beta| = {BENCH2} from the same normal approximation. Institution state = `institution_state` of "
      f"`pseo_all_institutions.csv` (V4.14.1). Unemployment: {src_txt}; files in `data/raw/laus_state_ur/` "
      f"(provenance in `data/raw/SOURCES.md`, section 10c); calendar-year means use years with all 12 monthly "
      f"values (the series have no value for October 2025, a year not used here).")
    w(f"- **Out-of-state work.** PSEO Flows V4.14.1 (`data/raw/pseo_flows_2026q2/pseof_all.csv.gz`): bachelor's, "
      f"institution level, all CIP, pooled cohorts, national row, all industries; 1 - employment-weighted mean "
      f"in-state share (the scripts/32 formula).\n")

    # ---------------- caveats
    w("## Caveats\n")
    os1 = OS[("Test 2 sample institutions", "y1")]["out_share"]
    os5 = OS[("Test 2 sample institutions", "y5")]["out_share"]
    os10 = OS[("Test 2 sample institutions", "y10")]["out_share"]
    osa = OS[("all PSEO institutions", "y5")]["out_share"]
    w("- **Coverage drift.** States join the PSEO coalition at different dates (e.g. MD and NM from 2011, TN from "
      "2010; `version_pseo.txt` as reported in PSEO_REFRESH_RESULT.md), so the set of institutions changes across windows. Every cross-window comparison here holds "
      "institutions fixed within field (triple-matched, fixed-17, balanced Test 2) or controls same-horizon "
      "coverage (the `all` sample); fixing institutions shrinks the sample and restricts it to the long-standing "
      "coalition states, and the coverage control is linear in ranks and does not correct selection within "
      "institution.")
    ne1 = {c: NE[("gr_y1", c)] for c in NE_COHORTS}
    nef = {c: NE[("fixed17", c)] for c in NE_COHORTS}
    ma_first = min(c for h in HZ for c in MA["win"][h])
    ma_fixed = [MA["samp"][k] for k in ("fixed17", "gr_y1", "gr_y5", "gr_y10")] + [MA["t2"][h] for h in HZ]
    ma_txt = (f"The {MA['n']} Massachusetts institutions among those analysed (PSEO cells with Wapman prestige) have "
              f"released cells only from the "
              f"{grad_years(ma_first)} window on (" + "; ".join(f"{h}: " + ", ".join(grad_years(c) for c in MA["win"][h])
                                                                for h in HZ if MA["win"][h])
              + f") and are in {'none' if not any(ma_fixed) else 'some'} of the fixed samples that span 2010 "
              f"(fixed-17 {MA['samp']['fixed17']}; Test 1 triples y1/y5/y10 {MA['samp']['gr_y1']}/"
              f"{MA['samp']['gr_y5']}/{MA['samp']['gr_y10']}; balanced Test 2 samples y1/y5/y10 {MA['t2']['y1']}/"
              f"{MA['t2']['y5']}/{MA['t2']['y10']}); the COVID triples, whose earnings years all lie after 2010, hold "
              f"{MA['samp']['covid_y1']} (y1) and {MA['samp']['covid_y5']} (y5), and the coverage-controlled "
              f"all-institution sample holds {MA['samp']['all']} of them from that window on (post hoc count, "
              f"REVISION NOTE 3).")
    w(f"- **LEHD earnings frame before 2010 (added in revision 2).** The PSEO documentation ({PSEO_DOC}, fetched "
      f"2026-09-24): \"Availability of state UI data in the LEHD system varies by state. LEHD has data for only about "
      f"ten states in the early 1990s, expanding rapidly to 40 states by the late 1990s, with Massachusetts being the "
      f"last state entering the data in 2010.\" A graduate who worked only in Massachusetts in a pre-2010 earnings "
      f"year has no UI earnings in the frame for that year, fails the attachment rule and drops out of the earnings "
      f"tabulation. This is a "
      f"period-specific measurement change, separate from institution coverage: it touches the y1 cells of the "
      f"2001-03, 2004-06 and 2007-09 windows (earnings 2002-2010) and the y5 cells of the 2001-03 and 2004-06 "
      f"windows (2006-2011), so it loads on the cross-window drift theta_c (hence on the lower end of the career "
      f"bound), on the period curvature, and on the Test 1 y1 contrast, whose 2004-06 and 2007-09 windows precede "
      f"2010 and whose 2010-12 window does not. {ma_txt} Size (PSEO Flows, post hoc): the share "
      f"of y1 employed graduates working in New England (census division 1, which contains Massachusetts; "
      f"institutions with both rows released) among the Test 1 y1 institutions is "
      + ", ".join(f"{fu(ne1[c]['share'], 4)} ({grad_years(c)}, {ne1[c]['n_inst']} inst.)" for c in NE_COHORTS)
      + "; among the fixed-17 institutions "
      + ", ".join(f"{fu(nef[c]['share'], 4)} ({grad_years(c)})" for c in NE_COHORTS)
      + f". From the 2010-12 window on (y1 earnings with Massachusetts in the frame) the New England share is an "
      f"upper bound on the share working in Massachusetts, hence on the share the pre-2010 frame can drop; the "
      f"change from the 2007-09 to the 2010-12 window ({ne1['2010']['share'] - ne1['2007']['share']:+.4f} for the "
      f"Test 1 institutions) shows the step at the frame change, up to the trend in moves to New England. For these "
      f"institutions the bound is small: at most {max(ne1[c]['share'] for c in ('2010', '2013', '2016')):.2%} of "
      f"employed y1 graduates (Test 1 institutions, windows 2010-12 to 2016-18), so the frame change can move a "
      f"window's institution medians only through a small, selected group. The linear surface restricted to cells "
      f"whose earnings years are all >= 2010 is in Answer 5(a).")
    w("- **Pre-specification timing is self-reported.** The script was neither committed nor hashed before its first "
      "estimating run (it was first committed only after the estimating runs behind the first write-up and REVISION "
      "NOTE 2), so that Tests 1-2 were fixed before any "
      "estimate cannot be checked "
      f"from the repository; the md5 of the PRE-SPECIFICATION block as run here is `{PREMD5}`.")
    w(f"- **Three-year windows blur timing.** Mean state unemployment (51 jurisdictions, unweighted; Appendix C) in "
      f"the graduation years is {ga:.2f} (2004-06), {gg:.2f} (2007-09) and {gb:.2f} (2010-12), and in the y1 "
      f"earnings years {ea:.2f}, {eg:.2f} and {eb:.2f}. Against the mean of its two neighbours the Great Recession "
      f"window is therefore {gg - gm:+.2f} percentage points by graduation years and {eg - em:+.2f} by entry years: "
      f"the neighbour contrast of Test 1 compares a severe-entry window with a pair of which one member (2010-12) "
      f"entered an equally weak or weaker market, which shrinks the entry-condition difference the test rests on. "
      f"The same applies to COVID (window 2019-21: 2019 graduates entered in 2020, 2021 graduates in 2022).")
    w("- **Only two shocks.** Test 1 rests on one recession and one pandemic window; any other event specific to "
      "those windows (e.g. changes in PSEO coverage or in post-graduate enrolment) is not separable from the entry "
      "shock. The Test 1 statistic mixes a cohort and a period second difference; the split in answer 1 depends on "
      "the piecewise-linear period restriction.")
    w(f"- **State of institution is not state of work.** In PSEO Flows, {os1:.0%} (y1), {os5:.0%} (y5) and "
      f"{os10:.0%} (y10) of the Test 2 sample institutions' employed graduates work outside the institution's "
      f"state ({osa:.0%} at y5 over all PSEO institutions). U measures the state where the institution is, not "
      f"the market the graduates enter, which attenuates beta.")
    w("- **Few clusters and state-level confounds.** Test 2 identifies beta from state x window variation in U; "
      "states differ in other ways that move with U (public-university funding, industry mix), so beta is an "
      "association. Webb wild bootstrap is used because the number of states is small.")
    w("- **Earnings are for attached workers.** PSEO drops graduates who earn less than the annual equivalent of "
      "full-time work at the federal minimum wage or have two or more quarters without earnings in the reference "
      "year (PSEO documentation). In a weak labour market fewer "
      "graduates pass this bar, and the share may differ by institution status, so a window's coupling mixes "
      "earnings differences with selection into attachment.")
    sgn = ["+" if T[f"D:fixed17:y1:{c}"]["est"] > 0 else "-" for c in CO["y1"][1:-1]]
    alt = all(a != b for a, b in zip(sgn, sgn[1:]))
    w(f"- **Window-level movement.** On the fixed-17 surface the signs of the y1 cohort-axis deviations for windows "
      f"{', '.join(grad_years(c) for c in CO['y1'][1:-1])} are {' '.join(sgn)}"
      + (" (alternating)" if alt else "") + ". A shift specific to one window and shared by fields makes the "
      "second differences of adjacent windows negatively correlated (in the direction of alternating signs), so a "
      "single window's deviation is best read against the deviations at the other windows (placebo windows), not "
      "only against 0. The two-way CIs cover field and institution sampling, not window-to-window variation.")
    w("- **Ecological aggregates.** Institution medians of CIP-4 cells aggregated to fields; no individual "
      "earnings histories, no tails beyond p50, and PSEO covers graduates with earnings in covered employment "
      "(selection into employment and into graduate school differs across windows).")
    w("- **Multiple comparisons and small cells.** The APC and surface sections report many curvature contrasts; "
      "a few CIs excluding 0 are expected by chance. Many cells have 15-30 institutions.")
    w("- **Descriptive.** None of these estimates is causal; earnings differences across institutions mix "
      "selection and value added.\n")

    # ---------------- appendices
    w("## Appendix A — mean coupling per cell (figure data)\n")
    for samp in ("fixed17", "all"):
        w(f"{SF[samp]['label']}; k={len(SF[samp]['fields'])} fields with all 17 cells. Estimate [two-way 95% CI].\n")
        w("| window | " + " | ".join(f"{h} (earnings)" for h in HZ) + " |")
        w("|---|---|---|---|")
        for c in WINDOWS:
            cells = []
            for h in HZ:
                if c in CO[h]:
                    r = T[f"cell:{samp}:{c}:{h}"]
                    cells.append(f"{ec(r)} ({earn_years(c, h)})")
                else:
                    cells.append("—")
            w(f"| {grad_years(c)} | " + " | ".join(cells) + " |")
        w("")
    w("## Appendix B — Test 1 per field (triple-matched institutions, Spearman(F, p50))\n")
    w("| field | n y1 | D y1 | n y5 | D y5 | n y10 | D y10 | n COVID | E COVID y1 |")
    w("|---|---|---|---|---|---|---|---|---|")
    allf = sorted(set(g1["fields"]) | set(g5["fields"]) | set(g10["fields"]) | set(cv["fields"]))
    for f in allf:
        cells = []
        for r, samp, c0 in ((g1, "gr_y1", ("2004", "y1")), (g5, "gr_y5", ("2004", "y5")),
                            (g10, "gr_y10", ("2004", "y10")), (cv, "covid_y1", ("2013", "y1"))):
            if f in r["fields"]:
                cells += [str(len(CELL[J[(samp, f) + c0]]["df"])), fm(float(r["per_field"][r["fields"].index(f)]))]
            else:
                cells += ["—", "—"]
        w(f"| {LAB.get(f, f)} | " + " | ".join(cells) + " |")
    w("")
    w("## Appendix C — unemployment rate at entry by window\n")
    w("U = mean over the graduation years of the calendar-year mean of the monthly seasonally adjusted state rate; "
      "'entry years' = mean over the y1 earnings years c+1..c+3.\n")
    w("| window | mean over 51 jurisdictions | SD | min | max | mean, entry years | mean in Test 2 y1 sample (rows) "
      "| SD in sample |")
    w("|---|---|---|---|---|---|---|---|")
    for _, r in Utab.iterrows():
        w(f"| {r.grad_years} | {r.mean51:.2f} | {r.sd51:.2f} | {r.min51:.2f} | {r.max51:.2f} | {r.mean51_entry:.2f} "
          f"| {r.mean_y1_rows:.2f} | {r.sd_y1_rows:.2f} |")
    w("")
    w("## Appendix D — Test 2, all specifications\n")
    w("CI (pre-specified) = unrestricted wild bootstrap-t (symmetric), the basis of the verdict; CI (WCR inversion) = "
      "{b0 : WCR p(b0) > 0.05}, same draws, post hoc; |t*| 95% = 95% quantile of the absolute bootstrap t under the "
      "unrestricted / restricted (b0 = 0) bootstrap. Verdict columns apply the pre-set rule to each CI.\n")
    w("| horizon | specification | coefficient | estimate | 95% CI (pre-specified) | 95% CI (WCR inversion) | WCR p | "
      "|t*| 95% (unrestr. / restr.) | verdict (pre-spec. CI / WCR CI) | CR1 SE | CR1 p | MDE | power vs 0.01 | "
      "U main effect (CR1 SE) | states | fields | institutions | N | SD of U after effects |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h in HZ:
        for spec in T2SPECS:
            r = T2[(h, spec)]
            for c in ("GU", "FU"):
                if c in r and "p_wcr" in r[c]:
                    x = r[c]
                    w(f"| {h} | {T2LAB[spec]} | {'G~' if c == 'GU' else 'F~'} x U | {fm(x['b'], 4)} | "
                      f"[{fm(x['lo'], 4)}, {fm(x['hi'], 4)}] | [{fm(x['lo_inv'], 4)}, {fm(x['hi_inv'], 4)}]"
                      f"{'' if x['inv_connected'] else ' (hull; not contiguous)'} | {fp(x['p_wcr'])} | "
                      f"{x['q95']:.2f} / {x['q95_wcr']:.2f} | {x['verdict']} / {x['verdict_inv']} | {x['se']:.4f} | "
                      f"{fp(x['p_cr1'])} | {x['mde']:.4f} | {x['power_bench']:.2f} | {fm(r['U']['b'], 4)} "
                      f"({r['U']['se']:.4f}) | {r['G']} | {r['fields']} | {r['n_inst']} | {r['N']} | "
                      f"{r['sd_Ut']:.3f} |")
    w("")
    cq = {h: T2[(h, "Gall")]["cov_q"] for h in HZ}
    w("Coverage share in the all-institution samples (min / median / 99th percentile / max): " + "; ".join(
        f"{h} {cq[h][0]:.3f} / {cq[h][1]:.3f} / {cq[h][2]:.3f} / {cq[h][3]:.3f}" for h in HZ)
      + ". The share is PSEO earners / IPEDS graduates over the same CIP-4 rows (scripts/52) and exceeds 1 in some "
        "cells; it enters linearly, as pre-specified.\n")
    w("## Appendix E — provenance and design checks\n")
    w(f"- Unemployment files ({len(ur_meta)}; source {src}; `data/raw/laus_state_ur/`): first "
      f"{ur_meta['first'].min()}, last {ur_meta['last'].max()}; md5 (first 8 hex digits) of each file: "
      + ", ".join(f"{r.series} {r.md5[:8]}" for r in ur_meta.itertuples()) + ".")
    w(f"- Earnings windows (first year t) and cells per window: " + ", ".join(
        f"{t}: {n}" for t, n in dchk["nobs"].items()) + ".")
    w(f"- Variance components (REML): " + "; ".join(
        f"{k}: s2_field={A[k]['_meta']['s2']:.3g}, tau2={A[k]['_meta']['t2']:.3g}, K={A[k]['_meta']['K']}, "
        f"cells={A[k]['_meta']['N']}" for k in A) + ". A tau2 below 1e-6 is at the lower bound of the search "
        f"(no residual heterogeneity beyond the Bonett-Wright sampling variance and the field intercept); the GLS "
        f"weights then rest on the sampling variances, and the reported CIs do not use the model variance (two-way "
        f"jackknife and bootstrap).")
    pi = binfo["placebo"]
    w(f"- Placebo triples (exploratory): {pi['n_cells']} cells over {pi['NI']} institutions, own institution draws "
      f"({NBOOT} shared, {NBOOT} per field; degenerate replicates redrawn: {pi['redrawn_shared']} shared, "
      f"{pi['redrawn_ind']} field-level), so the draws behind the pre-specified statistics are those of a run "
      f"without them.")
    w(f"- Estimability of each curvature contrast with a free effect per earnings window (design {dchk['free_cols']} "
      f"columns, rank {dchk['free_rank']}): " + ", ".join(
        f"{k} {'estimable' if v_ else 'not estimable'}" for k, v_ in dchk["free_est"].items())
      + f". Under the default piecewise-linear profile (rank {dchk['pw_rank']} of {dchk['pw_cols']}): " + ", ".join(
        f"{k} {'estimable' if v_ else 'not estimable'}" for k, v_ in dchk["pw_est"].items())
      + f". Estimable part of the span of all {dchk['n_all']} contrasts without restriction: {dchk['dim_all']} "
        f"dimensions (cohort contrasts alone {dchk['dim_dk']}; age curvature and period contrasts alone "
        f"{dchk['dim_ka_dp']}).")
    w("- Restriction-free cohort combinations as weights on the 17 cell means (Fisher z; from the free-period "
      "design, unique because it has full row rank): " + " | ".join(
        f"{n}: " + ", ".join(f"{grad_years(c)} {h} {a_:+.3f}" for (c, h), a_ in zip(CELLS17, dchk["combo_w"][n])
                             if abs(a_) > 1e-12) for n in DKCOMBO) + ".")
    w(f"- Lack-of-fit contrasts (orthonormal bases of the cell-mean contrasts that vanish under each model): plane "
      f"{dchk['N_lin'].shape[0]}, default piecewise-linear {dchk['N_pw'].shape[0]}, knots on y5 windows "
      f"{dchk['alt']['k5']['N'].shape[0]}, step function {dchk['alt']['step']['N'].shape[0]}. Two-way (CGM) "
      f"covariance in each joint test, minimum eigenvalue / directions with fallback / share of W in the largest "
      f"direction: " + "; ".join(
        f"{k}/{wn} {wd['mineig']:.2g} / {wd['nfb']} / {wd['dom']:.0%}" for k in A
        for wn, wd in A[k]["_meta"]["wald"].items()) + ".")
    ma_all = ", ".join(f"{h} {MA['nh'][h]}" for h in HZ)
    w(f"- Massachusetts institutions with released cells, by horizon: {ma_all}; windows as in the Caveats (post hoc, "
      f"REVISION NOTE 3).")
    fl_src = all(x_["in_sources"] for x_ in FLM.values())
    w("- PSEO Flows files read for the out-of-state and New England shares (downloaded for scripts/61; download "
      "record in FLOWS_PLACEMENT_RESULT.md; md5 " + ("recorded" if fl_src else "not recorded")
      + " in data/raw/SOURCES.md): " + "; ".join(
        f"`data/raw/pseo_flows_2026q2/{n_}` {x_['bytes']:,} bytes, md5 `{x_['md5']}`" for n_, x_ in FLM.items()) + ".")
    w(f"- md5 of the docstring's PRE-SPECIFICATION block as run: `{PREMD5}`.\n")

    # ---------------- revisions
    w("## Revisions (after an independent verification of the first full run)\n")
    w("Each item was checked against the data or the design before it was changed; every number below is printed by "
      "the script from this run. The pre-specified statistics of Tests 1 and 2 (estimates, CIs, p-values, verdicts) "
      "are unchanged; only their reading, the listed post hoc additions and the Test 2 all-institution sensitivity "
      "(now as pre-specified) changed.\n")
    dk_mv = "; ".join(f"{cshort(k)} {fm(apc[k]['est'])} (knots on y5 windows {fm(apk5[k]['est'])}, step "
                      f"{fm(apst[k]['est'])})" for k in dku) + (
        f"; age curvature {fm(apc['kA']['est'])} (knots on y5 windows {fm(apk5['kA']['est'])}; under the step profile "
        f"only the curvature on the metric 1, 4, 10 is estimable: {fm(apst['kAs']['est'])})")
    w(f"1. **APC identification (major).** The first write-up said the curvature contrasts are identified 'under a "
      f"piecewise-linear period profile' and that, with a free effect per earnings window, the age curvature is not "
      f"estimable, which read as if the cohort and period second differences were identified by the design. They "
      f"are not: with free period effects none of the {len(curv_names)} contrasts is estimable (printed in Answer 5 "
      f"and Appendix E; the check existed in the script but was not printed), so every one of them, including the "
      f"cohort contrasts whose CIs exclude 0, rests on the functional-form restriction. Changed: Answer 5, the Method "
      f"and the key-question paragraph say so; added (post hoc) the restriction-free cohort combinations, the "
      f"lack-of-fit tests and two alternative period profiles. The values move with the profile: {dk_mv}.")
    w(f"2. **No verdict for the curvature contrasts.** They had no pre-set hypothesis or rule; they are now "
      f"descriptive, with a joint Wald test, which {wverdict(wcur)} (default knots: {wtxt(wcur)}). The two-way (CGM) "
      f"covariance matrix of many contrasts is ill-conditioned when estimated from {wcur['K']} fields and a "
      f"difference of two bootstrap covariances (Answer 5c, Appendix E), so a conservative covariance is reported "
      f"next to it. The same holds for the lack-of-fit test of a plane through the 17 cell means, whose CGM "
      f"statistic is carried {wpl['dom']:.0%} by one direction ({wverdict(wpl)}).")
    w(f"3. **Two different slopes.** theta_h = {fm(th['est'], 4)} is the GLS slope of the linear surface on the "
      f"fixed-17 cells (k={th['k']}); the reproduction of scripts/59 is the unweighted mean of per-field career "
      f"slopes on the balanced scripts/52 panel ({repro['slope']:.6f}, k={repro['k']}; assert passes). The "
      f"write-up (Answer 8) already kept them apart; the run summary had mixed them.")
    w("4. **Test 2 verdict per horizon.** " + "; ".join(f"{h}: {t2v[h]}" for h in HZ)
      + f" (pre-set rule on the pre-specified CI); the y10 verdict now carries its fragility on the verdict line "
        f"(Answer 4).")
    pv = "; ".join(f"{h}: {T2[(h, 'G')]['GU']['verdict']} / {T2[(h, 'G')]['GU']['verdict_inv']}" for h in HZ)
    w(f"5. **WCR p vs the pre-specified CI.** The first write-up said the two 'can disagree near the 5% line'. They "
      f"disagree by more than that in {len(dis)} row(s), now listed with numbers in Answer 4: {dis_txt}. The CI from "
      f"inverting the WCR test (agrees with the WCR p by construction) is added to Answer 4 and Appendix D; the "
      f"verdict stays with the pre-specified CI. Primary verdicts, pre-specified CI / WCR-inverted CI: {pv}.")
    w(f"6. **Latest-window extrapolations are not separate evidence.** y10 2010-12 vs extrapolation "
      f"({fm(T['E:covid_y10']['est'])}) = -2 x D_y10 ({fm(g10['est'])}); y5 2016-18 vs extrapolation "
      f"({fm(T['E:covid_y5']['est'])}) = -2 x D(2013-15, y5) ({fm(T['PL:y5:2013']['est'])}); both identities are "
      f"asserted field by field and stated next to the numbers (Answer 7).")
    w(f"7. **Test 1 verdict qualifier.** The verdict line now carries the placebo comparison (rank p = "
      f"{p1['rank']}/{p1['n']} = {p1['rank'] / p1['n']:.2f}) and the entry-unemployment contrast ({dUg:+.2f} pp by "
      f"graduation years, {dUe:+.2f} pp by y1 earnings years); D per percentage point of the latter is reported, "
      f"and the placebo windows show their own unemployment contrasts (Answer 1).")
    rk = "; ".join(f"{h} {fm(T2[(h, 'Gallrk')]['GU']['b'], 4)} -> {fm(T2[(h, 'Gall')]['GU']['b'], 4)} "
                   f"[{fm(T2[(h, 'Gall')]['GU']['lo'], 4)}, {fm(T2[(h, 'Gall')]['GU']['hi'], 4)}]" for h in HZ)
    w(f"8. **Coverage covariate in the Test 2 all-institution sensitivity.** The docstring pre-specifies the "
      f"coverage share; the first run used its within-(field, window) percentile rank without marking the change. "
      f"The share is now used and the rank version kept, labelled a deviation (same draws as the first run, so its "
      f"numbers are the first run's). G~ x U, rank -> share: {rk}.")
    w(f"9. **LEHD frame before 2010.** New caveat quoting the PSEO documentation (Massachusetts UI records enter "
      f"LEHD in 2010), with the New England employment shares by window (PSEO Flows) and the linear surface on cells "
      f"whose earnings years are all >= 2010 (fixed institutions theta_h {fm(l10['theta_h']['est'], 4)}, theta_c "
      f"{fm(l10['theta_c']['est'], 4)}; all 17 cells: {fm(th['est'], 4)}, {fm(tc['est'], 4)}).")
    w(f"10. **Pre-specification timing.** Stated as self-reported (header and Caveats); the md5 of the "
      f"PRE-SPECIFICATION block is printed (`{PREMD5}`) so later edits of the block are detectable. For future "
      f"scripts the block should be committed or hashed before the first estimating run.")
    w("\n### Second pass (REVISION NOTE 3; checks, wording and provenance only)\n")
    w("The ten items above were checked again against the code and the data before this pass. An independent "
      "re-implementation outside this script (built from the panel file; its output is not part of this write-up) "
      "reproduced the design ranks, the estimability of each curvature contrast with free and with restricted "
      "period effects, the dimension of the estimable part of their span, the absence of the age curvature from it, "
      "and the default, y5-knot, step-profile and saturated-fit contrasts to the printed precision; the Massachusetts "
      "release pattern was re-read from the raw PSEO file. All ten items hold. No estimate changed in this pass; the "
      "results file gains only the design rows named below.\n")
    ac = "; ".join(f"{grad_years(c)}: t = " + ", ".join(str(t) for t in sorted(pv)) for c, pv in dchk["agecurv"].items())
    w(f"11. **Restriction-free set, age axis.** Answer 5(b) said the age curvature 'enters no estimable combination', "
      f"which is true only within the span of the {dchk['n_all']} curvature contrasts. The model-free within-window "
      f"age-axis curvature of Answer 6 is estimable; under the additive model it equals kappa_A plus the period "
      f"curvature over the window's earnings years ({ac}), and because A(5) enters only the single-cell y5 earnings "
      f"windows the two parts cannot be separated (design checks). The period-only combinations that are estimable "
      f"with free period effects ({', '.join(dchk['dp_combos'])}) are now named and checked. Answers 5(b), 5(e) and "
      f"6 and the Method now say so "
      f"and name the restriction-free quantities: the cohort combinations, the lack-of-fit tests and the model-free "
      f"surface quantities. The key-numbers rows of the M_APC fits now carry the period restriction in their sample "
      f"label.")
    w(f"12. **Massachusetts statement computed.** The LEHD-frame caveat said that Massachusetts institutions appear "
      f"only from the 2010-12 window and in none of the fixed samples that span 2010; this is now counted by the "
      f"script ({MA['n']} institutions; first window {grad_years(ma_first)}; fixed samples "
      f"{'all 0' if not any(ma_fixed) else 'not all 0'}; Caveats and Appendix E).")
    w("13. **Timing caveat.** The Caveats said the script is untracked in git; it was committed later, after the "
      "estimating runs behind the first write-up and REVISION NOTE 2, and the caveat now says that. The timing claims "
      "stay self-reported.")
    w("14. **Flows provenance.** The PSEO Flows V4.14.1 file read here had no subsection in `data/raw/SOURCES.md` "
      "(its download is recorded in FLOWS_PLACEMENT_RESULT.md, scripts/61). A subsection was appended; its md5 is "
      "printed in Appendix E, which checks that it is " + ("recorded" if fl_src else "NOT recorded")
      + " in `data/raw/SOURCES.md`.")
    inv_rows = [x for r_ in T2.values() for c_ in ("GU", "FU") if c_ in r_ and "p_wcr" in r_[c_] for x in [r_[c_]]]
    n_agree = sum((x["p_wcr"] > 0.05) == (x["lo_inv"] <= 0.0 <= x["hi_inv"]) for x in inv_rows)
    n_conn = sum(bool(x["inv_connected"]) for x in inv_rows)
    w(f"15. **WCR p and WCR-inverted CI.** In {n_agree} of {len(inv_rows)} Test 2 interaction estimates the WCR p "
      f"exceeds 0.05 exactly when the WCR-inverted CI contains 0, and in {n_conn} of {len(inv_rows)} the accepted "
      f"null values form one interval (counted by the script). The pre-specified unrestricted bootstrap-t CI and the "
      f"WCR p disagree in the {len(dis)} row(s) listed in item 5.")
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
