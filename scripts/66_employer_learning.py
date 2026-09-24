"""Employer learning, aggregate analogue -- does the loading of pay on institution-wide status fall, stay or
rise with years since graduation once selectivity is in the model?

QUESTION. scripts/52 (and scripts/59 on PSEO V4.14.1) found that, on fixed PSEO cohorts, the within-field
association of median earnings with the academia-wide Wapman rank G grows with years since graduation while the
association with field prestige F stays flat. Employer-learning models with statistical discrimination
(Farber & Gibbons 1996; Altonji & Pierret 2001; Lange 2007; Arcidiacono, Bayer & Hizmo 2010; bib keys
farber1996learning, altonji2001employer, lange2007speed, arcidiacono2010beyond in paper/refs_merged.bib) predict
that the pay loading on a characteristic employers observe at hiring falls with experience, while the loading on
a correlate of productivity that employers observe only imperfectly rises. This script asks which way the G
loading goes once institution selectivity (SAT_AVG or -ADM_RATE), Pell share, control and the state earnings
level are in the model, and which way the selectivity loading goes.

MAPPING (imperfect; stated before any result). The theory is about individuals. Here the unit is an institution
x field cell of PSEO medians, so this is an aggregate analogue, not the individual-level Altonji-Pierret test.
G stands in for the institution's name / status, which employers see on the CV. Institution SAT_AVG and ADM_RATE
are correlates of the average ability of an institution's students, but an institution's selectivity is itself
partly public (and correlated with its name), so it is not a clean "hard-to-observe" variable; it is measured on
recent entering classes (Scorecard "Most Recent" release), not on the panel's own entry cohorts. If employers
already price selectivity at hiring, neither loading needs to move under learning. If employers at hiring use the
name and learn average ability later, b_G should fall and b_SAT rise. Rank-scale loadings (standardized
coefficients on ranks) describe how well each variable orders institutions' medians, not dollar returns; a
log-earnings version is reported as exploratory.

PRE-SPECIFIED DESIGN AND TESTS (fixed in this docstring before any selectivity-controlled estimate was computed;
the uncontrolled F+G result of scripts/52/59 and the covariate coverage of the panel were known):
  Sample A (primary): PSEO V4.14.1 (data/raw/pseo_2026q2, scripts/59's canonical-join "new" variant) bachelor's
    fixed cohorts 2001/2004/2007/2010 at y1/y5/y10, balanced within field x cohort (scripts/52 fixed_panel).
    Covariates: Scorecard institution file joined on the 8-digit OPEID (= PSEO institution id; scripts/61
    scorecard_institutions: SAT_AVG, ADM_RATE, PCTPELL, CONTROL, STABBR, state earnings level ST_EARN of
    scripts/55). Controlled cells: institutions with SAT_AVG, ADM_RATE, PCTPELL, CONTROL and ST_EARN ("SAT
    sample"), n >= 18 (residual df >= 10 in the largest partial specification, scripts/55's DFMIN rule).
  Sample B (secondary): the y1/y5 panel, cohorts 2001/2004/2007/2010/2013/2016 (every bachelor's cohort with y5
    released), balanced on y1 and y5 within field x cohort; same covariates and cell rule.
  Primary loading model (spec S2), per field x cohort x horizon cell: z(rank earnings) on z(rank G), z(rank F),
    z(rank SAT_AVG), z(rank PCTPELL), z(rank ST_EARN) and a private-control dummy, OLS with intercept ->
    b_G(h), b_F(h), b_SAT(h), b_Pell(h). Secondary (spec S3): -ADM_RATE in place of SAT_AVG on the ADM sample
    (institutions with ADM_RATE, PCTPELL, CONTROL, ST_EARN; keeps institutions without SAT_AVG). Comparison
    (spec S1): F + G only on exactly the S2 cells.
  Slope: OLS slope of the loading on years since graduation (A: {1,5,10}; B: {1,5}) per cell, mean over a field's
    cohorts, mean over fields. Inference: two-way (field, institution) cluster variance of Cameron-Gelbach-Miller
    (scripts/59 twoway(), imported): V_field + V_institution - V_field x institution, 1000 institution multinomial
    draws shared by all cells and 1000 independent draws per field; normal 95% CI. Minimum detectable slope at 80%
    power (two-sided 5%): MDE80 = (1.960 + 0.842) x SE.
  T1 (signal decay): d b_G / dh < 0 under S2, sample A.  T2 (learning about ability): d b_SAT / dh > 0 under S2,
    sample A.  T3: scripts/52's "the rise loads on G, F flat" survives S2 (d b_G CI > 0 and d b_F CI includes 0).
    Also reported: d b_G - d b_SAT and d b_G - d b_F (paired, same draws).
  Verdicts: "supported" if the two-way 95% CI lies wholly on the predicted side of 0; "opposite" if wholly on the
    other side; otherwise "null" if MDE80 <= benchmark and "underpowered" if MDE80 > benchmark, where the
    benchmark is |d b_G| of the uncontrolled F+G model (S1) on the same cells (the size of the known rise).
    T3: "supported" if d b_G CI > 0 and d b_F CI includes 0; "opposite" if d b_G CI < 0, or d b_F CI > 0 while
    d b_G CI is not > 0; otherwise "null"/"underpowered" by the MDE rule applied to d b_G.
  Partial-correlation forms (comparability with scripts/55/58), same rules: T1p slope of the partial Spearman of G
    with earnings given the scripts/55 broad set (SAT, ADM, Pell, control, state level); T2p slope of the partial
    Spearman of SAT with earnings given G, F, Pell, control, state level; T3p slope of rho(G, Y | broad + F) > 0
    with slope of rho(F, Y | broad + G) including 0.
  The same T1-T3 on sample B and on S3 (ADM) are pre-specified secondary tests. Pre-specified robustness: the
    y5->y10 segment in A (y1 medians are depressed by post-graduate study); a coalition-fixed variant of A and B
    (institutions in states whose PSEO series starts in 2001, version_pseo.txt).
  UK analogue (C): DfE LEO provider x CAH2 subject, fixed cohorts 2013/14-2016/17 at YAG 1/3/5 (balanced providers,
    n >= 15, scripts/62); z(rank earnings) on z(rank G) + z(rank institution selectivity = provider share of
    graduates with >= 360 UCAS tariff points, scripts/62 inst_top); T1u d b_G < 0, T2u d b_SEL > 0, same rules
    with (subject, provider) two-way variance; benchmark = |slope of the G-only loading| on the same cells.
  Everything else (coverage control, + ADM_RATE, no-F model, log earnings, n >= 25 cells, cohort splits, per-field
    values) is exploratory and labelled so.

REVISION 1 (after an independent verification of the first run; post hoc, every addition is exploratory and does
not change a pre-specified rule or verdict; the block above is left verbatim and its sha256 is printed in the
write-up so later readers can check it has not changed since this revision; the script is untracked in git, so
nothing time-stamps the block before the first run):
  - Scale-free contrasts (the loadings are standardized rank coefficients; when the within-cell R^2 rises every
    loading rises with it): per-cell loadings divided by sqrt(R^2_h) (relative weights; specs S2n, U2n), the mean
    within-cell R^2 by horizon, and delta-method log-ratios of last- to first-horizon mean loadings (UK: G, SEL and
    G - SEL; US: SAT; each also net of 1/2 log R^2 ratio, the uniform-scaling benchmark).
  - Calendar time for the controlled loadings: scripts/52's triple-matched design (cohort c at y1 and y10 and cohort
    c+9 at y1 on the same institutions) gives within-cohort (y10-y1)/9 = a+b, calendar-matched (c@y10 - c+9@y1)/9
    = a-g and cross-cohort drift at y1 (c+9 - c)/9 = b+g; plus the drift at y5 on pair-matched institutions
    (c@y5 vs c+9@y5, c = 2001/2004/2007).
  - Verdict display: 'edge' also when |MDE80 - benchmark| is within 2 Monte Carlo SE of MDE80 (shown as
    'underpowered/null (edge)'); CSV verdict 'null' written as 'null_result' (a bare 'null' reads as missing).

Descriptive and not causal. Public data only. Existing scripts are imported, not edited. Seeded (SEED=66;
order-free streams per analysis); outputs byte-identical on re-run.
Run: OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
     .venv/bin/python scripts/66_employer_learning.py
Outputs: data/interim/employer_learning_loadings.csv, outputs/figures/employer_learning.png,
         EMPLOYER_LEARNING_RESULT.md (repo root, local only).
"""
from __future__ import annotations

import sys
import time
import zlib
import hashlib
import zipfile
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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


s59 = _load("s59", "59_pseo_refresh.py")     # V4.14.1 plumbing, weighted ranks, two-way variance (not edited)
s52 = s59.s52                                  # fixed-cohort panel builder (not edited)
s55 = _load("s55", "55_selectivity.py")        # partial_rank (unit-weight check of the partial correlations)
s61 = _load("s61", "61_flows_placement.py")    # Scorecard institution covariates keyed by OPEID
s62 = _load("s62", "62_uk_leo.py")             # UK constants, prestige table, career cells (not edited)

SEED = 66
NBOOT = 1000
NMIN = s52.NMIN          # 15: scripts/52 cell rule (full-panel reproduction)
NMIN_C = 18              # controlled cells: residual df >= 10 in the largest partial spec (7 columns + x)
NMIN_BIG = 25            # exploratory: larger cells only
Z975 = s59.Z975
Z80 = 0.8416212335729143
MDE_K = Z975 + Z80
FIELDS66, LAB = s52.FIELDS66, s52.LAB

COH_A = ["2001", "2004", "2007", "2010"]
HZ_A = ["y1", "y5", "y10"]
YRS_A = np.array([1.0, 5.0, 10.0])
COH_B = ["2001", "2004", "2007", "2010", "2013", "2016"]
HZ_B = ["y1", "y5"]
YRS_B = np.array([1.0, 5.0])
YRS_U = np.array([1.0, 3.0, 5.0])


def slope_w(yrs):
    return (yrs - yrs.mean()) / ((yrs - yrs.mean()) ** 2).sum()


W_A, W_B, W_U = slope_w(YRS_A), slope_w(YRS_B), slope_w(YRS_U)
assert np.allclose(W_A, s52.W_SLOPE) and np.allclose(W_U, s62.W_SLOPE)

NEW_DIR = s59.NEW_DIR
VERSION_FILE = NEW_DIR / "version_pseo.txt"
REF_S59 = ROOT / "data" / "interim" / "pseo_refresh.csv"
UK_PRES = s62.OUT_PRES
UK_CAREER = s62.OUT_CAREER
OUT_CSV = ROOT / "data" / "interim" / "employer_learning_loadings.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "employer_learning.png"
OUT_MD = ROOT / "EMPLOYER_LEARNING_RESULT.md"
PSEO_DOC = "https://lehd.ces.census.gov/data/pseo_documentation.html"

T0 = time.time()


def stage(msg: str):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rng_for(tag: str) -> np.random.Generator:
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


# ---------------------------------------------------------------------------------------------
# specifications. kind: coef (standardized rank regression), coef_log (log earnings on standardized ranks),
# coef_cov (adds same-horizon coverage), partial (partial Spearman of target with earnings given controls)
# ---------------------------------------------------------------------------------------------
REG_S2 = ["G", "F", "SAT", "PELL", "ST", "PRIV"]
BROAD = ["SAT", "NADM", "PELL", "ST", "PRIV"]          # scripts/55 broad (full_stlev) set
SPECS = {
    "S0": ("full", "coef", ["G", "F"]),
    "S1": ("sat", "coef", ["G", "F"]),
    "S2": ("sat", "coef", REG_S2),
    "S2L": ("sat", "coef_log", REG_S2),
    "S4": ("sat", "coef", ["G", "F", "SAT", "NADM", "PELL", "ST", "PRIV"]),
    "S6": ("sat", "coef", ["G", "SAT", "PELL", "ST", "PRIV"]),
    "pGb": ("sat", "partial", ("G", BROAD)),
    "pGbF": ("sat", "partial", ("G", BROAD + ["F"])),
    "pFbG": ("sat", "partial", ("F", BROAD + ["G"])),
    "pSAT": ("sat", "partial", ("SAT", ["G", "F", "PELL", "ST", "PRIV"])),
    "rG": ("sat", "partial", ("G", [])),
    "rF": ("sat", "partial", ("F", [])),
    "rSAT": ("sat", "partial", ("SAT", [])),
    "S3": ("adm", "coef", ["G", "F", "NADM", "PELL", "ST", "PRIV"]),
    "S3FG": ("adm", "coef", ["G", "F"]),
    "pGbA": ("adm", "partial", ("G", ["NADM", "PELL", "ST", "PRIV"])),
    "S2c": ("satcov", "coef", REG_S2),
    "S5": ("satcov", "coef_cov", REG_S2),
    "S2n": ("sat", "coef_norm", REG_S2),                 # revision 1: S2 loadings / sqrt(R^2_h), and R^2_h
    # UK
    "U1": ("sel", "coef", ["G"]),
    "U2": ("sel", "coef", ["G", "SEL"]),
    "U2n": ("sel", "coef_norm", ["G", "SEL"]),           # revision 1: U2 loadings / sqrt(R^2_h), and R^2_h
    "pGsel": ("sel", "partial", ("G", ["SEL"])),
    "pSELg": ("sel", "partial", ("SEL", ["G"])),
    "rSEL": ("sel", "partial", ("SEL", [])),
}
SPEC_DESC = {
    "S0": "F + G, full balanced panel (scripts/52/59 reproduction)",
    "S1": "F + G only, SAT-sample cells (comparison for S2)",
    "S2": "G + F + SAT_AVG + Pell share + state earnings level + private (primary)",
    "S2L": "S2 regressors, outcome log median earnings (exploratory)",
    "S4": "S2 + ADM_RATE (scripts/55 broad set + F + G; exploratory)",
    "S6": "S2 without F (exploratory)",
    "pGb": "partial rho(G, Y | SAT, ADM, Pell, control, state level)",
    "pGbF": "partial rho(G, Y | broad + F)",
    "pFbG": "partial rho(F, Y | broad + G)",
    "pSAT": "partial rho(SAT, Y | G, F, Pell, control, state level)",
    "rG": "raw rho(G, Y), SAT-sample cells",
    "rF": "raw rho(F, Y), SAT-sample cells",
    "rSAT": "raw rho(SAT, Y), SAT-sample cells",
    "S3": "G + F + (-ADM_RATE) + Pell + state level + private, ADM sample (secondary)",
    "S3FG": "F + G only, ADM-sample cells",
    "pGbA": "partial rho(G, Y | ADM, Pell, control, state level), ADM sample",
    "S2c": "S2 on coverage-complete cells (comparison for S5)",
    "S5": "S2 + same-horizon PSEO coverage (exploratory)",
    "S2n": "S2 loadings divided by sqrt(within-cell R^2) (relative weights; exploratory, revision 1)",
    "U2n": "U2 loadings divided by sqrt(within-cell R^2), UK (relative weights; exploratory, revision 1)",
    "U1": "G only (= coupling), UK",
    "U2": "G + institution selectivity (share >= 360 tariff points), UK (primary UK)",
    "pGsel": "partial rho(G, Y | selectivity), UK",
    "pSELg": "partial rho(selectivity, Y | G), UK",
    "rSEL": "raw rho(selectivity, Y), UK",
}
VAR_LAB = {"G": "academia-wide rank G", "F": "field prestige F", "SAT": "SAT_AVG", "NADM": "-ADM_RATE",
           "PELL": "Pell share", "ST": "state earnings level", "PRIV": "private", "SEL": "selectivity (>=360 pts)"}
DUMMIES = {"PRIV"}

# ---------------------------------------------------------------------------------------------
# weighted rank regression / partial correlation (exact under institution multinomial counts W)
# ---------------------------------------------------------------------------------------------
wrank, wcorr = s59.wrank, s59.wcorr


def wstd(r: np.ndarray, W: np.ndarray, dummy: bool = False) -> np.ndarray:
    sw = W.sum(1, keepdims=True)
    m = (W * r).sum(1, keepdims=True) / sw
    sd = np.sqrt((W * (r - m) ** 2).sum(1, keepdims=True) / sw)
    ok = sd > 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (r - m) / np.where(ok, sd, 1.0)
    if dummy:
        return np.where(ok, z, 0.0)                    # constant dummy: column drops out (pinv)
    return np.where(ok, z, np.nan)                      # constant rank: statistic undefined -> redraw


def wls(D: np.ndarray, Y: np.ndarray, W: np.ndarray) -> np.ndarray:
    """D (B,n,k), Y (B,n,H), W (B,n) -> coefficients (B,k,H) (pinv: exactly constant columns drop out)."""
    A = np.einsum("bnk,bn,bnl->bkl", D, W, D)
    c = np.einsum("bnk,bn,bnh->bkh", D, W, Y)
    bad = ~np.isfinite(A).all((1, 2)) | ~np.isfinite(c).all((1, 2))
    A = np.where(bad[:, None, None], 0.0, A)
    c = np.where(bad[:, None, None], 0.0, c)
    out = np.linalg.pinv(A) @ c
    out[bad] = np.nan
    return out


def eval_cell(cell: dict, W: np.ndarray, specs: list[str]) -> dict:
    """All statistics of one cell under weights W (B, n): stat key -> (B, H)."""
    B, H = W.shape[0], cell["Y"].shape[1]
    Z, R = {}, {}

    def zr(v):
        if v not in Z:
            x = cell["V"][v]
            if v in DUMMIES:
                Z[v] = wstd(np.broadcast_to(x, W.shape).astype(float), W, dummy=True)
            else:
                R[v] = wrank(x, W)
                Z[v] = wstd(R[v], W)
        return Z[v]
    Yz = np.stack([wstd(wrank(cell["Y"][:, h], W), W) for h in range(H)], -1)        # (B,n,H)
    one = np.ones(W.shape)
    out, cache = {}, {}
    for s in specs:
        sub, kind, arg = SPECS[s]
        if kind in ("coef", "coef_log", "coef_norm"):
            ck = ("log" if kind == "coef_log" else "z", tuple(arg))
            if ck not in cache:                            # S2n reuses the S2 fit (identical arithmetic)
                D = np.stack([one] + [zr(v) for v in arg], -1)
                Yv = Yz if kind != "coef_log" else np.broadcast_to(np.log(cell["Y"])[None], (B,) + cell["Y"].shape)
                cache[ck] = (D, wls(D, Yv, W))
            D, b = cache[ck]
            if kind == "coef_norm":
                # revision 1: within-cell R^2_h of the rank regression (weighted variance of the fitted values;
                # z(rank earnings) has weighted variance 1) and the loadings divided by sqrt(R^2_h), which are
                # unchanged when every loading is scaled by the same factor
                fit = np.einsum("bnk,bkh->bnh", D, b)
                sw = W.sum(1)[:, None]
                m = np.einsum("bn,bnh->bh", W, fit) / sw
                r2 = np.einsum("bn,bnh->bh", W, fit ** 2) / sw - m ** 2
                out[f"{s}:R2"] = r2
                with np.errstate(invalid="ignore", divide="ignore"):
                    sr = np.where(r2 > 0, np.sqrt(np.where(r2 > 0, r2, 1.0)), np.nan)
                for j, v in enumerate(arg):
                    if v not in DUMMIES:
                        out[f"{s}:{v}"] = b[:, 1 + j, :] / sr
                continue
            for j, v in enumerate(arg):
                out[f"{s}:{v}"] = b[:, 1 + j, :]
        elif kind == "coef_cov":
            cols = [one] + [zr(v) for v in arg]
            b = np.empty((B, len(arg), H))
            for h in range(H):
                zc = wstd(wrank(cell["COV"][:, h], W), W)
                D = np.stack(cols + [zc], -1)
                b[:, :, h] = wls(D, Yz[:, :, [h]], W)[:, 1:1 + len(arg), 0]
            for j, v in enumerate(arg):
                out[f"{s}:{v}"] = b[:, j, :]
        else:                                             # partial Spearman
            x, ctrl = arg
            D = np.stack([one] + [zr(v) for v in ctrl], -1)
            T = np.concatenate([zr(x)[:, :, None], Yz], -1)                           # (B,n,1+H)
            res = T - np.einsum("bnk,bkh->bnh", D, wls(D, T, W))
            out[s] = np.stack([wcorr(res[:, :, 0], res[:, :, 1 + h], W) for h in range(H)], -1)
    return out


# ---------------------------------------------------------------------------------------------
# passes: observed (unit weights), shared institution draw, independent draw per field
# ---------------------------------------------------------------------------------------------
def eval_pass(cells: list, W: np.ndarray, specs: dict, keep_cells: bool = False):
    """acc[(stat, grp)][field] = [sum (B,H), count]; bad (B,) rows with a non-finite statistic in any cell."""
    acc, bad, per = {}, np.zeros(W.shape[0], bool), []
    for c in cells:
        sp = [s for s in specs[c["sub"]]]
        if not sp:
            continue
        o = eval_cell(c, W[:, c["ix"]], sp)
        for k, v in o.items():
            bad |= ~np.isfinite(v).all(1)
            for g in c["grps"]:
                d = acc.setdefault((k, g), {}).setdefault(c["field"], [0.0, 0])
                d[0] = d[0] + v
                d[1] += 1
        if keep_cells:
            per.append((c, o))
    acc = {k: {f: s / n for f, (s, n) in d.items()} for k, d in acc.items()}
    return acc, bad, per


def draw_pass(cells: list, NI: int, tag: str, specs: dict, max_tries: int = 100):
    rng = rng_for(tag)
    p = np.full(NI, 1.0 / NI)
    W = rng.multinomial(NI, p, size=NBOOT).astype(float)
    acc, bad, _ = eval_pass(cells, W, specs)
    redrawn = 0
    for _ in range(max_tries):
        rows = np.flatnonzero(bad)
        if not len(rows):
            return acc, redrawn
        redrawn += len(rows)
        W[rows] = rng.multinomial(NI, p, size=len(rows)).astype(float)
        a2, b2, _ = eval_pass(cells, W[rows], specs)
        for k, d in a2.items():
            for f, v in d.items():
                acc[k][f][rows] = v
        bad = np.zeros(NBOOT, bool)
        bad[rows] = b2
    raise AssertionError(f"could not draw non-degenerate institution weights for {tag}")


def jack_pass(cells: list, NI: int, specs: dict) -> dict:
    """Delete-one-institution values per cell (weights 1 - I), assembled per (stat, grp, field) into
    Theta_f (NI, H): field f's value with institution i deleted from all of f's cells (= the full value when i is
    not in f's cells), plus the membership mask (NI,)."""
    per_field = {}
    for c in cells:
        sp = specs[c["sub"]]
        if not sp:
            continue
        n = c["n"]
        full = eval_cell(c, np.ones((1, n)), sp)
        dele = eval_cell(c, 1.0 - np.eye(n), sp)
        for k in full:
            assert np.isfinite(dele[k]).all(), (c["field"], c["cohort"], k)
            for g in c["grps"]:
                d = per_field.setdefault((k, g), {}).setdefault(c["field"], dict(cells=[]))
                d["cells"].append((c["ix"], full[k][0], dele[k]))
    out = {}
    for kg, d in per_field.items():
        for f, x in d.items():
            C = len(x["cells"])
            H = x["cells"][0][1].shape[0]
            base = np.mean([fv for _, fv, _ in x["cells"]], axis=0)
            Th = np.repeat(base[None, :], NI, 0)
            mask = np.zeros(NI, bool)
            for ix, fv, dv in x["cells"]:
                Th[ix] += (dv - fv[None, :]) / C
                mask[ix] = True
            out.setdefault(kg, {})[f] = (Th, mask)
    return out


def run_sample(name: str, cells: list, specs: dict) -> dict:
    univ = sorted(set().union(*[set(c["keys"]) for c in cells]))
    iid = {k: i for i, k in enumerate(univ)}
    for c in cells:
        c["ix"] = np.array([iid[k] for k in c["keys"]])
    NI = len(univ)
    obs, bad, per = eval_pass(cells, np.ones((1, NI)), specs, keep_cells=True)
    assert not bad.any(), f"{name}: a statistic is undefined in the observed data"
    stage(f"{name}: observed pass done ({len(cells)} cells, {NI} institutions)")
    jack = jack_pass(cells, NI, specs)
    stage(f"{name}: delete-one-institution jackknife done")
    shared, red_s = draw_pass(cells, NI, f"shared|{name}", specs)
    stage(f"{name}: shared institution draw done (redrawn {red_s})")
    fields = sorted({c["field"] for c in cells})
    indep, red_i = {}, 0
    for f in fields:
        a, r = draw_pass([c for c in cells if c["field"] == f], NI, f"indep|{name}|{f}", specs)
        red_i += r
        for k, d in a.items():
            indep.setdefault(k, {}).update(d)
    stage(f"{name}: independent per-field draws done (redrawn {red_i})")
    return dict(name=name, obs={k: {f: v[0] for f, v in d.items()} for k, d in obs.items()}, shared=shared,
                indep=indep, jack=jack, per=per, NI=NI, redrawn_shared=red_s, redrawn_indep=red_i, cells=cells)


# ---------------------------------------------------------------------------------------------
# aggregation and inference
# ---------------------------------------------------------------------------------------------
ROWS: list[dict] = []
RES: dict = {}


def ci(x):
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def field_arrays(S: dict, stat: str, grp: str, w: np.ndarray, fields=None):
    d = S["obs"].get((stat, grp), {})
    fs = sorted(d) if fields is None else [f for f in fields if f in d]
    pf = np.array([d[f] @ w for f in fs])
    cb = np.stack([S["shared"][(stat, grp)][f] @ w for f in fs])
    ib = np.stack([S["indep"][(stat, grp)][f] @ w for f in fs])
    Th = np.stack([S["jack"][(stat, grp)][f][0] @ w for f in fs])
    mk = np.stack([S["jack"][(stat, grp)][f][1] for f in fs])
    return fs, pf, cb, ib, Th, mk


def guard(v2: float, vf: float, vs: float) -> tuple[float, bool]:
    """Amendment (declared in the write-up): the two-way variance cannot fall below the between-field one-way
    variance V_field unless the estimated cross-field institution covariance (V_inst - V_field x inst) is
    negative, which here signals Monte Carlo / heavy-tail noise in V_field x inst. In that case (and when the
    estimate is <= 0, scripts/59's own fallback) the larger one-way variance is used."""
    if v2 >= vf and v2 > 0:
        return v2, False
    return max(vf, vs), True


def jack2(pf: np.ndarray, Th: np.ndarray, mk: np.ndarray) -> dict:
    """Two-way cluster jackknife in the CGM form: V_field (delete one field) + V_inst (delete one institution
    from every cell) - V_field x inst (delete one institution from one field's cells)."""
    K = len(pf)
    est = float(pf.mean())
    vf = float(pf.var(ddof=1) / K)
    th_i = Th[:, mk.any(0)].mean(0)
    N = len(th_i)
    vi = float((N - 1) / N * ((th_i - th_i.mean()) ** 2).sum())
    vals = est + ((Th - pf[:, None]) / K)[mk]
    M = len(vals)
    vfi = float((M - 1) / M * ((vals - vals.mean()) ** 2).sum())
    v, g = guard(vf + vi - vfi, vf, vi)
    se = float(np.sqrt(v))
    return dict(jse=se, jlo=est - Z975 * se, jhi=est + Z975 * se, jz=est / se if se > 0 else np.nan, jvf=vf,
                jvi=vi, jvfi=vfi, jguard=g, jnoise=float(np.sqrt(vfi * K) / pf.std(ddof=1)) if K > 1 else np.nan)


def infer(key: str, fs, pf, cb, ib, Th, mk, meta: dict) -> dict:
    K = len(fs)
    FI = rng_for(f"fields|{key}").integers(K, size=(NBOOT, K))
    est = float(pf.mean())
    cbm, ibm = cb.mean(0), ib.mean(0)
    fb = pf[FI].mean(1)
    xb = cb[FI, np.arange(NBOOT)[:, None]].mean(1)
    vfield = pf.var(ddof=1) / K
    tw = s59.twoway(est, vfield, cbm, ibm, fb, xb)
    v_raw = tw["tvf"] + tw["tvs"] - tw["tvi"]
    v, g = guard(v_raw, tw["tvf"], tw["tvs"])
    se = float(np.sqrt(v))
    # IQR-based (robust) two-way variance, same guard
    vr_f, vr_s, vr_i = s59.iqr_var(fb), s59.iqr_var(cbm), s59.iqr_var(ibm)
    vr, gr = guard(vr_f + vr_s - vr_i, vr_f, vr_s)
    se_r = float(np.sqrt(vr))
    J = jack2(pf, Th, mk)
    r = dict(key=key, est=est, k=K, fields=list(fs), pf=pf, se=se, z=est / se, lo=est - Z975 * se,
             hi=est + Z975 * se, se_raw=float(np.sqrt(v_raw)) if v_raw > 0 else np.nan, guarded=g,
             lo_u=tw["tlo"], hi_u=tw["thi"], mde_u=MDE_K * tw["tse"],
             se_r=se_r, z_r=est / se_r, lo_r=est - Z975 * se_r, hi_r=est + Z975 * se_r, guarded_r=gr,
             mcse=Z975 * float(np.sqrt(max(s59.varvar(cbm) + s59.varvar(ibm), 0.0))) / (2 * se),
             vf=tw["tvf"], vs=tw["tvs"], vi=tw["tvi"],
             noise=float(np.sqrt(tw["tvi"] * K) / pf.std(ddof=1)) if K > 1 else np.nan,
             xlo=ci(xb)[0], xhi=ci(xb)[1], clo=ci(cbm)[0], chi=ci(cbm)[1], mde=MDE_K * se,
             mde_r=MDE_K * se_r, pos=int((pf > 0).sum()))
    r.update(J)
    r["mde_j"] = MDE_K * r["jse"]
    r.update(meta)
    RES[key] = r
    ROWS.append(dict(section="aggregate", sample=meta.get("sample", ""), spec=meta.get("spec", ""),
                     stat=meta.get("stat", ""), contrast=meta.get("contrast", ""), group=meta.get("grp", "all"),
                     field="", cohort="", n=np.nan, estimate=est, se_twoway=se, z_twoway=r["z"],
                     ci_lo_twoway=r["lo"], ci_hi_twoway=r["hi"], twoway_guard_applied=g,
                     se_twoway_unguarded=r["se_raw"], mcse_ci_endpoint=r["mcse"],
                     se_twoway_iqr=se_r, ci_lo_twoway_iqr=r["lo_r"], ci_hi_twoway_iqr=r["hi_r"],
                     se_jackknife=r["jse"], ci_lo_jackknife=r["jlo"], ci_hi_jackknife=r["jhi"],
                     jackknife_guard_applied=r["jguard"],
                     ci_lo_crossed=r["xlo"], ci_hi_crossed=r["xhi"], ci_lo_fields_fixed=r["clo"],
                     ci_hi_fields_fixed=r["chi"], v_field=r["vf"], v_inst=r["vs"], v_field_x_inst=r["vi"],
                     noise_ratio_boot=r["noise"], noise_ratio_jack=r["jnoise"],
                     mde80=r["mde"], mde80_iqr=r["mde_r"], mde80_jackknife=r["mde_j"], k_fields=K,
                     n_positive_fields=r["pos"], verdict="", key=key, note=meta.get("note", "")))
    return r


def summ(S: dict, stat: str, w: np.ndarray, contrast: str, grp: str = "all", fields=None, key=None) -> dict:
    fs, pf, cb, ib, Th, mk = field_arrays(S, stat, grp, w, fields)
    key = key or f"{S['name']}|{stat}|{contrast}|{grp}"
    spec, _, var = stat.partition(":")
    return infer(key, fs, pf, cb, ib, Th, mk, dict(sample=S["name"], spec=spec, stat=stat, contrast=contrast,
                                                   grp=grp))


def summ_diff(S: dict, a: str, b: str, w: np.ndarray, contrast: str, grp: str = "all", Sb: dict | None = None,
              key=None) -> dict:
    Sb = Sb or S
    fa = sorted(S["obs"][(a, grp)])
    fb_ = sorted(Sb["obs"][(b, grp)])
    fs = [f for f in fa if f in fb_]
    _, pa, ca, ia, Ta, ma = field_arrays(S, a, grp, w, fs)
    _, pb, cb, ib, Tb, mb = field_arrays(Sb, b, grp, w, fs)
    key = key or f"{S['name']}|{a} - {b}|{contrast}|{grp}"
    return infer(key, fs, pa - pb, ca - cb, ia - ib, Ta - Tb, ma | mb,
                 dict(sample=S["name"], spec="diff", stat=f"{a} - {b}", contrast=contrast, grp=grp))


def summ_logratio(S: dict, terms: list, stat_label: str, contrast: str) -> dict | None:
    """Revision 1 (exploratory, scale-free): g = sum_j c_j log(mean over fields of stat_j at horizon h_j), e.g.
    log(mean b_G at YAG 5 / mean b_G at YAG 1). Inference by the delta method: every per-field array (observed,
    shared draw, independent draws, delete-one-institution values) is mapped through the linearization
    g + sum_j c_j (x_j - xbar_j) / xbar_j, so all three intervals are the delta-method versions of the same
    variances. Undefined (returns None) when a mean is <= 0."""
    fs0, parts = None, []
    for st, h, cf in terms:
        H = next(iter(S["obs"][(st, "all")].values())).shape[0]
        parts.append((st, h, cf, H))
        fx = sorted(S["obs"][(st, "all")])
        fs0 = fx if fs0 is None else [f for f in fs0 if f in fx]
    arrs = []
    for st, h, cf, H in parts:
        _, pf, cb, ib, Th, mk = field_arrays(S, st, "all", level_w(H, h), fs0)
        arrs.append((cf, pf.mean(), pf, cb, ib, Th, mk))
    if min(m for _, m, *_ in arrs) <= 0:
        return None
    g0 = float(sum(cf * np.log(m) for cf, m, *_ in arrs))
    lin = [g0 + sum(cf * (x[i] - m) / m for cf, m, *x in arrs) for i in range(4)]
    mk = np.logical_or.reduce([x[-1] for x in arrs])
    key = f"{S['name']}|{stat_label}|{contrast}|all"
    return infer(key, fs0, lin[0], lin[1], lin[2], lin[3], mk,
                 dict(sample=S["name"], spec="logratio", stat=stat_label, contrast=contrast, grp="all",
                      note="delta method; " + "; ".join(f"{cf:+g} x log mean {st} at horizon {h}"
                                                        for st, h, cf, _ in parts)))


def level_w(H: int, h: int) -> np.ndarray:
    w = np.zeros(H)
    w[h] = 1.0
    return w


CIK = {"tw": ("lo", "hi", "mde"), "iqr": ("lo_r", "hi_r", "mde_r"), "jack": ("jlo", "jhi", "mde_j"),
       "unguarded": ("lo_u", "hi_u", "mde_u")}


def verdict(r: dict, side: str, bench: float, kind: str = "tw") -> str:
    """Pre-specified rule. side 'neg' (prediction < 0) or 'pos' (prediction > 0). kind: which interval
    ('tw' = the pre-specified two-way bootstrap variance; 'iqr', 'jack' = sensitivity intervals)."""
    lo, hi, mde = (r[x] for x in CIK[kind])
    if (side == "neg" and hi < 0) or (side == "pos" and lo > 0):
        return "supported"
    if (side == "neg" and lo > 0) or (side == "pos" and hi < 0):
        return "opposite"
    return "null" if mde <= bench else "underpowered"


def verdict_t3(rg: dict, rf: dict, bench: float, kind: str = "tw") -> str:
    lo, hi, mde = (rg[x] for x in CIK[kind])
    flo, fhi, _ = (rf[x] for x in CIK[kind])
    if lo > 0 and flo <= 0 <= fhi:
        return "supported"
    if hi < 0 or (flo > 0 and not lo > 0):
        return "opposite"
    return "null" if mde <= bench else "underpowered"


def edge(r: dict) -> bool:
    """scripts/59 convention: a CI endpoint within 2 Monte Carlo SE of 0 (bootstrap interval only)."""
    return bool(min(abs(r["lo"]), abs(r["hi"])) <= 2 * r["mcse"])


# ---------------------------------------------------------------------------------------------
# data: US
# ---------------------------------------------------------------------------------------------
def pseo_state_start() -> dict:
    """PSEOE state -> first graduation year of its series (version_pseo.txt of V4.14.1)."""
    out = {}
    for line in VERSION_FILE.read_text().strip().splitlines():
        p = line.split()
        if p[0] == "PSEOE" and p[1] != "US":
            out[p[1]] = int(p[3].split("-")[0])
    return out


def us_data():
    s59.set_release("new")
    ar = load_ar_wapman(fields=FIELDS66)
    gen = s52._s28.load_generic()
    gen = gen.assign(G=-gen["g_rank"].astype(float))[["inst_key", "G"]]
    # y1 also for cohort 2019 (revision 1: the calendar-matched cohort c+9 of c = 2010); load_er_pseo groups by
    # (field, institution, cohort), so the extra cohort leaves every other row unchanged; panels A and B filter
    # their cohorts explicitly
    er = {"y1": s52.load_fixed("y1", COH_B + ["2019"]), "y5": s52.load_fixed("y5", COH_B),
          "y10": s52.load_fixed("y10", COH_A)}
    stage("PSEO V4.14.1 loaded (y1 cohorts 2001-2019; y5 cohorts 2001-2016; y10 cohorts 2001-2010)")
    panel_a = s52.fixed_panel(er, ar, gen)                        # scripts/52 builder, cohorts 2001-2010
    wide = None
    for h in HZ_B:
        x = er[h][er[h]["grad_cohort"].isin(COH_B)][["field", "inst_key", "grad_cohort", "earnings", "coverage"]] \
            .rename(columns={"earnings": f"e_{h}", "coverage": f"cov_{h}"})
        wide = x if wide is None else wide.merge(x, on=["field", "inst_key", "grad_cohort"], how="inner")
    panel_b = wide.merge(ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"}),
                         on=["field", "inst_key"], how="inner").merge(gen, on="inst_key", how="left")
    panel_b = panel_b[panel_b.G.notna()].sort_values(["field", "grad_cohort", "inst_key"]).reset_index(drop=True)
    ids = pd.concat([er[h][["field", "inst_key", "grad_cohort", "institution_id"]] for h in HZ_B]) \
        .drop_duplicates(["field", "inst_key", "grad_cohort"])
    assert ids.groupby("inst_key").institution_id.nunique().max() == 1
    ids = ids.drop_duplicates("inst_key")[["inst_key", "institution_id"]]
    sc = s61.scorecard_institutions()
    ins = s59.institutions("new").drop_duplicates("institution").rename(columns={"institution": "institution_id"})
    st = pd.read_csv(s59.NEW_INST, dtype=str)
    st.columns = [c.strip().lstrip("﻿") for c in st.columns]
    st = st.rename(columns={"institution": "institution_id"})[["institution_id", "institution_state"]]
    start = pseo_state_start()
    cov = ids.merge(sc, left_on="institution_id", right_on="institution", how="left") \
        .merge(st, on="institution_id", how="left")
    cov["state_start"] = cov.institution_state.map(start)
    cov["coal01"] = cov.state_start <= 2001
    cov["PRIV"] = (cov.CONTROL != 1).astype(float).where(cov.CONTROL.notna())
    cov = cov.rename(columns={"NEG_ADM": "NADM", "ST_EARN": "ST"})
    keep = ["inst_key", "institution_id", "INSTNM", "institution_state", "state_start", "coal01", "CONTROL", "PRIV",
            "SAT", "ADM_RATE", "NADM", "PELL", "ST"]
    cov = cov[keep]
    panel_a = panel_a.merge(cov, on="inst_key", how="left")
    panel_b = panel_b.merge(cov, on="inst_key", how="left")
    assert panel_a.institution_id.notna().all() and panel_b.institution_id.notna().all()
    del ins
    return ar, gen, er, panel_a, panel_b, cov, start


SUB_NEED = {"full": [], "sat": ["SAT", "NADM", "PELL", "ST", "PRIV"], "adm": ["NADM", "PELL", "ST", "PRIV"],
            "satcov": ["SAT", "NADM", "PELL", "ST", "PRIV"]}


def us_cells(panel: pd.DataFrame, hz: list[str], subs: list[str], big: bool = False, cohort_grps=None) -> list:
    cells = []
    for (fld, coh), g in panel.groupby(["field", "grad_cohort"], sort=True):
        for sub in subs:
            need = SUB_NEED[sub] + ([f"cov_{h}" for h in hz] if sub == "satcov" else [])
            x = g.dropna(subset=need) if need else g
            nmin = NMIN if sub == "full" else NMIN_C
            if len(x) < nmin:
                continue
            x = x.sort_values("inst_key")
            grps = ["all"]
            if big and len(x) >= NMIN_BIG:
                grps.append("n25")
            if cohort_grps:
                grps.append(cohort_grps(coh))
            V = {"G": x.G.to_numpy(float), "F": x.P.to_numpy(float)}
            for v in ("SAT", "NADM", "PELL", "ST", "PRIV"):
                V[v] = x[v].to_numpy(float)
            cells.append(dict(field=fld, cohort=coh, sub=sub, n=len(x), keys=list(x.inst_key), V=V, grps=grps,
                              Y=x[[f"e_{h}" for h in hz]].to_numpy(float),
                              COV=x[[f"cov_{h}" for h in hz]].to_numpy(float) if sub == "satcov" else None))
    return cells


D5_PAIRS = {"2001": "2010", "2004": "2013", "2007": "2016"}      # y5 released for cohorts up to 2016


def time_cells(er: dict, ar: pd.DataFrame, gen: pd.DataFrame, cov: pd.DataFrame) -> tuple[list, list]:
    """Revision 1 (exploratory). Calendar-time decomposition of the controlled loadings with scripts/52's design.
    CAL: per field and c in 2001/2004/2007/2010 the institutions with released earnings at (c, y1), (c, y10) and
    (c+9, y1) (triple-matched, scripts/52 LATE map) and the S2 covariates; Y columns = [c@y1, c@y10, c+9@y1].
    D5: per field and c in 2001/2004/2007 the institutions at (c, y5) and (c+9, y5) (pair-matched); Y = [c@y5,
    c+9@y5]. Institutions are held fixed within every comparison (matched), so PSEO coalition growth does not enter
    a contrast. Same SAT-sample rule and n >= 18 as the controlled cells."""
    a = ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"})
    assert not cov.inst_key.duplicated().any()
    cv = cov[["inst_key"] + SUB_NEED["sat"]]
    k2 = ["field", "inst_key"]

    def part(h, c, nm):
        x = er[h]
        return x[x.grad_cohort == c][k2 + ["earnings"]].rename(columns={"earnings": nm})

    def build(t, cohort, ycols):
        t = t.merge(a, on=k2, how="inner").merge(gen, on="inst_key", how="inner").merge(cv, on="inst_key", how="left")
        t = t.dropna(subset=["G", "P"] + SUB_NEED["sat"])
        out = []
        for fld, g in t.groupby("field", sort=True):
            if len(g) < NMIN_C:
                continue
            g = g.sort_values("inst_key")
            V = {"G": g.G.to_numpy(float), "F": g.P.to_numpy(float)}
            for v in ("SAT", "NADM", "PELL", "ST", "PRIV"):
                V[v] = g[v].to_numpy(float)
            out.append(dict(field=fld, cohort=cohort, sub="sat", n=len(g), keys=list(g.inst_key), V=V, grps=["all"],
                            Y=g[ycols].to_numpy(float), COV=None))
        return out

    cal, d5 = [], []
    for c in COH_A:
        c9 = s52.LATE[c]
        t = part("y1", c, "e0").merge(part("y10", c, "e1"), on=k2).merge(part("y1", c9, "e2"), on=k2)
        cal += build(t, c, ["e0", "e1", "e2"])
    for c, c9 in D5_PAIRS.items():
        t = part("y5", c, "e0").merge(part("y5", c9, "e1"), on=k2)
        d5 += build(t, c, ["e0", "e1"])
    return cal, d5


# ---------------------------------------------------------------------------------------------
# data: UK (lean re-read of the LEO provider file with scripts/62's filters and multi-region rule)
# ---------------------------------------------------------------------------------------------
def load_leo_all_graduates() -> tuple[pd.DataFrame, dict]:
    keep = ["tax_year", "academic_year", "YAG", "ukprn", "provider_region_name", "cah2_subject_name",
            "grads", "grads_earnings_include", "earnings_median", "characteristic_type"]
    z = zipfile.ZipFile(s62.LEO_ZIP)
    name = [n for n in z.namelist() if n.startswith("provider_data")][0]
    parts, ptot = [], []
    for ch in pd.read_csv(z.open(name), encoding="latin-1", dtype=str, usecols=keep, chunksize=200000):
        ch = ch[(ch.characteristic_type == "All graduates") & (ch.ukprn != "Total")]
        tax = ch.tax_year.str[:4].astype(int)
        coh = ch.academic_year.str[:4].astype(int)
        yag = ch.YAG.astype(int)
        t = ch.cah2_subject_name == "Total"
        ptot.append(pd.DataFrame({"ukprn": ch.ukprn[t].values, "cohort": coh[t].values, "yag": yag[t].values,
                                  "reg": ch.provider_region_name[t].values,
                                  "grads": pd.to_numeric(ch.grads[t], errors="coerce").values}))
        k = (~t & coh.isin(s62.COHORTS) & yag.isin(s62.YAGS)).values
        parts.append(pd.DataFrame({"ukprn": ch.ukprn.values[k], "subject": ch.cah2_subject_name.values[k],
                                   "cohort": coh.values[k], "yag": yag.values[k], "tax": tax.values[k],
                                   "reg": ch.provider_region_name.values[k],
                                   "earn": pd.to_numeric(ch.earnings_median, errors="coerce").values[k],
                                   "n_earn": pd.to_numeric(ch.grads_earnings_include, errors="coerce").values[k],
                                   "grads": pd.to_numeric(ch.grads, errors="coerce").values[k]}))
    d = pd.concat(parts, ignore_index=True)
    pt = pd.concat(ptot, ignore_index=True)
    # scripts/62 rule: multi-region providers -> keep the region with the most graduates in the provider's
    # all-subject, all-graduate row of that cohort x YAG (ties: region name)
    pt["g"] = pt.grads.fillna(-1.0)
    main = pt.sort_values(["ukprn", "cohort", "yag", "g", "reg"], ascending=[True, True, True, False, True],
                          kind="mergesort").drop_duplicates(["ukprn", "cohort", "yag"]) \
             .set_index(["ukprn", "cohort", "yag"]).reg
    multi = pt.groupby(["ukprn", "cohort", "yag"]).reg.nunique()
    multi = multi[multi > 1].index
    key = pd.MultiIndex.from_frame(d[["ukprn", "cohort", "yag"]])
    drop = key.isin(multi) & (d.reg.values != main.reindex(key).values)
    info = dict(minor_region_rows=int(drop.sum()))
    d = d[~drop].drop(columns="reg").reset_index(drop=True)
    assert not d.duplicated(["ukprn", "subject", "cohort", "yag"]).any()
    off = (d.tax - d.cohort - d.yag).value_counts()
    info["tax_offset"] = {int(k): int(v) for k, v in off.items()}
    return d, info


def uk_data():
    P = pd.read_csv(UK_PRES, dtype={"ukprn": str})
    leo, info = load_leo_all_graduates()
    T = leo[leo.ukprn.isin(P.ukprn)].copy()
    T["G"] = T.ukprn.map(P.set_index("ukprn").G)
    T["SEL"] = T.ukprn.map(P.set_index("ukprn").inst_top)
    x = T[T.earn.notna() & T.G.notna()]
    w = x.pivot_table(index=["subject", "cohort", "ukprn"], columns="yag", values="earn", aggfunc="first")
    w = w.reindex(columns=s62.YAGS).dropna()
    # reproduce scripts/62's all-graduate career cells (subject x cohort, n >= 15)
    ref = pd.read_csv(UK_CAREER)
    ref = ref[ref.analysis == "all_graduates"].set_index(["subject", "cohort"])
    Gm, Sm = P.set_index("ukprn").G, P.set_index("ukprn").inst_top
    cells, chk = [], []
    for (s, c), g in w.groupby(level=[0, 1]):
        provs = list(g.index.get_level_values(2))
        if len(provs) < s62.NMIN:
            continue
        Y = g.values.astype(float)
        G = Gm.reindex(provs).values
        rh = [spearmanr(G, Y[:, k])[0] for k in range(3)]
        r0 = ref.loc[(s, c)]
        chk.append(dict(subject=s, cohort=c, n=len(provs), n_ref=int(r0.n),
                        dmax=float(np.max(np.abs(np.array(rh) - r0[["y1", "y3", "y5"]].to_numpy(float))))))
        sel = Sm.reindex(provs).values
        ok = np.isfinite(sel)
        if ok.sum() >= s62.NMIN:
            cells.append(dict(field=s, cohort=str(c), sub="sel", n=int(ok.sum()), keys=[p for p, o in zip(provs, ok) if o],
                              V={"G": G[ok], "SEL": sel[ok]}, Y=Y[ok], COV=None, grps=["all"]))
    chk = pd.DataFrame(chk)
    assert len(chk) == len(ref), (len(chk), len(ref))
    assert (chk.n == chk.n_ref).all() and chk.dmax.max() < 6e-5, chk.dmax.max()
    info.update(n_ref_cells=len(ref), max_abs_diff_ref=float(chk.dmax.max()), n_providers=len(P),
                n_sel=int(P.inst_top.notna().sum()), rho_G_sel=float(spearmanr(P.G, P.inst_top, nan_policy="omit")[0]))
    return cells, info, P


# ---------------------------------------------------------------------------------------------
# checks against scripts/52/55/59
# ---------------------------------------------------------------------------------------------
def check_reproduction(SA: dict) -> dict:
    """S0 per-cell coefficients = scripts/52 cell_metrics; aggregate dG, dF = scripts/59 V4.14.1 values;
    partial correlations = scripts/55 partial_rank."""
    nchk, dmax = 0, 0.0
    for c, o in SA["per"]:
        if c["sub"] == "full":
            m = s52.cell_metrics(c["V"]["F"], c["Y"], c["V"]["G"], np.arange(c["n"])[None, :])
            dmax = max(dmax, float(np.abs(m["bF"][0] - o["S0:F"][0]).max()), float(np.abs(m["bG"][0] - o["S0:G"][0]).max()))
            nchk += 1
    assert dmax < 1e-9, dmax
    pmax, npc = 0.0, 0
    for c, o in SA["per"]:
        if c["sub"] == "sat" and npc < 25:
            V = c["V"]
            for h in range(3):
                r, _ = s55.partial_rank(V["G"], c["Y"][:, h], [V["SAT"], V["NADM"], V["PELL"], V["ST"]], [V["PRIV"]])
                pmax = max(pmax, abs(r - o["pGb"][0, h]))
            npc += 1
    assert pmax < 1e-9, pmax
    ref = pd.read_csv(REF_S59)
    ref = ref[(ref.section == "career_time") & (ref.variant == "new")].set_index("stat").estimate
    dG = np.mean([SA["obs"][("S0:G", "all")][f] @ W_A for f in SA["obs"][("S0:G", "all")]])
    dF = np.mean([SA["obs"][("S0:F", "all")][f] @ W_A for f in SA["obs"][("S0:F", "all")]])
    assert abs(dG - ref["dbG_all"]) < 1e-6 and abs(dF - ref["dbF_all"]) < 1e-6, (dG, dF)
    return dict(cells_checked=nchk, max_abs_diff_s52=dmax, partial_cells_checked=npc, max_abs_diff_s55=pmax,
                dG_s59=float(ref["dbG_all"]), dF_s59=float(ref["dbF_all"]), dG_here=float(dG), dF_here=float(dF))


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------
SPECS_A = {"full": ["S0"], "sat": ["S1", "S2", "S2n", "S2L", "S4", "S6", "pGb", "pGbF", "pFbG", "pSAT", "rG", "rF",
                                    "rSAT"],
           "adm": ["S3", "S3FG", "pGbA"], "satcov": ["S2c", "S5"]}
SPECS_B = {"full": ["S0"], "sat": ["S1", "S2", "pGb", "pGbF", "pFbG", "pSAT", "rG", "rSAT"], "adm": ["S3", "S3FG"],
           "satcov": []}
SPECS_C = {"full": [], "sat": ["S1", "S2", "pGb", "pSAT", "rG"], "adm": [], "satcov": []}
SPECS_U = {"sel": ["U1", "U2", "U2n", "pGsel", "pSELg", "rSEL"]}
# revision 1: calendar-time decomposition of the controlled loadings (triple-matched y1/y10/y1+9 and pair-matched y5)
SPECS_T = {"sat": ["S2", "pGb", "pSAT", "rG"]}
CAL_STATS = ["S2:G", "S2:SAT", "pGb", "pSAT", "rG"]


def coef_vars(spec: str) -> list[str]:
    sub, kind, arg = SPECS[spec]
    return [v for v in arg if v not in DUMMIES] if kind.startswith("coef") else []


def stats_of(specs: dict) -> list[str]:
    out = []
    for sub, sp in specs.items():
        for s in sp:
            out += [f"{s}:{v}" for v in coef_vars(s)] if SPECS[s][1].startswith("coef") else [s]
            if SPECS[s][1] == "coef_norm":
                out.append(f"{s}:R2")
    return out


def all_summaries(S: dict, specs: dict, hz: list[str], w: np.ndarray, extra_contrasts: dict, grps=("all",)):
    H = len(hz)
    for st in stats_of(specs):
        for g in grps:
            if (st, g) not in S["obs"]:
                continue
            summ(S, st, w, "slope", g)
            if g == "all":
                for i, h in enumerate(hz):
                    summ(S, st, level_w(H, i), h, g)
                for cname, cw in extra_contrasts.items():
                    summ(S, st, cw, cname, g)


def diffs(S: dict, pairs: list, w: np.ndarray, contrasts=("slope",), extra: dict | None = None):
    for a, b in pairs:
        if (a, "all") in S["obs"] and (b, "all") in S["obs"]:
            summ_diff(S, a, b, w, "slope")
            for cname, cw in (extra or {}).items():
                summ_diff(S, a, b, cw, cname)


def descriptives(S: dict, hz: list[str]) -> dict:
    """Observed within-cell collinearity and cell counts (primary SAT-sample cells)."""
    out = {}
    sat = [c for c, _ in S["per"] if c["sub"] == "sat"]
    if sat:
        def rr(a, b):
            return float(np.mean([spearmanr(c["V"][a], c["V"][b])[0] for c in sat]))
        out.update(r_GF=rr("G", "F"), r_GSAT=rr("G", "SAT"), r_FSAT=rr("F", "SAT"), r_GPELL=rr("G", "PELL"),
                   r_SATPELL=rr("SAT", "PELL"), r_GNADM=rr("G", "NADM"), r_SATNADM=rr("SAT", "NADM"))
        r2 = []
        for c in sat:
            X = np.column_stack([np.ones(c["n"])] + [s52.rk(c["V"][v]) for v in ("F", "SAT", "PELL", "ST")] +
                                ([c["V"]["PRIV"]] if np.ptp(c["V"]["PRIV"]) > 0 else []))
            y = s52.rk(c["V"]["G"])
            e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
            r2.append(1 - e.var() / y.var())
        out["r2_G_on_rest"] = float(np.mean(r2))
        out["n_sat_cells"] = len(sat)
        out["n_sat_fields"] = len({c["field"] for c in sat})
        out["n_sat_min"], out["n_sat_med"], out["n_sat_max"] = (int(np.min([c["n"] for c in sat])),
                                                               float(np.median([c["n"] for c in sat])),
                                                               int(np.max([c["n"] for c in sat])))
        out["n_sat_insts"] = len(set().union(*[set(c["keys"]) for c in sat]))
        out["n_priv_cells"] = int(sum(np.ptp(c["V"]["PRIV"]) > 0 for c in sat))
    for sub in ("full", "adm", "satcov", "sel"):
        cs = [c for c, _ in S["per"] if c["sub"] == sub]
        if cs:
            out[f"n_{sub}_cells"] = len(cs)
            out[f"n_{sub}_fields"] = len({c["field"] for c in cs})
            out[f"n_{sub}_insts"] = len(set().union(*[set(c["keys"]) for c in cs]))
    if S["name"] == "UK":
        cs = [c for c, _ in S["per"]]
        out["r_GSEL"] = float(np.mean([spearmanr(c["V"]["G"], c["V"]["SEL"])[0] for c in cs]))
        out["n_sel_min"], out["n_sel_max"] = int(np.min([c["n"] for c in cs])), int(np.max([c["n"] for c in cs]))
    out["redrawn_shared"], out["redrawn_indep"], out["NI"] = S["redrawn_shared"], S["redrawn_indep"], S["NI"]
    return out


def cell_rows(S: dict, hz: list[str], w: np.ndarray, keep_specs: set):
    for c, o in S["per"]:
        for k, v in o.items():
            if k.split(":")[0] not in keep_specs:
                continue
            vals = list(v[0]) + [float(v[0] @ w)]
            for h, x in zip(hz + ["slope"], vals):
                ROWS.append(dict(section="cell", sample=S["name"], spec=k.split(":")[0], stat=k, contrast=h,
                                 group="", field=c["field"], cohort=c["cohort"], n=c["n"], estimate=x,
                                 key=f"{S['name']}|{k}|{h}|{c['field']}|{c['cohort']}"))


TESTS: list[dict] = []


def mcse_mde(r: dict) -> float:
    """Monte Carlo SE of MDE80 (revision 1): MDE80 = MDE_K x SE, and the Monte Carlo SE of SE is the same one that
    gives the endpoint MC SE (r['mcse'] = Z975 x MC SE of SE)."""
    return r["mcse"] * MDE_K / Z975


def csv_verdict(v: str) -> str:
    """Revision 1: a bare 'null' is read as missing by pandas.read_csv; the CSV writes 'null_result'."""
    return "null_result" if v == "null" else v


def add_test(tid: str, label: str, r: dict, side: str, bench: float, rule: str = "one", rf: dict | None = None,
             prespec: str = "primary"):
    vs = {kd: (verdict_t3(r, rf, bench, kd) if rule == "t3" else verdict(r, side, bench, kd)) for kd in CIK}
    v = vs["tw"]
    e_ci = edge(r)
    # revision 1: the null/underpowered split is also 'edge' when MDE80 and the benchmark are within 2 Monte Carlo
    # SE of MDE80 of each other (the split then depends on bootstrap noise); displayed as 'underpowered/null'
    e_mde = bool(v in ("null", "underpowered") and abs(r["mde"] - bench) <= 2 * mcse_mde(r))
    v_disp = "underpowered/null" if e_mde else v
    eflag = "+".join(x for x, on in (("ci", e_ci), ("mde", e_mde)) if on)
    TESTS.append(dict(tid=tid, label=label, key=r["key"], est=r["est"], lo=r["lo"], hi=r["hi"], se=r["se"],
                      z=r["z"], mde=r["mde"], bench=bench, verdict=v, verdict_iqr=vs["iqr"], verdict_jack=vs["jack"],
                      verdict_unguarded=vs["unguarded"], verdict_disp=v_disp, edge_ci=e_ci, edge_mde=e_mde,
                      edge_flag=eflag, mcse_mde=mcse_mde(r), mcse=r["mcse"],
                      lo_r=r["lo_r"], hi_r=r["hi_r"], jlo=r["jlo"], jhi=r["jhi"], mde_r=r["mde_r"], mde_j=r["mde_j"],
                      edge=e_ci or e_mde, k=r["k"], prespec=prespec, side=side, rule=rule, guarded=r["guarded"],
                      rf_key=rf["key"] if rf is not None else "", rf_est=rf["est"] if rf is not None else np.nan,
                      rf_lo=rf["lo"] if rf is not None else np.nan, rf_hi=rf["hi"] if rf is not None else np.nan,
                      rf_jlo=rf["jlo"] if rf is not None else np.nan, rf_jhi=rf["jhi"] if rf is not None else np.nan,
                      rf_lo_r=rf["lo_r"] if rf is not None else np.nan,
                      rf_hi_r=rf["hi_r"] if rf is not None else np.nan))
    for row in ROWS:
        if row.get("key") == r["key"] and row["section"] == "aggregate":
            row["verdict"] = (row["verdict"] + "; " if row["verdict"] else "") + f"{tid}: {csv_verdict(v)}"
    ROWS.append(dict(section="test", sample=r.get("sample", ""), spec=r.get("spec", ""), stat=r.get("stat", ""),
                     contrast=r.get("contrast", ""), group=r.get("grp", "all"), field="", cohort="", n=np.nan,
                     estimate=r["est"], se_twoway=r["se"], z_twoway=r["z"], ci_lo_twoway=r["lo"], ci_hi_twoway=r["hi"],
                     mcse_ci_endpoint=r["mcse"],
                     ci_lo_twoway_iqr=r["lo_r"], ci_hi_twoway_iqr=r["hi_r"], ci_lo_jackknife=r["jlo"],
                     ci_hi_jackknife=r["jhi"], mde80=r["mde"], mde80_iqr=r["mde_r"], mde80_jackknife=r["mde_j"],
                     k_fields=r["k"], verdict=csv_verdict(v), edge_flag=eflag, key=f"test|{tid}",
                     note=f"{label}; prediction {'< 0' if side == 'neg' else '> 0'}; benchmark {bench:.4f}; "
                          f"pre-specified (self-attested, not time-stamped): {prespec}; "
                          f"post-hoc verdict under IQR-robust interval: {csv_verdict(vs['iqr'])}; under two-way "
                          f"jackknife: {csv_verdict(vs['jack'])}"
                          + ("; a CI endpoint within 2 MC SE of 0 (edge)" if e_ci else "")
                          + (f"; |MDE80 - benchmark| = {abs(r['mde'] - bench):.4f} within 2 MC SE of MDE80 "
                             f"({2 * mcse_mde(r):.4f}): shown as underpowered/null (edge)" if e_mde else "")
                          + (f"; F-loading slope {rf['est']:+.4f} [{rf['lo']:+.4f}, {rf['hi']:+.4f}]" if rf else "")))
    return v


def main():
    stage("start")
    ar, gen, er, panel_a, panel_b, cov, start = us_data()
    stage(f"panels: A {len(panel_a)} rows, B {len(panel_b)} rows")
    cells_a = us_cells(panel_a, HZ_A, ["full", "sat", "adm", "satcov"], big=True)
    cells_b = us_cells(panel_b, HZ_B, ["full", "sat", "adm"],
                       cohort_grps=lambda c: "c2001_2010" if int(c) <= 2010 else "c2013_2016")
    pa_c = panel_a[panel_a.coal01 == True]
    pb_c = panel_b[panel_b.coal01 == True]
    cells_ac = us_cells(pa_c, HZ_A, ["sat"])
    cells_bc = us_cells(pb_c, HZ_B, ["sat"])
    SA = run_sample("A", cells_a, SPECS_A)
    chk = check_reproduction(SA)
    stage(f"reproduction checks passed: {chk}")
    SB = run_sample("B", cells_b, SPECS_B)
    SAC = run_sample("A_coal", cells_ac, SPECS_C)
    SBC = run_sample("B_coal", cells_bc, SPECS_C)
    ucells, uinfo, UP = uk_data()
    stage(f"UK cells built ({len(ucells)}); reproduction of scripts/62 career cells max |diff| "
          f"{uinfo['max_abs_diff_ref']:.2g}")
    SU = run_sample("UK", ucells, SPECS_U)
    cal_cells, d5_cells = time_cells(er, ar, gen, cov)
    stage(f"calendar-time cells built (triple-matched {len(cal_cells)}, y5 pair-matched {len(d5_cells)})")
    SCAL = run_sample("CAL", cal_cells, SPECS_T)
    SD5 = run_sample("D5", d5_cells, SPECS_T)

    early = np.array([-0.25, 0.25, 0.0])
    late = np.array([0.0, -0.2, 0.2])
    all_summaries(SA, SPECS_A, HZ_A, W_A, {"y1->y5": early, "y5->y10": late}, grps=("all", "n25"))
    diffs(SA, [("S2:G", "S2:SAT"), ("S2:G", "S2:F"), ("S2:G", "S1:G"), ("S2:F", "S1:F"), ("S1:G", "S1:F"),
               ("S3:G", "S3:NADM"), ("S3:G", "S3:F"), ("S3:G", "S3FG:G"), ("pGbF", "pFbG"), ("pGb", "rG"),
               ("S0:G", "S0:F"), ("S5:G", "S2c:G"), ("S5:SAT", "S2c:SAT"), ("S4:G", "S2:G"), ("S6:G", "S2:G")],
          W_A, extra={"y5->y10": late})
    all_summaries(SB, SPECS_B, HZ_B, W_B, {}, grps=("all", "c2001_2010", "c2013_2016"))
    diffs(SB, [("S2:G", "S2:SAT"), ("S2:G", "S2:F"), ("S2:G", "S1:G"), ("S1:G", "S1:F"), ("S3:G", "S3:NADM"),
               ("S3:G", "S3:F"), ("pGbF", "pFbG"), ("S0:G", "S0:F")], W_B)
    all_summaries(SAC, SPECS_C, HZ_A, W_A, {})
    all_summaries(SBC, SPECS_C, HZ_B, W_B, {})
    diffs(SAC, [("S2:G", "S2:SAT"), ("S2:G", "S2:F")], W_A)
    diffs(SBC, [("S2:G", "S2:SAT"), ("S2:G", "S2:F")], W_B)
    all_summaries(SU, SPECS_U, ["yag1", "yag3", "yag5"], W_U, {})
    diffs(SU, [("U2:G", "U2:SEL"), ("U2:G", "U1:G"), ("pGsel", "pSELg")], W_U)
    # ---- revision 1 (exploratory): scale-free contrasts and the calendar-time decomposition
    diffs(SA, [("S2n:G", "S2n:SAT"), ("S2n:G", "S2n:F")], W_A)
    diffs(SU, [("U2n:G", "U2n:SEL")], W_U)
    LR = {}
    for nm, S, h0, h1, pairs in (("UK", SU, 0, 2, (("U2:G", "b_G"), ("U2:SEL", "b_SEL"))),
                                 ("A", SA, 0, 2, (("S2:G", "b_G"), ("S2:SAT", "b_SAT"), ("S2:F", "b_F")))):
        r2s = "U2n:R2" if nm == "UK" else "S2n:R2"
        half = [(r2s, h1, 0.5), (r2s, h0, -0.5)]
        LR[(nm, "half_r2")] = summ_logratio(S, half, "log ratio: 1/2 R2", "last/first")
        for st, lab in pairs:
            base = [(st, h1, 1.0), (st, h0, -1.0)]
            LR[(nm, lab)] = summ_logratio(S, base, f"log ratio: {lab}", "last/first")
            LR[(nm, lab + "_net")] = summ_logratio(S, base + [(t, h, -c) for t, h, c in half],
                                                   f"log ratio: {lab} minus 1/2 R2", "last/first")
    LR[("UK", "G-SEL")] = summ_logratio(SU, [("U2:G", 2, 1.0), ("U2:G", 0, -1.0), ("U2:SEL", 2, -1.0),
                                             ("U2:SEL", 0, 1.0)], "log ratio: b_G minus b_SEL", "last/first")
    TW = {"within": np.array([-1.0, 1.0, 0.0]) / 9, "calendar": np.array([0.0, 1.0, -1.0]) / 9,
          "drift_y1": np.array([-1.0, 0.0, 1.0]) / 9}
    for st in CAL_STATS:
        for cname, cw in TW.items():
            summ(SCAL, st, cw, cname)
        summ(SD5, st, np.array([-1.0, 1.0]) / 9, "drift_y5")
    stage("summaries done")

    R = RES
    k = lambda S, st, c="slope", g="all": f"{S}|{st}|{c}|{g}"
    # ---------------- pre-specified tests ----------------
    bA = abs(R[k("A", "S1:G")]["est"])
    add_test("T1", "d b_G/dh < 0 (signal decay), S2, sample A", R[k("A", "S2:G")], "neg", bA)
    add_test("T2", "d b_SAT/dh > 0 (learning about ability), S2, sample A", R[k("A", "S2:SAT")], "pos", bA)
    add_test("T3", "rise loads on G, F flat survives S2, sample A", R[k("A", "S2:G")], "pos", bA, rule="t3",
             rf=R[k("A", "S2:F")])
    bAp = abs(R[k("A", "rG")]["est"])
    add_test("T1p", "slope of rho(G, Y | broad selectivity set) < 0, sample A", R[k("A", "pGb")], "neg", bAp,
             prespec="primary (partial form)")
    add_test("T2p", "slope of rho(SAT, Y | G, F, Pell, control, state) > 0, sample A", R[k("A", "pSAT")], "pos", bAp,
             prespec="primary (partial form)")
    add_test("T3p", "slope rho(G,Y|broad+F) > 0 with rho(F,Y|broad+G) flat, sample A", R[k("A", "pGbF")], "pos", bAp,
             rule="t3", rf=R[k("A", "pFbG")], prespec="primary (partial form)")
    bB = abs(R[k("B", "S1:G")]["est"])
    add_test("T1B", "d b_G/dh < 0, S2, sample B (y1->y5, cohorts 2001-2016)", R[k("B", "S2:G")], "neg", bB,
             prespec="secondary")
    add_test("T2B", "d b_SAT/dh > 0, S2, sample B", R[k("B", "S2:SAT")], "pos", bB, prespec="secondary")
    add_test("T3B", "G rise, F flat, S2, sample B", R[k("B", "S2:G")], "pos", bB, rule="t3", rf=R[k("B", "S2:F")],
             prespec="secondary")
    bA3 = abs(R[k("A", "S3FG:G")]["est"])
    add_test("T1a", "d b_G/dh < 0, S3 (-ADM_RATE), sample A", R[k("A", "S3:G")], "neg", bA3, prespec="secondary")
    add_test("T2a", "d b_(-ADM)/dh > 0, S3, sample A", R[k("A", "S3:NADM")], "pos", bA3, prespec="secondary")
    add_test("T3a", "G rise, F flat, S3, sample A", R[k("A", "S3:G")], "pos", bA3, rule="t3", rf=R[k("A", "S3:F")],
             prespec="secondary")
    bL = abs(R[k("A", "S1:G", "y5->y10")]["est"])
    add_test("T1late", "d b_G/dh < 0 on y5->y10 only, S2, sample A", R[k("A", "S2:G", "y5->y10")], "neg", bL,
             prespec="robustness")
    add_test("T2late", "d b_SAT/dh > 0 on y5->y10 only, S2, sample A", R[k("A", "S2:SAT", "y5->y10")], "pos", bL,
             prespec="robustness")
    for nm, lab in (("A_coal", "sample A, states in PSEO from 2001"), ("B_coal", "sample B, states in PSEO from 2001")):
        bc = abs(R[k(nm, "S1:G")]["est"])
        add_test(f"T1_{nm}", f"d b_G/dh < 0, S2, {lab}", R[k(nm, "S2:G")], "neg", bc, prespec="robustness")
        add_test(f"T2_{nm}", f"d b_SAT/dh > 0, S2, {lab}", R[k(nm, "S2:SAT")], "pos", bc, prespec="robustness")
    bU = abs(R[k("UK", "U1:G")]["est"])
    add_test("T1u", "UK: d b_G/dYAG < 0 given selectivity", R[k("UK", "U2:G")], "neg", bU, prespec="UK analogue")
    add_test("T2u", "UK: d b_SEL/dYAG > 0 given G", R[k("UK", "U2:SEL")], "pos", bU, prespec="UK analogue")
    add_test("T1up", "UK: slope of rho(G, Y | selectivity) < 0", R[k("UK", "pGsel")], "neg", bU,
             prespec="UK analogue (partial form)")
    add_test("T2up", "UK: slope of rho(selectivity, Y | G) > 0", R[k("UK", "pSELg")], "pos", bU,
             prespec="UK analogue (partial form)")
    stage("tests done")
    for t in TESTS:
        print(f"{t['tid']:10s} {t['est']:+.4f} tw[{t['lo']:+.4f}, {t['hi']:+.4f}] iqr[{t['lo_r']:+.4f}, {t['hi_r']:+.4f}] "
              f"jk[{t['jlo']:+.4f}, {t['jhi']:+.4f}] mde {t['mde']:.4f}/{t['mde_r']:.4f}/{t['mde_j']:.4f} "
              f"bench {t['bench']:.4f} -> {t['verdict']} / {t['verdict_iqr']} / {t['verdict_jack']}"
              + (" EDGE" if t["edge"] else "") + (" guarded" if t["guarded"] else "")
              + (f" | F {t['rf_est']:+.4f} [{t['rf_lo']:+.4f}, {t['rf_hi']:+.4f}] jk[{t['rf_jlo']:+.4f}, {t['rf_jhi']:+.4f}]"
                 if t['rf_key'] else ""))
    D = {nm: descriptives(S, hz) for nm, S, hz in (("A", SA, HZ_A), ("B", SB, HZ_B), ("A_coal", SAC, HZ_A),
                                                    ("B_coal", SBC, HZ_B), ("UK", SU, ["yag1", "yag3", "yag5"]),
                                                    ("CAL", SCAL, ["c_y1", "c_y10", "c9_y1"]),
                                                    ("D5", SD5, ["c_y5", "c9_y5"]))}
    for nm, d in D.items():
        print(nm, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in d.items()})
    keep = {"S0", "S1", "S2", "S3", "pGb", "pGbF", "pFbG", "pSAT", "U1", "U2", "pGsel", "pSELg"}
    cell_rows(SA, HZ_A, W_A, keep)
    cell_rows(SB, HZ_B, W_B, keep)
    cell_rows(SU, ["yag1", "yag3", "yag5"], W_U, keep)
    for (nm, lab), r in LR.items():
        print(f"log ratio {nm} {lab}: " + ("undefined (a mean loading <= 0)" if r is None else
                                           f"{r['est']:+.4f} tw[{r['lo']:+.4f}, {r['hi']:+.4f}] "
                                           f"jk[{r['jlo']:+.4f}, {r['jhi']:+.4f}] x{np.exp(r['est']):.3f}"))
    for st in CAL_STATS:
        print(f"calendar {st}: " + " ".join(f"{c} {RES[f'CAL|{st}|{c}|all']['est']:+.4f}"
                                            for c in ("within", "calendar", "drift_y1"))
              + f" drift_y5 {RES[f'D5|{st}|drift_y5|all']['est']:+.4f}")
    ctx = dict(SA=SA, SB=SB, SAC=SAC, SBC=SBC, SU=SU, chk=chk, uinfo=uinfo, UP=UP, D=D, cov=cov, start=start,
               panel_a=panel_a, panel_b=panel_b, SCAL=SCAL, SD5=SD5, LR=LR)
    write_csv()
    make_figure(ctx)
    write_md(ctx)
    stage("outputs written")
    return ctx




# ---------------------------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------------------------
CSV_COLS = ["section", "sample", "spec", "stat", "contrast", "group", "field", "cohort", "n", "estimate",
            "se_twoway", "z_twoway", "ci_lo_twoway", "ci_hi_twoway", "twoway_guard_applied", "se_twoway_unguarded",
            "mcse_ci_endpoint", "se_twoway_iqr", "ci_lo_twoway_iqr", "ci_hi_twoway_iqr", "se_jackknife",
            "ci_lo_jackknife", "ci_hi_jackknife", "jackknife_guard_applied", "ci_lo_crossed", "ci_hi_crossed",
            "ci_lo_fields_fixed", "ci_hi_fields_fixed", "v_field", "v_inst", "v_field_x_inst", "noise_ratio_boot",
            "noise_ratio_jack", "mde80", "mde80_iqr", "mde80_jackknife", "k_fields", "n_positive_fields", "verdict",
            "edge_flag", "key", "note"]


def write_csv():
    df = pd.DataFrame(ROWS).reindex(columns=CSV_COLS)
    for c in ("twoway_guard_applied", "jackknife_guard_applied"):
        df[c] = df[c].map({True: "yes", False: "no"}).fillna("")
    df.to_csv(OUT_CSV, index=False, float_format="%.6g")
    # revision 1: every non-empty text cell must survive pandas' default missing-value parsing
    back = pd.read_csv(OUT_CSV, low_memory=False)
    for c in ("verdict", "edge_flag", "key", "stat", "contrast", "group", "sample", "spec", "section", "field"):
        lost = (df[c].fillna("").astype(str) != "") & back[c].isna()
        assert not lost.any(), (c, df.loc[lost, c].unique()[:5])
    print(f"[csv] {OUT_CSV} ({len(df)} rows)")


COLV = {"G": "#2a78d6", "F": "#eb6834", "SAT": "#1baf7a", "SEL": "#1baf7a", "PELL": "#eda100"}
MKV = {"G": "o", "F": "s", "SAT": "^", "SEL": "^", "PELL": "D"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.axhline(0, color=INK2, lw=0.8)


def _levels(ax, sample, spec, vars_, hz, yrs, ref=None, title=""):
    span = yrs[-1] - yrs[0]
    offs = np.linspace(-0.035, 0.035, len(vars_)) * span
    for j, v in enumerate(vars_):
        rr = [RES[f"{sample}|{spec}:{v}|{h}|all"] for h in hz]
        est = np.array([r["est"] for r in rr])
        lo = np.array([r["lo"] for r in rr])
        hi = np.array([r["hi"] for r in rr])
        ax.errorbar(yrs + offs[j], est, yerr=[est - lo, hi - est], color=COLV[v], marker=MKV[v], ms=6, lw=2,
                    capsize=3, label=f"b_{v} ({VAR_LAB[v]})")
    if ref is not None:
        rs, rv, lab = ref
        est = np.array([RES[f"{sample}|{rs}:{rv}|{h}|all"]["est"] for h in hz])
        ax.plot(yrs, est, color=INK2, ls="--", marker="o", mfc="white", ms=6, lw=1.4, label=lab)
    ax.set_xticks(yrs)
    ax.set_xlabel("years since graduation", color=INK2, fontsize=9)
    ax.set_ylabel("standardized rank-regression loading", color=INK2, fontsize=9)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    _style(ax)
    ax.legend(fontsize=7.2, frameon=False, loc="upper left")


FOREST = [("US A: b_G, F + G only (S1)", "A|S1:G|slope|all"),
          ("US A: b_G, + selectivity set (S2)", "A|S2:G|slope|all"),
          ("US A: b_F (S2)", "A|S2:F|slope|all"),
          ("US A: b_SAT (S2)", "A|S2:SAT|slope|all"),
          ("US A: b_Pell (S2)", "A|S2:PELL|slope|all"),
          ("US A: rho(G, Y) raw", "A|rG|slope|all"),
          ("US A: rho(G, Y | selectivity set)", "A|pGb|slope|all"),
          ("US A: rho(G, Y | set + F)", "A|pGbF|slope|all"),
          ("US A: rho(F, Y | set + G)", "A|pFbG|slope|all"),
          ("US A: rho(SAT, Y | G, F, Pell, ctrl, state)", "A|pSAT|slope|all"),
          ("US B (y1->y5): b_G (S2)", "B|S2:G|slope|all"),
          ("US B (y1->y5): b_SAT (S2)", "B|S2:SAT|slope|all"),
          ("UK: b_G, G only", "UK|U1:G|slope|all"),
          ("UK: b_G, + selectivity", "UK|U2:G|slope|all"),
          ("UK: b_SEL, + G", "UK|U2:SEL|slope|all")]


def make_figure(ctx):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), layout="constrained")
    fig.set_facecolor("#fcfcfb")
    for ax in axes.ravel():
        ax.set_facecolor("#fcfcfb")
    _levels(axes[0, 0], "A", "S2", ["G", "F", "SAT", "PELL"], HZ_A, YRS_A,
            ref=("S1", "G", "b_G with F + G only (S1, same cells)"),
            title="A. US fixed cohorts 2001-2010 (PSEO V4.14.1), S2: G + F + SAT + Pell + state + control\n"
                  "mean over fields; bars = pre-specified two-way 95% CI")
    _levels(axes[0, 1], "B", "S2", ["G", "F", "SAT", "PELL"], HZ_B, YRS_B,
            ref=("S1", "G", "b_G with F + G only (S1, same cells)"),
            title="B. US y1/y5 panel, cohorts 2001-2016, S2\nmean over fields; bars = pre-specified two-way 95% CI")
    _levels(axes[1, 0], "UK", "U2", ["G", "SEL"], ["yag1", "yag3", "yag5"], YRS_U,
            ref=("U1", "G", "b_G with G only (= coupling)"),
            title="C. UK LEO fixed cohorts 2013/14-2016/17, G + institution selectivity\n"
                  "mean over subjects; bars = two-way 95% CI")
    ax = axes[1, 1]
    n = len(FOREST)
    for i, (lab, key) in enumerate(FOREST):
        r = RES[key]
        y = n - i
        col = INK if not key.startswith("UK") else "#4a3aa7"
        ax.plot([r["lo"], r["hi"]], [y + 0.12, y + 0.12], color=col, lw=2.2, solid_capstyle="round")
        ax.plot([r["jlo"], r["jhi"]], [y - 0.14, y - 0.14], color=col, lw=1.0, alpha=0.6)
        ax.plot(r["est"], y + 0.12, "o", color=col, ms=5.5)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(n, 0, -1))
    ax.set_yticklabels([l for l, _ in FOREST], fontsize=7.6, color=INK)
    ax.set_xlabel("change per year since graduation", color=INK2, fontsize=9)
    ax.set_title("D. slopes: thick = pre-specified two-way CI, thin = two-way jackknife CI\n"
                 "(US A: years 1, 5, 10; US B: 1, 5; UK: 1, 3, 5)", fontsize=9.5, color=INK, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.savefig(OUT_FIG, dpi=140, facecolor="#fcfcfb", bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"[fig] {OUT_FIG}")


# --- markdown helpers ---------------------------------------------------------------------------
def fm(x, d=3):
    return "—" if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def cis(lo, hi, d=3):
    return f"[{lo:+.{d}f}, {hi:+.{d}f}]"


def esc(x: str) -> str:
    return x.replace("|", "\\|")


def side(lo, hi):
    return "above 0" if lo > 0 else "below 0" if hi < 0 else "includes 0"


def G(key):
    return RES[key]


def e3(key, d=3):
    """estimate + pre-specified two-way CI."""
    r = G(key)
    return f"{fm(r['est'], d)} {cis(r['lo'], r['hi'], d)}"


def e3all(key, d=3):
    r = G(key)
    return (f"{fm(r['est'], d)}; two-way {cis(r['lo'], r['hi'], d)}, IQR {cis(r['lo_r'], r['hi_r'], d)}, "
            f"jackknife {cis(r['jlo'], r['jhi'], d)}")


def sides3(key):
    r = G(key)
    s = [side(r["lo"], r["hi"]), side(r["lo_r"], r["hi_r"]), side(r["jlo"], r["jhi"])]
    if len(set(s)) == 1:
        return f"CI {s[0]} under all three intervals"
    names = ["the pre-specified two-way interval", "the IQR interval", "the jackknife interval"]
    parts = []
    for val in dict.fromkeys(s):
        who = [n for n, x in zip(names, s) if x == val]
        parts.append(f"{val} under {' and '.join(who)}")
    return "CI " + "; ".join(parts)


def T(tid):
    return next(t for t in TESTS if t["tid"] == tid)


def vtxt(tid):
    t = T(tid)
    base = f"verdict {t['verdict_disp']}" + (" (edge)" if t["edge"] else "")
    if not t["edge_mde"] and t["verdict"] == t["verdict_iqr"] == t["verdict_jack"]:
        return base + " (same under the post-hoc IQR and jackknife intervals)"
    return base + f" (post-hoc IQR interval: {t['verdict_iqr']}; post-hoc jackknife: {t['verdict_jack']})"


def prespec_block() -> str:
    """The pre-specification block of the module docstring, verbatim (hashed in the write-up)."""
    d = __doc__
    i, j = d.index("PRE-SPECIFIED DESIGN AND TESTS"), d.index("REVISION 1")
    return d[i:j].rstrip() + "\n"


def calendar_windows() -> list[tuple]:
    out = []
    for c in COH_B:
        cy = int(c)
        row = [f"{cy}-{cy + 2}"]
        for h in (1, 5, 10):
            row.append(f"{cy + h}-{cy + 2 + h}" if (h < 10 or c in COH_A) else "not released")
        out.append((c, *row))
    return out


def md5(p: Path) -> str:
    return s59.md5(p)


KEYROWS = [
    # (label, key, sample text, decimals)
    ("b_G slope, F + G only (scripts/59 reproduction)", "A|S0:G|slope|all", "A, full panel (n>=15)", 3),
    ("b_F slope, F + G only (scripts/59 reproduction)", "A|S0:F|slope|all", "A, full panel (n>=15)", 3),
    ("b_G slope - b_F slope, F + G only", "A|S0:G - S0:F|slope|all", "A, full panel", 3),
    ("b_G slope, F + G only (S1)", "A|S1:G|slope|all", "A, SAT-sample cells", 3),
    ("b_F slope, F + G only (S1)", "A|S1:F|slope|all", "A, SAT-sample cells", 3),
    ("b_G slope - b_F slope (S1)", "A|S1:G - S1:F|slope|all", "A, SAT-sample cells", 3),
    ("**b_G slope (S2, T1)**", "A|S2:G|slope|all", "A, S2", 3),
    ("b_F slope (S2)", "A|S2:F|slope|all", "A, S2", 3),
    ("**b_SAT slope (S2, T2)**", "A|S2:SAT|slope|all", "A, S2", 3),
    ("b_Pell slope (S2)", "A|S2:PELL|slope|all", "A, S2", 3),
    ("b_state-level slope (S2)", "A|S2:ST|slope|all", "A, S2", 3),
    ("b_G slope - b_SAT slope (S2)", "A|S2:G - S2:SAT|slope|all", "A, S2", 3),
    ("b_G slope - b_F slope (S2)", "A|S2:G - S2:F|slope|all", "A, S2", 3),
    ("b_G slope, S2 minus S1 (effect of the controls)", "A|S2:G - S1:G|slope|all", "A, same cells", 3),
    ("b_F slope, S2 minus S1", "A|S2:F - S1:F|slope|all", "A, same cells", 3),
    ("rho(G, Y) slope, raw", "A|rG|slope|all", "A, SAT-sample cells", 3),
    ("rho(F, Y) slope, raw", "A|rF|slope|all", "A, SAT-sample cells", 3),
    ("rho(SAT, Y) slope, raw", "A|rSAT|slope|all", "A, SAT-sample cells", 3),
    ("**rho(G, Y | SAT, ADM, Pell, control, state) slope (T1p)**", "A|pGb|slope|all", "A, partial", 3),
    ("rho(G, Y | broad + F) slope (T3p)", "A|pGbF|slope|all", "A, partial", 3),
    ("rho(F, Y | broad + G) slope (T3p)", "A|pFbG|slope|all", "A, partial", 3),
    ("**rho(SAT, Y | G, F, Pell, control, state) slope (T2p)**", "A|pSAT|slope|all", "A, partial", 3),
    ("rho(G, Y | broad) minus raw rho(G, Y), slope", "A|pGb - rG|slope|all", "A, partial", 3),
    ("rho(G, Y | broad + F) minus rho(F, Y | broad + G), slope", "A|pGbF - pFbG|slope|all", "A, partial", 3),
    ("b_G slope, S3 (-ADM_RATE) (T1a)", "A|S3:G|slope|all", "A, ADM sample", 3),
    ("b_F slope, S3", "A|S3:F|slope|all", "A, ADM sample", 3),
    ("b_(-ADM) slope, S3 (T2a)", "A|S3:NADM|slope|all", "A, ADM sample", 3),
    ("b_Pell slope, S3", "A|S3:PELL|slope|all", "A, ADM sample", 3),
    ("b_G slope, F + G only, ADM cells", "A|S3FG:G|slope|all", "A, ADM sample", 3),
    ("b_G slope, S3 minus F + G only", "A|S3:G - S3FG:G|slope|all", "A, ADM sample", 3),
    ("rho(G, Y | ADM, Pell, control, state) slope", "A|pGbA|slope|all", "A, ADM sample", 3),
    ("b_G, y1->y5 segment (S1)", "A|S1:G|y1->y5|all", "A", 3),
    ("b_G, y5->y10 segment (S1)", "A|S1:G|y5->y10|all", "A", 3),
    ("b_G, y1->y5 segment (S2)", "A|S2:G|y1->y5|all", "A", 3),
    ("b_G, y5->y10 segment (S2) (T1late)", "A|S2:G|y5->y10|all", "A", 3),
    ("b_SAT, y1->y5 segment (S2)", "A|S2:SAT|y1->y5|all", "A", 3),
    ("b_SAT, y5->y10 segment (S2) (T2late)", "A|S2:SAT|y5->y10|all", "A", 3),
    ("rho(G, Y | broad), y5->y10 segment", "A|pGb|y5->y10|all", "A", 3),
    ("rho(SAT, Y | ...), y5->y10 segment", "A|pSAT|y5->y10|all", "A", 3),
    ("b_G slope (S1), cells n>=25 (exploratory)", "A|S1:G|slope|n25", "A, n>=25", 3),
    ("b_G slope (S2), cells n>=25 (exploratory)", "A|S2:G|slope|n25", "A, n>=25", 3),
    ("b_F slope (S2), cells n>=25 (exploratory)", "A|S2:F|slope|n25", "A, n>=25", 3),
    ("b_SAT slope (S2), cells n>=25 (exploratory)", "A|S2:SAT|slope|n25", "A, n>=25", 3),
    ("rho(G, Y | broad) slope, cells n>=25 (exploratory)", "A|pGb|slope|n25", "A, n>=25", 3),
    ("rho(SAT, Y | ...) slope, cells n>=25 (exploratory)", "A|pSAT|slope|n25", "A, n>=25", 3),
    ("b_G slope, S2 + ADM_RATE (S4, exploratory)", "A|S4:G|slope|all", "A", 3),
    ("b_SAT slope, S4 (exploratory)", "A|S4:SAT|slope|all", "A", 3),
    ("b_G slope, S2 without F (S6, exploratory)", "A|S6:G|slope|all", "A", 3),
    ("b_SAT slope, S6 (exploratory)", "A|S6:SAT|slope|all", "A", 3),
    ("b_G slope, S2 + same-horizon coverage (S5, exploratory)", "A|S5:G|slope|all", "A, coverage-complete", 3),
    ("b_G slope, S5 minus S2 on the same cells", "A|S5:G - S2c:G|slope|all", "A, coverage-complete", 3),
    ("b_SAT slope, S5 (exploratory)", "A|S5:SAT|slope|all", "A, coverage-complete", 3),
    ("log earnings on S2 ranks: G (log points per SD per yr, exploratory)", "A|S2L:G|slope|all", "A", 4),
    ("log earnings on S2 ranks: SAT (exploratory)", "A|S2L:SAT|slope|all", "A", 4),
    ("log earnings on S2 ranks: Pell (exploratory)", "A|S2L:PELL|slope|all", "A", 4),
    ("b_G slope, F + G only", "B|S0:G|slope|all", "B, full panel", 3),
    ("b_F slope, F + G only", "B|S0:F|slope|all", "B, full panel", 3),
    ("b_G slope (S1)", "B|S1:G|slope|all", "B, SAT-sample cells", 3),
    ("**b_G slope (S2, T1B)**", "B|S2:G|slope|all", "B, S2", 3),
    ("b_F slope (S2)", "B|S2:F|slope|all", "B, S2", 3),
    ("**b_SAT slope (S2, T2B)**", "B|S2:SAT|slope|all", "B, S2", 3),
    ("b_Pell slope (S2)", "B|S2:PELL|slope|all", "B, S2", 3),
    ("b_G slope - b_SAT slope (S2)", "B|S2:G - S2:SAT|slope|all", "B, S2", 3),
    ("b_G slope - b_F slope (S2)", "B|S2:G - S2:F|slope|all", "B, S2", 3),
    ("b_G slope, S2 minus S1", "B|S2:G - S1:G|slope|all", "B, same cells", 3),
    ("rho(G, Y) slope, raw", "B|rG|slope|all", "B", 3),
    ("rho(SAT, Y) slope, raw", "B|rSAT|slope|all", "B", 3),
    ("rho(G, Y | broad) slope", "B|pGb|slope|all", "B, partial", 3),
    ("rho(G, Y | broad + F) slope", "B|pGbF|slope|all", "B, partial", 3),
    ("rho(F, Y | broad + G) slope", "B|pFbG|slope|all", "B, partial", 3),
    ("rho(SAT, Y | ...) slope", "B|pSAT|slope|all", "B, partial", 3),
    ("b_G slope, S3", "B|S3:G|slope|all", "B, ADM sample", 3),
    ("b_(-ADM) slope, S3", "B|S3:NADM|slope|all", "B, ADM sample", 3),
    ("b_G slope (S1), cohorts 2001-2010", "B|S1:G|slope|c2001_2010", "B, exploratory split", 3),
    ("b_G slope (S1), cohorts 2013-2016", "B|S1:G|slope|c2013_2016", "B, exploratory split", 3),
    ("b_G slope (S2), cohorts 2001-2010", "B|S2:G|slope|c2001_2010", "B, exploratory split", 3),
    ("b_G slope (S2), cohorts 2013-2016", "B|S2:G|slope|c2013_2016", "B, exploratory split", 3),
    ("b_SAT slope (S2), cohorts 2001-2010", "B|S2:SAT|slope|c2001_2010", "B, exploratory split", 3),
    ("b_SAT slope (S2), cohorts 2013-2016", "B|S2:SAT|slope|c2013_2016", "B, exploratory split", 3),
    ("rho(G, Y | broad) slope, cohorts 2001-2010", "B|pGb|slope|c2001_2010", "B, exploratory split", 3),
    ("rho(G, Y | broad) slope, cohorts 2013-2016", "B|pGb|slope|c2013_2016", "B, exploratory split", 3),
    ("rho(SAT, Y | ...) slope, cohorts 2001-2010", "B|pSAT|slope|c2001_2010", "B, exploratory split", 3),
    ("rho(SAT, Y | ...) slope, cohorts 2013-2016", "B|pSAT|slope|c2013_2016", "B, exploratory split", 3),
    ("b_G slope (S1)", "A_coal|S1:G|slope|all", "A, states in PSEO from 2001", 3),
    ("b_G slope (S2)", "A_coal|S2:G|slope|all", "A, states in PSEO from 2001", 3),
    ("b_F slope (S2)", "A_coal|S2:F|slope|all", "A, states in PSEO from 2001", 3),
    ("b_SAT slope (S2)", "A_coal|S2:SAT|slope|all", "A, states in PSEO from 2001", 3),
    ("rho(G, Y | broad) slope", "A_coal|pGb|slope|all", "A, states in PSEO from 2001", 3),
    ("rho(SAT, Y | ...) slope", "A_coal|pSAT|slope|all", "A, states in PSEO from 2001", 3),
    ("b_G slope (S1)", "B_coal|S1:G|slope|all", "B, states in PSEO from 2001", 3),
    ("b_G slope (S2)", "B_coal|S2:G|slope|all", "B, states in PSEO from 2001", 3),
    ("b_F slope (S2)", "B_coal|S2:F|slope|all", "B, states in PSEO from 2001", 3),
    ("b_SAT slope (S2)", "B_coal|S2:SAT|slope|all", "B, states in PSEO from 2001", 3),
    ("rho(G, Y | broad) slope", "B_coal|pGb|slope|all", "B, states in PSEO from 2001", 3),
    ("rho(SAT, Y | ...) slope", "B_coal|pSAT|slope|all", "B, states in PSEO from 2001", 3),
    ("b_G slope, G only (= coupling slope)", "UK|U1:G|slope|all", "UK", 3),
    ("**b_G slope, G + selectivity (T1u)**", "UK|U2:G|slope|all", "UK", 3),
    ("**b_SEL slope, G + selectivity (T2u)**", "UK|U2:SEL|slope|all", "UK", 3),
    ("b_G slope - b_SEL slope", "UK|U2:G - U2:SEL|slope|all", "UK", 3),
    ("b_G slope, with minus without selectivity", "UK|U2:G - U1:G|slope|all", "UK", 3),
    ("rho(G, Y | selectivity) slope (T1up)", "UK|pGsel|slope|all", "UK", 3),
    ("rho(selectivity, Y | G) slope (T2up)", "UK|pSELg|slope|all", "UK", 3),
    ("rho(G, Y | sel.) minus rho(sel., Y | G), slope", "UK|pGsel - pSELg|slope|all", "UK", 3),
    ("rho(selectivity, Y) slope, raw", "UK|rSEL|slope|all", "UK", 3),
]


def write_md(ctx):
    D, chk, ui = ctx["D"], ctx["chk"], ctx["uinfo"]
    SA, SB, SU = ctx["SA"], ctx["SB"], ctx["SU"]
    dA, dB, dU = D["A"], D["B"], D["UK"]
    R = RES
    L = []
    A = L.append
    t1, t2, t3, t1p, t2p, t3p = (T(x) for x in ("T1", "T2", "T3", "T1p", "T2p", "T3p"))
    s1g, s2g, s2f, s2s, s2p = (G(f"A|{x}|slope|all") for x in ("S1:G", "S2:G", "S2:F", "S2:SAT", "S2:PELL"))
    pgb, psat, rg, rsat = (G(f"A|{x}|slope|all") for x in ("pGb", "pSAT", "rG", "rSAT"))
    dctl, dctlp = G("A|S2:G - S1:G|slope|all"), G("A|pGb - rG|slope|all")
    u1, u2g, u2s = (G(f"UK|{x}|slope|all") for x in ("U1:G", "U2:G", "U2:SEL"))
    cov = ctx["cov"]
    pa_insts = set(ctx["panel_a"].inst_key)
    cova = cov[cov.inst_key.isin(pa_insts)]
    nosat = cova[cova.SAT.isna()]
    nosat_adm = nosat[nosat.ADM_RATE.notna()]
    A("# Employer learning, aggregate analogue — does the pay loading on institution-wide status fall with years "
      "since graduation once selectivity is in the model?\n")
    A(f"Script: `scripts/66_employer_learning.py` (seed {SEED}; every number below is printed by the script from its own "
      "computations; re-run byte-identical). Public data only: PSEO V4.14.1 (`data/raw/pseo_2026q2/`), College "
      "Scorecard institution file, the published Wapman et al. (2022) field and academia-wide ranks, DfE LEO provider "
      "data and the UK hiring-network prestige of scripts/62. scripts/52, 55, 59, 61 and 62 are imported, not edited. "
      "Descriptive, not causal. Tables: `data/interim/employer_learning_loadings.csv` (every aggregate, test and "
      "per-cell number); figure: `outputs/figures/employer_learning.png`.\n")

    # ------------------------------------------------------------------ what this tests
    A("## What this tests, and what it cannot (read first)\n")
    A("**Theory.** In employer-learning models with statistical discrimination, employers first pay on what they "
      "can see at hiring and learn productivity later. Farber & Gibbons (1996) show that the loading of wages on a "
      "variable employers see at hiring (schooling) need not change with experience, while the loading on a "
      "correlate of productivity that employers do not see (a test score) grows. Altonji & Pierret (2001) add that, "
      "with both in the model, the loading on the seen variable falls with experience and the loading on the hidden "
      "correlate rises. Lange (2007) estimates that this learning is fast, with employers' initial expectation "
      "errors falling by about half within the first three years. Arcidiacono, Bayer & Hizmo (2010) report that for college graduates the "
      "test-score loading is already high at entry and does not grow with experience. Bib keys "
      "(paper/refs_merged.bib): `farber1996learning`, `altonji2001employer`, `lange2007speed`, "
      "`arcidiacono2010beyond`.\n")
    A("**Mapping to this project, and why it is imperfect.**\n")
    A("- The unit here is an institution × field cell of median earnings, not a person. This is an aggregate analogue "
      "of the Altonji–Pierret design, not the individual-level test.")
    A("- G (the academia-wide Wapman rank) stands in for the institution's name or status, which every employer sees "
      "on the CV. Nothing in the data shows that employers use G itself.")
    A("- Institution SAT_AVG and ADM_RATE are correlates of the average ability of an institution's students. They "
      "are **not** a clean hidden variable: an institution's selectivity is public and correlated with its name "
      f"(mean within-cell Spearman of G with SAT_AVG {dA['r_GSAT']:+.2f}), so employers may price it at hiring. They "
      "are also measured on recent entering classes (Scorecard 'Most Recent' release), not on the entry cohorts "
      "of the graduates in the panel.")
    A("- What would count as the learning signature here: with selectivity in the model, the G loading falls with "
      "years since graduation and the selectivity loading rises. If employers already price what G and selectivity "
      "carry at hiring, neither needs to move. If G tracks something whose pay grows with experience (skills, "
      "networks, sorting into steeper careers), the G loading can rise.")
    A("- Loadings are standardized coefficients of a rank regression (or partial Spearman correlations). They "
      "describe how well each variable orders institutions' medians within a field, not dollar returns. A "
      "log-earnings version is reported as exploratory. When institution characteristics order the medians better "
      "(a higher within-cell R²), all loadings rise together, so the learning signature is a shift of relative "
      "weight from G toward selectivity, not only a larger absolute slope for selectivity; revision 1 adds "
      "exploratory scale-free contrasts for this.\n")
    A(f"**Timing (PSEO documentation, {PSEO_DOC}, accessed 2026-09-24).** 'For the Bachelor's degree level, the "
      "graduation cohorts are three-year cohorts, e.g. 2001-2003; 2004-2006; 2007-2009; 2010-2012; 2013-2015; "
      "2016-2018; 2019-2021.' 'For all post-secondary graduates, the first year post-graduation is defined as the "
      "first calendar year following their graduation year. So for a student who graduates in May of 2005, year one "
      "begins in January of 2006, year five in January 2010, etc.' 'Earnings are total annual earnings for attached "
      "workers from all jobs, converted to 2023 dollars using the CPI-U.' (scripts/52 records the same 3-year "
      "windows.) Hence the calendar years whose earnings enter each cell:\n")
    A("| cohort label | graduation years | y1 earnings years | y5 earnings years | y10 earnings years |")
    A("|---|---|---|---|---|")
    for c, gy, y1, y5, y10 in calendar_windows():
        A(f"| {c} | {gy} | {y1} | {y5} | {y10} |")
    A("")
    tax = ui["tax_offset"]
    A(f"**UK timing.** In the LEO provider file every analysed row has tax-year start = academic-year start + YAG + "
      f"{list(tax)[0]} ({sum(tax.values())} rows checked; e.g. the 2013/14 graduates are observed at YAG 1 in tax year "
      f"2015/16 and at YAG 5 in 2019/20). The UK cohort and YAG definitions are those of scripts/62.\n")

    # ------------------------------------------------------------------ answer
    # qualitative claims are computed and asserted, so a changed input cannot leave a stale sentence behind
    # (revision 1) headline rests only on the pre-specified status-decay tests
    STATUS_T = ["T1", "T1p", "T1B", "T1a", "T1late", "T1_A_coal", "T1_B_coal", "T1u", "T1up"]
    assert all(T(x)["side"] == "neg" for x in STATUS_T)
    st_below = [x for x in STATUS_T if min(T(x)["hi"], T(x)["hi_r"], T(x)["jhi"]) < 0]
    assert not st_below
    # exploratory tally of every selectivity-controlled status slope (moved out of the headline in revision 1)
    ctl_g = [k for k, r in R.items() if r["contrast"] in ("slope", "y1->y5", "y5->y10") and
             r["stat"] in ("S2:G", "S3:G", "S4:G", "S5:G", "S6:G", "S2L:G", "pGb", "pGbF", "pGbA", "U2:G", "pGsel")]
    n_below = sum(min(R[k]["hi"], R[k]["hi_r"], R[k]["jhi"]) < 0 for k in ctl_g)
    n_pos = sum(R[k]["est"] > 0 for k in ctl_g)
    neg_pt = sorted(k for k in ctl_g if R[k]["est"] <= 0)
    test_keys = {t["key"] for t in TESTS}
    n_testkeys = sum(k in test_keys for k in ctl_g)
    share_reg, share_par = s2g["est"] / s1g["est"], pgb["est"] / rg["est"]
    pgb_all = min(pgb["lo"], pgb["lo_r"], pgb["jlo"]) > 0
    u2s_all = min(u2s["lo"], u2s["lo_r"], u2s["jlo"]) > 0
    u2g_span = max(u2g["lo"], u2g["lo_r"], u2g["jlo"]) < 0 < min(u2g["hi"], u2g["hi_r"], u2g["jhi"])
    sat_span = s2s["lo"] < 0 < s2s["hi"]
    assert n_below == 0 and pgb_all and u2s_all and u2g_span and sat_span and share_reg < 1 and share_par < 1
    assert neg_pt and all(k.startswith("B") for k in neg_pt)
    # T1: post-data statement about the decays the pre-specified CI excludes
    assert t1["lo"] < 0 < t1["hi"] and -t1["lo"] < t1["bench"]
    # scale (revision 1): mean within-cell R^2 by horizon and what a uniform rise of all loadings would give
    lv = lambda key: G(key)["est"]
    r2a = [lv(f"A|S2n:R2|{h}|all") for h in HZ_A]
    r2u = [lv(f"UK|U2n:R2|{h}|all") for h in ("yag1", "yag3", "yag5")]
    sat1, sat10 = lv("A|S2:SAT|y1|all"), lv("A|S2:SAT|y10|all")
    fac_a = float(np.sqrt(r2a[2] / r2a[0]))
    sat_impl, sat_obs = sat1 * (fac_a - 1), sat10 - sat1
    us_scale_share = sat_impl / sat_obs
    ug1, ug5 = lv("UK|U2:G|yag1|all"), lv("UK|U2:G|yag5|all")
    us1, us5 = lv("UK|U2:SEL|yag1|all"), lv("UK|U2:SEL|yag5|all")
    fac_u = float(np.sqrt(r2u[2] / r2u[0]))
    uG_impl, uS_impl = ug1 * (fac_u - 1) / 4, us1 * (fac_u - 1) / 4          # per year, YAG 1 -> 5 endpoints
    ratio_g, ratio_s = ug5 / ug1, us5 / us1
    dgs, dgsp = G("UK|U2:G - U2:SEL|slope|all"), G("UK|pGsel - pSELg|slope|all")
    dgs_incl = all(lo < 0 < hi for lo, hi in ((dgs["lo"], dgs["hi"]), (dgs["lo_r"], dgs["hi_r"]),
                                              (dgs["jlo"], dgs["jhi"])))
    LR = ctx["LR"]
    lr_gs = LR[("UK", "G-SEL")]
    lr_gs_incl = lr_gs["lo"] < 0 < lr_gs["hi"] and lr_gs["jlo"] < 0 < lr_gs["jhi"] and lr_gs["lo_r"] < 0 < lr_gs["hi_r"]
    # the UK wording below depends on these four facts
    assert dgs_incl and ratio_g >= ratio_s and uS_impl > uG_impl and r2u[2] > r2u[0] and lr_gs_incl
    assert r2a[2] > r2a[0] and 0 < us_scale_share < 1
    assert t2p["edge_ci"] and s2s["lo"] < 0 < s2s["hi"]                  # "sits at the edge of 0 in partial form"
    A("## Answer\n")
    A("**Key question: once selectivity is in the model, does the loading of pay on institution-wide status G fall, "
      "stay or rise within cohort from y1 to y10 (career plus calendar time)?** No decay is detected. None of the "
      f"{len(STATUS_T)} pre-specified status-decay tests ({', '.join(STATUS_T)}) has an interval below 0, under the "
      "pre-specified interval or under either post-hoc sensitivity interval. Their pre-specified verdicts: "
      + "; ".join(f"{x} {T(x)['verdict_disp']}{' (edge)' if T(x)['edge'] else ''}" for x in STATUS_T) + ". "
      f"In the primary regression form (T1) the pre-specified CI excludes decays steeper than {-t1['lo']:.3f}/yr, "
      f"{-t1['lo'] / t1['bench']:.0%} of the uncontrolled rise ({t1['bench']:.3f}/yr), so a decay as large as the "
      "uncontrolled rise is excluded. In partial-correlation form (T1p) the status loading rises "
      f"({fm(pgb['est'])}/yr, {sides3('A|pGb|slope|all')}), the opposite of the decay prediction. "
      f"In the US the controls shrink the rise of the G loading to {share_reg:.0%} of its size in the rank regression "
      f"({fm(s1g['est'])} → {fm(s2g['est'])}/yr) and to {share_par:.0%} in partial-correlation form. The selectivity "
      f"loading also rises in point estimate ({fm(s2s['est'])}/yr in the regression, {fm(psat['est'])}/yr in partial "
      "form). Its interval includes 0 in the regression and sits at the edge of 0 in partial form, and a uniform rise "
      f"of all loadings with the within-cell R² would give about {us_scale_share:.0%} of its y1 → y10 rise (item 2). "
      f"In the UK, status net of selectivity is flat within its interval ({fm(u2g['est'])}/yr), and selectivity net "
      f"of status rises in absolute standardized units ({fm(u2s['est'])}/yr, {sides3('UK|U2:SEL|slope|all')}). The "
      f"regression-form difference of the two slopes includes 0 ({fm(dgs['est'])}; {sides3('UK|U2:G - U2:SEL|slope|all')}). "
      f"Relative to its level the status loading grows at least as fast (×{ratio_g:.2f} against ×{ratio_s:.2f} from "
      f"YAG 1 to YAG 5). A uniform rise in how well institution characteristics order the medians (mean within-cell "
      f"R² {r2u[0]:.2f} → {r2u[2]:.2f}) would give the same absolute-slope pattern. So the UK data do not single out "
      "selectivity, and they do not show a shift of relative weight toward it (item 6). The decay signature is absent "
      "in both countries. Aggregate data cannot say whether the remaining rises reflect learning, skills, networks, "
      "sorting or selection.\n")
    det1 = ("would be detected with at least 80% power" if t1["mde"] <= t1["bench"] else
            "would not quite reach 80% power")
    A(f"1. **T1, status loading decays (primary, S2, sample A): not supported.** Pre-specified {vtxt('T1')}. The slope "
      f"of b_G is {e3all('A|S2:G|slope|all')}; per field it is positive in {s2g['pos']} of {s2g['k']}. The benchmark "
      f"is {t1['bench']:.3f}, the uncontrolled rise on the same cells (S1: {e3('A|S1:G|slope|all')}). Pre-data power: "
      f"under the pre-specified interval (MDE80 {t1['mde']:.3f}) a decay of that size {det1}, but MDE80 and the "
      f"benchmark differ by {abs(t1['mde'] - t1['bench']):.3f}, "
      + ("within" if t1["edge_mde"] else "more than") + f" 2 Monte Carlo SE of MDE80 ({2 * t1['mcse_mde']:.3f}), "
      + ("so the bootstrap cannot separate 'underpowered' from 'null' (shown as underpowered/null)" if t1["edge_mde"]
         else "so the split is resolved") + f"; under the post-hoc IQR (MDE80 {t1['mde_r']:.3f}) and jackknife "
      f"({t1['mde_j']:.3f}) intervals it "
      + ("would be detected." if max(t1["mde_r"], t1["mde_j"]) <= t1["bench"] else "would not always be detected.")
      + f" Post-data: the pre-specified CI excludes decays steeper than {t1['lo']:+.3f}/yr, about "
      f"{-t1['lo'] / t1['bench']:.0%} of the benchmark, so a decay as large as the uncontrolled rise is excluded "
      + "(" + "; ".join(f"post-hoc {nm} interval: " + (f"every decay excluded, lower end {lo:+.3f}" if lo >= 0 else
                                                      f"decays steeper than {lo:+.3f}/yr excluded")
                      for nm, lo in (("IQR", t1["lo_r"]), ("jackknife", t1["jlo"]))) + ")."
      + f" Partial form, **T1p**: {vtxt('T1p')}. rho(G, Y | SAT, ADM, Pell, control, state level) rises "
      f"{e3all('A|pGb|slope|all')}. The controls remove part of the rise: b_G slope S2 minus S1 "
      f"{e3all('A|S2:G - S1:G|slope|all')} ({sides3('A|S2:G - S1:G|slope|all')}); partial minus raw coupling "
      f"{e3all('A|pGb - rG|slope|all')} ({sides3('A|pGb - rG|slope|all')}).")
    A(f"2. **T2, selectivity loading rises (primary): not established.** Pre-specified {vtxt('T2')}. Slope of b_SAT "
      f"{e3all('A|S2:SAT|slope|all')}; MDE80 {t2['mde']:.3f}. Partial form, **T2p**: {vtxt('T2p')}. "
      f"rho(SAT, Y | G, F, Pell, control, state level) {e3all('A|pSAT|slope|all')}"
      + ("; the lower end of the pre-specified interval is within 2 Monte Carlo SE of 0 (edge)" if t2p["edge_ci"] else "")
      + f". Raw selectivity coupling rises about as fast as raw status coupling: rho(SAT, Y) {e3('A|rSAT|slope|all')}, "
      f"rho(G, Y) {e3('A|rG|slope|all')}. The selectivity loading is large at every horizon (b_SAT "
      f"{fm(sat1)} at y1, {fm(sat10)} at y10; b_G "
      f"{fm(G('A|S2:G|y1|all')['est'])} and {fm(G('A|S2:G|y10|all')['est'])}). *Scale (revision 1).* These loadings "
      "are standardized rank coefficients, and the mean within-cell R² of the S2 regression rises "
      f"({r2a[0]:.3f} / {r2a[1]:.3f} / {r2a[2]:.3f} at y1 / y5 / y10; slope {e3('A|S2n:R2|slope|all', 4)}). If every "
      f"loading rose in proportion to sqrt(R²) (×{fac_a:.2f} from y1 to y10), b_SAT would rise by {sat_impl:.3f} of "
      f"its observed {sat_obs:.3f}. So part of the T2 point estimate is a common rise of all loadings rather than a "
      "shift of weight toward selectivity. The partial form T2p is scale-dependent in the same way: when the other "
      "regressors order the medians better, the residual spread falls and a partial correlation with unchanged "
      "relative weight rises. Scale-free check (exploratory): per-cell loadings divided by sqrt(R²_h) give a b_SAT "
      f"slope of {e3all('A|S2n:SAT|slope|all')} and a b_G slope of {e3all('A|S2n:G|slope|all')} (difference G − SAT "
      f"{e3all('A|S2n:G - S2n:SAT|slope|all')}); the log ratio of mean b_SAT at y10 to y1 is "
      f"{e3all('A|log ratio: b_SAT|last/first|all')}, and net of half the log ratio of mean R² it is "
      f"{e3all('A|log ratio: b_SAT minus 1/2 R2|last/first|all')} ({sides3('A|log ratio: b_SAT minus 1/2 R2|last/first|all')}). "
      "The G result is not a scale artifact: a proportional rise cannot turn a loading that starts near 0 "
      f"({fm(G('A|S2:G|y1|all')['est'])}) into {fm(G('A|S2:G|y10|all')['est'])} "
      f"(the log ratio for b_G is undefined because its y1 mean is ≤ 0).")
    dgf, dgfp = G("A|S2:G - S2:F|slope|all"), G("A|pGbF - pFbG|slope|all")
    dgfp_tw = dgfp["lo"] < 0 < dgfp["hi"]
    assert dgfp_tw and dgf["lo"] < 0 < dgf["hi"]
    A(f"3. **T3, 'the rise loads on G, F flat' survives the controls: the pre-specified rule is met in partial form, "
      f"not in the regression; the G − F difference itself is not established.** Pre-specified {vtxt('T3')}. Under "
      f"S2 the b_F slope is {e3all('A|S2:F|slope|all')}, flat as in scripts/52; the b_G slope is positive, but its "
      f"pre-specified interval includes 0. b_G − b_F slope {e3all('A|S2:G - S2:F|slope|all')}; b_G − b_SAT slope "
      f"{e3all('A|S2:G - S2:SAT|slope|all')}. Partial form, **T3p**: {vtxt('T3p')}. rho(G, Y | broad + F) "
      f"{e3('A|pGbF|slope|all')}, rho(F, Y | broad + G) {e3('A|pFbG|slope|all')}; difference "
      f"{e3all('A|pGbF - pFbG|slope|all')} ({sides3('A|pGbF - pFbG|slope|all')}). The T3-type rule compares the two "
      "intervals (G-loading slope above 0, F-loading slope including 0); the directly estimated G − F difference is "
      "not established under the pre-specified interval in either form.")
    s1b, dctlb = G("B|S1:G|slope|all"), G("B|S2:G - S1:G|slope|all")
    tb1, tb2 = T("T1B"), T("T2B")
    ctlb_txt = ("more than the whole S1 rise in point estimate" if abs(dctlb["est"]) >= abs(s1b["est"]) else
                f"{abs(dctlb['est']) / abs(s1b['est']):.0%} of the S1 rise")
    A(f"4. **Sample B (y1 → y5, cohorts 2001–2016; secondary): too imprecise to decide.** T1B: {vtxt('T1B')}; b_G "
      f"slope {e3all('B|S2:G|slope|all')}. T2B: {vtxt('T2B')}; b_SAT slope {e3all('B|S2:SAT|slope|all')}. MDE80 "
      f"{tb1['mde']:.3f} and {tb2['mde']:.3f}, against a benchmark of {tb1['bench']:.3f}. T3B: {vtxt('T3B')}. "
      f"Partial forms: rho(G, Y | broad) {e3('B|pGb|slope|all')}, rho(SAT, Y | ...) {e3all('B|pSAT|slope|all')}. "
      f"The controls take out {ctlb_txt} (S1 {e3('B|S1:G|slope|all')}; S2 minus S1 "
      f"{e3all('B|S2:G - S1:G|slope|all')}). Within a cell the institutions are the same at y1 and y5. Across "
      "cohorts the mix changes as states join PSEO; the coalition-fixed variant is in item 5, and the cohort split "
      "(Key numbers) is exploratory.")
    A(f"5. **Pre-specified robustness.** *−ADM_RATE instead of SAT_AVG* (S3, ADM sample, which keeps "
      f"{len(nosat_adm)} panel institutions without SAT_AVG): T1a {vtxt('T1a')}, b_G slope "
      f"{e3all('A|S3:G|slope|all')}; T2a {vtxt('T2a')}, b_(−ADM) slope {e3('A|S3:NADM|slope|all')}; T3a "
      f"{vtxt('T3a')}. *y5 → y10 only* (avoids the y1 medians depressed by post-graduate study): T1late "
      f"{vtxt('T1late')}, b_G {e3all('A|S2:G|y5->y10|all')}; T2late {vtxt('T2late')}, b_SAT "
      f"{e3('A|S2:SAT|y5->y10|all')}. In partial form the y5 → y10 segment gives rho(G, Y | set) "
      f"{e3('A|pGb|y5->y10|all')} ({sides3('A|pGb|y5->y10|all')}) and rho(SAT, Y | ...) "
      f"{e3('A|pSAT|y5->y10|all')} ({sides3('A|pSAT|y5->y10|all')}). *States in PSEO from 2001 only* (coalition "
      f"fixed): sample A T1 {vtxt('T1_A_coal')}, b_G {e3('A_coal|S2:G|slope|all')}; T2 {vtxt('T2_A_coal')}, "
      f"b_SAT {e3('A_coal|S2:SAT|slope|all')}; sample B T1 {vtxt('T1_B_coal')}, b_G {e3('B_coal|S2:G|slope|all')}; "
      f"T2 {vtxt('T2_B_coal')}, b_SAT {e3('B_coal|S2:SAT|slope|all')}.")
    t1u = T("T1u")
    A(f"6. **UK analogue (C): in absolute standardized units the selectivity loading rises and the status loading is "
      f"flat; relative to their levels, or to the overall fit, neither is singled out.** On the scripts/62 fixed "
      f"cohorts (reproduced: {ui['n_ref_cells']} subject × cohort cells; the largest coupling difference, "
      f"{ui['max_abs_diff_ref']:.1e}, is CSV rounding), G alone rises {e3('UK|U1:G|slope|all')}/yr. With institution "
      f"selectivity (share of graduates with ≥ 360 UCAS tariff points) in the model: **T1u**, {vtxt('T1u')}; b_G "
      f"{e3all('UK|U2:G|slope|all')} (MDE80 {t1u['mde']:.3f} "
      + ("≤" if t1u["mde"] <= t1u["bench"] else ">") + f" benchmark {t1u['bench']:.3f}"
      + (", so a decay as large as the raw rise would have been detected). " if t1u["mde"] <= t1u["bench"] else "). ")
      + f"**T2u**, {vtxt('T2u')}; b_SEL {e3all('UK|U2:SEL|slope|all')}. The regression-form paired difference, "
      f"b_G slope − b_SEL slope, is {e3all('UK|U2:G - U2:SEL|slope|all')} ({sides3('UK|U2:G - U2:SEL|slope|all')}). "
      f"Only the partial-form difference excludes 0: T1up {vtxt('T1up')} ({e3('UK|pGsel|slope|all')}); T2up "
      f"{vtxt('T2up')} ({e3('UK|pSELg|slope|all')}); difference {e3all('UK|pGsel - pSELg|slope|all')} "
      f"({sides3('UK|pGsel - pSELg|slope|all')}), upper end {dgsp['hi']:+.3f}. Levels: b_SEL {fm(us1)} at YAG 1 and "
      f"{fm(us5)} at YAG 5 (×{ratio_s:.2f}); b_G {fm(ug1)} and {fm(ug5)} (×{ratio_g:.2f}). *Scale (revision 1, "
      f"exploratory).* The mean within-cell R² of the G + selectivity regression is {r2u[0]:.3f} / {r2u[1]:.3f} / "
      f"{r2u[2]:.3f} at YAG 1 / 3 / 5 (slope {e3('UK|U2n:R2|slope|all', 4)}). If both loadings rose in proportion to "
      f"sqrt(R²) (×{fac_u:.3f} from YAG 1 to 5), b_SEL would rise by {uS_impl:+.4f}/yr and b_G by {uG_impl:+.4f}/yr "
      "(endpoint arithmetic on the means), the same ordering as the observed slopes. Scale-free contrasts: per-cell "
      f"loadings divided by sqrt(R²_h) give slopes of {e3all('UK|U2n:G|slope|all')} for G and "
      f"{e3all('UK|U2n:SEL|slope|all')} for selectivity (difference {e3all('UK|U2n:G - U2n:SEL|slope|all')}); the log "
      "of the YAG 5 / YAG 1 ratio of mean loadings (delta method) is "
      f"{e3all('UK|log ratio: b_G|last/first|all')} for G, {e3all('UK|log ratio: b_SEL|last/first|all')} for "
      f"selectivity, and G minus selectivity {e3all('UK|log ratio: b_G minus b_SEL|last/first|all')} "
      f"({sides3('UK|log ratio: b_G minus b_SEL|last/first|all')}); half the log ratio of mean R² is "
      f"{e3('UK|log ratio: 1/2 R2|last/first|all')}. So the UK data do not single out selectivity, and they do not "
      "show a shift of relative weight toward it. G and selectivity correlate "
      f"{dU['r_GSEL']:+.2f} within cells.")
    A(f"7. **Other loadings (reported, not pre-specified tests).** The Pell-share loading falls within cohort from y1 "
      f"to y10 (career plus calendar time): b_Pell "
      f"{fm(G('A|S2:PELL|y1|all')['est'])} at y1, {fm(G('A|S2:PELL|y5|all')['est'])} at y5, "
      f"{fm(G('A|S2:PELL|y10|all')['est'])} at y10; slope {e3all('A|S2:PELL|slope|all')} "
      f"({sides3('A|S2:PELL|slope|all')}; sample B {e3('B|S2:PELL|slope|all')}). Holding status and selectivity "
      "fixed, institutions with more Pell students have relatively high y1 medians and lose that position by y10. "
      "Differences in post-graduate study (which depresses y1 medians where more graduates enrol) would produce "
      "this; the data cannot say whether that is the cause. State earnings level: slope "
      f"{e3('A|S2:ST|slope|all')}. Log earnings (exploratory; log points per within-cell SD of the regressor's rank, "
      f"per year): G {e3('A|S2L:G|slope|all', 4)}, SAT {e3('A|S2L:SAT|slope|all', 4)}, Pell "
      f"{e3('A|S2L:PELL|slope|all', 4)}.")
    s0g, n25 = G("A|S0:G|slope|all"), G("A|S2:G|slope|n25")
    A(f"8. **How sure: the three intervals agree except for regression coefficients in small cells.** The "
      "pre-specified interval is scripts/59's two-way (field, institution) cluster variance, built from "
      f"institution-bootstrap replicates. For rank regressions with six regressors in cells of {dA['n_sat_min']} to "
      f"{NMIN_BIG - 1} institutions those replicates are heavy-tailed, because a resample holds only about two thirds of the "
      "distinct institutions and the design is then close to singular. For the T1 statistic the Monte Carlo SE of "
      f"the interval endpoints is {s2g['mcse']:.4f}, against {s0g['mcse']:.4f} for scripts/59's two-regressor b_G. "
      f"The bootstrap's per-field noise is {s2g['noise']:.2f} times the observed spread of the per-field values; a "
      "ratio above 1 means the bootstrap overstates the sampling noise. Every number therefore also gets two "
      "post-hoc sensitivity intervals: the same two-way variance with IQR-based variances (scripts/59's robust "
      "version), and a two-way cluster jackknife that deletes one field, one institution, or one institution within "
      "one field, with no resampled duplicates. For two-regressor and partial-correlation statistics the three SEs agree "
      f"closely (b_G slope, F + G only: {s0g['se']:.4f} / {s0g['se_r']:.4f} / {s0g['jse']:.4f}; T1p: "
      f"{pgb['se']:.4f} / {pgb['se_r']:.4f} / {pgb['jse']:.4f}). For the S2 b_G slope they are {s2g['se']:.4f} / "
      f"{s2g['se_r']:.4f} / {s2g['jse']:.4f}. In cells with at least {NMIN_BIG} institutions (exploratory; "
      f"{n25['k']} fields) they agree again ({n25['se']:.4f} / {n25['se_r']:.4f} / {n25['jse']:.4f}). There the S2 "
      f"b_G slope is {e3all('A|S2:G|slope|n25')} ({sides3('A|S2:G|slope|n25')}), while b_SAT is "
      f"{e3('A|S2:SAT|slope|n25')} and b_F {e3('A|S2:F|slope|n25')}. Statements that hold under all three "
      "intervals are firm. Where the intervals disagree, the verdicts are given separately.")
    # ---- item 9: calendar time for the controlled loadings (revision 1, exploratory)
    dC, d5 = D["CAL"], D["D5"]
    cal_lab = {"S2:G": "b_G (S2)", "S2:SAT": "b_SAT (S2)", "pGb": "rho(G, Y | selectivity set)",
               "pSAT": "rho(SAT, Y | G, F, Pell, control, state)", "rG": "raw rho(G, Y)"}
    parts = []
    win_pgb, cal_pgb, cal_s2g = G("CAL|pGb|within|all"), G("CAL|pGb|calendar|all"), G("CAL|S2:G|calendar|all")
    raw_d1 = G("CAL|rG|drift_y1|all")
    ctl_drifts = [G(f"{s}|{st}|{c}|all") for st in ("S2:G", "S2:SAT", "pGb", "pSAT")
                  for s, c in (("CAL", "drift_y1"), ("D5", "drift_y5"))]
    assert all(r["lo"] < 0 < r["hi"] for r in ctl_drifts) and raw_d1["lo"] > 0 and win_pgb["est"] > 0
    for st in CAL_STATS:
        w_, c_, x_ = (G(f"CAL|{st}|{c}|all") for c in ("within", "calendar", "drift_y1"))
        x5 = G(f"D5|{st}|drift_y5|all")
        assert abs(w_["est"] - c_["est"] - x_["est"]) < 1e-12          # exact identity on the triple-matched cells
        if x_["est"] >= 0:
            bnd = f"under b, g ≥ 0 the career-time part lies in [{c_['est']:+.4f}, {w_['est']:+.4f}]"
        else:
            bnd = (f"the y1 drift is below 0 in point estimate ({x_['est']:+.4f}), so the sign assumption fails at face "
                   "value and no bound is given")
        parts.append(f"{cal_lab[st]}: within {e3(w_['key'], 4)}, calendar-matched {e3(c_['key'], 4)}, drift at y1 "
                     f"{e3(x_['key'], 4)}, drift at y5 {e3(x5['key'], 4)}; {bnd}")
    A(f"9. **Career time versus calendar time for the controlled loadings (exploratory, revision 1).** Every slope "
      "above is within cohort, so it combines career time with calendar time (a+b in the linear age-period-cohort "
      "notation of scripts/52), and the y1 → y10 span crosses the 2008–2010 recession (y1 of the 2007 cohort) and "
      "2020–2022 (y10 of the 2010 cohort). scripts/52's design on triple-matched institutions (cohort c at y1 and y10 "
      f"and cohort c+9 at y1, c = 2001/2004/2007/2010; same SAT-sample rule, n ≥ {NMIN_C}: {dC['n_sat_cells']} cells, "
      f"{dC['n_sat_fields']} fields, {dC['n_sat_insts']} institutions) gives, per year and with the pre-specified "
      "two-way interval: " + "; ".join(parts) + ". The drift at y5 uses pair-matched institutions (c at y5 against "
      f"c+9 at y5, c = 2001/2004/2007: {d5['n_sat_cells']} cells, {d5['n_sat_fields']} fields). Institutions are "
      "matched within every comparison, so the growth of the PSEO coalition does not enter these contrasts. Under a "
      "linear age-period-cohort reading, within-cohort change = a+b, calendar-matched = a−g and cross-cohort drift at "
      "fixed age = b+g (exact identity on the triple-matched institutions). If b ≥ 0 and g ≥ 0, the career-time "
      "component a lies between the calendar-matched and the within-cohort estimates; the split is not identified "
      "without such an assumption. All three intervals are in Key numbers. The triple-matched institutions differ "
      "from the balanced y1/y5/y10 panel, so these within-cohort values are not the item 1–3 slopes. "
      f"Read descriptively: for the status loading net of the selectivity set the calendar-matched contrast is "
      f"{cal_pgb['est'] / win_pgb['est']:.0%} of the within-cohort change ({sides3(cal_pgb['key'])}); in regression "
      f"form the calendar-matched b_G is {fm(cal_s2g['est'], 4)} ({sides3(cal_s2g['key'])}). None of the "
      f"{len(ctl_drifts)} fixed-age drifts of the controlled loadings (b_G, b_SAT and the two partial forms, at y1 "
      "and y5) has a pre-specified interval that excludes 0, while the raw G coupling on the same cells drifts "
      f"{fm(raw_d1['est'], 4)}/yr at y1 ({sides3(raw_d1['key'])}).")
    A(f"10. **Exploratory tally (not a test).** Across the {len(ctl_g)} selectivity-controlled status slopes computed "
      "here (specifications, segments, samples and both countries; heavily overlapping statistics on the same cells, "
      f"of which {n_testkeys} are the statistics of pre-specified tests and {len(ctl_g) - n_testkeys} are other "
      "specifications, segments or subgroups), none has an interval below 0 under any of the three variances, and "
      f"{n_pos} have a positive point estimate. The {len(neg_pt)} negative point estimates (the most negative "
      f"{min(R[k]['est'] for k in neg_pt):+.3f}/yr) are all in sample B (y1 → y5), whose intervals are wide.\n")

    # ------------------------------------------------------------------ tests table
    A("## Pre-specified tests\n")
    A("Pre-specified (self-attested, not time-stamped): the rules below were written in the script's docstring "
      "before any selectivity-controlled estimate was computed, but the script is untracked in git and nothing "
      "time-stamps the docstring before the first run, so this cannot be checked. The sha256 of the pre-specification "
      f"block as it stands at this revision is `{hashlib.sha256(prespec_block().encode()).hexdigest()}`; it lets a "
      "later reader detect changes from here on, not establish when the block was written. 'supported' = "
      "95% CI wholly on the predicted side of 0; 'opposite' = wholly on the other side; otherwise 'null' if MDE80 ≤ "
      "benchmark, 'underpowered' if MDE80 > benchmark. Benchmark = |slope of the G loading without selectivity "
      "controls| on the same cells (for partial forms: |slope of the raw rho(G, Y)|, Amendment 3). MDE80 = "
      "(1.960 + 0.842) × SE. T3-type rules: 'supported' if the G-loading slope CI > 0 and the F-loading slope CI "
      "includes 0. 'edge' = a CI endpoint of the pre-specified interval is within 2 Monte Carlo SE of 0 (scripts/59 "
      "convention), or (revision 1) |MDE80 − benchmark| is within 2 Monte Carlo SE of MDE80, in which case the "
      "verdict is shown as 'underpowered/null' because the bootstrap cannot tell the two apart. The Monte Carlo SE is "
      "that of the unguarded two-way variance; where the guard applies (†) the same noise decides whether the guard "
      "fires, so the flag is conservative there.\n")
    A("**Table 1. Pre-specified interval and verdicts.**\n")
    A("| test | prediction | estimate (per yr) | two-way 95% CI | MDE80 (MC SE) | benchmark | verdict | k |")
    A("|---|---|---|---|---|---|---|---|")
    for t in TESTS:
        pred = ("G rises, F flat" if t["rule"] == "t3" else ("< 0" if t["side"] == "neg" else "> 0"))
        extra = (f"; F: {fm(t['rf_est'])} {cis(t['rf_lo'], t['rf_hi'])}" if t["rf_key"] else "")
        A(f"| {t['tid']}: {esc(t['label'])} | {pred} | {fm(t['est'])} | {cis(t['lo'], t['hi'])}"
          f"{' †' if t['guarded'] else ''}{extra} | "
          f"{t['mde']:.3f} ({t['mcse_mde']:.3f}) | {t['bench']:.3f} | **{t['verdict_disp']}**"
          f"{' (edge)' if t['edge'] else ''} | {t['k']} |")
    A("")
    A("**Table 2. Post-hoc sensitivity intervals (added after the first run, Amendment 2; not pre-specified).** Same "
      "verdict rule applied to each interval.\n")
    A("| test | IQR-based two-way 95% CI | MDE80 | → verdict | two-way cluster jackknife 95% CI | MDE80 | → verdict |")
    A("|---|---|---|---|---|---|---|")
    for t in TESTS:
        extra_r = (f"; F: {cis(t['rf_lo_r'], t['rf_hi_r'])}" if t["rf_key"] else "")
        extra_j = (f"; F: {cis(t['rf_jlo'], t['rf_jhi'])}" if t["rf_key"] else "")
        A(f"| {t['tid']} | {cis(t['lo_r'], t['hi_r'])}{extra_r} | {t['mde_r']:.3f} | {t['verdict_iqr']} | "
          f"{cis(t['jlo'], t['jhi'])}{extra_j} | {t['mde_j']:.3f} | {t['verdict_jack']} |")
    A("")

    # ------------------------------------------------------------------ interpretation
    A("## Interpretation (descriptive)\n")
    A("The notes behind this analysis name three patterns for the status loading once selectivity is held fixed. "
      "(i) It decays: status works as a hiring signal that loses weight as employers learn. (ii) It stays flat. (iii) "
      "It rises: status tracks something employers keep paying for, or pay more for with time (skills, networks, "
      "sorting into steeper careers).\n")
    n_incl = sum(lo < 0 < hi for lo, hi in ((s2g["lo"], s2g["hi"]), (s2g["lo_r"], s2g["hi_r"]),
                                           (s2g["jlo"], s2g["jhi"])))
    t3p_all = all(T("T3p")[k] == "supported" for k in ("verdict", "verdict_iqr", "verdict_jack"))
    assert t3p_all
    pgbf = G("A|pGbF|slope|all")
    A(f"- **US.** In partial-correlation form the data match (iii), under all three intervals: G net of the "
      f"selectivity set rises {fm(pgb['est'])}/yr. In the regression form they are consistent with (ii) or (iii): "
      f"{fm(s2g['est'])}/yr, with an interval that includes 0 under {n_incl} of the 3 variances. They do not "
      "match (i). The status rise is smaller than without controls, so part of what looked like a growing return "
      "to status in scripts/52 moves with selectivity. For what is left, the G-loading slope net of selectivity and "
      f"F is above 0 ({e3('A|pGbF|slope|all')}) and the F-loading slope is flat ({e3('A|pFbG|slope|all')}); the "
      f"G − F difference itself is not established under the pre-specified interval ({e3('A|pGbF - pFbG|slope|all')}). "
      "A common rise of all loadings with the within-cell R² accounts for part of the selectivity rise, but not for "
      "the G rise, which starts near 0 (item 2). The within-cohort slopes include calendar time; item 9 gives the "
      "calendar-matched and fixed-age contrasts.")
    A(f"- **UK.** For status the data match (ii) in absolute standardized units: {fm(u2g['est'])}/yr"
      + (", and a decay as large as the raw rise would have been detected" if T("T1u")["mde"] <= T("T1u")["bench"]
         else "") + f". The selectivity loading rises in absolute units ({fm(u2s['est'])}/yr), but the regression-form "
      "difference from the status slope includes 0, the status loading grows at least as fast relative to its level, "
      "and a uniform rise in the within-cell R² would give the same absolute pattern (item 6). So the UK data do not "
      "show the shift of relative weight toward selectivity that learning about average ability would give if "
      "employers did not fully price selectivity at hiring. They are equally consistent with both loadings rising "
      "in proportion as institution characteristics come to order the medians better.")
    A("- **Across the two countries** neither loading falls. Among other readings, this fits a market that prices "
      "most of what G and selectivity carry at hiring and keeps paying for it afterwards. It is also consistent with "
      "Arcidiacono, Bayer & Hizmo's (2010) finding of little post-entry learning for college graduates, but it is "
      "not a test of their model.\n")
    A("**What aggregate data cannot distinguish.**\n")
    A("- Learning about individual ability versus differences in skill growth, on-the-job training or networks by "
      "institution: all of them make an institution's median move with experience in the same way.")
    A("- Sorting into occupations and sectors with steeper earnings profiles (FLOWS_PLACEMENT_RESULT: placement into "
      "market sectors is already ordered by status at y1) versus rising pay within the same jobs.")
    A("- Timing of graduate study: y1 medians are depressed where more graduates enrol, and the y5 and y10 medians "
      "include people who completed further degrees. The y5 → y10 segment reduces this problem but does not remove it.")
    A("- Career time versus calendar time: item 9 gives scripts/52's calendar-matched contrast and fixed-age drift for "
      "the controlled loadings; the linear split rests on a sign assumption (cohort windows are three years).")
    A("- Selective exit from covered employment: coverage falls with prestige at y1 (scripts/52). It enters here only "
      "as an "
      "exploratory control (b_G slope with coverage minus without "
      f"{e3all('A|S5:G - S2c:G|slope|all')}).")
    A("- Whether employers see selectivity at hiring. The institution-level SAT_AVG is public, so the mapping to "
      "Altonji–Pierret's hidden variable is weak by construction.")
    A("- Relative weights versus a common change of scale: rank loadings and partial correlations all rise when "
      "institution characteristics order the medians better; the scale-free contrasts (items 2 and 6) are exploratory.")
    A("- Dollar returns. Rank loadings can move when the cross-institution spread of medians changes; the "
      "log-earnings version is exploratory and imprecise.\n")

    # ------------------------------------------------------------------ key numbers
    A("## Key numbers\n")
    A("Estimates are means over fields (US) or subjects (UK) of per-field slopes (per year since graduation: A "
      "{1,5,10}, B {1,5}, UK YAG {1,3,5}), each per-field slope being the mean over the field's cohorts. Two-way = "
      "pre-specified Cameron–Gelbach–Miller two-way (field, institution) cluster variance from 1000 institution "
      "multinomial draws shared by all cells and 1000 independent draws per field (scripts/59 `twoway`), with the "
      "declared guard (Amendments). IQR = the same with IQR-based variances. Jackknife = two-way cluster jackknife. "
      "All normal 95% CIs. k = fields (subjects).\n")
    A("| quantity | sample / spec | estimate | two-way 95% CI | IQR 95% CI | jackknife 95% CI | MDE80 | k |")
    A("|---|---|---|---|---|---|---|---|")
    for lab, key, samp, d in KEYROWS:
        r = G(key)
        A(f"| {esc(lab)} | {samp} | {fm(r['est'], d)} | {cis(r['lo'], r['hi'], d)}{' †' if r['guarded'] else ''} | "
          f"{cis(r['lo_r'], r['hi_r'], d)} | {cis(r['jlo'], r['jhi'], d)} | {r['mde']:.{d}f} | {r['k']} |")
    A("\n† guard applied (the unguarded two-way variance was below the between-field variance, or ≤ 0).\n")
    A("### Loadings by horizon (mean over fields; pre-specified two-way 95% CI)\n")
    A("| sample | spec | loading | " + " | ".join(["y1", "y5", "y10"]) + " |")
    A("|---|---|---|---|---|---|")
    for spec, vs in (("S1", ["G", "F"]), ("S2", ["G", "F", "SAT", "PELL", "ST"]), ("S3", ["G", "F", "NADM", "PELL"])):
        for v in vs:
            A(f"| A | {spec} | b_{v} | " + " | ".join(e3(f"A|{spec}:{v}|{h}|all") for h in HZ_A) + " |")
    for p in ("rG", "rSAT", "pGb", "pGbF", "pFbG", "pSAT"):
        A(f"| A | partial | {esc(SPEC_DESC[p])} | " + " | ".join(e3(f"A|{p}|{h}|all") for h in HZ_A) + " |")
    A("| A | S2n (revision 1) | within-cell R² of the S2 regression | "
      + " | ".join(e3(f"A|S2n:R2|{h}|all") for h in HZ_A) + " |")
    for v in ("G", "F", "SAT", "PELL"):
        A(f"| A | S2n (revision 1) | b_{v} / sqrt(R²) | " + " | ".join(e3(f"A|S2n:{v}|{h}|all") for h in HZ_A) + " |")
    A("")
    A("| sample | spec | loading | y1 | y5 |")
    A("|---|---|---|---|---|")
    for spec, vs in (("S1", ["G", "F"]), ("S2", ["G", "F", "SAT", "PELL", "ST"])):
        for v in vs:
            A(f"| B | {spec} | b_{v} | " + " | ".join(e3(f"B|{spec}:{v}|{h}|all") for h in HZ_B) + " |")
    for p in ("rG", "rSAT", "pGb", "pSAT"):
        A(f"| B | partial | {esc(SPEC_DESC[p])} | " + " | ".join(e3(f"B|{p}|{h}|all") for h in HZ_B) + " |")
    A("")
    A("| sample | spec | loading | YAG 1 | YAG 3 | YAG 5 |")
    A("|---|---|---|---|---|---|")
    for spec, vs in (("U1", ["G"]), ("U2", ["G", "SEL"])):
        for v in vs:
            A(f"| UK | {spec} | b_{v} | " + " | ".join(e3(f"UK|{spec}:{v}|{h}|all") for h in ("yag1", "yag3", "yag5"))
              + " |")
    for p in ("rSEL", "pGsel", "pSELg"):
        A(f"| UK | partial | {esc(SPEC_DESC[p])} | " + " | ".join(e3(f"UK|{p}|{h}|all") for h in ("yag1", "yag3", "yag5"))
          + " |")
    A("| UK | U2n (revision 1) | within-cell R² of the G + selectivity regression | "
      + " | ".join(e3(f"UK|U2n:R2|{h}|all") for h in ("yag1", "yag3", "yag5")) + " |")
    for v in ("G", "SEL"):
        A(f"| UK | U2n (revision 1) | b_{v} / sqrt(R²) | "
          + " | ".join(e3(f"UK|U2n:{v}|{h}|all") for h in ("yag1", "yag3", "yag5")) + " |")
    A("")
    # ---- revision 1 tables (exploratory)
    A("### Revision 1 (exploratory): scale-free contrasts\n")
    A("Per-cell loadings divided by sqrt(R²_h) (relative weights: unchanged when every loading is scaled by the same "
      "factor) and log ratios of last- to first-horizon mean loadings (delta-method versions of the same three "
      "variances). 'minus 1/2 R²' subtracts half the log ratio of the mean within-cell R², the change a uniform "
      "rescaling of all loadings would produce. Per year for slopes; log ratios are totals (US y1 → y10, UK YAG 1 → 5).\n")
    A("| quantity | sample | estimate | two-way 95% CI | IQR 95% CI | jackknife 95% CI | MDE80 | k |")
    A("|---|---|---|---|---|---|---|---|")
    rows1 = [("R² slope (S2)", "A|S2n:R2|slope|all", "A, S2"), ("b_G / sqrt(R²) slope", "A|S2n:G|slope|all", "A, S2"),
             ("b_SAT / sqrt(R²) slope", "A|S2n:SAT|slope|all", "A, S2"),
             ("b_F / sqrt(R²) slope", "A|S2n:F|slope|all", "A, S2"),
             ("b_Pell / sqrt(R²) slope", "A|S2n:PELL|slope|all", "A, S2"),
             ("b_G / sqrt(R²) minus b_SAT / sqrt(R²), slope", "A|S2n:G - S2n:SAT|slope|all", "A, S2"),
             ("b_G / sqrt(R²) minus b_F / sqrt(R²), slope", "A|S2n:G - S2n:F|slope|all", "A, S2"),
             ("b_G / sqrt(R²) slope, cells n>=25", "A|S2n:G|slope|n25", "A, n>=25"),
             ("b_SAT / sqrt(R²) slope, cells n>=25", "A|S2n:SAT|slope|n25", "A, n>=25"),
             ("log ratio y10/y1 of mean b_SAT", "A|log ratio: b_SAT|last/first|all", "A, S2"),
             ("log ratio y10/y1 of mean b_F", "A|log ratio: b_F|last/first|all", "A, S2"),
             ("1/2 log ratio y10/y1 of mean R²", "A|log ratio: 1/2 R2|last/first|all", "A, S2"),
             ("log ratio of mean b_SAT minus 1/2 R²", "A|log ratio: b_SAT minus 1/2 R2|last/first|all", "A, S2"),
             ("log ratio of mean b_F minus 1/2 R²", "A|log ratio: b_F minus 1/2 R2|last/first|all", "A, S2"),
             ("R² slope (U2)", "UK|U2n:R2|slope|all", "UK"), ("b_G / sqrt(R²) slope", "UK|U2n:G|slope|all", "UK"),
             ("b_SEL / sqrt(R²) slope", "UK|U2n:SEL|slope|all", "UK"),
             ("b_G / sqrt(R²) minus b_SEL / sqrt(R²), slope", "UK|U2n:G - U2n:SEL|slope|all", "UK"),
             ("log ratio YAG5/YAG1 of mean b_G", "UK|log ratio: b_G|last/first|all", "UK"),
             ("log ratio YAG5/YAG1 of mean b_SEL", "UK|log ratio: b_SEL|last/first|all", "UK"),
             ("log ratio of mean b_G minus log ratio of mean b_SEL", "UK|log ratio: b_G minus b_SEL|last/first|all", "UK"),
             ("1/2 log ratio YAG5/YAG1 of mean R²", "UK|log ratio: 1/2 R2|last/first|all", "UK"),
             ("log ratio of mean b_G minus 1/2 R²", "UK|log ratio: b_G minus 1/2 R2|last/first|all", "UK"),
             ("log ratio of mean b_SEL minus 1/2 R²", "UK|log ratio: b_SEL minus 1/2 R2|last/first|all", "UK")]
    for lab, key, samp in rows1:
        r = G(key)
        A(f"| {esc(lab)} | {samp} | {fm(r['est'], 4)} | {cis(r['lo'], r['hi'], 4)}{' †' if r['guarded'] else ''} | "
          f"{cis(r['lo_r'], r['hi_r'], 4)} | {cis(r['jlo'], r['jhi'], 4)} | {r['mde']:.4f} | {r['k']} |")
    A("\nThe US log ratio for b_G is undefined (its mean at y1 is ≤ 0).\n")
    A("### Revision 1 (exploratory): career time versus calendar time for the controlled loadings\n")
    A(f"Triple-matched institutions (cohort c at y1 and y10, cohort c+9 at y1; c = 2001/2004/2007/2010; "
      f"{D['CAL']['n_sat_cells']} cells, {D['CAL']['n_sat_fields']} fields) and, for the y5 drift, pair-matched "
      f"institutions (c and c+9 at y5; c = 2001/2004/2007; {D['D5']['n_sat_cells']} cells, {D['D5']['n_sat_fields']} "
      "fields). Per year: within = (c@y10 − c@y1)/9 = a+b; calendar-matched = (c@y10 − c+9@y1)/9 = a−g; drift = "
      "(c+9 − c)/9 at fixed y1 or y5 = b+g (linear APC notation of scripts/52).\n")
    A("| loading | contrast | estimate | two-way 95% CI | IQR 95% CI | jackknife 95% CI | MDE80 | k |")
    A("|---|---|---|---|---|---|---|---|")
    for st in CAL_STATS:
        for smp, c, lab in (("CAL", "within", "within cohort (y10 − y1)/9"),
                            ("CAL", "calendar", "calendar-matched (c@y10 − c+9@y1)/9"),
                            ("CAL", "drift_y1", "drift at y1 (c+9 − c)/9"), ("D5", "drift_y5", "drift at y5 (c+9 − c)/9")):
            r = G(f"{smp}|{st}|{c}|all")
            A(f"| {esc(SPEC_DESC.get(st, st) if ':' not in st else 'b_' + st.split(':')[1] + ' (S2)')} | {lab} | "
              f"{fm(r['est'], 4)} | {cis(r['lo'], r['hi'], 4)}{' †' if r['guarded'] else ''} | "
              f"{cis(r['lo_r'], r['hi_r'], 4)} | {cis(r['jlo'], r['jhi'], 4)} | {r['mde']:.4f} | {r['k']} |")
    A("")

    # ------------------------------------------------------------------ method
    A("## Method\n")
    A(f"- **Panels.** Sample A = scripts/52's `fixed_panel` on PSEO V4.14.1 (scripts/59's canonical-join 'new' "
      f"variant; `set_release('new')`): bachelor's cohorts 2001/2004/2007/2010 with released earnings at y1, y5 and "
      f"y10 in the same field × cohort (balanced), {dA['n_full_insts']} institutions, {dA['n_full_cells']} cells with "
      f"n ≥ {NMIN}, {dA['n_full_fields']} fields. Sample B = the same construction balanced on y1 and y5 for cohorts "
      f"2001–2016 ({dB['n_full_insts']} institutions, {dB['n_full_cells']} cells, {dB['n_full_fields']} fields). "
      "Earnings = cohort-size-weighted p50 over the field's CIP-4 rows (`load_er_pseo`, unchanged). Prestige: F = "
      "−(published Wapman field rank), G = −(published Wapman academia-wide rank) (scripts/28 `load_generic`).")
    A(f"- **Covariates.** Scorecard institution file joined on the 8-digit OPEID, which is the PSEO institution id "
      f"(scripts/61 `scorecard_institutions`: main campus, then largest undergraduate enrolment, when OPEIDs repeat); "
      f"SAT_AVG, ADM_RATE, PCTPELL, CONTROL and the scripts/55 state earnings level (leave-one-out mean log "
      f"MD_EARN_WNE_P10 over predominantly-bachelor's institutions in the state). All {dA['n_full_insts']} panel "
      f"institutions match. {len(nosat)} have no SAT_AVG: "
      + "; ".join(f"{r.INSTNM} ({r.institution_state}; ADM_RATE {'—' if not np.isfinite(r.ADM_RATE) else f'{r.ADM_RATE:.2f}'})"
                  for r in nosat.sort_values('INSTNM').itertuples())
      + ". The SAT sample drops them, so it is truncated at the least selective end; the ADM sample (S3) keeps those "
      "with an ADM_RATE. Private = CONTROL ≠ 1.")
    A(f"- **Controlled cells.** Institutions with SAT_AVG, ADM_RATE, PCTPELL, CONTROL and state level; n ≥ {NMIN_C} "
      f"(residual df ≥ 10 in the largest partial specification, scripts/55's rule). Sample A: {dA['n_sat_cells']} "
      f"cells, {dA['n_sat_fields']} fields, {dA['n_sat_insts']} institutions, n {dA['n_sat_min']}–{dA['n_sat_max']} "
      f"(median {dA['n_sat_med']:.0f}); {dA['n_priv_cells']} cells contain both public and private institutions. "
      f"Sample B: {dB['n_sat_cells']} cells, {dB['n_sat_fields']} fields. ADM sample A: {dA['n_adm_cells']} cells, "
      f"{dA['n_adm_fields']} fields.")
    A("- **Loadings.** In each cell and horizon: OLS of z(rank earnings) on z(rank) of each continuous regressor, "
      "the private dummy and an intercept (standardized coefficients; S0, G and F only on the full panel, reproduces scripts/52's "
      f"b_F, b_G exactly: {chk['cells_checked']} cells, largest difference {chk['max_abs_diff_s52']:.1e}; the "
      f"aggregate b_G and b_F slopes equal scripts/59's V4.14.1 values {chk['dG_s59']:+.6f} and "
      f"{chk['dF_s59']:+.6f}). Partial Spearman correlations: ranks of the target and of earnings residualized on "
      "[1, ranks of the controls] and correlated (identical to scripts/55 `partial_rank`: "
      f"{chk['partial_cells_checked']} cells × 3 horizons checked, largest difference {chk['max_abs_diff_s55']:.1e}).")
    A(f"- **Collinearity.** Mean within-cell Spearman (sample A, SAT cells): G–F {dA['r_GF']:+.2f}, G–SAT "
      f"{dA['r_GSAT']:+.2f}, F–SAT {dA['r_FSAT']:+.2f}, G–Pell {dA['r_GPELL']:+.2f}, SAT–Pell {dA['r_SATPELL']:+.2f}, "
      f"SAT–(−ADM) {dA['r_SATNADM']:+.2f}. The mean R² of rank(G) on the other S2 regressors is "
      f"{dA['r2_G_on_rest']:.2f}, so b_G rests on about {1 - dA['r2_G_on_rest']:.0%} of G's rank variance.")
    A("- **Slopes and aggregation.** Per cell, OLS slope of the loading on years since graduation (A: {1, 5, 10}; "
      "segments (y5 − y1)/4 and (y10 − y5)/5; B: (y5 − y1)/4; UK: {1, 3, 5}); mean over a field's cohorts; unweighted "
      "mean over fields. Differences are formed per field and share every draw.")
    A(f"- **Inference.** (a) Pre-specified: scripts/59 `twoway` — V_field (between-field variance / k) + V_inst "
      f"(fields fixed, one institution multinomial draw shared by every cell) − V_field×inst (fields fixed, an "
      f"independent draw per field), {NBOOT} replicates each; replicates where some cell's statistic is undefined "
      f"are redrawn (sample A: {dA['redrawn_shared']} shared / {dA['redrawn_indep']} per-field; B: "
      f"{dB['redrawn_shared']} / {dB['redrawn_indep']}; UK: {dU['redrawn_shared']} / {dU['redrawn_indep']}). (b) IQR "
      "version: each variance replaced by (IQR/1.349)², the field part from a field bootstrap. (c) Two-way cluster "
      "jackknife: V_field (delete one field) + V_inst (delete one institution from every cell; (N−1)/N Σ) − "
      "V_field×inst (delete one institution from one field's cells). All three with the guard below; crossed and "
      "fields-fixed percentile intervals are in the CSV.")
    A(f"- **UK.** scripts/62's interim tables hold cell-level couplings and band statistics but no provider-level "
      f"earnings, so the provider × subject × cohort × YAG medians were rebuilt from the raw LEO provider file "
      f"(all-graduate rows) with scripts/62's filters and its multi-region rule "
      f"({ui['minor_region_rows']} minor-region rows dropped); providers = scripts/62's {ui['n_providers']} primary "
      f"providers with G (`uk_leo_prestige.csv`); selectivity = scripts/62's `inst_top` (provider-total share of "
      f"graduates with prior attainment band PA1 or PA2, i.e. ≥ 360 UCAS tariff points, among graduates with a known "
      f"band; mean over cohorts 2013–2016 at YAG 5; {ui['n_sel']} providers). Balanced providers at YAG 1/3/5 per "
      f"subject × cohort, n ≥ {s62.NMIN}: {dU['n_sel_cells']} cells, {dU['n_sel_fields']} subjects, "
      f"{dU['n_sel_insts']} providers (n {dU['n_sel_min']}–{dU['n_sel_max']}). Across providers Spearman(G, "
      f"selectivity) = {ui['rho_G_sel']:+.2f}. Two-way variance over (subject, provider).")
    A("- **Coverage.** Within a cell the institutions are the same at every horizon (balanced panels), so every "
      "horizon comparison holds institutions fixed. Across cohorts the mix changes as states join PSEO "
      "(version_pseo.txt: " + ", ".join(f"{k} {v}" for k, v in sorted(ctx["start"].items()) if v > 2001)
      + "); the coalition-fixed variants keep institutions in states whose series starts in 2001 "
      f"(A: {D['A_coal']['n_sat_insts']} institutions, B: {D['B_coal']['n_sat_insts']}). Coverage (share of "
      "graduates with PSEO earnings) enters only as an exploratory regressor (S5).")
    A("- **Revision 1 additions (exploratory).** (a) Relative weights: in every cell and horizon the S2 (US) and G + "
      "selectivity (UK) loadings are divided by sqrt(R²_h), R²_h being the weighted variance of the fitted values "
      "of the same rank regression (the outcome z(rank) has weighted variance 1), computed under every bootstrap and "
      "jackknife weight; slopes and paired differences as above. (b) Log ratios of last- to first-horizon mean "
      "loadings, with the delta method applied to every replicate array (observed, shared draw, per-field draws, "
      "delete-one-institution values), so the three intervals are the delta-method versions of the same variances. "
      "(c) Calendar-time cells: PSEO y1 is also loaded for cohort 2019 (load_er_pseo groups by field × institution "
      "× cohort, so no other row changes); triple-matched (c@y1, c@y10, c+9@y1) and pair-matched (c@y5, c+9@y5) "
      f"cells with the SAT-sample rule and n ≥ {NMIN_C}; own institution draws (redrawn: triple-matched "
      f"{D['CAL']['redrawn_shared']} shared / {D['CAL']['redrawn_indep']} per-field; y5 pairs "
      f"{D['D5']['redrawn_shared']} / {D['D5']['redrawn_indep']}). None of the additions changes a draw of the "
      "earlier analyses: the redraw rule is unchanged and the new statistics are finite wherever the old ones are.\n")

    # ------------------------------------------------------------------ amendments
    A("## Amendments to the pre-specification (made after the first run, before this write-up)\n")
    n_guard = sum(1 for r in R.values() if r["guarded"])
    n_vchg = sum(t["verdict"] != t["verdict_unguarded"] for t in TESTS)
    s3f = G("A|S3:F|slope|all")
    A("1. **Guard on the two-way variance.** For some statistics the estimated V_field×inst exceeded V_inst, which "
      "pushed the two-way variance below the between-field variance V_field. The most extreme case is the b_F slope "
      f"under S3: unguarded SE {s3f['se_raw']:.4f}, against {np.sqrt(s3f['vf']):.4f} from V_field alone. In the "
      "population the two-way variance cannot be below V_field when the cross-field institution covariance is "
      "non-negative. Rule: if the two-way estimate is below V_field, or ≤ 0 (scripts/59's own fallback), the larger "
      f"one-way variance is used. The guard only widens intervals. It applies to {n_guard} of the {len(R)} "
      "aggregate statistics (marked † in the table and 'twoway_guard_applied' in the CSV) and changes "
      f"{n_vchg} of the {len(TESTS)} test verdicts.")
    A("2. **Two sensitivity intervals** (IQR-based two-way, two-way cluster jackknife) were added after the "
      "bootstrap replicates of the six-regressor coefficients turned out heavy-tailed (Answer 8). The verdicts under "
      "the pre-specified interval are unchanged and reported first.")
    A("3. **Benchmark for the partial forms.** The docstring gave the benchmark for the coefficient tests; for the "
      "partial forms it was set, when coding and before the results, to |slope of the raw rho(G, Y)| on the same "
      "cells.\n")

    # ------------------------------------------------------------------ caveats
    A("## Caveats\n")
    A("- **Aggregate, not individual.** An institution's median can rise because its graduates' productivity is "
      "revealed, because they gain skills faster, because they move into better-paid jobs, or because the "
      "composition of those with covered earnings changes. The design identifies none of these channels.")
    A("- **Selectivity is public and measured on recent entrants.** SAT_AVG/ADM_RATE are from the current Scorecard "
      "release, not from the years the panel's graduates entered, and employers can see them. Measurement error in "
      "selectivity leaves part of it in G, and that part can grow with its loading.")
    A("- **G and F are collinear with each other and with selectivity** (Method). The b_G of S2 rests on about "
      f"{1 - dA['r2_G_on_rest']:.0%} of G's rank variance, which makes it imprecise. That is why the regression and "
      "partial forms differ in precision.")
    A("- **Small cells and heavy tails.** See Answer 8. The firm statements are those that hold under all three "
      "intervals.")
    A("- **Truncated samples.** PSEO partner institutions are public-heavy and miss most elite privates. The SAT "
      f"sample also drops {len(nosat)} institutions without SAT_AVG (mostly less selective). Field coverage differs "
      "between samples A and B.")
    A("- **Calendar time.** Within-cohort slopes mix career time with calendar time (a+b). Item 9 reports scripts/52's "
      "calendar-matched contrast and fixed-age drifts for the controlled loadings (exploratory); the linear "
      "age-period-cohort split is not identified without a sign assumption, and no unemployment or recession "
      "control is used.")
    A("- **Scale.** Standardized rank loadings and partial correlations are not relative weights: when institution "
      "characteristics order the medians better (higher within-cell R²), every loading rises. Two variables with "
      "fixed relative weights can therefore have different absolute slopes, larger for the one with the larger "
      "starting loading. The pre-specified tests are on the absolute scale; the scale-free contrasts are exploratory.")
    A("- **Pre-specification is self-attested.** The script is untracked in git; the docstring's pre-specification "
      "block is hashed in the tests section from this revision on, which cannot date it. The IQR and jackknife "
      "intervals, the guard and the partial-form benchmark are declared post-hoc amendments.")
    A("- **UK selectivity** is the provider-wide share of graduates with ≥ 360 tariff points (an intake measure from "
      "the same LEO file), not the subject's own intake. G is scripts/62's UK hiring-network SpringRank.")
    A(f"- **Multiplicity.** {len(TESTS)} pre-specified or robustness tests are reported; no correction is applied. The firm "
      "findings are the ones that hold across intervals and samples, not single CIs.\n")

    # ------------------------------------------------------------------ revisions
    A("## Revisions\n")
    A("Revision 1 answers an independent verification of the first version. The verifier's rebuild reproduced the "
      "sample sizes, point estimates and jackknife intervals. Every estimate and interval of the first version is "
      "unchanged (checked by comparing the aggregate rows of the CSV before and after); the changes are in wording, "
      "in the verdict display and in added exploratory rows.\n")
    A("1. **UK reading was scale-dependent (major).** The first version said the UK rise 'loads on selectivity' and "
      "that status 'matches (ii)' while selectivity moves in 'the direction learning about average ability would "
      "give'. That compared which slope was significant rather than testing the difference, and it read absolute "
      "standardized slopes as relative weights. Changed: the headline, item 6 and the UK interpretation now report "
      "the regression-form paired difference next to T2u (its CI includes 0 under all three intervals), the growth "
      "of each loading relative to its level, the rise of the within-cell R² and what a uniform rise would give, and "
      "two exploratory scale-free contrasts (loadings over sqrt(R²); delta-method log ratios). The same caveat is "
      "added to T2/T2p for the US (item 2), where a uniform rise accounts for part of the b_SAT rise but not for the "
      "G rise.")
    A("2. **'Loads on G rather than on F' compared significance levels.** The T3p verdict is kept (pre-specified "
      "rule). The interpretive sentence now says that the G-loading slope net of selectivity and F is above 0, the "
      "F-loading slope is flat, and the G − F difference itself is not established under the pre-specified interval "
      "(items 3 and Interpretation).")
    A("3. **T1's null/underpowered split was inside Monte Carlo noise, and the post-data bound was missing.** The "
      "'edge' flag now also covers |MDE80 − benchmark| within 2 Monte Carlo SE of MDE80 (shown as "
      "'underpowered/null (edge)'); item 1 adds which decays the pre-specified CI excludes; the headline says 'no "
      "decay detected' instead of 'it does not fall'. (The verifier's own bootstrap, which redraws rank-deficient "
      "resamples where this script uses the pseudo-inverse, gave a smaller SE and the verdict 'null' for T1, "
      "consistent with the edge display.)")
    A("4. **Headline tally of overlapping, mostly exploratory slopes.** The headline now rests only on the nine "
      "pre-specified status-decay tests. The tally moved to item 10 and is labelled exploratory; the n ≥ 25 remark "
      "left the headline (it stays in item 8, labelled exploratory).")
    A("5. **Career language for within-cohort slopes.** The Answer now says 'within cohort from y1 to y10 (career plus "
      "calendar time)'. Item 9 and a Key-numbers table add scripts/52's calendar-matched contrast and the fixed-age "
      "drifts at y1 and y5 for the controlled loadings, with scripts/52's bounds wording.")
    A("6. **CSV verdict 'null' was read as missing by pandas.** The CSV writes 'null_result'; the script reads the file "
      "back with default settings and asserts that no text cell is lost. A column `edge_flag` ('ci', 'mde', "
      "'ci+mde') is added.")
    A("7. **Pre-specification could not be checked.** The tests section now labels the rules as pre-specified, "
      "self-attested and not time-stamped, prints the sha256 of the pre-specification block, and puts the post-hoc "
      "IQR and jackknife intervals in a separate table (Table 2). Committing the block before a run is left to the "
      "author (this workflow does not commit).\n")

    # ------------------------------------------------------------------ appendices
    A("## Appendix A — per-field slopes, sample A (SAT cells)\n")
    A("Observed per-field slopes (mean over the field's cohorts); no per-field CI.\n")
    A("| field | cohorts | n (min–max) | b_G slope, F + G (S1) | b_G slope (S2) | b_F slope (S2) | b_SAT slope (S2) | "
      "rho(G, Y \\| broad) slope | rho(SAT, Y \\| ...) slope |")
    A("|---|---|---|---|---|---|---|---|---|")
    sat_cells = [c for c in SA["cells"] if c["sub"] == "sat"]
    byf = {}
    for c in sat_cells:
        byf.setdefault(c["field"], []).append(c)
    cols = ["A|S1:G|slope|all", "A|S2:G|slope|all", "A|S2:F|slope|all", "A|S2:SAT|slope|all", "A|pGb|slope|all",
            "A|pSAT|slope|all"]
    pfv = {k: dict(zip(G(k)["fields"], G(k)["pf"])) for k in cols}
    order = sorted(byf, key=lambda f: -pfv[cols[1]].get(f, -9))
    for f in order:
        cs = byf[f]
        A(f"| {LAB.get(f, f)} | {','.join(sorted(c['cohort'] for c in cs))} | {min(c['n'] for c in cs)}–"
          f"{max(c['n'] for c in cs)} | " + " | ".join(fm(pfv[k].get(f, np.nan)) for k in cols) + " |")
    A("")
    A("## Appendix B — per-subject slopes, UK\n")
    A("| subject | cohorts | n (min–max) | b_G slope, G only | b_G slope, G + selectivity | b_SEL slope | "
      "rho(G, Y \\| sel.) slope | rho(sel., Y \\| G) slope |")
    A("|---|---|---|---|---|---|---|---|")
    ucols = ["UK|U1:G|slope|all", "UK|U2:G|slope|all", "UK|U2:SEL|slope|all", "UK|pGsel|slope|all",
             "UK|pSELg|slope|all"]
    upf = {k: dict(zip(G(k)["fields"], G(k)["pf"])) for k in ucols}
    byu = {}
    for c in SU["cells"]:
        byu.setdefault(c["field"], []).append(c)
    for f in sorted(byu, key=lambda f: -upf[ucols[2]].get(f, -9)):
        cs = byu[f]
        A(f"| {f} | {','.join(sorted(c['cohort'] for c in cs))} | {min(c['n'] for c in cs)}–{max(c['n'] for c in cs)} | "
          + " | ".join(fm(upf[k].get(f, np.nan)) for k in ucols) + " |")
    A("")
    A("## Appendix C — provenance and checks\n")
    A("| input | file | md5 |")
    A("|---|---|---|")
    for lab, p in (("PSEO V4.14.1 earnings", s59.NEW_EARN), ("PSEO V4.14.1 institutions", s59.NEW_INST),
                   ("PSEO V4.14.1 version file", VERSION_FILE), ("Scorecard institution file", s61.SC_INST),
                   ("Wapman ranks", ROOT / "data" / "raw" / "wapman2022" / "ranks.csv"),
                   ("LEO provider data (zip)", s62.LEO_ZIP), ("UK prestige (scripts/62 output)", UK_PRES),
                   ("UK career cells (scripts/62 output, reproduction check)", UK_CAREER),
                   ("scripts/59 output (reproduction check)", REF_S59)):
        A(f"| {lab} | `{p.relative_to(ROOT)}` | `{md5(p)}` |")
    A("")
    A(f"Reproduction checks (asserted by the script): S0 F + G coefficients equal scripts/52 `cell_metrics` in "
      f"{chk['cells_checked']} cells (largest difference {chk['max_abs_diff_s52']:.1e}); aggregate b_G and b_F slopes "
      f"{chk['dG_here']:+.6f} and {chk['dF_here']:+.6f} equal scripts/59's {chk['dG_s59']:+.6f} and {chk['dF_s59']:+.6f}; "
      f"partial correlations equal scripts/55 `partial_rank`; the UK G-only couplings equal scripts/62's "
      f"`uk_leo_career_cells.csv` in all {ui['n_ref_cells']} cells (largest difference {ui['max_abs_diff_ref']:.1e}, "
      "CSV rounding) with the same provider counts.\n")
    OUT_MD.write_text("\n".join(L))
    print(f"[md] {OUT_MD}")


if __name__ == "__main__":
    CTX = main()
