"""Direction 1 -- career-time coupling, canonical fixed-cohort version.

QUESTION. Does coupling_f(h) = Spearman_i(field prestige_{i,f}, median earnings_{i,f} at h years
after graduation) change with years since graduation h, and is any change specific to fields whose
prestige and pay are "integrated" (high baseline coupling)?

DEFECT IN THE PREVIOUS VERSION OF THIS SCRIPT (fixed here)
  1. It used PSEO grad_cohort "0000" (all cohorts pooled) and labelled y1/y5/y10 as the same
     graduates. They are not: the pooled y1 cell averages cohorts 2001-2019, y5 2001-2016 and y10
     2001-2010, so a pooled y1 -> y10 comparison mixes career time with cohort composition and
     calendar time.
  2. PSEO was loaded through the TIER0 30-field CIP map, so teacher_ed, management, marketing,
     human_dev, kinesiology and spanish never entered the PSEO panel although the field groups were
     built from FIELDS66.
  3. Institutions were not balanced across horizons (a field's y10 cell could hold other
     institutions than its y1 cell).
  4. It read "decoupled group pinned at ~0" off 2 fields; communication disorders never reaches
     n>=15 institutions in a single fixed cohort, so that claim had no fixed-cohort support.

FIX / WHAT THIS SCRIPT DOES
  (a) PSEO bachelor's FIXED cohorts 2001/2004/2007/2010 (3-year graduation windows), FIELDS66 map,
      load_er_pseo(level, h, fields=FIELDS66, grad_cohort=[...]) at y1/y5/y10. Balanced panel: in
      each field x cohort the same institutions at all three horizons; cells need n>=15.
      coupling per field x cohort x horizon; OLS slope on years {1,5,10} per field x cohort;
      field slope = mean over its cohorts; group mean = mean over fields. Field groups from the
      Scorecard 4YR baseline coupling (integrated >=0.45, decoupled <0.25, middle otherwise;
      compute_gap reliable_flag and n>=15; everything else "unclassified").
  (b) Inference: two-stage bootstrap (resample fields, then institutions within field x cohort,
      keeping each institution's three horizons together); institution-level permutation of
      horizon labels (each institution's within-horizon earnings ranks shuffled across y1/y5/y10);
      field-label permutation for the integrated-minus-others contrast.
  (c) Calendar-matched contrast: cohort c at y10 vs cohort c+9 at y1 (2001y10 vs 2010y1,
      2004y10 vs 2013y1, 2007y10 vs 2016y1, 2010y10 vs 2019y1), institutions matched within field.
      Together with the within-cohort y1->y10 change and the fixed-horizon cross-cohort drift this
      gives bounds on the career-time component under a linear age-period-cohort reading.
  (d) Coverage sensitivity: coverage = y{h}_grads_earn / y1_ipeds_count (PSEO earners over IPEDS
      graduates, summed over the field's CIP-4 codes); partial Spearman coupling controlling for
      same-horizon coverage; above-median-y1-coverage subsample.
  (e) Brand vs field: standardized rank regression earnings ~ F + G per cell (F = -field rank,
      G = -academia-wide rank from scripts/28 load_generic); slopes dF, dG with the two-stage
      bootstrap. F and G are collinear and G is estimated from far more hiring edges than F.
  (f) The pooled "0000" estimate (cohort-mixed) and the exact legacy spec, for comparison only.
  (g) Scorecard 1YR/4YR/5YR coupling, descriptive only: the three horizons of one Scorecard release
      are different cohorts, so the cross-horizon change is cohort-confounded.

Descriptive and not causal: earnings differences across institutions mix value-added and selection.
Seeded; outputs are byte-identical on re-run.
Run: python scripts/52_career_time_coupling.py
Outputs: data/interim/career_time_coupling.csv, data/interim/career_time_brand_field.csv,
         data/interim/career_time_calendar_contrast.csv, outputs/figures/career_time_coupling.png,
         CAREER_TIME_COUPLING_RESULT.md (repo root, local only).
"""
from __future__ import annotations

import sys
import zlib
import itertools
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.load_ar import load_ar_wapman
from src.load_er import (load_er_scorecard, load_er_pseo, _pseo_institutions, PSEO_EARN,
                         PSEO_DEGREE_LEVEL)
from src.gap import compute_gap_map
from src.crosswalks import fields as F

SEED = 52
NMIN = 15            # min institutions in a cell
B_GAP = 250          # bootstrap draws inside compute_gap (reliability flag of the Scorecard baseline)
NBOOT = 2000         # two-stage bootstrap replicates
NPERM = 2000         # horizon-label permutations
NPERM_FIELD = 10000  # field-label permutations (contrast)

# FIELDS66 + label map, exactly as scripts/28/32 build them
_spec = importlib.util.spec_from_file_location("s28", ROOT / "scripts" / "28_field_vs_generic_prestige.py")
_s28 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_s28)
FIELDS66, LAB = _s28.FIELDS66, _s28.LAB

COHORTS = ["2001", "2004", "2007", "2010"]
LATE = {"2001": "2010", "2004": "2013", "2007": "2016", "2010": "2019"}   # c -> c+9
HZ = ["y1", "y5", "y10"]
YRS = np.array([1.0, 5.0, 10.0])
W_SLOPE = (YRS - YRS.mean()) / ((YRS - YRS.mean()) ** 2).sum()   # OLS slope = coupling @ W_SLOPE
SC_HORIZONS = {"1YR": ("EARN_MDN_1YR", "EARN_COUNT_WNE_1YR"),
               "4YR": ("EARN_MDN_4YR", "EARN_COUNT_WNE_4YR"),
               "5YR": ("EARN_MDN_5YR", "EARN_COUNT_WNE_5YR")}
SC_YEARS = {"1YR": 1, "4YR": 4, "5YR": 5}
GROUPS = ["integrated", "middle", "decoupled", "unclassified"]
OUT_CSV = ROOT / "data" / "interim" / "career_time_coupling.csv"
OUT_BF = ROOT / "data" / "interim" / "career_time_brand_field.csv"
OUT_CAL = ROOT / "data" / "interim" / "career_time_calendar_contrast.csv"
OUT_FIG = ROOT / "outputs" / "figures" / "career_time_coupling.png"
OUT_MD = ROOT / "CAREER_TIME_COUPLING_RESULT.md"


def rng_for(tag: str) -> np.random.Generator:
    """Independent, order-free stream per analysis (deterministic)."""
    return np.random.default_rng([SEED, zlib.crc32(tag.encode())])


# ---------------------------------------------------------------------------------------------
# vectorized rank-correlation helpers
# ---------------------------------------------------------------------------------------------
def corr_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    den = np.sqrt((xc ** 2).sum(-1) * (yc ** 2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc * yc).sum(-1) / den


def rk(a: np.ndarray) -> np.ndarray:
    return rankdata(a, axis=-1)          # average ranks for ties (= scipy spearmanr convention)


def partial_rows(rp, re_, rc):
    """Partial Spearman of prestige & earnings given coverage, from rank arrays."""
    a, b, c = corr_rows(rp, re_), corr_rows(rp, rc), corr_rows(re_, rc)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a - b * c) / np.sqrt((1 - b ** 2) * (1 - c ** 2))


def boot_idx(n: int, rng) -> np.ndarray:
    return rng.integers(n, size=(NBOOT, n))


REDRAWN = {}   # analysis -> number of degenerate bootstrap resamples that were redrawn


def boot_valid(fn, n: int, rng, tag: str, max_tries: int = 100) -> dict:
    """Stage-2 bootstrap conditioned on non-degenerate resamples: rows whose statistics are not
    finite (e.g. resampled F and G ranks perfectly concordant, so the two-predictor regression is
    undefined) are redrawn. The count is kept in REDRAWN[tag] and reported."""
    out = fn(boot_idx(n, rng))
    def bad_rows(o):
        m = np.zeros(next(iter(o.values())).shape[0], bool)
        for v in o.values():
            m |= ~np.isfinite(v.reshape(v.shape[0], -1)).all(1)
        return m
    bad = bad_rows(out)
    for _ in range(max_tries):
        if not bad.any():
            break
        REDRAWN[tag] = REDRAWN.get(tag, 0) + int(bad.sum())
        new = fn(rng.integers(n, size=(int(bad.sum()), n)))
        for k in out:
            out[k][bad] = new[k]
        bad = bad_rows(out)
    assert not bad.any(), f"could not draw non-degenerate resamples for {tag}"
    return out


def two_stage(stack: np.ndarray, rng) -> np.ndarray:
    """stack: (K fields, NBOOT inner replicates, ...). Outer stage resamples fields with
    replacement; each drawn field contributes an independently chosen inner replicate."""
    K, B = stack.shape[:2]
    fi = rng.integers(K, size=(B, K))
    ci = rng.integers(B, size=(B, K))
    return stack[fi, ci].mean(axis=1)


def ci(bs) -> tuple[float, float]:
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def perm_p(null: np.ndarray, obs: float) -> float:
    return float((1 + np.sum(np.abs(null) >= abs(obs) - 1e-12)) / (1 + len(null)))


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------
def pseo_coverage(h: str, cohorts: list[str]) -> pd.DataFrame:
    """coverage = PSEO earners at horizon h / IPEDS graduates (y1_ipeds_count; identical across
    y1/y5/y10 within a fixed cohort), summed over the field's CIP-4 rows, using exactly the rows
    load_er_pseo keeps (inst 'I', CIP-4, bachelor's, released y{h} earnings, non-missing p50)."""
    p50, ge, st = f"{h}_p50_earnings", f"{h}_grads_earn", f"status_{h}_earnings"
    use = ["inst_level", "institution", "degree_level", "cip_level", "cipcode", "grad_cohort",
           p50, ge, st, "y1_ipeds_count"]
    df = pd.read_csv(PSEO_EARN, dtype=str, usecols=use)
    df = df[(df["inst_level"] == "I") & (df["cip_level"] == "4") &
            df["grad_cohort"].isin(cohorts) & (df["degree_level"] == PSEO_DEGREE_LEVEL["undergrad"]) &
            (df[st] == "1")].copy()
    df["field"] = (df["cipcode"].astype(str).str.replace(".", "", regex=False).str.zfill(4)
                   .map(F.cip4_to_field_key(FIELDS66)))
    df = df[df["field"].notna()].copy()
    for c in (p50, ge, "y1_ipeds_count"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[p50])
    inst = _pseo_institutions().drop_duplicates("institution")
    df = df.merge(inst, on="institution", how="left")
    df = df[df["inst_key"].notna() & (df["inst_key"] != "")]
    ok = df[ge].notna() & df["y1_ipeds_count"].notna() & (df["y1_ipeds_count"] > 0)
    df["n_earn"] = np.where(ok, df[ge], 0.0)
    df["n_grad"] = np.where(ok, df["y1_ipeds_count"], 0.0)
    g = df.groupby(["field", "inst_key", "grad_cohort"], as_index=False)[["n_earn", "n_grad"]].sum()
    g["coverage"] = np.where(g["n_grad"] > 0, g["n_earn"] / g["n_grad"].where(g["n_grad"] > 0), np.nan)
    return g[["field", "inst_key", "grad_cohort", "coverage", "n_grad"]]


def load_fixed(h: str, cohorts: list[str]) -> pd.DataFrame:
    er = load_er_pseo("undergrad", h, fields=FIELDS66, grad_cohort=cohorts)
    return er.merge(pseo_coverage(h, cohorts), on=["field", "inst_key", "grad_cohort"], how="left")


def fixed_panel(er: dict, ar: pd.DataFrame, gen: pd.DataFrame) -> pd.DataFrame:
    """Balanced field x cohort x institution panel: earnings at y1, y5, y10 all present + prestige."""
    wide = None
    for h in HZ:
        x = er[h][er[h]["grad_cohort"].isin(COHORTS)][
            ["field", "inst_key", "grad_cohort", "earnings", "coverage"]].rename(
            columns={"earnings": f"e_{h}", "coverage": f"cov_{h}"})
        wide = x if wide is None else wide.merge(x, on=["field", "inst_key", "grad_cohort"], how="inner")
    wide = wide.merge(ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"}),
                      on=["field", "inst_key"], how="inner")
    wide = wide.merge(gen, on="inst_key", how="left")
    return wide.sort_values(["field", "grad_cohort", "inst_key"]).reset_index(drop=True)


def cells_of(panel: pd.DataFrame, keys=("field", "grad_cohort")) -> list[tuple]:
    out = []
    for k, g in panel.groupby(list(keys), sort=True):
        if len(g) >= NMIN:
            out.append((k, g.reset_index(drop=True)))
    return out


def classify(sc4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in sc4.sort_values("field").iterrows():
        rho, n, rel = r["spearman"], int(r["n_institutions"]), bool(r["reliable_flag"])
        if not rel or n < NMIN or not np.isfinite(rho):
            grp = "unclassified"
        elif rho >= 0.45:
            grp = "integrated"
        elif rho < 0.25:
            grp = "decoupled"
        else:
            grp = "middle"
        rows.append(dict(field=r["field"], baseline_rho=rho, baseline_n=n, baseline_reliable=rel,
                         group=grp, licensed=r["field"] in F.LICENSED_FIELDS))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# (a)+(b)+(e) fixed-cohort engine
# ---------------------------------------------------------------------------------------------
def cell_metrics(P, E, G, idx):
    """P (n,), E (n,3), G (n,) or None, idx (B,n) -> dict of per-replicate arrays."""
    rP = rk(P[idx])
    rE = [rk(E[:, k][idx]) for k in range(E.shape[1])]
    rhoF = np.stack([corr_rows(rP, r) for r in rE], -1)
    out = dict(rhoF=rhoF)
    if G is not None:
        rG = rk(G[idx])
        rFG = corr_rows(rP, rG)
        rhoG = np.stack([corr_rows(rG, r) for r in rE], -1)
        d = (1 - rFG ** 2)[:, None]
        d = np.where(d > 1e-12, d, np.nan)
        out.update(rFG=rFG, rhoG=rhoG,
                   bF=(rhoF - rFG[:, None] * rhoG) / d, bG=(rhoG - rFG[:, None] * rhoF) / d)
    return out


def run_fixed(panel: pd.DataFrame):
    """Observed + bootstrap + horizon-permutation quantities per cell and per field."""
    cells = cells_of(panel)
    rb, rp, rbb = rng_for("fixed_boot"), rng_for("fixed_perm"), rng_for("brand_boot")
    perms = np.array(list(itertools.permutations(range(3))))
    cell_rows, per_field = [], {}
    for (fld, coh), g in cells:
        n = len(g)
        P, G = g["P"].to_numpy(float), g["G"].to_numpy(float)
        E = g[[f"e_{h}" for h in HZ]].to_numpy(float)
        obs = cell_metrics(P, E, G, np.arange(n)[None, :])
        bt = cell_metrics(P, E, None, boot_idx(n, rb))                           # coupling
        bt.update({k: v for k, v in boot_valid(lambda ix: cell_metrics(P, E, G, ix), n, rbb,
                                               "brand").items() if k != "rhoF"})   # b_F, b_G, rho_G
        # horizon-label permutation: each institution's within-horizon earnings ranks shuffled
        R = np.stack([rankdata(E[:, k]) for k in range(3)], -1)                 # (n,3)
        sel = perms[rp.integers(6, size=(NPERM, n))]                            # (NPERM,n,3)
        Rp = R[np.arange(n)[None, :, None], sel]
        rP = rankdata(P)[None, :]
        pc = np.stack([corr_rows(rP, rk(Rp[:, :, k])) for k in range(3)], -1)    # (NPERM,3)
        d = per_field.setdefault(fld, dict(cohorts=[], n=[], rhoF=[], rhoF_b=[], rhoG=[], rhoG_b=[],
                                           bF=[], bF_b=[], bG=[], bG_b=[], rFG=[], perm=[]))
        d["cohorts"].append(coh); d["n"].append(n)
        for key in ("rhoF", "rhoG", "bF", "bG"):
            d[key].append(obs[key][0]); d[key + "_b"].append(bt[key])
        d["rFG"].append(float(obs["rFG"][0])); d["perm"].append(pc @ W_SLOPE)
        for k, h in enumerate(HZ):
            cell_rows.append(dict(field=fld, grad_cohort=coh, horizon=h, years=int(YRS[k]), n=n,
                                  coupling=obs["rhoF"][0, k], rho_G=obs["rhoG"][0, k],
                                  bF=obs["bF"][0, k], bG=obs["bG"][0, k], r_FG=obs["rFG"][0],
                                  cov_mean=float(np.nanmean(g[f"cov_{h}"])),
                                  rho_prestige_cov=float(spearmanr(P, g[f"cov_{h}"], nan_policy="omit")[0])))
    # field-level aggregation (mean over cohorts)
    fl = {}
    for fld, d in per_field.items():
        agg = dict(cohorts=d["cohorts"], n=d["n"], rFG=float(np.mean(d["rFG"])))
        for key in ("rhoF", "rhoG", "bF", "bG"):
            o = np.stack(d[key]); b = np.stack(d[key + "_b"])                     # (C,3), (C,B,3)
            agg[key + "_h"] = o.mean(0); agg[key + "_h_b"] = b.mean(0)
            agg[key + "_slope"] = float((o @ W_SLOPE).mean()); agg[key + "_slope_b"] = (b @ W_SLOPE).mean(0)
            agg[key + "_slope_cells"] = list(o @ W_SLOPE)
        agg["perm"] = np.stack(d["perm"]).mean(0)                                 # (NPERM,)
        h, hb = agg["rhoF_h"], agg["rhoF_h_b"]
        agg["early_slope"], agg["early_slope_b"] = (h[1] - h[0]) / 4, (hb[:, 1] - hb[:, 0]) / 4
        agg["late_slope"], agg["late_slope_b"] = (h[2] - h[1]) / 5, (hb[:, 2] - hb[:, 1]) / 5
        fl[fld] = agg
    return pd.DataFrame(cell_rows), fl


def group_stat(fl: dict, fields: list[str], key: str, tag: str):
    """mean over fields of fl[f][key] with a two-stage bootstrap CI."""
    obs = np.mean([fl[f][key] for f in fields], axis=0)
    bs = two_stage(np.stack([fl[f][key + "_b"] for f in fields]), rng_for(tag))
    return obs, bs


def contrast(fl, A, Bf, key, tag):
    oa, ba = group_stat(fl, A, key, tag + "_A")
    ob, bb = group_stat(fl, Bf, key, tag + "_B")
    return oa - ob, ba - bb


def field_label_perm(vals: dict, A: list, pool: list, tag: str) -> float:
    """p-value for mean(A) - mean(pool \\ A) under random relabelling of which fields are A."""
    v = np.array([vals[f] for f in pool]); isA = np.array([f in A for f in pool])
    obs = v[isA].mean() - v[~isA].mean()
    rng = rng_for(tag); k = isA.sum(); null = np.empty(NPERM_FIELD)
    for i in range(NPERM_FIELD):
        m = np.zeros(len(v), bool); m[rng.choice(len(v), k, replace=False)] = True
        null[i] = v[m].mean() - v[~m].mean()
    return perm_p(null, obs)


# ---------------------------------------------------------------------------------------------
# (c) calendar-matched contrast
# ---------------------------------------------------------------------------------------------
def run_calendar(er: dict, ar: pd.DataFrame):
    """Triple-matched (c,y1),(c,y10),(c+9,y1) and pair-matched (c,y10),(c+9,y1) contrasts."""
    a = ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"})
    y1, y10 = er["y1"], er["y10"]
    rows, fl = [], {}
    rb = rng_for("cal_boot")
    for c in COHORTS:
        c9 = LATE[c]
        s0 = y1[y1.grad_cohort == c][["field", "inst_key", "earnings"]].rename(columns={"earnings": "e0"})
        s1 = y10[y10.grad_cohort == c][["field", "inst_key", "earnings"]].rename(columns={"earnings": "e1"})
        s2 = y1[y1.grad_cohort == c9][["field", "inst_key", "earnings"]].rename(columns={"earnings": "e2"})
        pair = s1.merge(s2, on=["field", "inst_key"]).merge(a, on=["field", "inst_key"])
        trip = pair.merge(s0, on=["field", "inst_key"])
        for fld in sorted(set(pair.field)):
            gp = pair[pair.field == fld].sort_values("inst_key")
            gt = trip[trip.field == fld].sort_values("inst_key")
            rec = dict(field=fld, cohort=c, late_cohort=c9, n_pair=len(gp), n_triple=len(gt))
            if len(gp) >= NMIN:
                Pp, Ep = gp.P.to_numpy(float), gp[["e1", "e2"]].to_numpy(float)
                o = cell_metrics(Pp, Ep, None, np.arange(len(gp))[None, :])["rhoF"][0]
                b = cell_metrics(Pp, Ep, None, boot_idx(len(gp), rb))["rhoF"]
                rec.update(pair_rho_c_y10=o[0], pair_rho_c9_y1=o[1], D_cal_pair=(o[0] - o[1]) / 9)
                fl.setdefault(fld, {}).setdefault("D_cal_pair", []).append(((o[0] - o[1]) / 9, (b[:, 0] - b[:, 1]) / 9))
            if len(gt) >= NMIN:
                Pt, Et = gt.P.to_numpy(float), gt[["e0", "e1", "e2"]].to_numpy(float)
                o = cell_metrics(Pt, Et, None, np.arange(len(gt))[None, :])["rhoF"][0]
                b = cell_metrics(Pt, Et, None, boot_idx(len(gt), rb))["rhoF"]
                dw, dc, dx = (o[1] - o[0]) / 9, (o[1] - o[2]) / 9, (o[2] - o[0]) / 9
                rec.update(rho_c_y1=o[0], rho_c_y10=o[1], rho_c9_y1=o[2],
                           D_within=dw, D_cal=dc, D_cross=dx)
                for key, ov, bv in (("D_within", dw, (b[:, 1] - b[:, 0]) / 9),
                                    ("D_cal", dc, (b[:, 1] - b[:, 2]) / 9),
                                    ("D_cross", dx, (b[:, 2] - b[:, 0]) / 9)):
                    fl.setdefault(fld, {}).setdefault(key, []).append((ov, bv))
            rows.append(rec)
    agg = {}
    for fld, d in fl.items():
        agg[fld] = {}
        for key, lst in d.items():
            agg[fld][key] = float(np.mean([o for o, _ in lst]))
            agg[fld][key + "_b"] = np.mean([b for _, b in lst], axis=0)
            agg[fld][key + "_k"] = len(lst)
    return pd.DataFrame(rows), agg


# ---------------------------------------------------------------------------------------------
# (d) coverage sensitivity
# ---------------------------------------------------------------------------------------------
def run_coverage(panel: pd.DataFrame):
    cp = panel.dropna(subset=[f"cov_{h}" for h in HZ]).copy()
    rb = rng_for("cov_boot")
    fl, cell_rows = {}, []
    for (fld, coh), g in cells_of(cp):
        n = len(g)
        P = g.P.to_numpy(float); E = g[[f"e_{h}" for h in HZ]].to_numpy(float)
        C = g[[f"cov_{h}" for h in HZ]].to_numpy(float)
        def fn(idx):
            rP = rk(P[idx])
            return dict(raw=np.stack([corr_rows(rP, rk(E[:, k][idx])) for k in range(3)], -1),
                        par=np.stack([partial_rows(rP, rk(E[:, k][idx]), rk(C[:, k][idx]))
                                      for k in range(3)], -1))
        o_, b_ = fn(np.arange(n)[None, :]), boot_valid(fn, n, rb, "coverage_partial")
        res = {"o": (o_["raw"], o_["par"]), "b": (b_["raw"], b_["par"])}
        d = fl.setdefault(fld, dict(raw=[], raw_b=[], par=[], par_b=[]))
        d["raw"].append(res["o"][0][0] @ W_SLOPE); d["raw_b"].append(res["b"][0] @ W_SLOPE)
        d["par"].append(res["o"][1][0] @ W_SLOPE); d["par_b"].append(res["b"][1] @ W_SLOPE)
        for k, h in enumerate(HZ):
            cell_rows.append(dict(field=fld, grad_cohort=coh, horizon=h, n_cov=n,
                                  coupling_covsample=res["o"][0][0, k],
                                  coupling_partial_cov=res["o"][1][0, k]))
    agg = {f: dict(raw=float(np.mean(d["raw"])), raw_b=np.mean(d["raw_b"], 0),
                   par=float(np.mean(d["par"])), par_b=np.mean(d["par_b"], 0)) for f, d in fl.items()}
    return pd.DataFrame(cell_rows), agg


def run_highcov(panel: pd.DataFrame):
    thr = float(panel["cov_y1"].median())
    hp = panel[panel["cov_y1"] >= thr]
    rb = rng_for("hicov_boot")
    fl = {}
    for (fld, coh), g in cells_of(hp):
        n = len(g)
        P = g.P.to_numpy(float); E = g[[f"e_{h}" for h in HZ]].to_numpy(float)
        o = cell_metrics(P, E, None, np.arange(n)[None, :])["rhoF"][0] @ W_SLOPE
        b = cell_metrics(P, E, None, boot_idx(n, rb))["rhoF"] @ W_SLOPE
        d = fl.setdefault(fld, dict(s=[], s_b=[], n=[]))
        d["s"].append(o); d["s_b"].append(b); d["n"].append(n)
    agg = {f: dict(s=float(np.mean(d["s"])), s_b=np.mean(d["s_b"], 0), k=len(d["s"]))
           for f, d in fl.items()}
    return thr, agg


# ---------------------------------------------------------------------------------------------
# (f) pooled "0000" (cohort-mixed) and legacy spec
# ---------------------------------------------------------------------------------------------
def run_pooled(ar: pd.DataFrame):
    a = ar[["field", "inst_key", "prestige_score"]].rename(columns={"prestige_score": "P"})
    er0 = {h: load_er_pseo("undergrad", h, fields=FIELDS66) for h in HZ}
    wide = None
    for h in HZ:
        x = er0[h][["field", "inst_key", "earnings"]].rename(columns={"earnings": f"e_{h}"})
        wide = x if wide is None else wide.merge(x, on=["field", "inst_key"], how="inner")
    wide = wide.merge(a, on=["field", "inst_key"]).sort_values(["field", "inst_key"])
    rb = rng_for("pooled_boot")
    fl, rows = {}, []
    for fld, g in cells_of(wide, keys=("field",)):
        fld = fld[0] if isinstance(fld, tuple) else fld
        n = len(g); P = g.P.to_numpy(float); E = g[[f"e_{h}" for h in HZ]].to_numpy(float)
        o = cell_metrics(P, E, None, np.arange(n)[None, :])["rhoF"][0]
        b = cell_metrics(P, E, None, boot_idx(n, rb))["rhoF"]
        fl[fld] = dict(s=float(o @ W_SLOPE), s_b=b @ W_SLOPE, h=o, n=n)
        for k, h in enumerate(HZ):
            rows.append(dict(field=fld, horizon=h, years=int(YRS[k]), n=n, coupling=o[k]))
    # legacy: TIER0 CIP map, pooled 0000, unbalanced cells, polyfit over available horizons
    leg_rows = []
    for h in HZ:
        e = load_er_pseo("undergrad", h)
        m = a.merge(e[["field", "inst_key", "earnings"]], on=["field", "inst_key"]).dropna()
        for fld, g in m.groupby("field", sort=True):
            leg_rows.append(dict(field=fld, horizon=h, years=int(YRS[HZ.index(h)]), n=len(g),
                                 coupling=spearmanr(g.P, g.earnings)[0]))
    leg = pd.DataFrame(leg_rows)
    leg_slope = {}
    for fld, g in leg[leg.n >= NMIN].groupby("field", sort=True):
        if g.years.nunique() >= 2:
            leg_slope[fld] = float(np.polyfit(g.years.to_numpy(float), g.coupling.to_numpy(float), 1)[0])
    return pd.DataFrame(rows), fl, leg, leg_slope


# ---------------------------------------------------------------------------------------------
# (g) Scorecard descriptive panel
# ---------------------------------------------------------------------------------------------
def run_scorecard(ar: pd.DataFrame):
    rows, gm4 = [], None
    for h, (ec, cc) in SC_HORIZONS.items():
        er = load_er_scorecard("undergrad", earn_col=ec, count_col=cc, fields=FIELDS66)
        gm = compute_gap_map(ar, er, "undergrad", B=B_GAP, seed=SEED)
        if h == "4YR":
            gm4 = gm.copy()
        for _, r in gm.iterrows():
            rows.append(dict(field=r["field"], horizon=h, years=SC_YEARS[h], n=int(r["n_institutions"]),
                             coupling=r["spearman"], reliable=bool(r["reliable_flag"])))
    return pd.DataFrame(rows), gm4


# ---------------------------------------------------------------------------------------------
def main():
    ar = load_ar_wapman(fields=FIELDS66)
    gen = _s28.load_generic()
    gen = gen.assign(G=-gen["g_rank"].astype(float))[["inst_key", "G"]]

    # (g) Scorecard: descriptive panel + 4YR baseline for the field groups
    sc_long, gm4 = run_scorecard(ar)
    cls = classify(gm4)
    grp_of = dict(zip(cls.field, cls.group))

    # (a) fixed cohorts
    er = {"y1": load_fixed("y1", COHORTS + [LATE[c] for c in COHORTS]),
          "y5": load_fixed("y5", COHORTS), "y10": load_fixed("y10", COHORTS)}
    panel = fixed_panel(er, ar, gen)
    assert panel["G"].notna().all(), "academia-wide rank missing for a panel institution"
    cell_df, fl = run_fixed(panel)
    fields_all = sorted(fl)
    gsets = {g: [f for f in fields_all if grp_of.get(f, "unclassified") == g] for g in GROUPS}
    INT, OTH = gsets["integrated"], [f for f in fields_all if grp_of.get(f, "unclassified") != "integrated"]

    R = {}   # key numbers: name -> dict(est, lo, hi, p, k, sample)

    def put(name, est, bs=None, p=np.nan, k=np.nan, sample=""):
        lo, hi = ci(bs) if bs is not None else (np.nan, np.nan)
        R[name] = dict(est=float(est), lo=lo, hi=hi, p=p, k=k, sample=sample)

    # slopes by group + inference
    perm_null = {}
    for name, fs in [("all", fields_all)] + [(g, gsets[g]) for g in GROUPS]:
        if not fs:
            continue
        o, bs = group_stat(fl, fs, "rhoF_slope", f"slope_{name}")
        null = np.mean([fl[f]["perm"] for f in fs], axis=0)
        perm_null[name] = null
        put(f"slope_{name}", o, bs, perm_p(null, o), len(fs),
            f"PSEO fixed cohorts 2001-2010, balanced, {name} fields")
    o, bs = contrast(fl, INT, OTH, "rhoF_slope", "slope_int_minus_oth")
    p_lab = field_label_perm({f: fl[f]["rhoF_slope"] for f in fields_all}, INT, fields_all, "lab_slope")
    put("slope_int_minus_others", o, bs, p_lab, f"{len(INT)} vs {len(OTH)}",
        "PSEO fixed cohorts; integrated minus all other fields")
    # group trajectories (field-mean coupling at each horizon)
    traj = {}
    for name, fs in [("all", fields_all)] + [(g, gsets[g]) for g in GROUPS]:
        if fs:
            o, bs = group_stat(fl, fs, "rhoF_h", f"traj_{name}")
            traj[name] = (o, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0), len(fs))
    # per-cohort group slopes (observed; unbalanced field sets across cohorts)
    by_coh = []
    for c in COHORTS:
        for name, fs in (("all", fields_all), ("integrated", INT)):
            v = [fl[f]["rhoF_slope_cells"][fl[f]["cohorts"].index(c)] for f in fs if c in fl[f]["cohorts"]]
            by_coh.append(dict(cohort=c, group=name, k=len(v), mean=float(np.mean(v)) if v else np.nan,
                               n_pos=int(np.sum(np.array(v) > 0)) if v else 0))
    by_coh = pd.DataFrame(by_coh)
    # segment slopes: y1->y5 and y5->y10 (the latter avoids the y1 point, where post-graduation
    # enrollment depresses earnings most)
    for seg in ("early", "late"):
        for name, fs in (("all", fields_all), ("integrated", INT)):
            o, bs = group_stat(fl, fs, f"{seg}_slope", f"{seg}_{name}")
            put(f"{seg}_slope_{name}", o, bs, np.nan, len(fs),
                f"PSEO fixed cohorts; ({'y5-y1)/4' if seg == 'early' else 'y10-y5)/5'}; {name}")
        o, bs = contrast(fl, INT, OTH, f"{seg}_slope", f"{seg}_c")
        put(f"{seg}_slope_int_minus_others", o, bs,
            field_label_perm({f: fl[f][f"{seg}_slope"] for f in fields_all}, INT, fields_all, f"lab_{seg}"),
            f"{len(INT)} vs {len(OTH)}", f"PSEO fixed cohorts; {seg} segment; integrated minus others")
    # grouping by baseline rho alone (ignoring the reliability flag)
    base = cls.set_index("field")
    INT_R = [f for f in fields_all if f in base.index and base.loc[f, "baseline_rho"] >= 0.45
             and base.loc[f, "baseline_n"] >= NMIN]
    OTH_R = [f for f in fields_all if f not in INT_R]
    o, bs = contrast(fl, INT_R, OTH_R, "rhoF_slope", "slope_intR")
    put("slope_intrho_minus_others", o, bs,
        field_label_perm({f: fl[f]["rhoF_slope"] for f in fields_all}, INT_R, fields_all, "lab_intR"),
        f"{len(INT_R)} vs {len(OTH_R)}", "PSEO fixed cohorts; Scorecard 4YR rho>=0.45 (no reliability flag) minus others")
    # continuous: across fields, Spearman(field slope, Scorecard baseline rho)
    fb = [f for f in fields_all if f in base.index and np.isfinite(base.loc[f, "baseline_rho"])]
    xr = np.array([base.loc[f, "baseline_rho"] for f in fb]); ys = np.array([fl[f]["rhoF_slope"] for f in fb])
    o = spearmanr(xr, ys)[0]
    rng = rng_for("xfield"); K = len(fb)
    fi = rng.integers(K, size=(NBOOT, K)); cix = rng.integers(NBOOT, size=(NBOOT, K))
    Sb = np.stack([fl[f]["rhoF_slope_b"] for f in fb])[fi, cix]
    bs = corr_rows(rk(xr[fi]), rk(Sb))
    null = np.array([spearmanr(rng.permutation(xr), ys)[0] for _ in range(NPERM_FIELD)])
    put("xfield_spearman_slope_baseline", o, bs[np.isfinite(bs)], perm_p(null, o), K,
        "across fields: Spearman(fixed-cohort slope, Scorecard 4YR baseline rho)")

    # (e) brand vs field
    for name, fs in (("all", fields_all), ("integrated", INT), ("others", OTH)):
        for key in ("bF", "bG", "rhoG", "rhoF"):
            o, bs = group_stat(fl, fs, f"{key}_slope", f"bf_{name}_{key}")
            put(f"d{key}_{name}", o, bs, np.nan, len(fs), f"PSEO fixed cohorts; {name}; slope of {key}")
        o, bs = group_stat(fl, fs, "bG_slope", f"bf_{name}_bG2")
        o2, bs2 = group_stat(fl, fs, "bF_slope", f"bf_{name}_bG2")
        put(f"dG_minus_dF_{name}", o - o2, bs - bs2, np.nan, len(fs), f"PSEO fixed cohorts; {name}")
    bfh = {}
    for name, fs in (("all", fields_all), ("integrated", INT)):
        for key in ("bF", "bG"):
            o, bs = group_stat(fl, fs, f"{key}_h", f"bfh_{name}_{key}")
            bfh[(name, key)] = (o, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0))
    rFG_cells = cell_df.drop_duplicates(["field", "grad_cohort"])["r_FG"]
    bcount = {h: (int((cell_df[cell_df.horizon == h].bF > cell_df[cell_df.horizon == h].bG).sum()),
                  int((cell_df.horizon == h).sum())) for h in HZ}
    fcount = dict(dG_pos=sum(fl[f]["bG_slope"] > 0 for f in fields_all),
                  dF_pos=sum(fl[f]["bF_slope"] > 0 for f in fields_all), k=len(fields_all))

    # (c) calendar-matched
    cal_df, cal = run_calendar(er, ar)
    cal_fields_t = sorted(f for f in cal if "D_cal" in cal[f])
    cal_fields_p = sorted(f for f in cal if "D_cal_pair" in cal[f])
    for name, fs in (("all", cal_fields_t), ("integrated", [f for f in cal_fields_t if f in INT])):
        for key in ("D_within", "D_cal", "D_cross"):
            o = np.mean([cal[f][key] for f in fs])
            bs = two_stage(np.stack([cal[f][key + "_b"] for f in fs]), rng_for(f"cal_{name}_{key}"))
            put(f"{key}_{name}", o, bs, np.nan, len(fs), f"triple-matched institutions; {name}")
    for name, fs in (("all", cal_fields_p), ("integrated", [f for f in cal_fields_p if f in INT])):
        o = np.mean([cal[f]["D_cal_pair"] for f in fs])
        bs = two_stage(np.stack([cal[f]["D_cal_pair_b"] for f in fs]), rng_for(f"calp_{name}"))
        put(f"D_cal_pair_{name}", o, bs, np.nan, len(fs), f"pair-matched institutions; {name}")

    # (d) coverage
    cov_cells, cov = run_coverage(panel)
    cov_fields = sorted(cov)
    for name, fs in (("all", cov_fields), ("integrated", [f for f in cov_fields if f in INT])):
        rng = rng_for(f"cov_{name}")
        K = len(fs)
        fi = rng.integers(K, size=(NBOOT, K)); ci_ = rng.integers(NBOOT, size=(NBOOT, K))
        raw_s = np.stack([cov[f]["raw_b"] for f in fs]); par_s = np.stack([cov[f]["par_b"] for f in fs])
        raw_b, par_b = raw_s[fi, ci_].mean(1), par_s[fi, ci_].mean(1)   # same draws -> paired
        ro, po = np.mean([cov[f]["raw"] for f in fs]), np.mean([cov[f]["par"] for f in fs])
        put(f"cov_raw_{name}", ro, raw_b, np.nan, K, f"coverage-complete panel; raw coupling; {name}")
        put(f"cov_partial_{name}", po, par_b, np.nan, K, f"coverage-complete panel; partial given coverage; {name}")
        put(f"cov_share_removed_{name}", 1 - po / ro, 1 - par_b / raw_b, np.nan, K,
            f"1 - partial/raw slope; {name}")
    thr, hic = run_highcov(panel)
    hic_fields = sorted(hic)
    for name, fs in (("all", hic_fields), ("integrated", [f for f in hic_fields if f in INT])):
        o = np.mean([hic[f]["s"] for f in fs])
        bs = two_stage(np.stack([hic[f]["s_b"] for f in fs]), rng_for(f"hicov_{name}"))
        put(f"hicov_{name}", o, bs, np.nan, len(fs), f"y1 coverage >= {thr:.3f}; {name}")
        put(f"hicov_fullsample_samefields_{name}", np.mean([fl[f]["rhoF_slope"] for f in fs]),
            group_stat(fl, fs, "rhoF_slope", f"hicov_ref_{name}")[1], np.nan, len(fs),
            f"full fixed-cohort panel, same fields as high-coverage; {name}")
    rpc = cell_df.groupby("horizon")["rho_prestige_cov"].mean()

    # (f) pooled 0000 + legacy
    pooled_long, pfl, leg, leg_slope = run_pooled(ar)
    pf_all = sorted(pfl)
    pf_int = [f for f in pf_all if f in INT]
    pf_oth = [f for f in pf_all if f not in INT]
    for name, fs in (("all", pf_all), ("integrated", pf_int)):
        o = np.mean([pfl[f]["s"] for f in fs])
        bs = two_stage(np.stack([pfl[f]["s_b"] for f in fs]), rng_for(f"pooled_{name}"))
        put(f"pooled_{name}", o, bs, np.nan, len(fs), f"PSEO pooled 0000 (cohort-mixed), balanced; {name}")
    oa = np.mean([pfl[f]["s"] for f in pf_int]); ob = np.mean([pfl[f]["s"] for f in pf_oth])
    ba = two_stage(np.stack([pfl[f]["s_b"] for f in pf_int]), rng_for("pooled_c_A"))
    bb = two_stage(np.stack([pfl[f]["s_b"] for f in pf_oth]), rng_for("pooled_c_B"))
    put("pooled_int_minus_others", oa - ob, ba - bb, np.nan, f"{len(pf_int)} vs {len(pf_oth)}",
        "PSEO pooled 0000 (cohort-mixed)")
    leg_int = [f for f in sorted(leg_slope) if grp_of.get(f) == "integrated"]
    put("legacy_integrated", np.mean([leg_slope[f] for f in leg_int]), None, np.nan, len(leg_int),
        "legacy spec: pooled 0000, TIER0 CIP map, unbalanced")

    # (g) Scorecard descriptive slopes (no inference)
    sc_slope = {}
    for fld, g in sc_long[sc_long.n >= NMIN].groupby("field", sort=True):
        if g.years.nunique() >= 2:
            sc_slope[fld] = float(np.polyfit(g.years.to_numpy(float), g.coupling.to_numpy(float), 1)[0])
    sc_int = [f for f in sorted(sc_slope) if grp_of.get(f) == "integrated"]
    put("scorecard_integrated", np.mean([sc_slope[f] for f in sc_int]), None, np.nan, len(sc_int),
        "Scorecard 1/4/5YR, one release = different cohorts (descriptive)")
    put("scorecard_all", np.mean([sc_slope[f] for f in sorted(sc_slope)]), None, np.nan, len(sc_slope),
        "Scorecard 1/4/5YR (descriptive)")

    # ---------------- write tables ----------------
    write_tables(cell_df, fl, cls, grp_of, cov_cells, pooled_long, pfl, leg, sc_long, cal_df, R)
    make_figure(fl, fields_all, gsets, traj, bfh, R)
    write_md(R, cls, grp_of, fl, fields_all, gsets, traj, by_coh, bfh, rFG_cells, cal_df, rpc, thr,
             hic, cov, pfl, leg_slope, sc_long, panel, bcount, fcount, INT_R)
    for k, v in R.items():
        print(f"{k:38s} {v['est']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}] p={v['p']:.4f} k={v['k']}")


# ---------------------------------------------------------------------------------------------
def write_tables(cell_df, fl, cls, grp_of, cov_cells, pooled_long, pfl, leg, sc_long, cal_df, R):
    lab = lambda f: LAB.get(f, f)
    # career_time_coupling.csv: cell rows (all sources) + field-slope rows + summary rows
    fx = cell_df.merge(cov_cells, on=["field", "grad_cohort", "horizon"], how="left")
    fx = fx.assign(record="cell", source="pseo_fixed_cohort")[
        ["record", "source", "field", "grad_cohort", "horizon", "years", "n", "coupling",
         "cov_mean", "rho_prestige_cov", "n_cov", "coupling_covsample", "coupling_partial_cov"]]
    po = pooled_long.assign(record="cell", source="pseo_pooled_0000_cohort_mixed", grad_cohort="0000")
    lg = leg.assign(record="cell", source="pseo_pooled_0000_legacy_tier0_unbalanced", grad_cohort="0000")
    sc = sc_long.assign(record="cell", source="scorecard_cohort_confounded", grad_cohort="n/a")
    fs_rows = []
    for f in sorted(fl):
        d = fl[f]
        lo, hi = ci(d["rhoF_slope_b"])
        fs_rows.append(dict(record="field_slope", source="pseo_fixed_cohort", field=f,
                            grad_cohort="+".join(d["cohorts"]), n=int(min(d["n"])), n_max=int(max(d["n"])),
                            slope_per_yr=d["rhoF_slope"], slope_lo=lo, slope_hi=hi,
                            coupling_y1=d["rhoF_h"][0], coupling_y5=d["rhoF_h"][1], coupling_y10=d["rhoF_h"][2]))
    for f in sorted(pfl):
        lo, hi = ci(pfl[f]["s_b"])
        fs_rows.append(dict(record="field_slope", source="pseo_pooled_0000_cohort_mixed", field=f,
                            grad_cohort="0000", n=pfl[f]["n"], n_max=pfl[f]["n"], slope_per_yr=pfl[f]["s"],
                            slope_lo=lo, slope_hi=hi, coupling_y1=pfl[f]["h"][0],
                            coupling_y5=pfl[f]["h"][1], coupling_y10=pfl[f]["h"][2]))
    sm_rows = [dict(record="summary", source="see_sample", stat=k, sample=v["sample"], estimate=v["est"],
                    ci_lo=v["lo"], ci_hi=v["hi"], p_value=v["p"], k_fields=str(v["k"])) for k, v in R.items()]
    out = pd.concat([fx, po, lg, sc, pd.DataFrame(fs_rows), pd.DataFrame(sm_rows)], ignore_index=True)
    out["label"] = out["field"].map(lab)
    out["group"] = out["field"].map(lambda f: grp_of.get(f, "unclassified") if isinstance(f, str) else np.nan)
    out = out.merge(cls[["field", "baseline_rho", "baseline_n"]], on="field", how="left")
    first = ["record", "source", "field", "label", "group", "baseline_rho", "grad_cohort", "horizon", "years"]
    out = out[first + [c for c in out.columns if c not in first]]
    out.to_csv(OUT_CSV, index=False, float_format="%.6g")

    # career_time_brand_field.csv
    bf = cell_df[["field", "grad_cohort", "horizon", "years", "n", "r_FG", "coupling", "rho_G", "bF", "bG"]]
    bf = bf.rename(columns={"coupling": "rho_F"}).assign(record="cell")
    br = []
    for f in sorted(fl):
        d = fl[f]
        rec = dict(record="field_slope", field=f, grad_cohort="+".join(d["cohorts"]), r_FG=d["rFG"])
        for key in ("bF", "bG", "rhoF", "rhoG"):
            lo, hi = ci(d[f"{key}_slope_b"])
            rec.update({f"d{key}": d[f"{key}_slope"], f"d{key}_lo": lo, f"d{key}_hi": hi})
        br.append(rec)
    bf = pd.concat([bf, pd.DataFrame(br)], ignore_index=True)
    bf["label"] = bf["field"].map(lab); bf["group"] = bf["field"].map(lambda f: grp_of.get(f, "unclassified"))
    first = ["record", "field", "label", "group", "grad_cohort", "horizon", "years", "n"]
    bf[first + [c for c in bf.columns if c not in first]].to_csv(OUT_BF, index=False, float_format="%.6g")

    # career_time_calendar_contrast.csv
    cd = cal_df.copy()
    cd["label"] = cd["field"].map(lab); cd["group"] = cd["field"].map(lambda f: grp_of.get(f, "unclassified"))
    cd["calendar_window_y10_old"] = cd["cohort"].map(lambda c: f"cohort {c}-{int(c)+2} +10y")
    cd["calendar_window_y1_young"] = cd["late_cohort"].map(lambda c: f"cohort {c}-{int(c)+2} +1y")
    first = ["field", "label", "group", "cohort", "late_cohort", "n_triple", "n_pair"]
    cd = cd[first + [c for c in cd.columns if c not in first]].sort_values(["field", "cohort"])
    cd.to_csv(OUT_CAL, index=False, float_format="%.6g")
    print(f"[csv] {OUT_CSV}\n[csv] {OUT_BF}\n[csv] {OUT_CAL}")


# ---------------------------------------------------------------------------------------------
def make_figure(fl, fields_all, gsets, traj, bfh, R):
    colors = {"integrated": "#1b7837", "middle": "#8c6d31", "decoupled": "#b2182b",
              "unclassified": "#9e9e9e", "all": "#111111"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), layout="constrained",
                             gridspec_kw=dict(width_ratios=[1.1, 1, 1.0]))
    ax = axes[0]
    for g in GROUPS:
        for f in gsets[g]:
            ax.plot(YRS, fl[f]["rhoF_h"], color=colors[g], alpha=0.16, lw=0.9)
    for g in ["all"] + GROUPS:
        if g not in traj:
            continue
        o, lo, hi, k = traj[g]
        ax.plot(YRS, o, color=colors[g], lw=2.4 if g != "all" else 1.8, marker="o",
                ls="-" if g != "all" else "--", label=f"{g} (k={k} fields)")
        if g in ("integrated", "all"):
            ax.fill_between(YRS, lo, hi, color=colors[g], alpha=0.10)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(YRS); ax.set_xlabel("years since graduation")
    ax.set_ylabel("coupling = Spearman(field prestige, median earnings)")
    ax.set_title("A. PSEO fixed cohorts 2001-2010, balanced panels\n"
                 "field lines = mean over cohorts; band = two-stage 95% CI", fontsize=9.5)
    ax.legend(fontsize=7.5, loc="lower right")

    ax = axes[1]
    for (name, key), (o, lo, hi) in bfh.items():
        off = {"bF": -0.25, "bG": 0.25}[key] + (0 if name == "all" else 0.1)
        col = "#2166ac" if key == "bF" else "#d6604d"
        mk = "o" if name == "all" else "s"
        ax.errorbar(YRS + off, o, yerr=[o - lo, hi - o], color=col, marker=mk, lw=1.8 if name == "all" else 1,
                    alpha=1 if name == "all" else 0.55, capsize=3,
                    label=f"{'b_F field prestige' if key == 'bF' else 'b_G academia-wide rank'} ({name})")
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_xticks(YRS); ax.set_xlabel("years since graduation")
    ax.set_ylabel("standardized rank-regression coefficient")
    ax.set_title("B. earnings ~ F + G per field x cohort x horizon\n"
                 "mean over fields; two-stage 95% CI (F, G collinear)", fontsize=9.5)
    ax.legend(fontsize=7.2, loc="upper left")

    ax = axes[2]
    items = [("within-cohort slope, all fields", "slope_all"),
             ("within-cohort slope, integrated", "slope_integrated"),
             ("integrated minus others", "slope_int_minus_others"),
             ("y5->y10 segment only, all fields", "late_slope_all"),
             ("within-cohort y1->y10 (triple-matched), all", "D_within_all"),
             ("calendar-matched c@y10 vs c+9@y1, all", "D_cal_all"),
             ("cross-cohort drift at y1, all", "D_cross_all"),
             ("partial | coverage, all (cov. sample)", "cov_partial_all"),
             ("raw, same coverage sample, all", "cov_raw_all"),
             ("above-median y1 coverage, all", "hicov_all"),
             ("dG (academia-wide-rank coef. slope), all", "dbG_all"),
             ("dF (field coef. slope), all", "dbF_all"),
             ("pooled 0000 (cohort-mixed), all", "pooled_all"),
             ("pooled 0000 (cohort-mixed), integrated", "pooled_integrated"),
             ("Scorecard 1/4/5YR, integrated (descr.)", "scorecard_integrated")]
    for i, (lab, key) in enumerate(items):
        v = R[key]; y = len(items) - i
        col = "#555555" if key.startswith(("pooled", "scorecard")) else "#1b7837" if "integrated" in key else "#111111"
        if np.isfinite(v["lo"]):
            ax.plot([v["lo"], v["hi"]], [y, y], color=col, lw=1.6)
        ax.plot(v["est"], y, "o", color=col, ms=5, mfc="white" if key.startswith(("pooled", "scorecard")) else col)
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_yticks(range(len(items), 0, -1)); ax.set_yticklabels([l for l, _ in items], fontsize=7.6)
    ax.set_xlabel("change in coupling (or coefficient) per year")
    ax.set_title("C. per-year estimates with 95% CI\n(open markers = cohort-mixed or cohort-confounded; "
                 "Scorecard: no CI)", fontsize=9.5)
    fig.savefig(OUT_FIG, dpi=140, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    print(f"[fig] {OUT_FIG}")


# ---------------------------------------------------------------------------------------------
def fm(v, d=3):
    return "n/a" if v is None or not np.isfinite(v) else f"{v:+.{d}f}"


def row(R, key, label):
    v = R[key]
    cis = (f"[{fm(v['lo'])}, {fm(v['hi'])}]" if np.isfinite(v["lo"]) else
           "no CI (descriptive)" if "descriptive" in v["sample"] else "point estimate only")
    if not np.isfinite(v["p"]):
        p = "—"
    elif v["p"] <= 1.5 / (NPERM + 1) and "vs" not in str(v["k"]):
        p = f"<0.001 (0 of {NPERM})"
    else:
        p = f"{v['p']:.3f}"
    return f"| {label} | {v['sample']} | {fm(v['est'])} | {cis} | {p} | {v['k']} |"


def excl0(v):
    return (v["lo"] > 0) or (v["hi"] < 0)


def write_md(R, cls, grp_of, fl, fields_all, gsets, traj, by_coh, bfh, rFG_cells, cal_df, rpc, thr,
             hic, cov, pfl, leg_slope, sc_long, panel, bcount, fcount, INT_R):
    L = []
    s_all, s_int, s_c = R["slope_all"], R["slope_integrated"], R["slope_int_minus_others"]
    dcal, dwit, dcro = R["D_cal_all"], R["D_within_all"], R["D_cross_all"]
    coh_int = by_coh[by_coh.group == "integrated"]; coh_all = by_coh[by_coh.group == "all"]
    dec_fields = gsets["decoupled"]
    comm_n = panel[panel.field == "communication_disorders"].groupby("grad_cohort").size().to_dict()
    L.append("# Direction 1 — career-time coupling (canonical fixed-cohort version)\n")
    L.append("Script: `scripts/52_career_time_coupling.py` (seeded, byte-identical on re-run). "
             "coupling = Spearman(field prestige, median earnings) across institutions within a field. "
             "Descriptive, not causal. This replaces the earlier version, which used the PSEO pooled "
             "cohort \"0000\" (y1/y5/y10 average different graduation cohorts) and a 30-field CIP map; "
             "those labels (\"clean\", \"same-cohort\") are withdrawn.\n")
    L.append("## Answer\n")
    ea, la, lc = R["early_slope_all"], R["late_slope_all"], R["late_slope_int_minus_others"]
    ei, li, ec = R["early_slope_integrated"], R["late_slope_integrated"], R["early_slope_int_minus_others"]
    cr, xf = R["slope_intrho_minus_others"], R["xfield_spearman_slope_baseline"]
    dcal_i, dwit_i = R["D_cal_integrated"], R["D_within_integrated"]
    cs = R["cov_share_removed_all"]; hi_ = R["hicov_all"]; hir = R["hicov_fullsample_samefields_all"]
    dF, dG, dd = R["dbF_all"], R["dbG_all"], R["dG_minus_dF_all"]
    bF_o, bG_o = bfh[("all", "bF")][0], bfh[("all", "bG")][0]
    L.append(
        "**Key question: does coupling change with years since graduation, and is the change specific "
        "to integrated fields?** On fixed PSEO cohorts, coupling rises with years since graduation "
        f"({fm(s_all['est'])}/yr averaged over all {s_all['k']} fields). It rises in every cohort, and it "
        "rises from y5 to y10 as well as from y1 to y5. None of three tests shows the rise to be specific to "
        "integrated fields. Under a sign assumption (stated below), the career-time component is between "
        f"{fm(dcal['est'])} and {fm(dwit['est'])}/yr, the remainder being calendar time. A "
        f"coverage control removes about {cs['est']:.0%} of the slope; restricting to high-coverage "
        "institutions removes none. In a rank regression on field prestige F and the academia-wide rank G, "
        "the rise appears in the G coefficient and the F coefficient stays flat. The earlier claims "
        "(\"clean same-cohort\", \"integrated fields rise, decoupled fields pinned at 0\") are withdrawn.\n")
    L.append(
        f"1. **Rise on fixed cohorts.** Mean per-field OLS slope over y1/y5/y10, PSEO bachelor's cohorts "
        f"2001/2004/2007/2010, balanced panels: all {s_all['k']} fields {fm(s_all['est'])}/yr "
        f"[{fm(s_all['lo'])}, {fm(s_all['hi'])}] (horizon-label permutation: 0 of {NPERM} permutations as "
        f"extreme); integrated ({s_int['k']}) {fm(s_int['est'])} [{fm(s_int['lo'])}, {fm(s_int['hi'])}]. "
        f"Mean slope by cohort is positive in {int((coh_all['mean'] > 0).sum())}/{len(coh_all)} cohorts "
        f"(all fields) and {int((coh_int['mean'] > 0).sum())}/{len(coh_int)} (integrated). Segments: y1->y5 "
        f"{fm(ea['est'])} [{fm(ea['lo'])}, {fm(ea['hi'])}], y5->y10 {fm(la['est'])} [{fm(la['lo'])}, "
        f"{fm(la['hi'])}] (all fields), so the rise does not come only from the y1 point.")
    L.append(
        f"2. **Not specific to integrated fields.** Integrated minus others {fm(s_c['est'])}/yr "
        f"[{fm(s_c['lo'])}, {fm(s_c['hi'])}] (field-label permutation p={s_c['p']:.2f}). Grouping by the "
        f"Scorecard baseline rho alone (>=0.45, ignoring the reliability flag; {len(INT_R)} fields) gives "
        f"{fm(cr['est'])} [{fm(cr['lo'])}, {fm(cr['hi'])}] (p={cr['p']:.2f}). Across fields, "
        f"Spearman(slope, baseline rho) = {fm(xf['est'], 2)} [{fm(xf['lo'], 2)}, {fm(xf['hi'], 2)}] "
        f"(p={xf['p']:.2f}, k={xf['k']}). Any integrated-specific part sits in y1->y5 (contrast "
        f"{fm(ec['est'])} [{fm(ec['lo'])}, {fm(ec['hi'])}]); in y5->y10 it is {fm(lc['est'])} "
        f"[{fm(lc['lo'])}, {fm(lc['hi'])}]. All of these CIs span 0.")
    L.append(
        f"3. **Career time vs calendar time.** Triple-matched institutions, all fields (k={dcal['k']}): "
        f"within-cohort (y10-y1)/9 = {fm(dwit['est'])} [{fm(dwit['lo'])}, {fm(dwit['hi'])}]; calendar-"
        f"matched (cohort c at y10 minus cohort c+9 at y1, about the same calendar years)/9 = "
        f"{fm(dcal['est'])} [{fm(dcal['lo'])}, {fm(dcal['hi'])}]; cross-cohort drift at y1 (c+9 minus c)/9 "
        f"= {fm(dcro['est'])} [{fm(dcro['lo'])}, {fm(dcro['hi'])}]. "
        + ("If the cross-cohort drift is a sum of non-negative calendar and cohort effects, the career-time "
           f"effect lies in [{fm(dcal['est'])}, {fm(dwit['est'])}]/yr for all fields and "
           f"[{fm(dcal_i['est'])}, {fm(dwit_i['est'])}]/yr for integrated fields. The calendar-matched "
           f"contrast is {dcal['est'] / dwit['est']:.0%} of the within-cohort change (all fields), and "
           + ("its CI excludes 0." if excl0(dcal) else "its CI spans 0.")
           if dcro["est"] >= 0 else
           "The cross-cohort drift is negative, so the bounds require the reverse sign assumption."))
    L.append(
        f"4. **Coverage.** Mean within-cell Spearman(prestige, coverage) = {fm(rpc['y1'])} at y1, "
        f"{fm(rpc['y5'])} at y5 and {fm(rpc['y10'])} at y10: higher-prestige institutions have a smaller "
        "share of graduates with PSEO earnings. Partial coupling given same-horizon coverage lowers the "
        f"all-field slope from {fm(R['cov_raw_all']['est'])} to {fm(R['cov_partial_all']['est'])}/yr "
        f"(share removed {cs['est']:.2f} [{cs['lo']:.2f}, {cs['hi']:.2f}]). Rows with y1 coverage >= "
        f"{thr:.3f} (panel median) give {fm(hi_['est'])} [{fm(hi_['lo'])}, {fm(hi_['hi'])}] vs "
        f"{fm(hir['est'])} for the same {hi_['k']} fields in the full panel. The two checks disagree on "
        "whether coverage matters; neither removes most of the slope.")
    L.append(
        f"5. **Field prestige F vs academia-wide rank G.** Standardized rank regression per cell, mean over "
        f"all fields: b_F {bF_o[0]:+.3f} / {bF_o[1]:+.3f} / {bF_o[2]:+.3f} and b_G {bG_o[0]:+.3f} / "
        f"{bG_o[1]:+.3f} / {bG_o[2]:+.3f} at y1 / y5 / y10; dF {fm(dF['est'])}/yr [{fm(dF['lo'])}, "
        f"{fm(dF['hi'])}], dG {fm(dG['est'])}/yr [{fm(dG['lo'])}, {fm(dG['hi'])}], dG-dF {fm(dd['est'])} "
        f"[{fm(dd['lo'])}, {fm(dd['hi'])}]. b_F > b_G in {bcount['y1'][0]}/{bcount['y1'][1]} cells at y1 "
        f"and {bcount['y10'][0]}/{bcount['y10'][1]} at y10; dG > 0 in {fcount['dG_pos']}/{fcount['k']} "
        f"fields, dF > 0 in {fcount['dF_pos']}/{fcount['k']}. Within-cell Spearman(F, G) averages "
        f"{rFG_cells.mean():.2f} (range {rFG_cells.min():.2f} to {rFG_cells.max():.2f}). Suppose F were "
        "only a noisier copy of G. Then G would take the larger coefficient at every horizon, and the "
        "b_G/b_F ratio would stay fixed as the earnings loading changed. At y1, however, F carries the "
        "association and b_G is about 0. So the data need at least two components: one tracked better "
        "by F, whose weight is flat, and one tracked better by G, whose weight grows. Two things remain "
        "unknown. First, whether the G-tracked component is reputation or brand, or another "
        "institution-wide trait correlated with the academia-wide rank (student selectivity, resources, "
        "location). Second, how much of the split comes from differential measurement error.")
    dec_txt = "; ".join(f"{LAB.get(f, f)} {fm(fl[f]['rhoF_slope'])}/yr [{fm(ci(fl[f]['rhoF_slope_b'])[0])}, "
                        f"{fm(ci(fl[f]['rhoF_slope_b'])[1])}]" for f in dec_fields) or "none"
    L.append(
        f"6. **\"Decoupled fields pinned at 0\" is not supported as a group claim.** Decoupled fields with "
        f"a fixed-cohort cell at n>=15: {dec_txt} (institution-bootstrap CIs). Communication Disorders "
        f"never reaches n>=15 (balanced n by cohort: {', '.join(f'{k}: {v}' for k, v in comm_n.items())}). "
        "Nursing alone looks flat. Two fields do not support a group-level test.")
    po, pi_, lg = R["pooled_all"], R["pooled_integrated"], R["legacy_integrated"]
    L.append(
        f"7. **Comparison, cohort-mixed or cohort-confounded (not evidence for career time):** pooled "
        f"\"0000\" balanced slope all fields {fm(po['est'])} (k={po['k']}), integrated {fm(pi_['est'])} "
        f"(k={pi_['k']}). The legacy spec (pooled, 30-field map, unbalanced) reproduces the withdrawn "
        f"integrated {fm(lg['est'])} (k={lg['k']}). Scorecard 1/4/5YR integrated "
        f"{fm(R['scorecard_integrated']['est'])} is descriptive.\n")

    L.append("## Key numbers\n")
    L.append("Slopes are change in coupling per year since graduation (or per year of the stated contrast). "
             "CI = 95% two-stage bootstrap (fields, then institutions within field x cohort), "
             f"{NBOOT} replicates. p for group slopes = institution-level horizon-label permutation "
             f"({NPERM} permutations, two-sided); p for contrasts and the cross-field Spearman = field-label "
             f"permutation ({NPERM_FIELD}). k = fields entering the statistic.\n")
    L.append("| quantity | sample / spec | estimate | 95% CI | p | k fields |")
    L.append("|---|---|---|---|---|---|")
    for key, lab in [("slope_all", "within-cohort slope, all fields"),
                     ("slope_integrated", "within-cohort slope, integrated"),
                     ("slope_middle", "within-cohort slope, middle"),
                     ("slope_decoupled", "within-cohort slope, decoupled"),
                     ("slope_unclassified", "within-cohort slope, unclassified"),
                     ("slope_int_minus_others", "integrated minus others"),
                     ("slope_intrho_minus_others", "integrated (baseline rho only) minus others"),
                     ("xfield_spearman_slope_baseline", "across fields: Spearman(slope, baseline rho)"),
                     ("early_slope_all", "segment y1->y5, all"),
                     ("late_slope_all", "segment y5->y10, all"),
                     ("early_slope_integrated", "segment y1->y5, integrated"),
                     ("late_slope_integrated", "segment y5->y10, integrated"),
                     ("early_slope_int_minus_others", "segment y1->y5, integrated minus others"),
                     ("late_slope_int_minus_others", "segment y5->y10, integrated minus others"),
                     ("D_within_all", "within-cohort (y10-y1)/9, all"),
                     ("D_cal_all", "calendar-matched (c@y10 - c+9@y1)/9, all"),
                     ("D_cross_all", "cross-cohort drift at y1 (c+9 - c)/9, all"),
                     ("D_within_integrated", "within-cohort (y10-y1)/9, integrated"),
                     ("D_cal_integrated", "calendar-matched, integrated"),
                     ("D_cross_integrated", "cross-cohort drift at y1, integrated"),
                     ("D_cal_pair_all", "calendar-matched, pair-matched institutions, all"),
                     ("D_cal_pair_integrated", "calendar-matched, pair-matched, integrated"),
                     ("cov_raw_all", "slope, coverage-complete sample, all"),
                     ("cov_partial_all", "slope of partial coupling given coverage, all"),
                     ("cov_share_removed_all", "share of slope removed by coverage control, all"),
                     ("cov_raw_integrated", "slope, coverage-complete sample, integrated"),
                     ("cov_partial_integrated", "slope of partial coupling given coverage, integrated"),
                     ("cov_share_removed_integrated", "share removed by coverage control, integrated"),
                     ("hicov_all", "slope, above-median y1 coverage, all"),
                     ("hicov_fullsample_samefields_all", "slope, full panel, same fields"),
                     ("hicov_integrated", "slope, above-median y1 coverage, integrated"),
                     ("hicov_fullsample_samefields_integrated", "slope, full panel, same fields, integrated"),
                     ("dbF_all", "dF: slope of field-prestige coef., all"),
                     ("dbG_all", "dG: slope of academia-wide-rank coef., all"),
                     ("dG_minus_dF_all", "dG - dF, all"),
                     ("dbF_integrated", "dF, integrated"),
                     ("dbG_integrated", "dG, integrated"),
                     ("dG_minus_dF_integrated", "dG - dF, integrated"),
                     ("drhoF_all", "slope of Spearman(F, earnings), all"),
                     ("drhoG_all", "slope of Spearman(G, earnings), all"),
                     ("pooled_all", "pooled 0000 slope, all (cohort-mixed)"),
                     ("pooled_integrated", "pooled 0000 slope, integrated (cohort-mixed)"),
                     ("pooled_int_minus_others", "pooled 0000 integrated minus others"),
                     ("legacy_integrated", "legacy spec slope, integrated"),
                     ("scorecard_integrated", "Scorecard 1/4/5YR slope, integrated"),
                     ("scorecard_all", "Scorecard 1/4/5YR slope, all")]:
        if key in R:
            L.append(row(R, key, lab))
    L.append("")

    L.append("## Method\n")
    L.append(
        "- **Prestige.** F = -(Wapman 2011-2020 field rank) via `load_ar_wapman(fields=FIELDS66)`; held "
        "fixed across horizons, so any change is on the earnings side. G = -(Wapman academia-wide rank) "
        "from `scripts/28 load_generic`.\n"
        "- **Earnings, fixed cohorts.** `load_er_pseo('undergrad', h, fields=FIELDS66, grad_cohort=[...])` "
        "for h in y1/y5/y10; PSEO bachelor's cohorts are 3-year graduation windows (2001 = 2001-2003, "
        "etc.). Institution-level, CIP-4 rows with released earnings, cohort-size-weighted median over "
        "the field's CIP-4 codes.\n"
        "- **Balanced panel.** Within each field x cohort keep institutions with earnings at all of y1, "
        f"y5, y10 and a field rank; cell kept if n>={NMIN}. Coupling per field x cohort x horizon; OLS "
        "slope on years {1,5,10} per field x cohort; field slope = unweighted mean over the field's "
        "cohorts; group statistic = unweighted mean over fields.\n"
        "- **Field groups.** Scorecard 4YR baseline coupling via `compute_gap_map` (FIELDS66, B=250): "
        "integrated >=0.45, decoupled <0.25, middle otherwise, among fields with `reliable_flag` and "
        f"n>={NMIN}; all other fields = unclassified (still in \"all fields\").\n"
        "- **Two-stage bootstrap.** Stage 2: institutions resampled with replacement within each field x "
        "cohort cell (an institution's three horizons move together), coupling and slope recomputed. "
        "Stage 1: fields resampled with replacement; each drawn field contributes an independently "
        "chosen stage-2 replicate. Contrasts use independent draws per group.\n"
        "- **Permutation.** Within each cell, each institution's three within-horizon earnings ranks are "
        "shuffled across the y1/y5/y10 labels (null: horizon exchangeable within institution); the "
        "statistic is recomputed over the same fields. The integrated-minus-others contrast is tested by "
        "shuffling the integrated label across fields.\n"
        "- **Calendar-matched contrast.** For c in 2001/2004/2007/2010: rho(c, y10), rho(c, y1), "
        "rho(c+9, y1) on institutions present in all three (triple-matched; also the pair-matched "
        "version c@y10 vs c+9@y1 only). c@y10 and c+9@y1 fall in approximately the same calendar years "
        "(3-year windows). Under a linear age-period-cohort reading coupling = a*age + b*year + g*cohort: "
        "within-cohort change = a+b, calendar-matched = a-g, cross-cohort drift at fixed age = b+g "
        "(exact identity on the triple-matched sample). If b>=0 and g>=0, a lies between the "
        "calendar-matched and the within-cohort estimates. The linear APC split is not identified "
        "without such an assumption.\n"
        "- **Coverage.** coverage = sum of y{h}_grads_earn / sum of y1_ipeds_count over the same CIP-4 "
        "rows used for earnings (the IPEDS count is identical across y1/y5/y10 within a fixed cohort). "
        "Partial Spearman coupling given same-horizon coverage on the sample with coverage at all three "
        "horizons (re-balanced, n>=15). High-coverage subsample: institution-cohort rows with y1 coverage "
        f"at or above the panel median ({thr:.3f}), cells re-formed with n>={NMIN}.\n"
        "- **Brand vs field.** Per cell, standardized OLS of rank(earnings) on rank(F) and rank(G) "
        "(unweighted): b_F = (r_yF - r_FG r_yG)/(1 - r_FG^2), b_G symmetric. Slopes dF, dG as above.\n"
        "- **Specificity tests.** (i) integrated minus all other fields; (ii) the same with the group "
        "defined by Scorecard 4YR rho>=0.45 and n>=15 only (no reliability flag); (iii) across fields, "
        "Spearman(fixed-cohort slope, Scorecard 4YR baseline rho), CI from the two-stage bootstrap, p from "
        "shuffling baseline rho across fields. Segment slopes (y5-y1)/4 and (y10-y5)/5 use the "
        "cohort-mean coupling per field.\n"
        "- **Pooled \"0000\"** (cohort-mixed): same pipeline with `grad_cohort='0000'`, balanced within "
        "field. **Legacy spec**: pooled 0000, 30-field default CIP map, unbalanced cells, polyfit slope "
        "over available horizons (reproduces the number in the withdrawn version).\n"
        "- **Scorecard** 1YR/4YR/5YR of one release are different cohorts; shown only as a descriptive "
        "coverage panel.\n")

    n_gt1 = int((panel[[f"cov_{h}" for h in HZ]] > 1).any(axis=1).sum())
    kc = [len(fl[f]["cohorts"]) for f in fields_all]
    kmin, kmax, k4 = min(kc), max(kc), sum(k == 4 for k in kc)
    n_cells = sum(len(fl[f]["cohorts"]) for f in fields_all)
    L.append("## Caveats\n")
    L.append(
        "- **Career time is not separately identified.** The within-cohort slope contains calendar-time "
        "change (the labour market in 2011-2013 vs 2002-2004); the calendar-matched contrast contains "
        "cohort differences instead. The bounds rest on the sign assumption stated in Method.\n"
        "- **Coverage/selection.** PSEO earnings cover graduates with sufficient earnings in covered "
        "employment; at y1 higher-prestige institutions have lower coverage (graduate school, out-of-"
        "scope employment). The partial-correlation control is linear in ranks and coverage is an "
        "aggregate share; it does not correct selection within institution. Coverage >1 (IPEDS vs PSEO "
        f"institution definitions) occurs in {n_gt1} of {len(panel)} panel rows at some horizon; rank "
        "methods limit their weight.\n"
        "- **F vs G is weakly identified.** Within-cell Spearman(F, G) averages "
        f"{rFG_cells.mean():.2f}. G pools hiring edges across all fields, so it is measured with less "
        "noise than a single field's F, and with two collinear predictors of unequal reliability OLS "
        "shifts shared signal toward the better-measured one. The y1 pattern (b_F > 0, b_G about 0) "
        "argues against reading the whole split as measurement error. Still, the regression does not "
        "say what the G-tracked component is (reputation, student selectivity, resources, location), "
        "and cell-level coefficients are noisy (n often 15-30).\n"
        "- **Group definitions and ceilings.** Integrated/middle/decoupled come from the reliability-"
        "filtered Scorecard classification, so several high-baseline fields (e.g. Mechanical and Computer "
        "Engineering, Sociology) sit in \"unclassified\". The rho-only grouping and the continuous test "
        "address this. Fields that start high at y1 have less room to rise, which works against finding "
        "an integrated-specific rise; none of the tests adjusts for that.\n"
        "- **y1 earnings.** Several fields have negative y1 coupling (Kinesiology, Music, History, "
        "Psychology; Appendix A), consistent with post-graduation enrollment at higher-prestige "
        "institutions depressing y1 medians. The coverage measure counts graduates with earnings, not "
        "enrollment, so it captures this only in part; the y5->y10 segment avoids the y1 point but "
        "still contains calendar time.\n"
        f"- **Degenerate resamples.** In small cells some institution resamples make F and G ranks "
        f"perfectly concordant, where b_F and b_G are undefined; these were redrawn "
        f"({REDRAWN.get('brand', 0)} of {n_cells * NBOOT} brand-decomposition resamples; "
        f"{REDRAWN.get('coverage_partial', 0)} coverage-partial resamples). Near-collinear resamples "
        "that are not exactly degenerate stay in and give the b_F and b_G CIs heavy tails.\n"
        "- **Sample.** PSEO partner institutions are public-skewed and miss most elite privates, so the "
        "prestige range is truncated at the top. Fields enter only where a fixed cohort has n>=15 "
        f"balanced institutions; cohorts per field vary ({kmin}-{kmax}; {k4} of {len(fields_all)} fields "
        "have all 4). Decoupled fields are too few for a group test.\n"
        "- **Medians.** Institution medians cannot show tail effects; an unchanged median ranking is "
        "compatible with widening upper tails.\n"
        "- **What public aggregates cannot resolve here.** Individual-level earnings histories linked to "
        "institution and field (not available in public data) would allow tracking the same graduates "
        "with person fixed effects, separating selection into employment from earnings, and splitting "
        "calendar from career time with more cohorts per calendar year.\n")

    # appendix: per-field table
    L.append("## Appendix A — all fields, fixed cohorts\n")
    L.append("Coupling = mean over the field's cohorts; slope CI = stage-2 (institution) bootstrap only.\n")
    L.append("| field | group | Scorecard 4YR baseline rho | cohorts | n (min-max) | y1 | y5 | y10 | slope/yr [95% CI] |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    cb = cls.set_index("field")
    order = sorted(fields_all, key=lambda f: (GROUPS.index(grp_of.get(f, "unclassified")), -fl[f]["rhoF_slope"]))
    for f in order:
        d = fl[f]; lo, hi = ci(d["rhoF_slope_b"])
        base = cb.loc[f, "baseline_rho"] if f in cb.index else np.nan
        lic = "*" if f in F.LICENSED_FIELDS else ""
        L.append(f"| {LAB.get(f, f)}{lic} | {grp_of.get(f, 'unclassified')} | {fm(base)} | "
                 f"{','.join(d['cohorts'])} | {min(d['n'])}-{max(d['n'])} | {d['rhoF_h'][0]:+.3f} | "
                 f"{d['rhoF_h'][1]:+.3f} | {d['rhoF_h'][2]:+.3f} | {d['rhoF_slope']:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    L.append("\n(*) licensed field.\n")
    L.append("## Appendix B — group trajectories and per-cohort slopes\n")
    L.append("| group | k | y1 | y5 | y10 |")
    L.append("|---|---|---|---|---|")
    for g in ["all"] + GROUPS:
        if g in traj:
            o, lo, hi, k = traj[g]
            L.append(f"| {g} | {k} | " + " | ".join(f"{o[i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]" for i in range(3)) + " |")
    L.append("\n| cohort | group | k fields | mean slope/yr | fields with slope > 0 |")
    L.append("|---|---|---|---|---|")
    for _, r in by_coh.iterrows():
        L.append(f"| {r['cohort']} | {r['group']} | {r['k']} | {fm(r['mean'], 4)} | {r['n_pos']}/{r['k']} |")
    L.append("\n## Appendix C — field prestige F vs academia-wide rank G by horizon (mean over fields, two-stage 95% CI)\n")
    L.append("| sample | coef | y1 | y5 | y10 |")
    L.append("|---|---|---|---|---|")
    for (name, key), (o, lo, hi) in bfh.items():
        L.append(f"| {name} | {key} | " + " | ".join(f"{o[i]:+.3f} [{lo[i]:+.3f}, {hi[i]:+.3f}]" for i in range(3)) + " |")
    L.append("\n## Appendix D — calendar-matched contrast by cohort pair (mean over fields, observed)\n")
    L.append("| pair | k fields (triple, n>=15) | within (y10-y1)/9 | calendar-matched /9 | cross-cohort /9 |")
    L.append("|---|---|---|---|---|")
    for c in COHORTS:
        x = cal_df[(cal_df.cohort == c) & cal_df["D_cal"].notna()] if "D_cal" in cal_df else cal_df.iloc[0:0]
        if len(x):
            L.append(f"| {c}y10 vs {LATE[c]}y1 | {len(x)} | {x.D_within.mean():+.4f} | {x.D_cal.mean():+.4f} | "
                     f"{x.D_cross.mean():+.4f} |")
    L.append("\n## Appendix E — Scorecard 1/4/5YR (descriptive, cohort-confounded)\n")
    L.append("| group | 1YR | 4YR | 5YR |")
    L.append("|---|---|---|---|")
    scx = sc_long[sc_long.n >= NMIN].assign(group=lambda d: d.field.map(lambda f: grp_of.get(f, "unclassified")))
    for g in GROUPS:
        s = scx[scx.group == g].groupby("horizon").coupling.agg(["mean", "count"])
        if len(s):
            L.append(f"| {g} | " + " | ".join(
                f"{s.loc[h, 'mean']:+.3f} (k={int(s.loc[h, 'count'])})" if h in s.index else "—"
                for h in SC_HORIZONS) + " |")
    L.append("")
    OUT_MD.write_text("\n".join(L))
    print(f"[result] {OUT_MD}")


if __name__ == "__main__":
    main()
