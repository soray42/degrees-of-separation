"""Non-truncated field-level earnings-dispersion proxies for the P1 re-test (Task 1).

The prior P1 used PSEO within-institution dispersion, measured on PSEO's public-skewed,
elite-private-truncated institution sample. These proxies are NOT truncated:

  * ACS PUMS (full US population incl. private-institution graduates, no suppression):
    weighted within-field cross-individual earnings spread by field-of-bachelor's (FOD1P).
  * PSEO STATE-level: pooling institutions within a state dissolves the institution-level
    small-cell suppression. (Bachelor's survives; master's/doctoral are still suppressed
    even at state level — a finding, not a usable proxy.)

Dispersion = (p75 − p25)/p50  and  CV = SD/mean, per field. Higher = employers price
individual productivity more (P1's observability proxy).
"""
from __future__ import annotations

from pathlib import Path
import glob
import numpy as np
import pandas as pd

from .crosswalks import fields as F

ROOT = Path(__file__).resolve().parents[1]
ACS_GLOB = str(ROOT / "data" / "raw" / "acs" / "psam_pus*.csv")
ACS_CACHE = ROOT / "data" / "interim" / "acs_workers.parquet"
PSEO_EARN = ROOT / "data" / "raw" / "pseo" / "pseoe_all.csv.gz"


def weighted_quantile(values, weights, q):
    v = np.asarray(values, float); w = np.asarray(weights, float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[ok], w[ok]
    if v.size == 0:
        return np.nan
    order = np.argsort(v); v, w = v[order], w[order]
    cw = np.cumsum(w) - 0.5 * w
    cw /= w.sum()
    return float(np.interp(q, cw, v))


# ---------------------------------------------------------------------------
# ACS PUMS
# ---------------------------------------------------------------------------
def load_acs_workers(earn_col: str = "PERNP", refresh: bool = False) -> pd.DataFrame:
    """Bachelor's-or-higher, full-time-employed persons with a field of degree, mapped
    to the 30 project fields. Cached to parquet after first build."""
    if ACS_CACHE.exists() and not refresh:
        return pd.read_parquet(ACS_CACHE)
    need = ["FOD1P", "PERNP", "WAGP", "SCHL", "AGEP", "WKHP", "ESR", "PWGTP"]
    frames = []
    for f in sorted(glob.glob(ACS_GLOB)):
        cols = pd.read_csv(f, nrows=0).columns
        frames.append(pd.read_csv(f, usecols=[c for c in need if c in cols], dtype="float64"))
    acs = pd.concat(frames, ignore_index=True)
    acs = acs[(acs.SCHL >= 21) & acs.FOD1P.notna() & (acs.WKHP >= 35) &
              acs.ESR.isin([1, 2]) & (acs[earn_col] > 0)].copy()
    acs["fod1p"] = acs.FOD1P.astype(int).astype(str).str.zfill(4)
    acs["field"] = acs.fod1p.map(F.fod1p_to_field_key())
    acs = acs[acs.field.notna()].copy()
    ACS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    acs.to_parquet(ACS_CACHE)
    return acs


def field_dispersion_acs(workers: pd.DataFrame, age_lo: int, age_hi: int,
                         earn_col: str = "PERNP", min_n: int = 50) -> pd.DataFrame:
    """Weighted within-field dispersion over an age band. Returns field, disp_acs (IQR/p50),
    cv_acs, p50, n. `disp_acs` column name encodes the band via the caller."""
    w = workers[(workers.AGEP >= age_lo) & (workers.AGEP <= age_hi)]
    rows = []
    for fld, g in w.groupby("field"):
        if len(g) < min_n:
            rows.append(dict(field=fld, p50=np.nan, disp_acs=np.nan, cv_acs=np.nan, n=len(g)))
            continue
        p25 = weighted_quantile(g[earn_col], g.PWGTP, 0.25)
        p50 = weighted_quantile(g[earn_col], g.PWGTP, 0.50)
        p75 = weighted_quantile(g[earn_col], g.PWGTP, 0.75)
        mean = np.average(g[earn_col], weights=g.PWGTP)
        var = np.average((g[earn_col] - mean) ** 2, weights=g.PWGTP)
        rows.append(dict(field=fld, p50=p50, disp_acs=(p75 - p25) / p50,
                         cv_acs=np.sqrt(var) / mean, n=len(g)))
    return pd.DataFrame(rows)


def field_grad_share_acs(workers: pd.DataFrame, age_lo: int = 25, age_hi: int = 64) -> pd.DataFrame:
    """Ex-ante pipeline intensity from ACS (external, NOT the gap): weighted share of a
    field's bachelor's-holders who hold a graduate degree (SCHL >= 22). Covers all 30 fields
    (NY Fed misses 6); NY Fed's share-with-grad-degree is the cross-check."""
    w = workers[(workers.AGEP >= age_lo) & (workers.AGEP <= age_hi)]
    rows = []
    for fld, g in w.groupby("field"):
        tot = g.PWGTP.sum()
        grad = g.loc[g.SCHL >= 22, "PWGTP"].sum()
        rows.append(dict(field=fld, grad_share_acs=grad / tot if tot else np.nan, n=len(g)))
    return pd.DataFrame(rows)


def acs_earnings_by_attainment(workers: pd.DataFrame, age_lo: int = 25, age_hi: int = 64,
                               earn_col: str = "PERNP") -> pd.DataFrame:
    """Bonus: field-level weighted median earnings by attainment of the BA-field holders —
    BA (SCHL 21), Master's (22), Doctorate (24). A field-level glance at PhD-holder earnings
    by undergraduate field (the institution-level PhD-ER that PSEO suppressed)."""
    w = workers[(workers.AGEP >= age_lo) & (workers.AGEP <= age_hi)]
    out = []
    for fld, g in w.groupby("field"):
        rec = dict(field=fld)
        for lab, schl in [("ba", [21]), ("ma", [22]), ("phd", [24])]:
            s = g[g.SCHL.isin(schl)]
            rec[f"med_{lab}"] = weighted_quantile(s[earn_col], s.PWGTP, 0.50) if len(s) >= 30 else np.nan
            rec[f"n_{lab}"] = len(s)
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# PSEO state-level
# ---------------------------------------------------------------------------
def field_dispersion_pseo_state(level: str = "undergrad", horizon: str = "y5") -> pd.DataFrame:
    """State-level (inst_level='S') PSEO dispersion, aggregated to national field.
    Pooling institutions within a state avoids institution-level cell suppression."""
    dl = {"undergrad": "05", "masters": "07", "phd": "17"}[level]
    p25, p50, p75 = f"{horizon}_p25_earnings", f"{horizon}_p50_earnings", f"{horizon}_p75_earnings"
    grads, status = f"{horizon}_grads_earn", f"status_{horizon}_earnings"
    df = pd.read_csv(PSEO_EARN, dtype=str, usecols=["inst_level", "institution", "degree_level",
                    "cip_level", "cipcode", "grad_cohort", p25, p50, p75, grads, status])
    df = df[(df.inst_level == "S") & (df.cip_level == "4") & (df.grad_cohort == "0000") &
            (df.degree_level == dl) & (df[status] == "1")].copy()
    if df.empty:
        return pd.DataFrame(columns=["field", "disp_pseo_state", "n_states"])
    df["cip4"] = df.cipcode.str.replace(".", "", regex=False).str.zfill(4)
    df["field"] = df.cip4.map(F.cip4_to_field_key())
    df = df[df.field.notna()].copy()
    for c in (p25, p50, p75, grads):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[p50])
    df["disp"] = (df[p75] - df[p25]) / df[p50]
    df["w"] = df[grads].fillna(1).clip(lower=1)
    df["wd"] = df["disp"] * df["w"]
    g = df.groupby("field").agg(wd=("wd", "sum"), w=("w", "sum"),
                                n_states=("disp", "size")).reset_index()
    g["disp_pseo_state"] = g.wd / g.w
    return g[["field", "disp_pseo_state", "n_states"]]
