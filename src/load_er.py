"""Employer-reputation loaders → a SOURCE-NORMALIZED tidy ER table.

Common schema (so `compute_gap` is source-agnostic):

    inst_key | institution_id | institution_name | field | level | earnings | dispersion | cohort_n

- `level` ∈ {"undergrad","masters","phd"}.
- `earnings` is the median (p50) for the field at the institution (cohort-weighted over the
  field's CIP-4 codes).
- `dispersion` is within-institution cross-individual spread `(p75-p25)/p50` — the P1
  observability proxy. Only PSEO provides it; Scorecard rows get NaN.
- `cohort_n` is the earnings-cohort size (for the null-model sampling-noise calibration).

Two sources, very different coverage (probed in Tier 0):
  * Scorecard FoS — near-universal (IRS-linked), spans elite privates; NO within-inst spread.
  * PSEO (LEHD)   — ~985 partner institutions, public-skewed, misses the elite-private top;
                    HAS percentiles; but **PhD/master cells are disclosure-suppressed at the
                    4-digit-CIP grain** (0 institutions) → PSEO usable at undergrad only.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .crosswalks import fields as F
from .crosswalks.institutions import normalize_institution_name

ROOT = Path(__file__).resolve().parents[1]
SCORECARD_FOS = ROOT / "data" / "raw" / "scorecard_fos" / "Most-Recent-Cohorts-Field-of-Study.csv"
PSEO_EARN = ROOT / "data" / "raw" / "pseo" / "pseoe_all.csv.gz"
PSEO_INST = ROOT / "data" / "raw" / "pseo" / "pseo_all_institutions.csv"

SCORECARD_CREDLEV = {"undergrad": "3", "masters": "5", "phd": "6"}
PSEO_DEGREE_LEVEL = {"undergrad": "05", "masters": "07", "phd": "17"}


# ---------------------------------------------------------------------------
# Scorecard Field-of-Study (undergraduate ER, broad coverage)
# ---------------------------------------------------------------------------
def load_er_scorecard(level: str = "undergrad", earn_col: str = "EARN_MDN_4YR",
                      count_col: str = "EARN_COUNT_WNE_4YR", fields=None) -> pd.DataFrame:
    credlev = SCORECARD_CREDLEV[level]
    use = ["UNITID", "INSTNM", "CIPCODE", "CREDLEV", earn_col, count_col]
    df = pd.read_csv(SCORECARD_FOS, usecols=use, dtype=str)
    df = df[df["CREDLEV"] == credlev].copy()
    df["cip4"] = df["CIPCODE"].astype(str).str.zfill(4)
    df["field"] = df["cip4"].map(F.cip4_to_field_key(fields))
    df = df[df["field"].notna()].copy()
    df["earn"] = pd.to_numeric(df[earn_col], errors="coerce")
    df["cohort"] = pd.to_numeric(df[count_col], errors="coerce")
    df = df.dropna(subset=["earn"])
    df["inst_key"] = df["INSTNM"].map(normalize_institution_name)
    df = df[df["inst_key"] != ""]
    df["w"] = df["cohort"].fillna(1.0).clip(lower=1.0)
    df["we"] = df["earn"] * df["w"]
    g = df.groupby(["field", "inst_key"]).agg(
        institution_id=("UNITID", "first"), institution_name=("INSTNM", "first"),
        we=("we", "sum"), wsum=("w", "sum"), cohort_n=("cohort", "sum")).reset_index()
    g["earnings"] = g["we"] / g["wsum"]
    g["dispersion"] = np.nan                          # Scorecard has no within-inst spread
    g["level"] = level
    return g[["inst_key", "institution_id", "institution_name", "field", "level",
              "earnings", "dispersion", "cohort_n"]]


# ---------------------------------------------------------------------------
# PSEO (degree-level ER + within-institution dispersion)
# ---------------------------------------------------------------------------
def _pseo_institutions() -> pd.DataFrame:
    ins = pd.read_csv(PSEO_INST, dtype=str)
    ins.columns = [c.strip().lstrip("﻿") for c in ins.columns]
    ins["inst_key"] = ins["label"].map(normalize_institution_name)
    return ins[["institution", "label", "inst_key"]]


def load_er_pseo(level: str = "undergrad", horizon: str = "y5") -> pd.DataFrame:
    """PSEO earnings + dispersion for a degree level. Uses pooled cohorts
    (grad_cohort=='0000'), institution-level rows (inst_level=='I'), 4-digit CIP
    (cip_level=='4'), and only released earnings (status_y{h}_earnings=='1')."""
    dl = PSEO_DEGREE_LEVEL[level]
    p25, p50, p75 = f"{horizon}_p25_earnings", f"{horizon}_p50_earnings", f"{horizon}_p75_earnings"
    grads, status = f"{horizon}_grads_earn", f"status_{horizon}_earnings"
    use = ["inst_level", "institution", "degree_level", "cip_level", "cipcode",
           "grad_cohort", p25, p50, p75, grads, status]
    df = pd.read_csv(PSEO_EARN, dtype=str, usecols=use)
    df = df[(df["inst_level"] == "I") & (df["cip_level"] == "4") &
            (df["grad_cohort"] == "0000") & (df["degree_level"] == dl) &
            (df[status] == "1")].copy()
    df["cip4"] = df["cipcode"].astype(str).str.replace(".", "", regex=False).str.zfill(4)
    df["field"] = df["cip4"].map(F.cip4_to_field_key())
    df = df[df["field"].notna()].copy()
    for c in (p25, p50, p75, grads):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[p50])
    df["disp"] = (df[p75] - df[p25]) / df[p50]
    df["w"] = df[grads].fillna(1.0).clip(lower=1.0)
    inst = _pseo_institutions().drop_duplicates("institution")
    df = df.merge(inst, on="institution", how="left")
    df = df[df["inst_key"].notna() & (df["inst_key"] != "")]
    df["we"] = df[p50] * df["w"]
    df["wd"] = df["disp"] * df["w"]
    g = df.groupby(["field", "inst_key"]).agg(
        institution_id=("institution", "first"), institution_name=("label", "first"),
        we=("we", "sum"), wd=("wd", "sum"), wsum=("w", "sum"),
        cohort_n=(grads, "sum")).reset_index()
    g["earnings"] = g["we"] / g["wsum"]
    g["dispersion"] = g["wd"] / g["wsum"]
    g["level"] = f"{level}_pseo"
    return g[["inst_key", "institution_id", "institution_name", "field", "level",
              "earnings", "dispersion", "cohort_n"]]
