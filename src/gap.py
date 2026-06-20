"""Generic gap machinery — the Tier-1 hook. KEEP SOURCE-AGNOSTIC.

`compute_gap(ar_table, er_table, field, level, period)` takes ANY normalized AR table
(src/load_ar.py schema) and ANY normalized ER table (src/load_er.py schema) and returns one
tidy reliability-annotated record. No Wapman- or Scorecard-specific logic lives here, so
Tier 1 calls the exact same function with an ORCID-rebuilt AR table and a new `period` label,
and the two-period out-of-sample test (proposal v2 §7) is just a groupby/append.

    gap = 1 − spearman(prestige_score, earnings)   # within field, across matched institutions

Returned record:
    {field, level, period, n_institutions, gap, spearman, signal_frac, ci_lo, ci_hi,
     synth_lo, synth_hi, reliable_flag}

Reliability (proposal v2 §5.4): a field is `reliable` iff
  (i) signal_frac ≥ `signal_threshold` (default 0.5) — earnings carry real cross-institution
      signal beyond cohort sampling noise (null model), AND
  (ii) the observed gap's bootstrap CI does NOT overlap the null-model noise band — the gap
      is distinguishable from what pure noise under perfect agreement would produce.
Fields with n < `min_n` (default 10) are computed but flagged `reliable_flag = False`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .diagnostic import null_model_field, bootstrap_field  # generic array-level helpers


def compute_gap(ar_table: pd.DataFrame, er_table: pd.DataFrame, field: str, level: str,
                period: str = "2011-2020", *, alpha: float = 0.6, B: int = 1000,
                min_n: int = 10, signal_threshold: float = 0.5, seed: int = 1234) -> dict:
    a = ar_table[ar_table["field"] == field][["inst_key", "prestige_score"]]
    e = er_table[(er_table["field"] == field) & (er_table["level"] == level)][
        ["inst_key", "earnings", "cohort_n"]]
    m = a.merge(e, on="inst_key", how="inner").dropna(subset=["prestige_score", "earnings"])
    n = len(m)
    rec = dict(field=field, level=level, period=period, n_institutions=n,
               gap=np.nan, spearman=np.nan, signal_frac=np.nan,
               ci_lo=np.nan, ci_hi=np.nan, synth_lo=np.nan, synth_hi=np.nan,
               reliable_flag=False)
    if n < 4:
        return rec

    prest = m["prestige_score"].to_numpy(float)
    earn = m["earnings"].to_numpy(float)
    rho, _ = spearmanr(prest, earn)
    rec["spearman"] = float(rho)
    rec["gap"] = float(1.0 - rho)

    # cohort sizes for the null model; fill gaps with the field median
    cohort = m["cohort_n"].to_numpy(float)
    if np.isnan(cohort).any():
        med = np.nanmedian(cohort) if np.isfinite(np.nanmedian(cohort)) else 30.0
        cohort = np.where(np.isnan(cohort), med, cohort)
    cohort = np.clip(cohort, 5.0, None)

    rng = np.random.default_rng(seed)
    nm = null_model_field(prest, earn, cohort, alpha=alpha, B=B, rng=rng)
    bs = bootstrap_field(prest, earn, B=B, rng=rng)
    rec["signal_frac"] = float(nm["signal_frac"])
    rec["synth_lo"], rec["synth_hi"] = float(nm["synth_gap_lo"]), float(nm["synth_gap_hi"])
    # bootstrap CI on the GAP (= 1 − spearman); flip the spearman CI
    rec["ci_lo"], rec["ci_hi"] = float(1 - bs["boot_hi"]), float(1 - bs["boot_lo"])

    overlaps_noise = not (rec["ci_hi"] < rec["synth_lo"] or rec["ci_lo"] > rec["synth_hi"])
    rec["reliable_flag"] = bool(
        n >= min_n and rec["signal_frac"] >= signal_threshold and not overlaps_noise)
    return rec


def compute_gap_map(ar_table: pd.DataFrame, er_table: pd.DataFrame, level: str,
                    period: str = "2011-2020", **kw) -> pd.DataFrame:
    """Run `compute_gap` for every field present in both tables at `level`."""
    fields = sorted(set(ar_table["field"]) &
                    set(er_table.loc[er_table["level"] == level, "field"]))
    return pd.DataFrame([compute_gap(ar_table, er_table, f, level, period, **kw) for f in fields])
