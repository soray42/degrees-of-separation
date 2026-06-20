"""Tier-0 robustness sweep: does the verdict survive reasonable specification changes?

Varies the Employer-Reputation earnings measure (credential level x horizon) — the
proposal's #1 caveat is the undergraduate-earnings vs PhD-prestige level mismatch —
and re-evaluates the three GO conditions + the gap~industry-share mechanism each time.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import tier0 as T
from src.crosswalks import fields as F

prestige = T.load_prestige()
proxy = T.industry_share_table()

SPECS = [
    ("Bachelor 4yr", "3", "EARN_MDN_4YR"),
    ("Bachelor 1yr", "3", "EARN_MDN_1YR"),
    ("Master 4yr",   "5", "EARN_MDN_4YR"),
    ("Master 1yr",   "5", "EARN_MDN_1YR"),
    ("Doctoral 4yr", "6", "EARN_MDN_4YR"),
    ("Doctoral 1yr", "6", "EARN_MDN_1YR"),
]

# coverage check: how many fields clear min_n at each credential level?
print("=== ER earnings coverage by credential level (matched institutions per field) ===")
for name, cred, col in SPECS:
    earn = T.load_earnings(credlev=cred, earn_col=col)
    gaps = T.all_field_gaps(prestige, earn, min_n=8)
    ok = gaps["ok"].sum()
    med_matched = gaps["n_matched"].median()
    print(f"  {name:<14} fields_usable={ok:2d}/20  median_matched_institutions={med_matched:.0f}")

print("\n=== three conditions + mechanism per spec ===")
print(f"{'spec':<14}{'nF':>4}{'gapSD':>7}{'gapMed':>8}{'CSgap':>7}{'CSrank':>8}"
      f"{'rho_all':>9}{'rho_SDR':>9}{'verdict':>9}")
rows = []
for name, cred, col in SPECS:
    earn = T.load_earnings(credlev=cred, earn_col=col)
    gaps = T.all_field_gaps(prestige, earn, min_n=8)
    res = T.evaluate_go(gaps, proxy)
    cs_rank, n = res["cs_rank_of_n"]
    print(f"{name:<14}{res['n_fields']:>4}{res['gap_sd']:>7.3f}{res['gap_median']:>8.3f}"
          f"{res['cs_gap']:>7.3f}{f'{cs_rank}/{n}':>8}"
          f"{res['spearman_gap_vs_industry_all']:>9.3f}"
          f"{res['spearman_gap_vs_industry_sdr_only']:>9.3f}{res['verdict']:>9}")
    rows.append(dict(spec=name, **{k: v for k, v in res.items() if k != "merged"}))

pd.DataFrame(rows).drop(columns=["cs_rank_of_n"], errors="ignore").to_csv(
    "results/tables/tier0_robustness.csv", index=False)
print("\nwrote results/tables/tier0_robustness.csv")
