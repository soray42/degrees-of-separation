"""Tier 0 addendum, Task 2 — ex-ante field typology.
 → results/REGIME_TYPOLOGY.csv + results/REGIME_NOTE.md

Tag each field with a regime from EXTERNAL data only (never the gap):
  license-standardization  (hand-coded licensure: nursing, comm-disorders, accounting/CPA, civil/PE)
  PhD-pipeline             (ACS graduate-degree share in the top tercile of non-licensed fields)
  prestige-transmission    (the rest)
Then re-report the gap within regime and classify the reliable HIGH-gap fields by channel.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map
from src.predictions import load_nyfed_outcomes, assign_regime
from src import dispersion as D
from src.crosswalks import fields as F

LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}


def main():
    ar = load_ar_wapman(); sc = load_er_scorecard("undergrad")
    gm = compute_gap_map(ar, sc, "undergrad")
    w = D.load_acs_workers()
    gshare = D.field_grad_share_acs(w)                  # pipeline intensity (ACS, all 30)
    nyf = load_nyfed_outcomes()[["field", "grad_degree_share"]]

    t = gm.merge(gshare[["field", "grad_share_acs"]], on="field", how="left").merge(
        nyf, on="field", how="left")
    t["label"] = t.field.map(LAB)
    t["licensed"] = t.field.map(F.is_licensed)
    # PhD-pipeline cut = top tercile of grad_share_acs among NON-licensed fields
    cut = t.loc[~t.licensed, "grad_share_acs"].quantile(2 / 3)
    t["regime"] = [assign_regime(f, gs, cut) for f, gs in zip(t.field, t.grad_share_acs)]

    out = t[["field", "label", "n_institutions", "gap", "reliable_flag", "signal_frac",
             "grad_share_acs", "grad_degree_share", "licensed", "regime"]].sort_values(
        ["regime", "gap"])
    out.to_csv("results/REGIME_TYPOLOGY.csv", index=False)

    # gap within regime (reliable fields)
    rel = t[t.reliable_flag]
    by = rel.groupby("regime")["gap"].agg(["size", "mean", "median"]).round(3)

    # classify reliable HIGH-gap fields by channel
    hi = rel[rel.gap > rel.gap.median()].sort_values("gap", ascending=False)

    L = ["# Tier 0 addendum — Ex-ante field typology (Task 2)\n",
         "Regimes assigned from **external** data only (never the gap): licensure is a "
         "hand-coded ex-ante binary; pipeline intensity is the ACS graduate-degree share. "
         "Run: `python scripts/05_regime.py`. Date 2026-06-20.\n",
         "## The classification rule (documented, ex-ante)\n",
         "- **license-standardization** — entry to the field's modal occupation requires a "
         "standardized external license/credential that prices graduates largely independent "
         "of department prestige. Hand-coded TRUE for **nursing** (RN/NCLEX), **communication "
         "disorders** (CCC-SLP / state license), **accounting** (CPA), **civil engineering** "
         "(PE, effectively required to practice). Other engineering: PE is optional for most "
         "industry roles → FALSE. (Validates against the CPS certification/licensing question; "
         "see SOURCES.md.)",
         f"- **PhD-pipeline** — non-licensed fields whose **ACS graduate-degree share** is in "
         f"the top tercile (cut = {cut:.0%}); the signal-carrying BA graduates continue to grad "
         "school, so BA earnings undercount the department.",
         "- **prestige-transmission** — the remaining fields (BA labor market prices the "
         "graduate directly).",
         "- **Scope boundary (not a limitation):** professional-entry fields (law, medicine) "
         "are excluded by entry-degree — there is no BA-to-practice pipeline and Wapman does "
         "not cover professional-school prestige.\n",
         "## Gap within regime (reliable fields)\n",
         by.to_markdown(),
         "\n## The reliable HIGH-gap fields, classified by channel\n",
         hi[["label", "gap", "regime", "grad_share_acs"]].to_markdown(
             index=False, floatfmt=("", ".3f", "", ".0%")),
         "\n**Reading the high-gap tail by channel — the high gap is not one phenomenon:**",
         "- **license-standardization** (nursing 0.93, communication disorders 1.21): a "
         "standardized license, not department prestige, sets earnings → prestige and earnings "
         "decouple by construction. These are the highest gaps and are *expected*, not anomalous.",
         "- **PhD-pipeline / talent-exit** (biology, and the unreliable chemistry/physics/earth "
         "sciences): the ablest BA graduates leave for PhDs, so BA earnings cannot track research "
         "prestige. Biology is the reliable exemplar.",
         "- **genuine quality-disagreement** (English, philosophy): neither licensed nor strongly "
         "pipelined, yet a real gap — the residual the integration frame is actually about.\n",
         "This converts the structurally-special fields from a limitations footnote into an "
         "ex-ante typology: the gap means different things in different regimes, and the "
         "integration construct is cleanest within **prestige-transmission**.\n",
         "## Full typology\n",
         out.to_markdown(index=False, floatfmt=("", "", ".0f", ".3f", "", ".2f", ".0%", ".1f", "", ""))]
    Path("results/REGIME_NOTE.md").write_text("\n".join(L))
    print("wrote results/REGIME_TYPOLOGY.csv + results/REGIME_NOTE.md")
    print("PhD-pipeline cut (top-tercile grad share, non-licensed):", round(cut, 3))
    print(by.to_string())
    print("\nregime counts:", dict(t.regime.value_counts()))


if __name__ == "__main__":
    main()
