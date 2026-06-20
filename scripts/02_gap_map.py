"""Tier 0 — the reliability-gated gap map.  → results/GAP_MAP.csv + figures

$A$ = Wapman; $E$ = Scorecard (undergrad). Generic `compute_gap` per field with the full
reliability framework. Confirms the established anchors (CS low+reliable, biology
high+reliable). PhD-level map is omitted: PSEO PhD cells are suppressed at the field grain
(see COVERAGE_REPORT.md).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map
from src.crosswalks import fields as F

LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}


def main():
    ar = load_ar_wapman()
    er = load_er_scorecard("undergrad")
    gm = compute_gap_map(ar, er, "undergrad", period="2011-2020")
    gm["label"] = gm["field"].map(LAB)
    gm = gm.sort_values("gap").reset_index(drop=True)
    cols = ["field", "label", "level", "period", "n_institutions", "gap", "spearman",
            "signal_frac", "ci_lo", "ci_hi", "synth_lo", "synth_hi", "reliable_flag"]
    gm[cols].to_csv("results/GAP_MAP.csv", index=False)

    rel = gm[gm.reliable_flag]
    print(f"gap map: {len(gm)} fields, {len(rel)} reliable "
          f"(signal_frac>=0.5 & bootstrap CI excludes noise band)")
    # anchors
    def show(k):
        r = gm[gm.field == k]
        return None if r.empty else r.iloc[0]
    cs, bio = show("computer_science"), show("biology")
    anchors_ok = (cs is not None and cs.gap < gm.gap.median() and cs.reliable_flag and
                  bio is not None and bio.gap > gm.gap.median() and bio.reliable_flag)
    print(f"ANCHORS: CS gap={cs.gap:.3f} reliable={cs.reliable_flag}; "
          f"biology gap={bio.gap:.3f} reliable={bio.reliable_flag} -> "
          f"{'reproduce' if anchors_ok else 'FAIL — investigate'}")

    # ---- figure 1: gaps sorted with CI, reliable colored ----
    fig, ax = plt.subplots(figsize=(9, 9))
    y = np.arange(len(gm))
    for i, r in gm.iterrows():
        c = "#2C7BB6" if r.reliable_flag else "#cccccc"
        ax.errorbar(r.gap, i, xerr=[[max(r.gap - r.ci_lo, 0)], [max(r.ci_hi - r.gap, 0)]],
                    fmt="o", color=c, ecolor=c, capsize=2,
                    markersize=9 if r.field in ("computer_science", "biology") else 6,
                    markeredgecolor="black" if r.field in ("computer_science", "biology") else c)
    ax.set_yticks(y); ax.set_yticklabels(gm.label, fontsize=8)
    ax.axvline(gm.gap.median(), color="#bbb", ls=":", lw=.8)
    ax.set_xlabel("Gap = 1 − Spearman(prestige, earnings)  [undergrad, Scorecard]")
    ax.set_title("Tier 0 gap map (CI whiskers; blue=reliable, grey=unreliable)\n"
                 "CS (low) and Biology (high) anchors outlined")
    fig.tight_layout(); fig.savefig("results/figures/gap_map.png", dpi=150)

    # ---- figure 2: signal fraction per field ----
    g2 = gm.sort_values("signal_frac")
    fig2, ax2 = plt.subplots(figsize=(8.5, 9))
    colors = ["#2C7BB6" if x else "#cccccc" for x in g2.reliable_flag]
    ax2.barh(np.arange(len(g2)), g2.signal_frac, color=colors)
    ax2.axvline(0.5, color="#D6202A", ls="--", lw=1, label="reliability threshold 0.5")
    ax2.set_yticks(np.arange(len(g2))); ax2.set_yticklabels(g2.label, fontsize=8)
    ax2.set_xlabel("signal_frac (earnings signal vs cohort sampling noise)")
    ax2.set_title("Per-field reliability: signal fraction (null model)")
    ax2.legend(); fig2.tight_layout(); fig2.savefig("results/figures/gap_signalfrac.png", dpi=150)

    print("wrote results/GAP_MAP.csv, figures/gap_map.png, figures/gap_signalfrac.png")
    return gm


if __name__ == "__main__":
    main()
