"""Run the Tier-0 go/no-go analysis end to end and write outputs.

Outputs:
  results/tables/tier0_field_gaps.csv     per-field gap + proxy table
  results/figures/tier0_gap_vs_industry.png   labeled scatter (CS highlighted)
Prints the GO/NO-GO verdict to stdout.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import tier0 as T
from src.crosswalks import fields as F

OUT_TBL = Path("results/tables/tier0_field_gaps.csv")
OUT_FIG = Path("results/figures/tier0_gap_vs_industry.png")


def main():
    prestige = T.load_prestige()
    earnings = T.load_earnings()
    gaps = T.all_field_gaps(prestige, earnings)
    proxy = T.industry_share_table()

    tbl = gaps.merge(proxy, on="field_key")
    OUT_TBL.parent.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUT_TBL, index=False)

    print("=== Per-field gaps (sorted low->high) ===")
    show = tbl[["label", "domain", "n_matched", "rho", "gap", "industry_share", "proxy_source"]]
    with pd.option_context("display.width", 160, "display.max_rows", None):
        print(show.to_string(index=False,
              formatters={"rho": "{:.3f}".format, "gap": "{:.3f}".format,
                          "industry_share": "{:.1f}".format}))

    res = T.evaluate_go(gaps, proxy)
    print("\n=== GO / NO-GO ===")
    for k, v in res.items():
        if k == "merged":
            continue
        print(f"  {k}: {v}")

    # ---- figure: gap vs industry share; SDR vs SED-humanities distinguished, CS highlighted ----
    m = res["merged"]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    is_cs = m["field_key"] == F.HEADLINE_FIELD
    is_hum = m["proxy_source"] != "SDR fine"
    # SDR fine fields (consistently measured)
    base = m[~is_cs & ~is_hum]
    ax.scatter(base["industry_share"], base["gap"], s=46, c="#4878CF",
               alpha=0.85, edgecolor="white", linewidth=0.6, label="SDR fine field")
    # humanities (SED-fallback proxy, different universe) — hollow, flagged
    hum = m[is_hum]
    ax.scatter(hum["industry_share"], hum["gap"], s=60, facecolors="none",
               edgecolors="#999999", linewidth=1.3, label="Humanities (SED proxy, diff. universe)")
    # CS
    ax.scatter(m.loc[is_cs, "industry_share"], m.loc[is_cs, "gap"], s=160, c="#D6202A",
               edgecolor="black", linewidth=1.0, zorder=5, label="Computer science")
    for _, row in m.iterrows():
        ax.annotate(row["label"], (row["industry_share"], row["gap"]),
                    fontsize=7.3, xytext=(4, 3), textcoords="offset points",
                    color=("#D6202A" if row["field_key"] == F.HEADLINE_FIELD else "#333333"))
    # OLS fit on consistently-measured SDR fields only
    msdr = m[~is_hum]
    b1, b0 = np.polyfit(msdr["industry_share"], msdr["gap"], 1)
    xs = np.linspace(m["industry_share"].min(), m["industry_share"].max(), 50)
    ax.plot(xs, b0 + b1 * xs, "--", color="#888888", linewidth=1.2,
            label=f"OLS (SDR fields) slope={b1:.4f}")
    ax.axhline(res["gap_median"], color="#cccccc", linewidth=0.8, linestyle=":")
    ax.annotate("median gap", (xs[0], res["gap_median"]), fontsize=7, color="#999999",
                xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Industry employment share (%)  — integration proxy (SDR 2021 fine field; SED for humanities)")
    ax.set_ylabel(r"Gap$_{field}$ = 1 − Spearman(prestige rank, earnings rank)")
    ax.set_title("Tier 0 — Academic–Employer reputation gap vs. industry integration\n"
                 f"Spearman(gap, industry): full={res['spearman_gap_vs_industry_all']:.2f}, "
                 f"SDR-only={res['spearman_gap_vs_industry_sdr_only']:.2f}  |  "
                 f"CS gap={res['cs_gap']:.2f} (rank {res['cs_rank_of_n'][0]}/{res['cs_rank_of_n'][1]})  |  "
                 f"verdict: {res['verdict']}", fontsize=10)
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
    fig.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150)
    print(f"\nwrote {OUT_TBL} and {OUT_FIG}")
    return res


if __name__ == "__main__":
    main()
