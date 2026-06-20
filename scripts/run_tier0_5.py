"""Tier 0.5 — run the four-explanator diagnosis and write outputs.

Outputs:
  results/tables/tier0_5_explanators.csv   per-field gap + 4 explanators
  results/tables/tier0_5_tests.csv         per-explanator Spearman + combined OLS coefs
  results/figures/tier0_5_gap_vs_gradschool.png   winner (grad-school pull), labeled
  results/figures/tier0_5_diagnostic_panel.png    2x2 gap vs all four explanators
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import tier0 as T, tier0_5 as T5
from src.crosswalks import fields as F

CS, CHEM = "computer_science", "chemistry"
LABELS = {"task_distance": "Task distance (deg)  [H2]",
          "earn_cv": "Earnings dispersion (CV)",
          "doc_ba_ratio": "Grad-school pull (PhD/BA ratio)",
          "price_wedge_abs": "|Price wedge| (SDR)"}


def _scatter(ax, m, xvar, logx=False):
    is_cs = m.field_key == CS
    is_ch = m.field_key == CHEM
    other = m[~is_cs & ~is_ch]
    ax.scatter(other[xvar], other["gap"], s=40, c="#4878CF", alpha=.8, edgecolor="white", lw=.5)
    ax.scatter(m[is_ch][xvar], m[is_ch]["gap"], s=130, c="#E8902A", edgecolor="black", lw=1, zorder=5)
    ax.scatter(m[is_cs][xvar], m[is_cs]["gap"], s=140, c="#D6202A", edgecolor="black", lw=1, zorder=5)
    for _, r in m.iterrows():
        col = "#D6202A" if r.field_key == CS else ("#E8902A" if r.field_key == CHEM else "#444")
        ax.annotate(r.label, (r[xvar], r["gap"]), fontsize=6.6, xytext=(3, 2),
                    textcoords="offset points", color=col)
    if logx:
        ax.set_xscale("log")
    mm = m.dropna(subset=[xvar, "gap"])
    if len(mm) > 2 and not logx:
        b1, b0 = np.polyfit(mm[xvar], mm["gap"], 1)
        xs = np.linspace(mm[xvar].min(), mm[xvar].max(), 40)
        ax.plot(xs, b0 + b1 * xs, "--", c="#999", lw=1)


def main():
    prestige = T.load_prestige(); earn = T.load_earnings()
    gaps = T.all_field_gaps(prestige, earn)
    df = T5.assemble(gaps)
    df.to_csv("results/tables/tier0_5_explanators.csv", index=False)

    tests = T5.per_explanator(df)
    print("=== per-explanator Spearman vs gap ===")
    print(tests.to_string(index=False))

    mod, n, used = T5.combined_ols(df)
    print(f"\n=== combined standardized OLS (n={n}): gap ~ z(4 predictors) ===")
    rows = []
    if mod is not None:
        for name in mod.params.index:
            rows.append(dict(term=name, coef=mod.params[name], se=mod.bse[name],
                             t=mod.tvalues[name], p=mod.pvalues[name]))
            print(f"  {name:<16} coef={mod.params[name]:+.4f}  p={mod.pvalues[name]:.3f}")
        print(f"  R2={mod.rsquared:.3f}  adjR2={mod.rsquared_adj:.3f}")
    # write tests table
    out = tests.copy()
    out.to_csv("results/tables/tier0_5_tests.csv", index=False)
    if rows:
        pd.DataFrame(rows).to_csv("results/tables/tier0_5_combined_ols.csv", index=False)

    # CS & chemistry placement per explanator
    print("\n=== CS & Chemistry positions (rank within 20; 1 = lowest) ===")
    g = df[df.ok]
    for v in LABELS:
        gg = g.dropna(subset=[v]).sort_values(v).reset_index(drop=True)
        def rk(k):
            idx = gg.index[gg.field_key == k]
            return f"{int(idx[0])+1}/{len(gg)}" if len(idx) else "n/a"
        print(f"  {v:<16} CS={rk(CS)}  Chem={rk(CHEM)}")

    # ---- primary figure: gap vs winning explanator (grad-school pull) ----
    m = df[df.ok]
    fig, ax = plt.subplots(figsize=(8.8, 6))
    _scatter(ax, m.dropna(subset=["doc_ba_ratio"]), "doc_ba_ratio", logx=True)
    rho, p = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
        m.doc_ba_ratio, m.gap)
    ax.set_xlabel("Grad-school pull = PhD/Bachelor's completions ratio (log)   [IPEDS 2023]")
    ax.set_ylabel(r"Gap$_{field}$ = 1 − Spearman(prestige, earnings)")
    ax.set_title(f"Tier 0.5 winner — gap RISES with grad-school pull  "
                 f"(Spearman={rho:+.2f}, p={p:.3f})\nred=CS, orange=Chemistry")
    fig.tight_layout(); fig.savefig("results/figures/tier0_5_gap_vs_gradschool.png", dpi=150)

    # ---- diagnostic panel: gap vs all four ----
    fig2, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, v in zip(axes.ravel(), LABELS):
        _scatter(ax, m.dropna(subset=[v]), v, logx=(v == "doc_ba_ratio"))
        rho, p = __import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(
            m.dropna(subset=[v])[v], m.dropna(subset=[v]).gap)
        pred = T5.EXPLANATORS[v][0]
        ok = "sign OK" if np.sign(rho) == pred else "SIGN WRONG"
        ax.set_xlabel(LABELS[v]); ax.set_ylabel("Gap")
        ax.set_title(f"{LABELS[v]}  |  Spearman={rho:+.2f} (p={p:.2f}) — {ok}", fontsize=9)
    fig2.suptitle("Tier 0.5 diagnostic — gap vs four competing explanators "
                  "(red=CS, orange=Chemistry)", fontsize=12)
    fig2.tight_layout(); fig2.savefig("results/figures/tier0_5_diagnostic_panel.png", dpi=140)
    print("\nwrote results tables + 2 figures")


if __name__ == "__main__":
    main()
