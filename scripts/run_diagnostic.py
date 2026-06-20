"""Run the four-test diagnostic: is the AR-ER gap real signal or a low-SNR artifact?

Outputs:
  results/tables/diagnostic_nullmodel.csv     per-field observed vs synthetic (null) gap
  results/tables/diagnostic_bootstrap.csv      per-field Spearman + bootstrap CI
  results/tables/diagnostic_summary.csv        verdict-relevant scalars
  results/figures/diagnostic_nullmodel.png      observed gap vs null-model gap (decisive)
  results/figures/diagnostic_bootstrap.png      per-field Spearman with bootstrap CI
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from src import tier0 as T, tier0_5 as T5, diagnostic as D
from src.crosswalks import fields as F

CS, CHEM = "computer_science", "chemistry"
LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}


def main():
    m = D.load_matched()
    # observed gap on the SAME matched set used by the null/bootstrap (internal consistency)
    obs = (m.groupby("field_key")
             .apply(lambda g: 1 - spearmanr(g["prestige_score"], g["earn"])[0]))
    obs.name = "observed_gap"

    # ---- Test 1: null model (alpha sensitivity) ----
    nm = {a: D.null_model(m, obs, alpha=a, B=1000) for a in (0.4, 0.6, 0.8)}
    nm06 = nm[0.6].copy()
    nm06["label"] = nm06["field_key"].map(LAB)
    nm06.to_csv("results/tables/diagnostic_nullmodel.csv", index=False)

    rho_s, _ = spearmanr(nm06["observed_gap"], nm06["synth_gap_mean"])
    rho_p, _ = pearsonr(nm06["observed_gap"], nm06["synth_gap_mean"])
    covered = nm06["covered"].mean()
    above = (nm06["observed_gap"] > nm06["synth_gap_hi"]).mean()
    print("=== TEST 1 — NULL MODEL (alpha=0.6) ===")
    cols = ["label", "N", "observed_gap", "synth_gap_mean", "synth_gap_lo", "synth_gap_hi",
            "signal_frac", "covered", "excess_gap"]
    print(nm06.sort_values("observed_gap")[cols].to_string(index=False,
          formatters={c: "{:.3f}".format for c in
                      ["observed_gap", "synth_gap_mean", "synth_gap_lo", "synth_gap_hi",
                       "signal_frac", "excess_gap"]}))
    print(f"\n across-field corr(observed, synth_mean): Spearman={rho_s:.2f}  Pearson={rho_p:.2f}")
    print(f" observed gap COVERED by null 95% band: {covered:.0%} of fields")
    print(f" observed gap ABOVE null band (real excess disagreement): {above:.0%} of fields")
    print(" alpha sensitivity (corr, %covered):")
    for a in (0.4, 0.6, 0.8):
        x = nm[a]
        rs, _ = spearmanr(x["observed_gap"], x["synth_gap_mean"])
        print(f"   alpha={a}: corr={rs:.2f}  covered={x['covered'].mean():.0%}  "
              f"mean synth_gap={x['synth_gap_mean'].mean():.3f} vs observed {x['observed_gap'].mean():.3f}")

    # ---- Test 2: bootstrap reliability ----
    bs = D.bootstrap(m, B=1000)
    bs["label"] = bs["field_key"].map(LAB)
    bs.to_csv("results/tables/diagnostic_bootstrap.csv", index=False)
    print("\n=== TEST 2 — BOOTSTRAP Spearman CIs (sorted) ===")
    print(bs.sort_values("spearman")[["label", "N", "spearman", "boot_lo", "boot_hi", "boot_se"]]
          .to_string(index=False, formatters={c: "{:.3f}".format for c in
                     ["spearman", "boot_lo", "boot_hi", "boot_se"]}))
    # do the high-gap (low-spearman) fields have CIs separated from the low-gap fields?
    lowsp = bs.nsmallest(5, "spearman"); highsp = bs.nlargest(5, "spearman")
    sep = lowsp["boot_hi"].max() < highsp["boot_lo"].min()
    print(f" highest-gap-5 upper CI ({lowsp['boot_hi'].max():.2f}) vs lowest-gap-5 lower CI "
          f"({highsp['boot_lo'].min():.2f}) -> separated: {sep}")
    print(f" median bootstrap SE: {bs['boot_se'].median():.3f}")

    # ---- Test 3: residualization ----
    expl = T5.assemble(T.all_field_gaps(T.load_prestige(), T.load_earnings()))
    d3 = obs.reset_index().merge(expl[["field_key", "earn_cv", "doc_ba_ratio"]], on="field_key")
    X = sm.add_constant(d3[["earn_cv", "doc_ba_ratio"]])
    fit = sm.OLS(d3["observed_gap"], X).fit()
    d3["resid"] = fit.resid
    d3["label"] = d3["field_key"].map(LAB)
    cs_resid = d3.loc[d3.field_key == CS, "resid"].iloc[0]
    cs_z = cs_resid / d3["resid"].std(ddof=1)
    print("\n=== TEST 3 — RESIDUALIZATION (gap ~ earn_cv + grad-school pull) ===")
    print(f" model R2={fit.rsquared:.2f}; residual SD={d3['resid'].std(ddof=1):.3f}")
    print(d3.sort_values("resid")[["label", "observed_gap", "resid"]].to_string(index=False,
          formatters={"observed_gap": "{:.3f}".format, "resid": "{:+.3f}".format}))
    print(f" CS residual = {cs_resid:+.3f} (z={cs_z:+.2f}) -> CS {'still stands out' if abs(cs_z)>1 else 'does NOT stand out'}")

    # ---- Test 4: restriction to high-CV (well-measured) fields ----
    d4 = obs.reset_index().merge(
        expl[["field_key", "earn_cv", "task_distance", "price_wedge_abs"]], on="field_key")
    cut = d4["earn_cv"].median()
    clean = d4[d4["earn_cv"] >= cut]
    print(f"\n=== TEST 4 — RESTRICTION to top-half CV (well-measured) fields (n={len(clean)}) ===")
    for v in ["task_distance", "price_wedge_abs"]:
        c = clean.dropna(subset=[v])
        rho, p = spearmanr(c[v], c["observed_gap"])
        print(f"   Spearman(gap, {v}) on clean subset = {rho:+.2f} (p={p:.2f}, n={len(c)})")

    # ---- summary scalars ----
    pd.DataFrame([dict(nullmodel_corr_spearman=rho_s, nullmodel_corr_pearson=rho_p,
                       frac_covered=covered, frac_above_band=above,
                       mean_observed_gap=obs.mean(), mean_synth_gap=nm06["synth_gap_mean"].mean(),
                       boot_se_median=bs["boot_se"].median(), ci_separated_hi_lo=sep,
                       resid_sd=d3["resid"].std(ddof=1), cs_resid_z=cs_z,
                       resid_model_r2=fit.rsquared)]).to_csv(
        "results/tables/diagnostic_summary.csv", index=False)

    # ===== FIGURES =====
    # Fig 1: observed vs null-model gap
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    x = nm06["synth_gap_mean"]; y = nm06["observed_gap"]
    yerr = np.vstack([x - nm06["synth_gap_lo"], nm06["synth_gap_hi"] - x])
    ax.errorbar(x, y, xerr=yerr, fmt="none", ecolor="#cccccc", elinewidth=1, zorder=1)
    for _, r in nm06.iterrows():
        col = "#D6202A" if r.field_key == CS else "#E8902A" if r.field_key == CHEM else "#4878CF"
        sz = 130 if r.field_key in (CS, CHEM) else 40
        ax.scatter(r["synth_gap_mean"], r["observed_gap"], s=sz, c=col, edgecolor="black" if sz > 100 else "white", lw=.6, zorder=3)
        ax.annotate(r["label"], (r["synth_gap_mean"], r["observed_gap"]), fontsize=6.6,
                    xytext=(3, 2), textcoords="offset points",
                    color=col if r.field_key in (CS, CHEM) else "#555")
    lim = [min(x.min(), y.min()) - .03, max(x.max(), y.max()) + .03]
    ax.plot(lim, lim, "--", c="#888", lw=1, label="observed = null (pure artifact)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Null-model gap (perfect agreement + sampling noise; x-err = 95% band)")
    ax.set_ylabel("Observed gap")
    ax.set_title(f"Test 1 — observed vs null-model gap   (corr={rho_p:.2f}, "
                 f"{covered:.0%} covered, {above:.0%} above band)\nred=CS, orange=Chemistry")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig("results/figures/diagnostic_nullmodel.png", dpi=150)

    # Fig 2: bootstrap Spearman CIs
    fig2, ax2 = plt.subplots(figsize=(8.5, 7))
    b = bs.sort_values("spearman").reset_index(drop=True)
    ypos = np.arange(len(b))
    ax2.errorbar(b["spearman"], ypos,
                 xerr=np.vstack([b["spearman"] - b["boot_lo"], b["boot_hi"] - b["spearman"]]),
                 fmt="o", color="#4878CF", ecolor="#aaaaaa", capsize=2)
    for k, c in [(CS, "#D6202A"), (CHEM, "#E8902A")]:
        i = b.index[b.field_key == k]
        if len(i):
            ax2.errorbar(b.loc[i, "spearman"], i, xerr=[[b.loc[i, "spearman"].iloc[0] - b.loc[i, "boot_lo"].iloc[0]],
                         [b.loc[i, "boot_hi"].iloc[0] - b.loc[i, "spearman"].iloc[0]]],
                         fmt="o", color=c, ecolor=c, capsize=3, zorder=5)
    ax2.set_yticks(ypos); ax2.set_yticklabels(b["label"], fontsize=7.5)
    ax2.axvline(0, color="#ccc", lw=.8)
    ax2.set_xlabel("within-field Spearman(prestige, earnings)  [gap = 1 − this]")
    ax2.set_title("Test 2 — bootstrap reliability of the within-field rank correlation\n"
                  "tight CI at low Spearman = genuine weak agreement; wide = noise-dominated")
    fig2.tight_layout(); fig2.savefig("results/figures/diagnostic_bootstrap.png", dpi=150)
    print("\nwrote 3 tables + 2 figures")


if __name__ == "__main__":
    main()
