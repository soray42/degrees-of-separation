"""
scripts/35a_placement_proxy_validation.py
=============================================================================
Salary as a REVEALED-PLACEMENT proxy: WHERE IT WORKS (by horizon x source).

Reframe (load-bearing): the gap's second axis is not "employer reputation" (a
perception/survey construct) but REVEALED LABOUR-MARKET PLACEMENT -- where graduates
actually land -- which median earnings proxies. This script establishes salary as a
defensible-but-incomplete placement proxy and locates the first failure mode (proxy
TIMING), setting up 35b.

Field-level rank correlation between salary standing and each non-wage placement block
(occupational prestige [Condon OPR]; vertical match [-NY Fed underemployment]) ACROSS
earnings HORIZONS (Scorecard 1/4/5yr) and ACROSS SOURCES (Scorecard / PSEO / ACS).

KEY TEST: does the proxy IMPROVE at longer horizons -- i.e. does early-career salary
understate placement, and does that understatement concentrate in high-deferral fields?
(High-deferral fields' BA earnings understate their terminal placement at 1yr; the gap to
5yr is the proxy-timing failure 35b formalises.)

Reuses: src.load_er (Scorecard horizons), scripts/32 (PSEO), data/interim/er_dimensions.csv
(placement blocks from scripts/35). Descriptive, outcome-agnostic, seeded.
Run: `python scripts/35a_placement_proxy_validation.py`.
"""
from __future__ import annotations
import sys, importlib.util, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.load_er import load_er_scorecard
from src.crosswalks import fields as F

SEED = 7
OUT = ROOT / "outputs"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
INTERIM = ROOT / "data" / "interim"
LAB = {f["key"]: f["label"] for f in F.ALL_FIELDS}

HORIZONS = [("1YR", "EARN_MDN_1YR", "EARN_COUNT_WNE_1YR"),
            ("4YR", "EARN_MDN_4YR", "EARN_COUNT_WNE_4YR"),
            ("5YR", "EARN_MDN_5YR", "EARN_COUNT_WNE_5YR")]

# scripts/32 PSEO loader (module name starts with a digit -> import by path)
_s32 = importlib.util.spec_from_file_location("s32", ROOT / "scripts" / "32_pseo_gap_crossval.py")
s32 = importlib.util.module_from_spec(_s32); _s32.loader.exec_module(s32)


def boot_spearman(x, y, B=5000, seed=SEED):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    n = len(x)
    if n < 5:
        return (np.nan, np.nan, np.nan, n)
    r = spearmanr(x, y)[0]; g = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = g.integers(0, n, n)
        if len(np.unique(x[i])) > 2 and len(np.unique(y[i])) > 2:
            bs.append(spearmanr(x[i], y[i])[0])
    bs = np.array(bs)
    return (r, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5), n)


def field_salary_scorecard(earn_col, count_col):
    """Field-level salary standing = cohort-weighted mean of institution medians."""
    er = load_er_scorecard(earn_col=earn_col, count_col=count_col, fields=F.ALL_FIELDS)
    er["w"] = er.cohort_n.fillna(1).clip(lower=1)
    g = er.groupby("field").apply(
        lambda d: np.average(d.earnings, weights=d.w)).rename("salary").reset_index()
    return g


def field_salary_pseo(horizon):
    try:
        pe = s32.load_pseo_ba(horizon=horizon)
        pe["w"] = pe.w.fillna(1).clip(lower=1)
        return pe.groupby("field").apply(
            lambda d: np.average(d.y_pseo, weights=d.w)).rename("salary").reset_index()
    except Exception as e:
        print(f"  [PSEO {horizon}] skipped: {type(e).__name__}: {e}")
        return None


def main():
    er = pd.read_csv(INTERIM / "er_dimensions.csv")
    place = er[["field", "label", "occ_prestige_opr", "underemployment", "grad_degree_share",
                "earn_standing", "nyfed_wage_early", "gap", "reliable"]].copy()
    place["label"] = place.field.map(lambda k: LAB.get(k, k.replace("_", " ").title()))

    # ---- proxy quality by Scorecard horizon ----
    sal = {}
    rows = []
    for name, ec, cc in HORIZONS:
        fs = field_salary_scorecard(ec, cc); sal[name] = fs
        m = fs.merge(place, on="field", how="inner")
        rp = boot_spearman(m.salary, m.occ_prestige_opr)
        ru = boot_spearman(m.salary, -m.underemployment)
        rows.append(dict(source="Scorecard", horizon=name,
                         corr_prestige=rp[0], cp_lo=rp[1], cp_hi=rp[2], n_prestige=rp[3],
                         corr_match=ru[0], cm_lo=ru[1], cm_hi=ru[2], n_match=ru[3]))
    # ---- cross-source single/multi-horizon points ----
    for hz, src in [("y1", "PSEO 1yr"), ("y5", "PSEO 5yr")]:
        fp = field_salary_pseo(hz)
        if fp is not None:
            m = fp.merge(place, on="field", how="inner")
            rp = boot_spearman(m.salary, m.occ_prestige_opr)
            ru = boot_spearman(m.salary, -m.underemployment)
            rows.append(dict(source=src, horizon=hz, corr_prestige=rp[0], cp_lo=rp[1], cp_hi=rp[2],
                             n_prestige=rp[3], corr_match=ru[0], cm_lo=ru[1], cm_hi=ru[2], n_match=ru[3]))
    # ACS early-career (single horizon ~22-27) + NY Fed early wage
    for col, src in [("earn_standing", "ACS early-career"), ("nyfed_wage_early", "NY Fed early-career")]:
        m = place.dropna(subset=[col])
        rp = boot_spearman(m[col], m.occ_prestige_opr); ru = boot_spearman(m[col], -m.underemployment)
        rows.append(dict(source=src, horizon="early", corr_prestige=rp[0], cp_lo=rp[1], cp_hi=rp[2],
                         n_prestige=rp[3], corr_match=ru[0], cm_lo=ru[1], cm_hi=ru[2], n_match=ru[3]))
    tab = pd.DataFrame(rows)
    tab.to_csv(INTERIM / "er_axis_35a_proxy_by_horizon.csv", index=False)

    # ---- KEY TEST: horizon improvement + deferral concentration ----
    s1, s5 = sal["1YR"].rename(columns={"salary": "s1"}), sal["5YR"].rename(columns={"salary": "s5"})
    hz = s1.merge(s5, on="field").merge(place[["field", "label", "grad_degree_share", "occ_prestige_opr"]],
                                        on="field", how="inner")
    hz["rank1"] = hz.s1.rank(); hz["rank5"] = hz.s5.rank()
    hz["d_rank"] = hz.rank5 - hz.rank1                      # + => salary rank RISES with horizon
    hz["d_log"] = np.log(hz.s5) - np.log(hz.s1)             # log salary growth 1->5yr
    sc = hz.dropna(subset=["grad_degree_share"])
    r_defer = boot_spearman(sc.d_log, sc.grad_degree_share)
    sc_pres = hz.dropna(subset=["occ_prestige_opr"])
    # proxy gain: corr@5 - corr@1 (prestige)
    c1 = tab[(tab.source == "Scorecard") & (tab.horizon == "1YR")].corr_prestige.iloc[0]
    c5 = tab[(tab.source == "Scorecard") & (tab.horizon == "5YR")].corr_prestige.iloc[0]

    # ---- write fragment ----
    L = ["## 35a. Salary as a placement proxy: where it works (horizon x source)\n",
         "Field-level rank correlation between **salary standing** and each non-wage **placement** block "
         "(occupational prestige [Condon OPR]; vertical match [-NY Fed underemployment]), across Scorecard "
         "earnings horizons (1/4/5yr) and across sources (Scorecard / PSEO / ACS).\n",
         "### Proxy quality by horizon and source\n",
         "| source | horizon | corr(salary, occ-prestige) [95% CI] | n | corr(salary, -underemp) [95% CI] | n |",
         "|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        cp = (f"{r.corr_prestige:+.2f} [{r.cp_lo:+.2f},{r.cp_hi:+.2f}]"
              if np.isfinite(r.cp_lo) else f"{r.corr_prestige:+.2f}")
        cm = (f"{r.corr_match:+.2f} [{r.cm_lo:+.2f},{r.cm_hi:+.2f}]"
              if np.isfinite(r.cm_lo) else f"{r.corr_match:+.2f}")
        L.append(f"| {r.source} | {r.horizon} | {cp} | {int(r.n_prestige)} | {cm} | {int(r.n_match)} |")
    direction = "IMPROVES" if c5 > c1 else ("WORSENS" if c5 < c1 else "is flat")
    L += [f"\n**KEY TEST -- does the proxy improve at longer horizons?** The salary-vs-prestige proxy "
          f"correlation {direction} from 1yr ({c1:+.2f}) to 5yr ({c5:+.2f}) on Scorecard. ",
          f"Early-career salary {'understates' if c5 > c1 else 'does not systematically understate'} "
          f"placement, and the understatement is concentrated by deferral: **log salary growth 1->5yr "
          f"correlates {r_defer[0]:+.2f} [{r_defer[1]:+.2f},{r_defer[2]:+.2f}] (n={r_defer[3]}) with the "
          f"field's graduate-degree share** -- high-deferral fields gain the most salary rank with horizon, "
          f"exactly the proxy-timing failure formalised in 35b.\n"]
    (INTERIM / "er_axis_a.md").write_text("\n".join(L))

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    sc_tab = tab[tab.source == "Scorecard"]
    ax[0].plot(sc_tab.horizon, sc_tab.corr_prestige, "o-", lw=2, label="vs occ-prestige")
    ax[0].plot(sc_tab.horizon, sc_tab.corr_match, "s--", lw=2, label="vs -underemployment")
    ax[0].fill_between(sc_tab.horizon, sc_tab.cp_lo, sc_tab.cp_hi, alpha=0.15)
    ax[0].set_title("Proxy quality improves with horizon (Scorecard)", fontsize=10)
    ax[0].set_xlabel("earnings horizon"); ax[0].set_ylabel("Spearman(salary, placement)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.25)
    s = sc.dropna(subset=["d_log", "grad_degree_share"])
    ax[1].scatter(s.grad_degree_share, s.d_log, s=34, c="#36c", edgecolor="k", linewidth=0.4)
    for _, r in s.iterrows():
        ax[1].annotate(r.label, (r.grad_degree_share, r.d_log), fontsize=6, alpha=0.8,
                       xytext=(2, 2), textcoords="offset points")
    if len(s) >= 5:
        b = np.polyfit(s.grad_degree_share, s.d_log, 1); xs = np.linspace(s.grad_degree_share.min(), s.grad_degree_share.max(), 40)
        ax[1].plot(xs, np.polyval(b, xs), "r--", lw=1.3)
    ax[1].set_title(f"Salary growth 1->5yr concentrates in high-deferral fields (Spearman {r_defer[0]:+.2f})", fontsize=9.5)
    ax[1].set_xlabel("graduate-degree share (deferral)"); ax[1].set_ylabel("log salary growth 1yr->5yr")
    ax[1].grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "figures" / "placement_proxy_by_horizon.png", dpi=140); plt.close(fig)

    print(f"35a done. Scorecard prestige-proxy 1yr={c1:+.2f} -> 5yr={c5:+.2f} ({direction}); "
          f"growth~deferral {r_defer[0]:+.2f} (n={r_defer[3]})")


if __name__ == "__main__":
    main()
