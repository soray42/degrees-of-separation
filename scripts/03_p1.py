"""Tier 0 — P1 test.  → results/P1_RESULT.md + figure

P1 (proposal v2 §4.3): the gap is *smaller* where graduate productivity is more observable,
proxied by within-field earnings dispersion. Predicted sign: NEGATIVE. Tested on
reliability-passing fields; the included-unreliable variant is reported for honesty.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard, load_er_pseo
from src.gap import compute_gap_map
from src.predictions import field_dispersion_pseo, field_dispersion_cv, test_p1
from src.crosswalks import fields as F

LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}


def main():
    ar = load_ar_wapman(); sc = load_er_scorecard("undergrad"); ps = load_er_pseo("undergrad")
    gm = compute_gap_map(ar, sc, "undergrad")
    dp = field_dispersion_pseo(ps); dcv = field_dispersion_cv(sc)

    res = {}
    for name, proxy, tbl in [("PSEO within-inst (primary)", "disp_pseo", dp),
                             ("cross-inst CV (robustness)", "disp_cv", dcv)]:
        for ro in (True, False):
            res[(name, ro)] = test_p1(gm, tbl, proxy, reliable_only=ro)

    # scatter on the primary proxy, reliable fields
    m = (gm[gm.reliable_flag].merge(dp, on="field").merge(dcv, on="field")
         .dropna(subset=["gap", "disp_pseo"]))
    m["label"] = m.field.map(LAB)
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for _, r in m.iterrows():
        c = "#D6202A" if r.field == "computer_science" else "#E8902A" if r.field == "biology" else "#4878CF"
        ax.scatter(r.disp_pseo, r.gap, s=70 if r.field in ("computer_science", "biology") else 45,
                   c=c, edgecolor="black" if r.field in ("computer_science", "biology") else "white")
        ax.annotate(r.label, (r.disp_pseo, r.gap), fontsize=7, xytext=(3, 2), textcoords="offset points")
    if len(m) > 2:
        b1, b0 = np.polyfit(m.disp_pseo, m.gap, 1)
        xs = np.linspace(m.disp_pseo.min(), m.disp_pseo.max(), 30)
        ax.plot(xs, b0 + b1 * xs, "--", c="#888")
    rr = res[("PSEO within-inst (primary)", True)]
    ax.set_xlabel("Within-institution earnings dispersion (p75−p25)/p50 [PSEO] — observability")
    ax.set_ylabel("Gap")
    ax.set_title(f"P1: gap vs productivity observability (reliable fields, n={rr['n']})\n"
                 f"Spearman={rr['spearman']:+.2f} (p={rr['p']:.2f}); P1 predicts negative — red=CS, orange=Biology")
    fig.tight_layout(); fig.savefig("results/figures/p1_scatter.png", dpi=150)

    # ---- write report ----
    L = ["# Tier 0 — P1 result\n",
         "**P1 (proposal v2 §4.3 / H2):** the gap is *smaller* where graduate productivity is "
         "more observable, proxied by within-field earnings dispersion. **Predicted sign: "
         "negative.** Run: `python scripts/03_p1.py`. Date 2026-06-20.\n",
         "## Spearman(gap, dispersion)\n",
         "| proxy | fields | n | Spearman | p | sign | holds (neg & p<.10) |",
         "|---|---|---|---|---|---|---|"]
    for (name, ro), r in res.items():
        scope = "reliable only" if ro else "all (incl. unreliable)"
        sign = "negative ✓" if r["sign_ok"] else "positive ✗"
        L.append(f"| {name} | {scope} | {r['n']} | {r['spearman']:+.3f} | {r['p']:.3f} | {sign} | {r['holds']} |")
    pp = res[("PSEO within-inst (primary)", True)]
    cv = res[("cross-inst CV (robustness)", True)]
    pp_all = res[("PSEO within-inst (primary)", False)]
    L.append("\n## Verdict on P1\n")
    L.append(f"- On **reliable fields**, the sign is **negative on both proxies** "
             f"(PSEO primary {pp['spearman']:+.2f}, p={pp['p']:.2f}; CV robustness "
             f"{cv['spearman']:+.2f}, p={cv['p']:.2f}) — the **predicted direction**, "
             "consistent with the prior diagnostic (≈ −0.41), **but not statistically "
             "significant** at n=%d." % pp["n"])
    L.append(f"- It is **fragile**: including unreliable fields **flips the primary PSEO proxy "
             f"to {pp_all['spearman']:+.2f}** (wrong sign) while the CV proxy stays weakly "
             "negative. Noise-dominated fields (chemistry, earth sciences, several "
             "engineering branches) must be excluded for the sign to hold.")
    L.append("- **Conclusion: P1 holds in *direction* but is *not confirmed*** — correct-signed,"
             " underpowered, and dependent on the reliability gate. This is 'initial support', "
             "not a clean prediction success. See `figures/p1_scatter.png`.\n")
    Path("results/P1_RESULT.md").write_text("\n".join(L))
    print("wrote results/P1_RESULT.md, figures/p1_scatter.png")
    for (name, ro), r in res.items():
        print(f"  {name:<28} {'reliable' if ro else 'all':<9} n={r['n']:>2} rho={r['spearman']:+.3f} p={r['p']:.3f}")


if __name__ == "__main__":
    main()
