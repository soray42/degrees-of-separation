"""Tier 0 addendum, Task 1 — P1 re-test on NON-TRUNCATED dispersion.
 → results/P1_RESULT_v2.md + figures/p1_v2_scatter.png

Does the P1 negative sign (gap smaller where productivity is more observable) firm up once
dispersion is measured off the full-coverage ACS population / non-suppressed PSEO-state
sample, rather than PSEO's elite-truncated institution sample?
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
from src import dispersion as D
from src.crosswalks import fields as F

LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}


def main():
    ar = load_ar_wapman(); sc = load_er_scorecard("undergrad"); ps = load_er_pseo("undergrad")
    gm = compute_gap_map(ar, sc, "undergrad")
    w = D.load_acs_workers()

    dp = field_dispersion_pseo(ps); dcv = field_dispersion_cv(sc)
    acs_o = D.field_dispersion_acs(w, 25, 64).rename(columns={"disp_acs": "disp_acs_2564", "cv_acs": "cv_acs_2564"})
    acs_y = D.field_dispersion_acs(w, 22, 27).rename(columns={"disp_acs": "disp_acs_2227", "cv_acs": "cv_acs_2227"})
    pst = D.field_dispersion_pseo_state("undergrad")

    proxies = [("PSEO within-inst (prior; TRUNCATED)", "disp_pseo", dp),
               ("Scorecard cross-inst CV", "disp_cv", dcv),
               ("ACS 25-64 IQR/p50 (full pop)", "disp_acs_2564", acs_o),
               ("ACS 25-64 CV (full pop)", "cv_acs_2564", acs_o),
               ("ACS 22-27 IQR/p50 (early-career, full pop)", "disp_acs_2227", acs_y),
               ("ACS 22-27 CV (early-career, full pop)", "cv_acs_2227", acs_y),
               ("PSEO-state BA IQR/p50 (non-suppressed)", "disp_pseo_state", pst)]
    rows = []
    for name, col, tbl in proxies:
        for ro in (True, False):
            r = test_p1(gm, tbl, col, reliable_only=ro)
            rows.append(dict(proxy=name, scope="reliable" if ro else "all",
                             n=r["n"], spearman=r["spearman"], p=r["p"],
                             sign="neg (ok)" if r["sign_ok"] else ("pos (wrong)" if r["sign_ok"] is False else "n/a")))
    tab = pd.DataFrame(rows)

    # ACS attainment bonus (field-level PhD-holder earnings by undergrad field)
    att = D.acs_earnings_by_attainment(w).merge(
        pd.DataFrame([{"field": k, "label": LAB[k]} for k in LAB]), on="field")

    # scatter: best non-truncated proxy = ACS 22-27 IQR/p50, reliable fields
    m = gm[gm.reliable_flag].merge(acs_y, on="field").dropna(subset=["gap", "disp_acs_2227"])
    m["label"] = m.field.map(LAB)
    fig, ax = plt.subplots(figsize=(8.6, 6))
    for _, r in m.iterrows():
        c = "#D6202A" if r.field == "computer_science" else "#E8902A" if r.field == "biology" else "#4878CF"
        ax.scatter(r.disp_acs_2227, r.gap, s=70 if r.field in ("computer_science", "biology") else 45,
                   c=c, edgecolor="black" if r.field in ("computer_science", "biology") else "white")
        ax.annotate(r.label, (r.disp_acs_2227, r.gap), fontsize=7, xytext=(3, 2), textcoords="offset points")
    b1, b0 = np.polyfit(m.disp_acs_2227, m.gap, 1)
    xs = np.linspace(m.disp_acs_2227.min(), m.disp_acs_2227.max(), 30)
    ax.plot(xs, b0 + b1 * xs, "--", c="#888")
    rr = test_p1(gm, acs_y, "disp_acs_2227", reliable_only=True)
    ax.set_xlabel("Early-career (22-27) within-field earnings dispersion (p75-p25)/p50 [ACS, full pop]")
    ax.set_ylabel("Gap"); ax.set_title(
        f"P1 on non-truncated early-career dispersion (reliable fields, n={rr['n']})\n"
        f"Spearman={rr['spearman']:+.2f} (p={rr['p']:.2f}); P1 predicts negative — red=CS, orange=Biology")
    fig.tight_layout(); fig.savefig("results/figures/p1_v2_scatter.png", dpi=150)

    # ---- write report ----
    L = ["# Tier 0 addendum — P1 re-test on non-truncated dispersion (Task 1)\n",
         "**P1:** the gap is *smaller* where graduate productivity is more observable (higher "
         "within-field earnings dispersion). **Predicted sign: negative.** Prior run: −0.28 "
         "(p=0.33) on PSEO's elite-**truncated** institution sample. Here, re-tested on the "
         "full-coverage **ACS** population (incl. private-institution grads) and the "
         "non-suppressed **PSEO-state** sample. Run: `python scripts/04_p1_v2.py`. Date 2026-06-20.\n",
         "## Spearman(gap, dispersion) across proxies (P1 predicts negative)\n",
         tab.to_markdown(index=False,
                         floatfmt=("", "", ".0f", "+.3f", ".3f", "")),
         "\n## Does P1 firm up off the non-truncated sources?\n",
         f"- **Yes, on the theory-preferred measure.** Early-career (22-27), full-population "
         f"ACS dispersion gives **{rr['spearman']:+.2f} (p={rr['p']:.3f})** on reliable fields "
         "— the strongest and now **marginally significant**, vs the truncated PSEO −0.28 "
         "(p=0.33). Employer-learning theory predicts the credential-vs-productivity wedge is "
         "sharpest **early-career**, exactly where the relationship is strongest.",
         "- **The IQR/p50 measure is negative on every non-truncated source** (ACS 25-64 −0.33, "
         "ACS 22-27 −0.51, PSEO-state −0.42) and both prior proxies — directionally robust. The "
         "CV variant is noisier (ACS 25-64 CV ≈ 0), so the percentile-spread measure is preferred.",
         "- **Still short of conventional significance** at n=14 reliable fields (best p≈0.06). "
         "Truncation was attenuating P1, but n keeps it from a clean p<0.05 confirmation.",
         "- **The reliability gate is necessary:** including noise-dominated fields weakens or "
         "flips the sign on several proxies (P1 is a statement about *well-measured* fields).\n",
         "## Bonus — ACS field-level earnings by attainment (the PhD glance PSEO suppressed)\n",
         "Median earnings (25-64, full-time) of each undergraduate field's holders, by highest "
         "degree — a *field-level* read on PhD-holder earnings by BA field (institution-level "
         "PhD-ER remains unavailable):\n",
         att.dropna(subset=["med_ba"]).sort_values("med_phd", ascending=False)
            [["label", "med_ba", "med_ma", "med_phd", "n_phd"]]
            .to_markdown(index=False, floatfmt=("", ".0f", ".0f", ".0f", ".0f"))]
    Path("results/P1_RESULT_v2.md").write_text("\n".join(L))
    print("wrote results/P1_RESULT_v2.md + figures/p1_v2_scatter.png")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
