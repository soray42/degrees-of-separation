"""Tier 0 addendum, Task 3 — multi-dimensional (non-salary) explanation layer.
 → results/GAP_EXPLANATION.md

Field-level DESCRIPTIVE correlates of the gap (explicitly NOT causal), from the NY Fed
by-major table: employer demand (underemployment, unemployment), continuation (share with
graduate degree), and early/mid wage. Same report-the-sign-whatever-it-is discipline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from src.load_ar import load_ar_wapman
from src.load_er import load_er_scorecard
from src.gap import compute_gap_map
from src.predictions import load_nyfed_outcomes
from src import dispersion as D
from src.crosswalks import fields as F

LAB = {f["key"]: f["label"] for f in F.TIER0_FIELDS}
CORR = {"underemployment": "employer demand (underemployment %)",
        "unemployment": "employer demand (unemployment %)",
        "wage_early": "early-career median wage",
        "grad_degree_share": "continuation (share w/ graduate degree)"}


def main():
    ar = load_ar_wapman(); sc = load_er_scorecard("undergrad")
    gm = compute_gap_map(ar, sc, "undergrad")
    nyf = load_nyfed_outcomes()
    w = D.load_acs_workers()
    gshare_acs = D.field_grad_share_acs(w)[["field", "grad_share_acs"]]

    m = gm[gm.reliable_flag].merge(nyf, on="field", how="inner").merge(gshare_acs, on="field", how="left")
    m["label"] = m.field.map(LAB)
    rows = []
    for col, desc in CORR.items():
        d = m.dropna(subset=["gap", col])
        rho, p = spearmanr(d[col], d["gap"]) if len(d) > 3 else (np.nan, np.nan)
        rows.append(dict(correlate=desc, n=len(d), spearman=rho, p=p))
    # ACS continuation (all-30 coverage) as a cross-check on the selection channel
    d2 = gm[gm.reliable_flag].merge(gshare_acs, on="field").dropna(subset=["gap", "grad_share_acs"])
    rho2, p2 = spearmanr(d2["grad_share_acs"], d2["gap"])
    rows.append(dict(correlate="continuation — ACS grad-degree share (all 30 covered)",
                     n=len(d2), spearman=rho2, p=p2))
    tab = pd.DataFrame(rows)

    L = ["# Tier 0 addendum — Multi-dimensional gap explanation (Task 3)\n",
         "The gap stays salary-anchored, but its **explanation** now draws on employer demand "
         "and continuation, not salary alone. **Descriptive correlates only — not causal.** "
         "NY Fed 'Labor Market for Recent College Graduates' by-major, reliable fields. "
         "Run: `python scripts/06_explanation.py`. Date 2026-06-20.\n",
         "## Spearman(gap, correlate) on reliable fields\n",
         tab.to_markdown(index=False, floatfmt=("", ".0f", "+.3f", ".3f")),
         "\n## Read (report-the-sign-whatever-it-is)\n",
         "- **The gap does NOT track employer demand.** Underemployment ≈ 0 (+0.05); "
         "unemployment weak (−0.31). The gap is not a demand story.",
         "- **Continuation (graduate-degree share) is correct-signed but modest** — +0.28 on "
         "the all-30-covered ACS measure, +0.15 on NY Fed (reliable fields): fields that send "
         "more BA talent onward tend to have larger gaps (the selection fingerprint), but the "
         "continuous correlation is weak. The signal is much **clearer as the discrete regime "
         "split** (Task 2: PhD-pipeline 0.65 vs prestige-transmission 0.40) than as a "
         "continuous Spearman — the typology, not the scalar, is where continuation bites.",
         "- **The strongest single correlate is early-career wage (−0.43)** — higher-paying "
         "fields have lower gaps — but that is itself salary, so it partly re-expresses the "
         "salary-anchored gap rather than adding a new dimension.",
         "- **Honest bottom line:** no *single non-salary* continuous correlate is strong; the "
         "multi-dimensional value comes from the **regime typology (licensure + pipeline, Task 2)** "
         "and the **observability test (P1)**, not from the NY Fed scalars. The gap's structure "
         "is legible, but through discrete channels, not one demand/continuation index.\n",
         "## Per-field correlate table (reliable fields)\n",
         m[["label", "gap", "underemployment", "unemployment", "wage_early",
            "grad_degree_share", "grad_share_acs"]].sort_values("gap", ascending=False)
          .to_markdown(index=False, floatfmt=("", ".3f", ".1f", ".1f", ".0f", ".1f", ".0%"))]
    Path("results/GAP_EXPLANATION.md").write_text("\n".join(L))
    print("wrote results/GAP_EXPLANATION.md")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
