"""Figures for the OpenAlex field-tagging re-validation (reads the CSVs from scripts/17 & 18).
 → outputs/figures/oa_field_validation.png, oa_research_vs_degree_drift.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = Path("outputs/figures"); OUT.mkdir(parents=True, exist_ok=True)
v = pd.read_csv("data/interim/openalex_field_validation.csv")
r = pd.read_csv("data/interim/oa_vs_ft_robust.csv")

# ---------- Fig 1: per-field rho, OpenAlex vs field_text vs benchmark ----------
v = v.sort_values("rho_benchmark", ascending=False).reset_index(drop=True)
x = np.arange(len(v)); w = 0.27
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.bar(x - w, v.rho_openalex, w, label="OpenAlex research-field tag", color="#D6202A")
ax.bar(x,     v.rho_fieldtext, w, label="degree free-text (field_text)", color="#4878CF")
ax.bar(x + w, v.rho_benchmark, w, label="Wapman public-edge benchmark (ceiling)", color="#9aa0a6")
for i, row in v.iterrows():
    ax.text(i - w, row.rho_openalex + .01, f"{int(row.oa_edges)}", ha="center", va="bottom", fontsize=7, color="#D6202A")
    ax.text(i,     row.rho_fieldtext + .01, f"{int(row.ft_edges)}", ha="center", va="bottom", fontsize=7, color="#4878CF")
ax.axhline(v.rho_benchmark.mean(), ls="--", lw=1, color="#666",
           label=f"mean ceiling {v.rho_benchmark.mean():.2f}")
ax.set_xticks(x); ax.set_xticklabels(v.field, rotation=20, ha="right")
ax.set_ylabel("Spearman(ORCID-AR SpringRank, Wapman rank)")
ax.set_ylim(0, .95)
ax.set_title("Per-field ORCID-AR vs Wapman — research-field tags do NOT beat degree text\n"
             "(numbers above bars = edges retained; OpenAlex retains fewer, the coverage effect)")
ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
fig.savefig(OUT / "oa_field_validation.png", dpi=140)

# ---------- Fig 2: coverage thinning (left) + symmetric equal-n parity (right) ----------
m = v.merge(r[["field", "rho_oa_eqn", "rho_ft_eqn", "p_oa_ge_ft_eqn"]], on="field", how="left")
m["retain"] = m.oa_edges / m.ft_edges
m = m.sort_values("retain")
y = np.arange(len(m))
fig2, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
a1.barh(y, m.retain, color="#D6202A")
a1.axvline(1.0, ls="--", lw=1, color="#666")
a1.set_yticks(y); a1.set_yticklabels(m.field)
a1.set_xlabel("OpenAlex edges / field_text edges  (per Wapman field)")
a1.set_title("Coverage thinning: OpenAlex retains\nfewer edges per degree-field")
# symmetric equal-n parity: ρ_oa@n vs ρ_ft@n (both subsampled to shared min n)
a2.scatter(m.rho_ft_eqn, m.rho_oa_eqn, s=60, color="#4878CF", zorder=3)
lim = [0.35, 0.68]; a2.plot(lim, lim, ls="--", lw=1, color="#666", label="parity")
for _, row in m.iterrows():
    if np.isfinite(row.rho_ft_eqn):
        a2.annotate(row.field, (row.rho_ft_eqn, row.rho_oa_eqn), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
a2.set_xlabel("ρ field_text @ shared-min n")
a2.set_ylabel("ρ OpenAlex @ shared-min n")
a2.set_xlim(lim); a2.set_ylim(lim)
a2.set_title("At EQUAL n, OpenAlex ≈ degree text\n(points on/above parity → tags not worse)")
a2.legend(fontsize=8)
fig2.tight_layout(); fig2.savefig(OUT / "oa_research_vs_degree_drift.png", dpi=140)
print("saved oa_field_validation.png, oa_research_vs_degree_drift.png")
