"""Task 2 — ORCID-AR overlap validation vs Wapman (2011–2020 window).
 → outputs/ORCID_OVERLAP_VALIDATION_RESULT.md + figures

Build ORCID-derived SpringRank on the Wapman window and check it reproduces Wapman's
published ranks. Benchmark: canonical SpringRank on the PUBLIC Wapman edges vs published
(≈0.77 ceiling — pure edge-coverage loss vs the proprietary AARC census). Same SpringRank
settings as the recalibration run: α=0.5, binary (presence) weights.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import springrank

from src.crosswalks.institutions import normalize_institution_name

OUT = Path("outputs"); (OUT / "figures").mkdir(parents=True, exist_ok=True)
ALPHA, BINARY = 0.5, True
# best-covered fields to test (field_text keyword -> Wapman field name)
FIELD_KW = {"Computer Science": "computer", "Biological Sciences, General": "biolog",
            "Chemistry": "chemist", "Physics, General": "physics", "Mathematics": "math",
            "Psychology, General": "psycholog", "Economics, General": "econ",
            "Mechanical Engineering": "mechanical engineer", "Civil Engineering": "civil engineer"}


def springrank_scores(pairs):
    """pairs: list of (src_name, dst_name). Binary (presence) adjacency over the node union."""
    nodes = sorted(set(s for s, _ in pairs) | set(d for _, d in pairs))
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, d in pairs:
        A[idx[s], idx[d]] = 1.0 if BINARY else A[idx[s], idx[d]] + 1
    m = springrank.SpringRank(alpha=ALPHA, inverse_temp_fit_warning=False); m.fit(A)
    return dict(zip(nodes, m.ranks))


def spearman_vs_wapman(scores, wap):
    """scores: {inst_key: springrank}; wap: DataFrame[nk, Rank]. Spearman on shared insts.
    high score = top; Wapman Rank 0 = top → correlate score with -Rank (positive = agreement)."""
    s = pd.DataFrame({"nk": list(scores), "score": list(scores.values())}).merge(wap, on="nk")
    if len(s) < 8:
        return np.nan, len(s), s
    return spearmanr(s.score, -s.Rank)[0], len(s), s


def main():
    e = pd.read_parquet("data/interim/orcid_phd_faculty_edges.parquet")
    e["emp_nk"] = e.org_to_ror_name.map(normalize_institution_name)
    e["deg_nk"] = e.org_from_ror_name.map(normalize_institution_name)
    win = e[(e.year >= 2011) & (e.year <= 2020)].copy()
    rk = pd.read_csv("data/raw/wapman2022/ranks.csv")
    ed = pd.read_csv("data/raw/wapman2022/edge_lists.csv")

    def wap_ranks(level, value):
        r = rk[(rk.TaxonomyLevel == level) & (rk.TaxonomyValue == value)].copy()
        r["nk"] = r.InstitutionName.map(normalize_institution_name)
        return r[["nk", "Rank"]]

    def wap_pairs(level, value):
        d = ed[(ed.TaxonomyLevel == level) & (ed.TaxonomyValue == value)]
        d = d[d.DegreeInstitutionName != d.InstitutionName]
        return [(normalize_institution_name(a), normalize_institution_name(b))
                for a, b in zip(d.DegreeInstitutionName, d.InstitutionName)]

    rows = []
    # ---------- Academia level ----------
    wap = wap_ranks("Academia", "Academia"); wapk = set(wap.nk)
    op = [(a, b) for a, b in zip(win.deg_nk, win.emp_nk) if a in wapk and b in wapk and a and b]
    orcid_rho, orcid_n, sc_ac = spearman_vs_wapman(springrank_scores(op), wap)
    bench_rho, bench_n, _ = spearman_vs_wapman(springrank_scores(
        [(a, b) for a, b in wap_pairs("Academia", "Academia") if a in wapk and b in wapk]), wap)
    rows.append(dict(level="Academia (all fields)", orcid_edges=len(op), orcid_n=orcid_n,
                     orcid_vs_wapman=orcid_rho, public_wapman_benchmark=bench_rho))

    # ---------- per-field ----------
    for field, kw in FIELD_KW.items():
        wapf = wap_ranks("Field", field); wfk = set(wapf.nk)
        sub = win[win.field_text.str.contains(kw, na=False)]
        op = [(a, b) for a, b in zip(sub.deg_nk, sub.emp_nk) if a in wfk and b in wfk and a and b]
        if len(set(b for _, b in op)) < 8:
            rows.append(dict(level=field, orcid_edges=len(op), orcid_n=np.nan,
                             orcid_vs_wapman=np.nan, public_wapman_benchmark=np.nan)); continue
        o_rho, o_n, sc = spearman_vs_wapman(springrank_scores(op), wapf)
        b_rho, _, _ = spearman_vs_wapman(springrank_scores(
            [(a, b) for a, b in wap_pairs("Field", field) if a in wfk and b in wfk]), wapf)
        rows.append(dict(level=field, orcid_edges=len(op), orcid_n=o_n,
                         orcid_vs_wapman=o_rho, public_wapman_benchmark=b_rho))
        if field == "Computer Science":
            sc_cs = sc
    res = pd.DataFrame(rows)

    # ---------- figures ----------
    # academia scatter
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(-sc_ac.Rank, sc_ac.score, s=18, alpha=.6, c="#4878CF")
    ax.set_xlabel("Wapman prestige (−Rank; right = top)"); ax.set_ylabel("ORCID-AR SpringRank")
    ax.set_title(f"ORCID-AR vs Wapman — Academia (n={int(res.iloc[0].orcid_n)})\n"
                 f"Spearman={res.iloc[0].orcid_vs_wapman:.2f}  (public-Wapman benchmark "
                 f"{res.iloc[0].public_wapman_benchmark:.2f})")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "orcid_vs_wapman_academia.png", dpi=140)
    # CS scatter
    if "sc_cs" in dir():
        cs = res[res.level == "Computer Science"].iloc[0]
        fig2, ax2 = plt.subplots(figsize=(6.5, 6))
        ax2.scatter(-sc_cs.Rank, sc_cs.score, s=24, alpha=.7, c="#D6202A")
        ax2.set_xlabel("Wapman CS prestige (−Rank)"); ax2.set_ylabel("ORCID-AR SpringRank")
        ax2.set_title(f"ORCID-AR vs Wapman — Computer Science (n={int(cs.orcid_n)})\n"
                      f"Spearman={cs.orcid_vs_wapman:.2f} (benchmark {cs.public_wapman_benchmark:.2f})")
        fig2.tight_layout(); fig2.savefig(OUT / "figures" / "orcid_vs_wapman_cs.png", dpi=140)
    # edges by year
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    vc = e[(e.year >= 2005) & (e.year <= 2024)].year.astype(int).value_counts().sort_index()
    ax3.bar(vc.index, vc.values, color="#4878CF"); ax3.axvspan(2010.5, 2020.5, color="#ffe9b0", alpha=.5, zorder=0)
    ax3.set_xlabel("year (first faculty job)"); ax3.set_ylabel("ORCID PhD→faculty edges")
    ax3.set_title("ORCID PhD→faculty edges by year (shaded = Wapman 2011–2020 window)")
    fig3.tight_layout(); fig3.savefig(OUT / "figures" / "orcid_edges_by_year.png", dpi=140)

    res.to_csv(OUT / "orcid_validation_table.csv", index=False)
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    main()
