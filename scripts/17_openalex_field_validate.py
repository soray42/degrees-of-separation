"""Task — per-field ORCID-AR re-validation vs Wapman, OpenAlex tags vs field_text heuristic.

Replaces the noisy free-text `field_text` field assignment (scripts/11) with OpenAlex
publication-topic field tags (scripts/16 -> data/interim/orcid_field.parquet), then asks the
"顶上去" question: does principled field tagging push per-field ORCID-AR↔Wapman agreement up
from the 0.45–0.68 field_text baseline toward the ~0.77 public-Wapman ceiling?

For each cleanly 1:1-mappable field we report three SpringRank-vs-Wapman Spearmans on the
SAME Wapman published field ranks (2011–2020 window, canonical SpringRank α=0.5 binary):
  ρ_openalex   OpenAlex-tagged ORCID subnetwork
  ρ_fieldtext  field_text-keyword ORCID subnetwork (the scripts/11 baseline)
  ρ_benchmark  Wapman's OWN public field edges (the data ceiling, ~0.77)
plus edge / employer-node counts per side, since added coverage (OpenAlex tags edges with
empty/uninformative free text) is a candidate mechanism distinct from cleaner tagging.

Mechanical / Civil Engineering from the scripts/11 baseline are NOT separable at OpenAlex's
26-field grain (both collapse into "Engineering"); they are dropped from the head-to-head and
noted. Outcome-agnostic — report gains, nulls, and regressions plainly.

 → outputs/OPENALEX_FIELD_VALIDATION_RESULT table (printed) + data/interim/openalex_field_validation.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import springrank

from src.crosswalks.institutions import normalize_institution_name

ALPHA, BINARY, WIN = 0.5, True, (2011, 2020)

# (label, OpenAlex field set, Wapman field, field_text keyword)  — cleanly mappable fields
TESTS = [
    ("Computer Science", {"Computer Science"}, "Computer Science", "computer"),
    ("Mathematics",      {"Mathematics"},      "Mathematics",      "math"),
    ("Physics",          {"Physics and Astronomy"}, "Physics, General", "physics"),
    ("Psychology",       {"Psychology"},        "Psychology, General", "psycholog"),
    ("Economics",        {"Economics, Econometrics and Finance"}, "Economics, General", "econ"),
    ("Chemistry",        {"Chemistry"},         "Chemistry",        "chemist"),
    ("Biology",          {"Biochemistry, Genetics and Molecular Biology",
                          "Agricultural and Biological Sciences",
                          "Immunology and Microbiology"}, "Biological Sciences, General", "biolog"),
]


def sr_scores(pairs):
    """pairs: (src_name, dst_name). Binary presence adjacency; canonical SpringRank."""
    pairs = [(a, b) for a, b in pairs if a and b]
    nodes = sorted(set(s for s, _ in pairs) | set(d for _, d in pairs))
    if len(nodes) < 8:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, d in pairs:
        A[idx[s], idx[d]] = 1.0 if BINARY else A[idx[s], idx[d]] + 1
    m = springrank.SpringRank(alpha=ALPHA, inverse_temp_fit_warning=False); m.fit(A)
    return dict(zip(nodes, m.ranks))


def rho_vs_wap(scores, wap):
    """scores {nk:springrank}; wap DataFrame[nk,Rank]. Wapman Rank 0 = top -> corr score vs -Rank."""
    if not scores:
        return np.nan, 0
    s = pd.DataFrame({"nk": list(scores), "score": list(scores.values())}).merge(wap, on="nk")
    if len(s) < 8:
        return np.nan, len(s)
    return spearmanr(s.score, -s.Rank)[0], len(s)


def main():
    e = pd.read_parquet("data/interim/orcid_phd_faculty_edges.parquet")
    fld = pd.read_parquet("data/interim/orcid_field.parquet").set_index("orcid").field
    e["oa_field"] = e.person_orcid.map(fld)
    e["emp_nk"] = e.org_to_ror_name.map(normalize_institution_name)
    e["deg_nk"] = e.org_from_ror_name.map(normalize_institution_name)
    win = e[(e.year >= WIN[0]) & (e.year <= WIN[1])].copy()

    rk = pd.read_csv("data/raw/wapman2022/ranks.csv")
    ed = pd.read_csv("data/raw/wapman2022/edge_lists.csv")

    def wap_ranks(value):
        r = rk[(rk.TaxonomyLevel == "Field") & (rk.TaxonomyValue == value)].copy()
        r["nk"] = r.InstitutionName.map(normalize_institution_name)
        return r[["nk", "Rank"]]

    def wap_pairs(value, keep):
        d = ed[(ed.TaxonomyLevel == "Field") & (ed.TaxonomyValue == value)]
        d = d[d.DegreeInstitutionName != d.InstitutionName]
        return [(normalize_institution_name(a), normalize_institution_name(b))
                for a, b in zip(d.DegreeInstitutionName, d.InstitutionName)
                if normalize_institution_name(a) in keep and normalize_institution_name(b) in keep]

    rows = []
    for label, oaset, wapfield, kw in TESTS:
        wap = wap_ranks(wapfield); wk = set(wap.nk)

        # OpenAlex-tagged subnetwork
        oa = win[win.oa_field.isin(oaset)]
        oa_pairs = [(a, b) for a, b in zip(oa.deg_nk, oa.emp_nk) if a in wk and b in wk]
        rho_oa, n_oa = rho_vs_wap(sr_scores(oa_pairs), wap)

        # field_text-keyword subnetwork (scripts/11 baseline)
        ft = win[win.field_text.str.contains(kw, na=False)]
        ft_pairs = [(a, b) for a, b in zip(ft.deg_nk, ft.emp_nk) if a in wk and b in wk]
        rho_ft, n_ft = rho_vs_wap(sr_scores(ft_pairs), wap)

        # Wapman public-edge benchmark (ceiling)
        rho_bench, n_bench = rho_vs_wap(sr_scores(wap_pairs(wapfield, wk)), wap)

        rows.append(dict(
            field=label, wapman_field=wapfield,
            oa_edges=len(oa_pairs), oa_nodes=len(set(b for _, b in oa_pairs)), rho_openalex=rho_oa,
            ft_edges=len(ft_pairs), ft_nodes=len(set(b for _, b in ft_pairs)), rho_fieldtext=rho_ft,
            rho_benchmark=rho_bench,
            delta_oa_minus_ft=(rho_oa - rho_ft) if np.isfinite(rho_oa) and np.isfinite(rho_ft) else np.nan,
            ceiling_gap=(rho_bench - rho_oa) if np.isfinite(rho_bench) and np.isfinite(rho_oa) else np.nan,
        ))
    res = pd.DataFrame(rows)
    res.to_csv("data/interim/openalex_field_validation.csv", index=False)

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=========  PER-FIELD ORCID-AR vs WAPMAN — OpenAlex tags vs field_text  =========")
    print(res[["field", "oa_edges", "rho_openalex", "ft_edges", "rho_fieldtext",
               "rho_benchmark", "delta_oa_minus_ft", "ceiling_gap"]]
          .to_string(index=False, float_format=lambda x: f"{x:6.3f}"))

    fin = res.dropna(subset=["delta_oa_minus_ft"])
    print("\n  fields where OpenAlex > field_text :",
          f"{(fin.delta_oa_minus_ft > 0).sum()}/{len(fin)}",
          f"(median Δ = {fin.delta_oa_minus_ft.median():+.3f})")
    print(f"  mean ρ_openalex = {res.rho_openalex.mean():.3f} | "
          f"mean ρ_fieldtext = {res.rho_fieldtext.mean():.3f} | "
          f"mean ρ_benchmark = {res.rho_benchmark.mean():.3f}")
    print(f"  mean OA edges = {res.oa_edges.mean():.0f} | mean field_text edges = {res.ft_edges.mean():.0f}"
          f"  (coverage gain ×{res.oa_edges.sum()/max(1,res.ft_edges.sum()):.2f})")
    print("\n  NOTE: Mechanical/Civil Engineering (scripts/11 baseline) are not separable at")
    print("        OpenAlex 26-field grain (both -> 'Engineering'); dropped from head-to-head.")
    return res


if __name__ == "__main__":
    main()
