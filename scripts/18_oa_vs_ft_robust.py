"""Adversarial robustness for scripts/17: is OpenAlex's per-field shortfall a COVERAGE (n) effect
or a TAG-QUALITY (mis-alignment) effect? Hardened after adversarial review.

(1) SYMMETRIC equal-n: subsample BOTH subnetworks to the shared min edge count (paired, K seeds),
    report mean ρ each side and P(ρ_OA ≥ ρ_ft). (The earlier one-sided test only subsampled when
    field_text was larger, silently dropping Biology/Psychology — fixed here.)
(2) DE-LEAKED field_text: the raw `str.contains(kw)` baseline leaks cross-field degrees ("computer"
    catches electrical-&-computer engineering, "biolog" catches chemistry), inflating field_text's n
    and ρ. Re-run with competing-field exclusions and report the cleaner baseline.
(3) Chemistry leverage: how much of the headline mean gap is one pathological field.
(4) Research-vs-degree drift: where OpenAlex sends each degree cohort.

 → printed; data/interim/oa_vs_ft_robust.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import springrank
from src.crosswalks.institutions import normalize_institution_name

ALPHA, WIN, KSEED = 0.5, (2011, 2020), 200
# (label, OpenAlex fields, Wapman field, field_text keyword, competing-field exclusions for de-leak)
TESTS = [
    ("Computer Science", {"Computer Science"}, "Computer Science", "computer", ["electrical"]),
    ("Mathematics",      {"Mathematics"},      "Mathematics",      "math",     ["biolog"]),
    ("Physics",          {"Physics and Astronomy"}, "Physics, General", "physics", ["medical", "health"]),
    ("Psychology",       {"Psychology"},        "Psychology, General", "psycholog", ["educational"]),
    ("Economics",        {"Economics, Econometrics and Finance"}, "Economics, General", "econ", ["agricultur"]),
    ("Chemistry",        {"Chemistry"},         "Chemistry",        "chemist",  ["biochem", "engineer"]),
    ("Biology",          {"Biochemistry, Genetics and Molecular Biology",
                          "Agricultural and Biological Sciences",
                          "Immunology and Microbiology"}, "Biological Sciences, General", "biolog", ["chem"]),
]


def sr(pairs):
    pairs = [(a, b) for a, b in pairs if a and b]
    nodes = sorted(set(s for s, _ in pairs) | set(d for _, d in pairs))
    if len(nodes) < 8:
        return {}
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for s, d in pairs:
        A[idx[s], idx[d]] = 1.0
    m = springrank.SpringRank(alpha=ALPHA, inverse_temp_fit_warning=False); m.fit(A)
    return dict(zip(nodes, m.ranks))


def rho(scores, wap):
    if not scores:
        return np.nan
    s = pd.DataFrame({"nk": list(scores), "score": list(scores.values())}).merge(wap, on="nk")
    return spearmanr(s.score, -s.Rank)[0] if len(s) >= 8 else np.nan


def main():
    e = pd.read_parquet("data/interim/orcid_phd_faculty_edges.parquet")
    fld = pd.read_parquet("data/interim/orcid_field.parquet").set_index("orcid").field
    e["oa_field"] = e.person_orcid.map(fld)
    e["emp_nk"] = e.org_to_ror_name.map(normalize_institution_name)
    e["deg_nk"] = e.org_from_ror_name.map(normalize_institution_name)
    win = e[(e.year >= WIN[0]) & (e.year <= WIN[1])].copy()
    rk = pd.read_csv("data/raw/wapman2022/ranks.csv")

    def wap_ranks(v):
        r = rk[(rk.TaxonomyLevel == "Field") & (rk.TaxonomyValue == v)].copy()
        r["nk"] = r.InstitutionName.map(normalize_institution_name)
        return r[["nk", "Rank"]]

    rows, drift_lines = [], []
    for label, oaset, wapfield, kw, excl in TESTS:
        wap = wap_ranks(wapfield); wk = set(wap.nk)

        oa = win[win.oa_field.isin(oaset)]
        oap = [(a, b) for a, b in zip(oa.deg_nk, oa.emp_nk) if a in wk and b in wk]

        ftmask = win.field_text.str.contains(kw, na=False)
        ft = win[ftmask]
        ftp = [(a, b) for a, b in zip(ft.deg_nk, ft.emp_nk) if a in wk and b in wk]
        # de-leaked: drop rows that also mention a competing field keyword
        exmask = ftmask & ~win.field_text.str.contains("|".join(excl), na=False)
        ftd = win[exmask]
        ftdp = [(a, b) for a, b in zip(ftd.deg_nk, ftd.emp_nk) if a in wk and b in wk]

        rho_oa = rho(sr(oap), wap)
        rho_ft = rho(sr(ftp), wap)
        rho_ftd = rho(sr(ftdp), wap)

        # symmetric paired equal-n
        nmin = min(len(oap), len(ftp))
        po = np.array(oap, dtype=object); pf = np.array(ftp, dtype=object)
        rng = np.random.default_rng(0); oo, ff, wins = [], [], 0
        if nmin >= 8:
            for _ in range(KSEED):
                ro = rho(sr([tuple(x) for x in po[rng.choice(len(po), nmin, replace=False)]]), wap)
                rf = rho(sr([tuple(x) for x in pf[rng.choice(len(pf), nmin, replace=False)]]), wap)
                if np.isfinite(ro) and np.isfinite(rf):
                    oo.append(ro); ff.append(rf); wins += (ro >= rf)
        rows.append(dict(field=label, oa_edges=len(oap), ft_edges=len(ftp), ftd_edges=len(ftdp),
                         rho_oa=rho_oa, rho_ft=rho_ft, rho_ft_deleaked=rho_ftd,
                         eqn_min=nmin, rho_oa_eqn=np.mean(oo) if oo else np.nan,
                         rho_ft_eqn=np.mean(ff) if ff else np.nan,
                         p_oa_ge_ft_eqn=(wins / len(oo)) if oo else np.nan))

        ftppl = win[ftmask & win.oa_field.notna()]
        dist = ftppl.oa_field.value_counts(normalize=True)
        top = "; ".join(f"{k} {v:.0%}" for k, v in dist.head(3).items())
        drift_lines.append(f"  {label:<16} in-target {dist[dist.index.isin(oaset)].sum():.0%} | {top}")

    res = pd.DataFrame(rows)
    res.to_csv("data/interim/oa_vs_ft_robust.csv", index=False)
    pd.set_option("display.width", 220)

    print("\n=====  SYMMETRIC equal-n (subsample BOTH to shared min edges, 200 paired seeds)  =====")
    print(res[["field", "oa_edges", "ft_edges", "eqn_min", "rho_oa_eqn", "rho_ft_eqn", "p_oa_ge_ft_eqn"]]
          .to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
    fin = res.dropna(subset=["p_oa_ge_ft_eqn"])
    print(f"\n  OA matches-or-beats ft at matched n (P≥0.5) in: "
          f"{(fin.p_oa_ge_ft_eqn >= 0.5).sum()}/{len(fin)} fields")

    print("\n=====  DE-LEAKED field_text baseline (drop cross-field degree leakage)  =====")
    print(res[["field", "rho_oa", "rho_ft", "rho_ft_deleaked", "ft_edges", "ftd_edges"]]
          .to_string(index=False, float_format=lambda x: f"{x:6.3f}"))
    d = res.dropna(subset=["rho_oa", "rho_ft_deleaked"])
    print(f"\n  OA ≥ de-leaked field_text in: {(d.rho_oa >= d.rho_ft_deleaked).sum()}/{len(d)} fields")
    print(f"  mean ρ: OA {res.rho_oa.mean():.3f} | ft(raw) {res.rho_ft.mean():.3f} | "
          f"ft(de-leaked) {res.rho_ft_deleaked.mean():.3f}")

    print("\n=====  CHEMISTRY leverage on the headline  =====")
    res["delta"] = res.rho_oa - res.rho_ft
    noc = res[res.field != "Chemistry"]
    print(f"  mean(ρ_oa−ρ_ft) all 7 fields     : {res.delta.mean():+.3f}")
    print(f"  mean(ρ_oa−ρ_ft) excl. Chemistry  : {noc.delta.mean():+.3f}  (-> a wash)")
    print(f"  Chemistry share of total OA edge deficit: "
          f"{(res.loc[res.field=='Chemistry','ft_edges'].iat[0]-res.loc[res.field=='Chemistry','oa_edges'].iat[0])/(res.ft_edges.sum()-res.oa_edges.sum()):.0%}")

    print("\n=====  RESEARCH-vs-DEGREE DRIFT  =====")
    for ln in drift_lines:
        print(ln)
    return res


if __name__ == "__main__":
    main()
