"""OpenAlex field-tagging PILOT — 2,000 random ORCIDs from the placement population.

Measures Route A / Route B-rescue / union / residual hit rates BEFORE committing to the
full 63,711-ORCID run (per the guide: "先拿 2,000 个随机 ORCID 跑一遍"). Also reports the
OpenAlex 26-field distribution and a consistency check against the existing free-text
`field_text` heuristic, so we can judge whether OpenAlex tagging will actually change the
per-field cut. Outcome-agnostic.

 → data/interim/openalex_pilot.parquet + printed report
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from src.openalex_tag import tag_orcids

N_PILOT = 2000
SEED = 42

# coarse field_text-keyword -> plausible OpenAlex field(s); for a consistency diagnostic only
KW_EXPECT = {
    "computer": {"Computer Science"},
    "economic": {"Economics, Econometrics and Finance", "Social Sciences", "Business, Management and Accounting"},
    "mathematic": {"Mathematics"},
    "statistic": {"Mathematics", "Decision Sciences"},
    "physic": {"Physics and Astronomy"},
    "chemist": {"Chemistry", "Chemical Engineering", "Biochemistry, Genetics and Molecular Biology"},
    "psycholog": {"Psychology", "Social Sciences", "Neuroscience"},
    "sociolog": {"Social Sciences"},
    "political": {"Social Sciences"},
    "biolog": {"Agricultural and Biological Sciences", "Biochemistry, Genetics and Molecular Biology",
               "Immunology and Microbiology", "Neuroscience"},
    "mechanical engineer": {"Engineering", "Materials Science"},
    "electrical engineer": {"Engineering", "Computer Science", "Materials Science"},
    "civil engineer": {"Engineering", "Environmental Science"},
    "nursing": {"Nursing", "Health Professions", "Medicine"},
    "geolog": {"Earth and Planetary Sciences", "Environmental Science"},
}


def main():
    e = pd.read_parquet("data/interim/orcid_phd_faculty_edges.parquet")
    orcids = e.person_orcid.dropna().unique().tolist()
    rng = np.random.default_rng(SEED)
    sample = rng.choice(np.array(orcids), size=min(N_PILOT, len(orcids)), replace=False).tolist()
    print(f"population ORCIDs: {len(orcids)} | pilot sample: {len(sample)} (seed {SEED})\n")

    rows = tag_orcids(sample, do_route_b=True)
    df = pd.DataFrame(rows)
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    df.to_parquet("data/interim/openalex_pilot.parquet")

    N = len(df)
    nA = (df.route == "A").sum(); nB = (df.route == "B").sum(); nM = (df.route == "miss").sum()
    print("\n================  HIT RATES  ================")
    print(f"  Route A (ORCID->author)   : {nA:5d}  {nA/N:6.1%}")
    print(f"  Route B rescue (works)    : {nB:5d}  {nB/N:6.1%}")
    print(f"  Union (ID-based tagged)   : {nA+nB:5d}  {(nA+nB)/N:6.1%}")
    print(f"  Residual (miss -> name)   : {nM:5d}  {nM/N:6.1%}")
    print(f"  median works (route A)    : {df[df.route=='A'].n_works.median():.0f}")
    print(f"  median works (route B)    : {df[df.route=='B'].n_works.median():.0f}")

    print("\n================  OPENALEX FIELD DISTRIBUTION (tagged)  ================")
    vc = df[df.field.notna()].field.value_counts()
    for f, c in vc.items():
        print(f"  {f:<48} {c:4d}  {c/vc.sum():5.1%}")
    print(f"  distinct fields: {df.field.nunique()}")

    # ---- consistency vs free-text field_text on the same people ----
    ft = e.set_index("person_orcid").field_text
    df["field_text"] = df.orcid.map(lambda o: ft.get(o, ""))
    def expect(ftxt):
        s = set()
        for kw, fs in KW_EXPECT.items():
            if kw in (ftxt or ""):
                s |= fs
        return s
    df["expect"] = df.field_text.map(expect)
    chk = df[(df.field.notna()) & (df.expect.map(len) > 0)]
    agree = chk.apply(lambda r: r.field in r.expect, axis=1)
    print("\n================  CONSISTENCY vs field_text heuristic  ================")
    print(f"  rows with a determinable field_text discipline & an OpenAlex tag: {len(chk)}")
    if len(chk):
        print(f"  OpenAlex field ∈ field_text-expected set: {agree.mean():6.1%}")
        print("  (disagreements are where OpenAlex publication-topic field departs from the")
        print("   degree/dept free text — the signal we expect to sharpen the per-field cut)")

    print("\nsaved -> data/interim/openalex_pilot.parquet")
    return df


if __name__ == "__main__":
    main()
