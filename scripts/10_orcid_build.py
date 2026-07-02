"""Task 1 — construct ORCID PhD→faculty placement edges from the ORCID mobility edges.
 → data/interim/orcid_phd_faculty_edges.parquet (+ printed stats)

Population filter to match Wapman: doctoral EDUCATION → faculty EMPLOYMENT, US→US, both
orgs ROR-resolved. Per person: the FIRST faculty employment (earliest start year). Each such
mobility edge IS a PhD→faculty placement (org_from = doctoral institution, org_to = employer).
"""
import sys, glob, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

SHARDS = sorted(glob.glob("data/orcid/all/edge_aff/*.parquet"))
COLS = ["role_type_from", "role_type_to", "role_from", "role_to",
        "org_country_from", "org_country_to", "org_from_ror_id", "org_from_ror_name",
        "org_to_ror_id", "org_to_ror_name", "org_dept_from", "org_dept_to",
        "epi_start_year_to", "person_orcid"]
DOC = re.compile(r"ph\.?\s?d|d\.phil|dphil|sc\.?d|\bdoctor|doctoral", re.I)
FAC = re.compile(r"professor|lecturer|\bfaculty\b|assistant prof|associate prof|\breader\b|instructor|tenure|assoc\. prof|asst\. prof", re.I)


def main():
    parts = []
    for f in SHARDS:
        d = pd.read_parquet(f, columns=COLS)
        d = d[(d.role_type_from == "education") & (d.role_type_to == "employment") &
              (d.org_country_from == "us") & (d.org_country_to == "us") &
              d.org_from_ror_id.notna() & d.org_to_ror_id.notna()]
        if len(d):
            rf = d.role_from.fillna(""); rt = d.role_to.fillna("")
            d = d[rf.str.contains(DOC) & rt.str.contains(FAC)]
            if len(d):
                parts.append(d)
    e = pd.concat(parts, ignore_index=True)
    e["year"] = pd.to_numeric(e.epi_start_year_to, errors="coerce")
    e["field_text"] = (e.role_from.fillna("") + " | " + e.org_dept_from.fillna("") + " | " +
                       e.org_dept_to.fillna("")).str.lower()
    # first faculty employment per person (earliest year; ties -> first)
    e = e.sort_values("year").drop_duplicates("person_orcid", keep="first")
    Path("data/interim").mkdir(parents=True, exist_ok=True)
    keep = ["person_orcid", "org_from_ror_id", "org_from_ror_name", "org_to_ror_id",
            "org_to_ror_name", "year", "field_text", "role_from", "role_to"]
    e[keep].to_parquet("data/interim/orcid_phd_faculty_edges.parquet")

    print(f"US PhD->faculty placement edges (first faculty job/person): {len(e)}")
    print(f"distinct degree institutions: {e.org_from_ror_name.nunique()}, "
          f"employers: {e.org_to_ror_name.nunique()}")
    yr = e.year
    print(f"year range: {yr.min():.0f}-{yr.max():.0f}; in 2011-2020: {((yr>=2011)&(yr<=2020)).sum()}")
    print("\nedges by year (2005-2024):")
    vc = yr[(yr >= 2005) & (yr <= 2024)].astype(int).value_counts().sort_index()
    for y, c in vc.items():
        print(f"  {y}: {'#'*int(c/ max(1,vc.max())*40):<40} {c}")
    print("\ntop-10 employer ROR names (faculty destinations):")
    print(e.org_to_ror_name.value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
