"""OpenAlex field-tagging — FULL placement population (63,711 ORCIDs), Route A only.

Pilot (scripts/15) verdict: Route A (ORCID->author->dominant field) covers ~80% of the
population; Route B (works?filter=author.orcid) rescues only ~0.6% (it is keyed on the same
ORCID->OpenAlex linkage as A, so it is near-redundant — the genuine residual lever is the
DOI path via the ORCID public API, deferred). So the full run is Route A only, reusing the
on-disk cache from the pilot. Residual (~20%) is tagged field=None, route='miss' for a later
low-confidence name+affiliation pass (quarantined, sensitivity-tested).

 → data/interim/orcid_field.parquet  (orcid, field, route, n_works, confidence)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

from src.openalex_tag import tag_orcids

WIN = (2011, 2020)


def main():
    e = pd.read_parquet("data/interim/orcid_phd_faculty_edges.parquet")
    orcids = e.person_orcid.dropna().unique().tolist()
    print(f"tagging {len(orcids)} ORCIDs (Route A only; cache reused)\n")

    rows = tag_orcids(orcids, do_route_b=False)
    df = pd.DataFrame(rows)
    df.to_parquet("data/interim/orcid_field.parquet")

    N = len(df); tagged = df.field.notna()
    print("\n================  FULL-POPULATION COVERAGE  ================")
    print(f"  ORCIDs                  : {N}")
    print(f"  Route A tagged          : {tagged.sum():6d}  {tagged.mean():6.1%}")
    print(f"  residual (miss)         : {(~tagged).sum():6d}  {(~tagged).mean():6.1%}")
    print(f"  distinct OpenAlex fields: {df.field.nunique()}")

    # coverage on the Wapman validation window (2011-2020), edge-weighted
    fmap = df.set_index("orcid").field
    e["oa_field"] = e.person_orcid.map(fmap)
    win = e[(e.year >= WIN[0]) & (e.year <= WIN[1])]
    print(f"\n  placement edges (all)   : {len(e)}, tagged {e.oa_field.notna().mean():.1%}")
    print(f"  placement edges {WIN[0]}-{WIN[1]}: {len(win)}, tagged {win.oa_field.notna().mean():.1%}")

    # per-field edge counts in the window — which fields have n>=threshold for SpringRank
    print(f"\n================  TAGGED PLACEMENT EDGES per OpenAlex FIELD ({WIN[0]}-{WIN[1]})  ===========")
    vc = win.oa_field.value_counts()
    for f, c in vc.items():
        ndeg = win[win.oa_field == f].org_to_ror_name.nunique()
        print(f"  {f:<48} edges {c:5d}   employer-nodes {ndeg:4d}")

    print("\nsaved -> data/interim/orcid_field.parquet")
    return df


if __name__ == "__main__":
    main()
