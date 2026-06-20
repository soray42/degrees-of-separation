"""Academic-reputation loader → a SOURCE-NORMALIZED tidy AR table.

`compute_gap` (src/gap.py) must never see Wapman-specific columns, so every AR source
is normalized here to the same schema:

    inst_key | institution_id | institution_name | field | period | prestige_score

`prestige_score` is oriented so **higher = more prestigious** (we negate Wapman's `Rank`,
where 0 = best). `inst_key` is the normalized-name join key shared with the ER side.
`period` is carried now ("2011-2020") so Tier 1 can append an ORCID-rebuilt AR table
for 2021-2026 and the two-period test is a groupby — no refactor (proposal v2 §5.1, §7).
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .crosswalks import fields as F
from .crosswalks.institutions import normalize_institution_name

ROOT = Path(__file__).resolve().parents[1]
WAPMAN_RANKS = ROOT / "data" / "raw" / "wapman2022" / "ranks.csv"
WAPMAN_PERIOD = "2011-2020"


def load_ar_wapman(path: Path = WAPMAN_RANKS, period: str = WAPMAN_PERIOD) -> pd.DataFrame:
    """Wapman SpringRank (pooled decade) → normalized AR table over the Tier-0 fields.

    `Rank` is the SpringRank prestige ordinal (0 = most prestigious). One row per
    (institution, field). Restricted to `TaxonomyLevel == "Field"` and the curated fields.
    """
    r = pd.read_csv(path)
    r = r[r["TaxonomyLevel"] == "Field"].copy()
    wf2key = {f["wapman_field"]: f["key"] for f in F.TIER0_FIELDS}
    r = r[r["TaxonomyValue"].isin(wf2key)].copy()
    out = pd.DataFrame({
        "inst_key": r["InstitutionName"].map(normalize_institution_name),
        "institution_id": r["InstitutionId"].astype("string"),
        "institution_name": r["InstitutionName"].astype("string"),
        "field": r["TaxonomyValue"].map(wf2key),
        "period": period,
        "prestige_score": -r["Rank"].astype(float),   # higher = more prestigious
    })
    out = out[out["inst_key"] != ""].drop_duplicates(["inst_key", "field"])
    return out.reset_index(drop=True)
