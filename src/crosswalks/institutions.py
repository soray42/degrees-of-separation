"""Institution-name normalization for joining Wapman prestige (named institutions)
to College Scorecard (``UNITID`` / ``INSTNM``).

The Wapman Zenodo tables key institutions by name; Scorecard keys by IPEDS ``UNITID``
with an ``INSTNM`` string. There is no shared numeric id, so the Tier-0 join is on a
normalized name. This module centralizes that normalization plus a small alias table
for the handful of names that differ in surface form across the two sources.

Keep this conservative: it is better to drop an ambiguous institution from the
within-field correlation than to mis-join it. The Tier-0 notebook reports match rates
and unmatched names so coverage is auditable.
"""
from __future__ import annotations

import re

# Common surface-form differences (normalized-key -> canonical normalized-key).
# Extend as the Tier-0 unmatched list reveals real cases.
_ALIASES: dict[str, str] = {
    "mit": "massachusetts institute of technology",
    "caltech": "california institute of technology",
    "uc berkeley": "university of california berkeley",
    "uc los angeles": "university of california los angeles",
    "ucla": "university of california los angeles",
    "penn": "university of pennsylvania",
    "upenn": "university of pennsylvania",
    "uiuc": "university of illinois urbana champaign",
    "university of illinois at urbana champaign": "university of illinois urbana champaign",
    "ohio state": "ohio state university",
    "the ohio state university": "ohio state university",
    "texas a m university": "texas a and m university",
    "suny stony brook": "stony brook university",
    "university of texas at austin": "university of texas austin",
    "university of michigan ann arbor": "university of michigan",
    "university of wisconsin madison": "university of wisconsin",
    "university of minnesota twin cities": "university of minnesota",
}

# Tokens stripped as noise during normalization.
_STOPWORDS = {"the", "at", "of", "and"}


def normalize_institution_name(name: str) -> str:
    """Return a normalized key for an institution name.

    Steps: lowercase; map '&'->'and'; drop punctuation; drop generic stopwords
    ('the','at','of','and'); collapse whitespace; apply the alias table. The result
    is a stable key to join Wapman names against Scorecard ``INSTNM``.
    """
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = s.replace("&", " and ")
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)        # drop punctuation
    s = re.sub(r"\s+", " ", s).strip()
    if s in _ALIASES:                          # alias on the pre-stopword form too
        s = _ALIASES[s]
    tokens = [t for t in s.split(" ") if t not in _STOPWORDS]
    key = " ".join(tokens)
    return _ALIASES.get(key, key)


def build_name_index(names) -> dict[str, str]:
    """Map normalized-key -> original name, for an iterable of institution names.
    Later duplicates are kept under their key (last wins); collisions are rare after
    normalization and surfaced by the caller comparing index size to input size."""
    return {normalize_institution_name(n): n for n in names if isinstance(n, str) and n.strip()}
