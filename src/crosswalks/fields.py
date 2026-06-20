"""Field <-> CIP <-> SDR crosswalk — the centralized join layer.

The **CIP code is the join key** across College Scorecard, SDR, O*NET and IPEDS.
This module is the single source of truth for the curated Tier-0 field set and how
each field maps onto:

  * ``wapman_field``  the field label as it appears in the Wapman et al. (2022)
                      Zenodo prestige tables (the academic-reputation source);
  * ``cip4``          4-digit CIP families used to filter College Scorecard
                      Field-of-Study earnings (stored zero-padded, no dot);
  * ``sdr_field``     the SDR field-of-degree label (salary + sector tables);
  * ``domain``        Wapman 8-domain grouping (for clustering / plots).

Names and CIP codes here are curated best-guesses based on CIP-2020 and the Wapman
taxonomy; they are **reconciled against the actual downloaded data in the Tier-0
notebook** via :func:`match_wapman_field` (fuzzy) and explicit verification. Do not
redefine field<->CIP maps ad hoc elsewhere — import from here.
"""
from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Curated Tier-0 field set (~20 fields spanning the full integration range:
# from fully integrated academic/industry markets (CS) to credential/periphery).
# ---------------------------------------------------------------------------
# Each cip4 is a 4-character, dot-free, zero-padded CIP-2020 family prefix.
TIER0_FIELDS: list[dict] = [
    # --- Mathematics & computing (expected LOW gap end) ---
    dict(key="computer_science", label="Computer Science",
         wapman_field="Computer Science", domain="Mathematics and computing",
         cip4=["1107", "1101", "1102", "1104"], sdr_field="Computer and information sciences"),
    dict(key="mathematics", label="Mathematics",
         wapman_field="Mathematics", domain="Mathematics and computing",
         cip4=["2701", "2703"], sdr_field="Mathematics and statistics"),
    dict(key="statistics", label="Statistics",
         wapman_field="Statistics", domain="Mathematics and computing",
         cip4=["2705"], sdr_field="Mathematics and statistics"),

    # --- Engineering (expected LOW-to-MID gap) ---
    dict(key="electrical_engineering", label="Electrical Engineering",
         wapman_field="Electrical Engineering", domain="Engineering",
         cip4=["1410"], sdr_field="Electrical and computer engineering"),
    dict(key="mechanical_engineering", label="Mechanical Engineering",
         wapman_field="Mechanical Engineering", domain="Engineering",
         cip4=["1419"], sdr_field="Mechanical engineering"),
    dict(key="civil_engineering", label="Civil Engineering",
         wapman_field="Civil Engineering", domain="Engineering",
         cip4=["1408"], sdr_field="Civil engineering"),
    dict(key="chemical_engineering", label="Chemical Engineering",
         wapman_field="Chemical Engineering", domain="Engineering",
         cip4=["1407"], sdr_field="Chemical engineering"),
    dict(key="materials_science", label="Materials Science",
         wapman_field="Materials Engineering", domain="Engineering",
         cip4=["1418", "4010"], sdr_field="Materials engineering"),

    # --- Natural sciences (expected MID gap) ---
    dict(key="physics", label="Physics",
         wapman_field="Physics, General", domain="Natural sciences",
         cip4=["4008"], sdr_field="Physics and astronomy"),
    dict(key="chemistry", label="Chemistry",
         wapman_field="Chemistry", domain="Natural sciences",
         cip4=["4005"], sdr_field="Chemistry"),
    dict(key="biology", label="Biology",
         wapman_field="Biological Sciences, General", domain="Natural sciences",
         cip4=["2601", "2602", "2603", "2604"], sdr_field="Biological sciences"),
    dict(key="earth_sciences", label="Earth Sciences",
         wapman_field="Geology", domain="Natural sciences",
         cip4=["4006"], sdr_field="Geosciences"),

    # --- Social sciences (expected MID-to-HIGH gap; economics->finance wedge) ---
    dict(key="economics", label="Economics",
         wapman_field="Economics, General", domain="Social sciences",
         cip4=["4506"], sdr_field="Economics"),
    dict(key="political_science", label="Political Science",
         wapman_field="Political Science", domain="Social sciences",
         cip4=["4510"], sdr_field="Political science"),
    dict(key="sociology", label="Sociology",
         wapman_field="Sociology", domain="Social sciences",
         cip4=["4511"], sdr_field="Sociology"),
    dict(key="anthropology", label="Anthropology",
         wapman_field="Anthropology", domain="Social sciences",
         cip4=["4502"], sdr_field="Anthropology"),
    dict(key="psychology", label="Psychology",
         wapman_field="Psychology, General", domain="Social sciences",
         cip4=["4201", "4228", "4227"], sdr_field="Psychology"),

    # --- Humanities (expected HIGH gap / periphery) ---
    dict(key="english", label="English",
         wapman_field="English Language and Literature", domain="Humanities",
         cip4=["2301"], sdr_field="English language and literature"),
    dict(key="history", label="History",
         wapman_field="History", domain="Humanities",
         cip4=["5401"], sdr_field="History"),
    dict(key="philosophy", label="Philosophy",
         wapman_field="Philosophy", domain="Humanities",
         cip4=["3801"], sdr_field="Philosophy"),
]

# Quick reference: the headline field whose low gap is hypothesis H4.
HEADLINE_FIELD = "computer_science"

# ---------------------------------------------------------------------------
# Field -> SED "historical broad field" crosswalk.
# The SED 2021 sector table (NSF 23-300, Table 2-6) reports the academia/
# industry/government split only at ~5 S&E broad fields + non-S&E. This maps each
# Tier-0 field onto its SED broad field so we can attach a (coarse) industry-share
# integration proxy. Replaced by a finer SDR table if one is available.
# ---------------------------------------------------------------------------
SED_BROAD_BY_KEY: dict[str, str] = {
    "computer_science": "Mathematics and computer sciences",
    "mathematics": "Mathematics and computer sciences",
    "statistics": "Mathematics and computer sciences",
    "electrical_engineering": "Engineering",
    "mechanical_engineering": "Engineering",
    "civil_engineering": "Engineering",
    "chemical_engineering": "Engineering",
    "materials_science": "Engineering",
    "physics": "Physical sciences and earth sciences",
    "chemistry": "Physical sciences and earth sciences",
    "earth_sciences": "Physical sciences and earth sciences",
    "biology": "Life sciences",
    "economics": "Psychology and social sciences",
    "political_science": "Psychology and social sciences",
    "sociology": "Psychology and social sciences",
    "anthropology": "Psychology and social sciences",
    "psychology": "Psychology and social sciences",
    "english": "Humanities and arts",
    "history": "Humanities and arts",
    "philosophy": "Humanities and arts",
}


def sed_broad_for(key: str) -> Optional[str]:
    """SED broad-field label for a Tier-0 field key (for the industry-share proxy)."""
    return SED_BROAD_BY_KEY.get(key)


# ---------------------------------------------------------------------------
# Field -> SDR fine-field crosswalk (the PREFERRED integration proxy).
# SDR 2021 Table 12-3 (NSF 23-319) gives Educational / Business-or-industry /
# Government employment COUNTS by ~98 fine fields of doctorate. industry_share =
# Business-or-industry / All employed. Covers SEH only -> the 3 humanities fields
# have no SDR row and fall back to SED_BROAD_BY_KEY ("Humanities and arts").
# Labels below verified present in nsf23319-tab012-003.xlsx with non-trivial cells.
# ---------------------------------------------------------------------------
SDR_FINE_BY_KEY: dict[str, str] = {
    "computer_science": "Computer science",
    "mathematics": "Mathematics",
    "statistics": "Statistics",
    "electrical_engineering": "Electrical, electronics, and communications engineering",
    "mechanical_engineering": "Mechanical engineering",
    "civil_engineering": "Civil engineering",
    "chemical_engineering": "Chemical engineering",
    "materials_science": "Metallurgical and materials engineering",
    "physics": "Physics",
    "chemistry": "Chemistry, except biochemistry",
    "earth_sciences": "Geological and earth sciences, geosciences",
    "biology": "Biological and biomedical sciences, general",
    "economics": "Economics",
    "political_science": "Political science and government",
    "sociology": "Sociology, demography, and population studies",
    "anthropology": "Anthropology",
    "psychology": "Psychology, general",
    # english / history / philosophy: not in SDR (SEH only) -> SED broad fallback
}


def sdr_fine_for(key: str) -> Optional[str]:
    """SDR fine-field-of-doctorate label for a Tier-0 field key (industry-share proxy).
    Returns None for the humanities fields not covered by SDR."""
    return SDR_FINE_BY_KEY.get(key)


def tier0_fields() -> list[dict]:
    """Return the curated Tier-0 field list (a copy)."""
    return [dict(f) for f in TIER0_FIELDS]


def field_keys() -> list[str]:
    return [f["key"] for f in TIER0_FIELDS]


def by_key(key: str) -> Optional[dict]:
    for f in TIER0_FIELDS:
        if f["key"] == key:
            return dict(f)
    return None


def cip4_prefixes(key: str) -> list[str]:
    """4-digit CIP prefixes for a field key (for filtering Scorecard CIPCODE)."""
    f = by_key(key)
    return list(f["cip4"]) if f else []


def normalize_field_name(s: str) -> str:
    """Lowercase, strip punctuation/'and'/'&', collapse whitespace — for matching."""
    s = (s or "").lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_wapman_field(name: str) -> Optional[dict]:
    """Match an observed Wapman field label to a curated field (exact-normalized,
    then substring). Returns the field dict or None. Used in Tier 0 to reconcile
    the curated ``wapman_field`` values against the real Zenodo field names."""
    target = normalize_field_name(name)
    if not target:
        return None
    # exact normalized match on the curated wapman_field
    for f in TIER0_FIELDS:
        if normalize_field_name(f["wapman_field"]) == target:
            return dict(f)
    # substring either direction (e.g. "computer science" in "computer sciences")
    for f in TIER0_FIELDS:
        wf = normalize_field_name(f["wapman_field"])
        if wf and (wf in target or target in wf):
            return dict(f)
    return None


def cip4_to_field_key() -> dict[str, str]:
    """Inverse map: every curated CIP-4 prefix -> field key (for tagging Scorecard rows)."""
    out: dict[str, str] = {}
    for f in TIER0_FIELDS:
        for c in f["cip4"]:
            out[c] = f["key"]
    return out
