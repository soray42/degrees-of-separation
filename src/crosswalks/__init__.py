"""Centralized crosswalks. CIP is the join key across Scorecard, SDR, O*NET, IPEDS.

Import field/CIP/SDR mappings and institution-name normalization from here; never
redefine them ad hoc in notebooks or scripts.
"""
from .fields import (  # noqa: F401
    TIER0_FIELDS,
    HEADLINE_FIELD,
    tier0_fields,
    field_keys,
    by_key,
    cip4_prefixes,
    cip4_to_field_key,
    normalize_field_name,
    match_wapman_field,
)
from .institutions import (  # noqa: F401
    normalize_institution_name,
    build_name_index,
)
