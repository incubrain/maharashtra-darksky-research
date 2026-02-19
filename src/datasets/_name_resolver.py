"""
District name normalization and fuzzy matching.

Resolves name mismatches between VNL shapefile district names (from
datta07/INDIAN-SHAPEFILES) and external dataset names (Census, AQI, etc.).

Strategy:
1. Normalize both sides: lowercase, strip whitespace, remove diacritics
2. Try exact match
3. If no match, use an explicit override mapping (curated for known mismatches)
4. If still no match, use difflib.get_close_matches() with threshold 0.8
5. Log all mappings for audit trail
"""

import difflib
import logging
import unicodedata

from src.logging_config import get_pipeline_logger

log = get_pipeline_logger(__name__)

# Known overrides: VNL shapefile name (normalized) -> Census/external name (normalized)
# These are manually curated for Maharashtra districts.
DISTRICT_NAME_OVERRIDES = {
    "bid": "beed",
    "gondiya": "gondia",
    "raigarh": "raigad",
    "mumbai suburban": "mumbai (suburban)",
}

# Reverse mapping (external -> VNL) built automatically
_REVERSE_OVERRIDES = {v: k for k, v in DISTRICT_NAME_OVERRIDES.items()}


def normalize_name(name: str) -> str:
    """Normalize a district name for matching.

    Lowercases, strips whitespace, removes diacritics/accents, and
    collapses multiple spaces.
    """
    if not isinstance(name, str):
        return ""
    # Remove diacritics
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase, strip, collapse whitespace
    return " ".join(ascii_only.lower().split())


def resolve_names(
    vnl_names: list[str],
    dataset_names: list[str],
    fuzzy_threshold: float = 0.8,
) -> tuple[dict[str, str], list[str]]:
    """Match dataset district names to VNL district names.

    Parameters
    ----------
    vnl_names : list[str]
        District names from the VNL shapefile / pipeline.
    dataset_names : list[str]
        District names from the external dataset.
    fuzzy_threshold : float
        Minimum similarity ratio for fuzzy matching (0.0-1.0).

    Returns
    -------
    mapping : dict[str, str]
        Maps dataset name -> VNL name for matched districts.
    unmatched : list[str]
        Dataset names that could not be matched.
    """
    # Build normalized lookup: normalized_vnl_name -> original_vnl_name
    vnl_lookup = {}
    for name in vnl_names:
        vnl_lookup[normalize_name(name)] = name

    mapping = {}  # dataset_original -> vnl_original
    unmatched = []

    for ds_name in dataset_names:
        ds_norm = normalize_name(ds_name)

        # 1. Exact match on normalized names
        if ds_norm in vnl_lookup:
            mapping[ds_name] = vnl_lookup[ds_norm]
            log.debug("Name match (exact): '%s' -> '%s'", ds_name, vnl_lookup[ds_norm])
            continue

        # 2. Check overrides (dataset_norm -> vnl_norm)
        if ds_norm in _REVERSE_OVERRIDES:
            vnl_norm = _REVERSE_OVERRIDES[ds_norm]
            if vnl_norm in vnl_lookup:
                mapping[ds_name] = vnl_lookup[vnl_norm]
                log.debug(
                    "Name match (override): '%s' -> '%s'", ds_name, vnl_lookup[vnl_norm]
                )
                continue

        # Also check forward overrides (vnl_norm -> dataset_norm)
        for vnl_norm_key, ds_norm_val in DISTRICT_NAME_OVERRIDES.items():
            if ds_norm == ds_norm_val and vnl_norm_key in vnl_lookup:
                mapping[ds_name] = vnl_lookup[vnl_norm_key]
                log.debug(
                    "Name match (override fwd): '%s' -> '%s'",
                    ds_name,
                    vnl_lookup[vnl_norm_key],
                )
                break
        else:
            # 3. Fuzzy match
            vnl_norm_list = list(vnl_lookup.keys())
            matches = difflib.get_close_matches(
                ds_norm, vnl_norm_list, n=1, cutoff=fuzzy_threshold
            )
            if matches:
                best = matches[0]
                mapping[ds_name] = vnl_lookup[best]
                log.info(
                    "Name match (fuzzy %.2f): '%s' -> '%s'",
                    difflib.SequenceMatcher(None, ds_norm, best).ratio(),
                    ds_name,
                    vnl_lookup[best],
                )
            else:
                unmatched.append(ds_name)
                log.warning("Name unmatched: '%s' (normalized: '%s')", ds_name, ds_norm)

    log.info(
        "Name resolution: %d matched, %d unmatched out of %d dataset names",
        len(mapping),
        len(unmatched),
        len(dataset_names),
    )
    return mapping, unmatched
