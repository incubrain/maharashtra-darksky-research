"""Protected-area-specific geocoding strategies."""

from src.geocoding.nominatim import geocode_with_fallback


def geocode_site(rate_limiter, name: str, district: str | None) -> dict:
    """Geocode a single protected area with fallback strategies.

    Returns dict with lat, lon, geocode_status.
    """
    clean = name.strip()
    for suffix in [
        " Wildlife Sanctuary", " National Park", " Tiger Reserve",
        " Conservation Reserve", " Bird Sanctuary",
    ]:
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)].strip()
            break

    strategies = []
    if district:
        primary_district = district.split(",")[0].strip()
        strategies.extend([
            f"{name}, {primary_district} district, Maharashtra, India",
            f"{clean}, {primary_district}, Maharashtra, India",
            f"{name}, Maharashtra, India",
        ])
    else:
        strategies.extend([
            f"{name}, Maharashtra, India",
            f"{clean}, Maharashtra, India",
        ])

    return geocode_with_fallback(rate_limiter, strategies)
