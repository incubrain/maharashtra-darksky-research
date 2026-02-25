"""Monument-specific geocoding strategies."""

from src.geocoding.nominatim import geocode_with_fallback


def geocode_monument(
    rate_limiter,
    name: str,
    place: str,
    taluka: str,
    district: str,
    monument_type: str,
) -> dict:
    """Geocode a single monument with fallback strategies.

    Monuments often share generic names (Mahadev Temple, Shiva Temple)
    so the *place* (village) is the primary disambiguator.

    Returns dict with lat, lon, geocode_status.
    """
    is_fort = monument_type == "Fort"

    strategies = []
    if is_fort:
        strategies.extend([
            f"{name}, {district} district, Maharashtra, India",
            f"{name}, {place}, {district}, Maharashtra, India",
            f"{place}, {taluka}, {district}, Maharashtra",
        ])
    else:
        strategies.extend([
            f"{name}, {place}, {district} district, Maharashtra, India",
            f"{place}, {taluka}, {district} district, Maharashtra, India",
            f"{name}, {district}, Maharashtra, India",
            f"{place}, {district}, Maharashtra, India",
        ])

    return geocode_with_fallback(rate_limiter, strategies)
