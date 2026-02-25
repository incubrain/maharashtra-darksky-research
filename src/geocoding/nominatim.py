"""
Nominatim geocoding with fallback strategies and bounding-box validation.

Shared by the monuments and protected-areas geocoding pipelines.
"""

from src import config

USER_AGENT = "maharashtra-darksky-research/1.0"
RATE_LIMIT_SECONDS = 1.1  # Slightly over 1s to respect Nominatim ToS
MAHARASHTRA_BBOX = config.MAHARASHTRA_BBOX


def is_in_maharashtra(lat: float, lon: float, margin: float = 0.5) -> bool:
    """Check whether *lat*/*lon* fall within the Maharashtra bounding box."""
    return (
        MAHARASHTRA_BBOX["south"] - margin <= lat <= MAHARASHTRA_BBOX["north"] + margin
        and MAHARASHTRA_BBOX["west"] - margin <= lon <= MAHARASHTRA_BBOX["east"] + margin
    )


def geocode_with_fallback(rate_limiter, strategies: list[str]) -> dict:
    """Try *strategies* in order; return the first result inside the bbox.

    Parameters
    ----------
    rate_limiter : callable
        A rate-limited geocoder (e.g. ``geopy.extra.rate_limiter.RateLimiter``
        wrapping ``Nominatim.geocode``).
    strategies : list[str]
        Ordered geocoding query strings to try.

    Returns
    -------
    dict with ``lat``, ``lon``, ``geocode_status`` (``"ok"``, ``"fallback"``,
    or ``"failed"``).
    """
    for i, query in enumerate(strategies):
        try:
            location = rate_limiter(query)
            if location is not None:
                lat, lon = location.latitude, location.longitude
                if is_in_maharashtra(lat, lon):
                    status = "ok" if i == 0 else "fallback"
                    return {
                        "lat": round(lat, 6),
                        "lon": round(lon, 6),
                        "geocode_status": status,
                    }
        except Exception as exc:
            print(f"    Geocode error for '{query}': {exc}")

    return {"lat": None, "lon": None, "geocode_status": "failed"}
