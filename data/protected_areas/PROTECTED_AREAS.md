# Maharashtra Protected Areas Dataset

## Data Description

Comprehensive geocoded inventory of Maharashtra's protected area network, compiled for spatial analysis of Artificial Light At Night (ALAN) impacts on ecologically sensitive zones. The dataset covers four categories under the Wildlife (Protection) Act, 1972:

| Category | Count | Total Area (sq km) |
|---|---|---|
| National Parks | 6 | ~1,273 |
| Tiger Reserves | 6 | ~3,775 (core) |
| Wildlife Sanctuaries | 50 | ~8,950 |
| Conservation Reserves | 17 | ~1,180 |
| **Total** | **79** | **~15,178** |

Coverage represents approximately 3% of Maharashtra's land area (~308,000 sq km).

## Data Sources

Primary sources used to compile the protected area listings:

1. **Maharashtra Forest Department Annual Report 2023-24** ([mahaforest.gov.in](https://mahaforest.gov.in/index.php/report/index/en)) - Official state-level inventory of all protected areas with gazetted areas and establishment years.

2. **National Tiger Conservation Authority (NTCA)** ([ntca.gov.in/tiger-reserves](https://ntca.gov.in/tiger-reserves)) - Tiger reserve boundaries, core/buffer area designations, and management status.

3. **Wildlife Trust of India** ([wti.org.in](https://www.wti.org.in)) - News reports on Nagzira-Nawegaon expansion (2025) and corridor designations.

4. **Wikipedia: List of Wildlife Sanctuaries of India** - Cross-referenced for area, establishment year, and district attribution. Used as secondary verification, not primary source.

5. **PMF IAS Maharashtra Protected Areas** ([pmfias.com](https://www.pmfias.com/maharashtra-national-parks)) - Summary tables with district and area data.

## Geocoding Methodology

Coordinates were obtained through a three-stage process:

### Stage 1: Automated Geocoding (Nominatim)
All 79 sites were submitted to OpenStreetMap Nominatim using the `scripts/geocode_protected_areas.py` script. Query strategies:
- Primary: `"{name}, {district} district, Maharashtra, India"`
- Fallback 1: `"{clean_name}, {district}, Maharashtra, India"` (category suffix removed)
- Fallback 2: `"{name}, Maharashtra, India"` (district dropped)

Results validated against Maharashtra bounding box (15.5-22.1N, 72.5-81.0E) with 0.5 degree margin.

**Automated results:** 13 ok, 38 fallback, 4 pre-provided, 24 failed (69.6% success rate).

### Stage 2: Manual Research
The 23 sites that failed automated geocoding were researched individually using:
- Wikipedia infoboxes and Wikidata entries (highest priority)
- BirdLife International / Ramsar site coordinates
- India Biodiversity Portal
- Nearest named village or tehsil headquarters for newly designated reserves

### Stage 3: Verification
Random sample of 7 sites plotted against Maharashtra district boundaries (GeoJSON) to confirm:
- All points fall within state borders
- Points cluster in expected ecological zones (Western Ghats, Vidarbha dry forests)
- District attributions are spatially consistent

## Coordinate Accuracy

| Geocode Status | Count | Accuracy |
|---|---|---|
| `ok` | 13 | High - primary query matched, bbox validated |
| `fallback` | 38 | Medium - secondary query matched, bbox validated |
| `provided` | 4 | High - from published literature with precise coords |
| `manual` | 24 | Variable - see notes below |

For manually geocoded sites:
- **High confidence (14 sites):** Coordinates from Wikipedia/Wikidata infoboxes or Ramsar/BirdLife databases with explicit lat/lon.
- **Medium confidence (7 sites):** Coordinates derived from nearest named settlement within or adjacent to the protected area.
- **Low confidence (3 sites):** Approximate tehsil/taluka centroids for newly designated conservation reserves (Alaldari, Chivatibari, Panhalgad) with limited online presence.

## File Structure

```
data/protected_areas/
  national_parks.csv          # 6 national parks
  tiger_reserves.csv          # 6 tiger reserves (includes core/buffer areas)
  wildlife_sanctuaries.csv    # 50 wildlife sanctuaries
  conservation_reserves.csv   # 17 conservation reserves
  verification_map.png        # Spatial verification plot
  PROTECTED_AREAS.md          # This file
```

## CSV Schema

All files share a common schema:

| Column | Type | Description |
|---|---|---|
| `name` | str | Official protected area name |
| `category` | str | `national_park`, `tiger_reserve`, `wildlife_sanctuary`, `conservation_reserve` |
| `district` | str | Primary district(s), comma-separated if spanning multiple |
| `area_sq_km` | float | Gazetted area in square kilometres |
| `establishment_year` | int | Year of official notification/establishment |
| `responsible_department` | str | Managing authority |
| `lat` | float | Latitude (decimal degrees, WGS84) |
| `lon` | float | Longitude (decimal degrees, WGS84) |
| `geocode_status` | str | `ok`, `fallback`, `provided`, `manual`, or `failed` |

Tiger reserves additionally include `core_area_sq_km` and `buffer_area_sq_km`.

## Temporal Coverage

Data is current as of early 2025:
- No new national parks or tiger reserves since 2014
- Kolamarka and Muktai Bhavani upgraded to wildlife sanctuaries in 2022
- Nagzira-Nawegaon area doubled to 559 sq km in 2025 notification
- Atpadi Conservation Reserve added in 2023
- Multiple conservation reserves added 2021-2022

## Limitations

1. **Point coordinates only:** These are centroid approximations, not polygon boundaries. Protected areas cover large spatial extents (e.g., Melghat Tiger Reserve core = 1,500 sq km).
2. **Conservation reserves:** Several newly designated reserves (post-2021) lack precise online geocoding data; coordinates are approximate.
3. **Multi-district sites:** For areas spanning multiple districts, the coordinate represents the approximate centre of the protected area, which may fall in one district.

## Citation

When using this dataset:

> Maharashtra protected areas inventory compiled from Maharashtra Forest Department Annual Reports (2023-24), NTCA tiger reserve records, and Wildlife Trust of India publications. Geocoordinates obtained via OpenStreetMap Nominatim API with manual verification from Wikipedia/Wikidata. Dataset compiled February 2026 for the Maharashtra VIIRS Dark Sky Research project.
