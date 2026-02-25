# Maharashtra State-Protected Monuments Dataset

## Data Description

Geocoded inventory of 384 state-protected monuments and archaeological sites in Maharashtra, sourced from the Maharashtra State Archaeology Department's official division-wise monument lists. The dataset spans 33 districts across all 6 administrative divisions.

| Division | Districts | Monument Count |
|---|---|---|
| Ratnagiri | Palghar, Thane, Mumbai, Raigad, Ratnagiri, Sindhudurg | 44 |
| Nashik | Nandurbar, Dhule, Nashik, Jalgaon, Ahmednagar | 47 |
| Pune | Pune, Satara, Solapur, Sangli, Kolhapur | 40 |
| Aurangabad | Aurangabad, Jalna, Beed, Osmanabad | 94 |
| Nanded | Parbhani, Hingoli, Latur, Nanded | 78 |
| Nagpur | Buldhana, Akola, Amravati, Washim, Nagpur, Yavatmal, Bhandara, Chandrapur, Gondia | 81 |
| **Total** | **33** | **384** |

### Monument Types

| Type | Count | Description |
|---|---|---|
| Temple | ~175 | Hindu, Jain, and other religious structures |
| Fort | ~55 | Maratha, Mughal, and pre-colonial fortifications |
| Other | ~120 | Masjids, dargahs, palaces, gates, samadhi sites, wadas |
| Caves | ~15 | Buddhist, Jain, and Hindu rock-cut cave complexes |
| Archaeological Site | ~20 | Prehistoric, Neolithic, and Chalcolithic excavation sites |

## Data Source

**Maharashtra State Archaeology Department** — Division-wise official monument lists:
- Ratnagiri Division: 44 monuments
- Nashik Division: 49 monuments (47 with valid entries)
- Pune Division: 40 monuments
- Aurangabad Division: 93 monuments (94 entries, some multi-district)
- Nanded Division: 78 monuments
- Nagpur Division: 81 monuments

Source URL: Maharashtra State Archaeology & Museums Department official website, monument listing pages by division (accessed February 2026).

All listed monuments carry either "Final" notification (officially gazetted) or "First" notification (preliminary notification under the Maharashtra Ancient Monuments and Archaeological Sites and Remains Act).

## Geocoding Methodology

### Stage 1: Automated Geocoding (Nominatim)
All 384 sites submitted to OpenStreetMap Nominatim via `scripts/geocode_monuments.py`. Monument-specific query strategy prioritises the **place/village name** over the monument name, since many monuments share generic names (e.g., "Mahadev Temple" appears 15+ times):

For **Forts** (well-known landmarks):
1. `"{name}, {district} district, Maharashtra, India"`
2. `"{name}, {place}, {district}, Maharashtra, India"`
3. `"{place}, {taluka}, {district}, Maharashtra"`

For **Temples and Other**:
1. `"{name}, {place}, {district} district, Maharashtra, India"`
2. `"{place}, {taluka}, {district} district, Maharashtra, India"`
3. `"{name}, {district}, Maharashtra, India"`
4. `"{place}, {district}, Maharashtra, India"`

Results validated against Maharashtra bounding box (15.5-22.1N, 72.5-81.0E) with 0.5 degree margin.

### Stage 2: Manual Research
Failed sites researched individually using Wikipedia, Wikidata, Google Maps references, and archaeological survey publications.

### Stage 3: Verification
Random sample of 7 sites plotted against Maharashtra district boundaries to confirm spatial plausibility.

## File Structure

```
data/monuments/
  ahmednagar.csv       # 9 monuments
  akola.csv            # 5 monuments
  amravati.csv         # 4 monuments
  aurangabad.csv       # 40 monuments
  beed.csv             # 20 monuments
  bhandara.csv         # 3 monuments
  buldhana.csv         # 6 monuments
  chandrapur.csv       # 10 monuments
  dhule.csv            # 5 monuments
  gondia.csv           # 1 monument
  hingoli.csv          # 13 monuments
  jalgaon.csv          # 6 monuments
  jalna.csv            # 7 monuments
  kolhapur.csv         # 8 monuments
  latur.csv            # 5 monuments
  mumbai.csv           # 10 monuments
  nagpur.csv           # 46 monuments
  nanded.csv           # 34 monuments
  nandurbar.csv        # 1 monument
  nashik.csv           # 26 monuments
  osmanabad.csv        # 27 monuments
  palghar.csv          # 2 monuments
  parbhani.csv         # 26 monuments
  pune.csv             # 23 monuments
  raigad.csv           # 6 monuments
  ratnagiri.csv        # 21 monuments
  sangli.csv           # 2 monuments
  satara.csv           # 5 monuments
  sindhudurg.csv       # 3 monuments
  solapur.csv          # 2 monuments
  thane.csv            # 2 monuments
  washim.csv           # 4 monuments
  yavatmal.csv         # 2 monuments
  verification_map.png # Spatial verification plot
  MONUMENTS.md         # This file
```

## CSV Schema

All district files share a common schema:

| Column | Type | Description |
|---|---|---|
| `name` | str | Monument name (disambiguated with place suffix where needed) |
| `monument_type` | str | `Fort`, `Temple`, `Caves`, `Archaeological Site`, `Other` |
| `place` | str | Village or locality where monument is situated |
| `taluka` | str | Sub-district (taluka) administrative unit |
| `district` | str | District name |
| `notification_status` | str | `Final` (gazetted) or `First` (preliminary notification) |
| `lat` | float | Latitude (decimal degrees, WGS84) |
| `lon` | float | Longitude (decimal degrees, WGS84) |
| `geocode_status` | str | `ok`, `fallback`, `manual`, or `failed` |

## Limitations

1. **Point coordinates only.** Monuments are represented as single points, not polygonal footprints. For forts covering large areas, the coordinate represents the main entrance or central feature.
2. **Generic monument names.** Many temples share names (e.g., "Mahadev Temple"). Disambiguation relies on the `place` column. Geocoding accuracy depends on Nominatim's coverage of small villages.
3. **Aurangabad/Beed/Osmanabad concentration.** The Marathwada region has a high density of Nizam-era transferred monuments with relatively sparse online geocoding data.
4. **Archaeological sites.** Multiple sites at "Ter" (Osmanabad) and "Ambala Talav" (Nagpur) share similar coordinates since they are clustered within the same archaeological zone.

## Citation

> Maharashtra state-protected monuments inventory compiled from the Maharashtra State Archaeology & Museums Department official division-wise monument lists. Geocoordinates obtained via OpenStreetMap Nominatim API with manual verification. Dataset compiled February 2026 for the Maharashtra VIIRS Dark Sky Research project.
