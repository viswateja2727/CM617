# AURA: Assessment of Urban Regreening Areas (Bengaluru) — code and data

The AURA report (`AURA_report.docx`) with the code and data behind it. Folders follow the report's three research questions.

```
AURA/
├── AURA_report.docx            the 2-page report
├── heat_overview/              city-wide heat and greenness overview
├── rq1_greening_change/        RQ1: did places that greened between 2000 and 2025 cool relative to their surroundings?
├── rq2_park_cooling/           RQ2: did individual new parks cool their own sites?
├── rq3_parks_vs_heat/          RQ3: are BBMP parks located in the hotter and more populous wards?
├── supporting_green_vs_heat/   supporting check: green cover vs heat across places at one time
├── data/                       all inputs and outputs (raw/ holds the original OpenCity downloads)
└── figures/                    charts produced by the scripts
```

## Findings

- **RQ1. Places that greened cooled; places that lost green warmed.** About 25,000 points, Jan–Apr 2000–04 vs
  2022–26, each compared with its own past. Among points with little new building, places that greened (NDVI +0.10
  or more) cooled by about 1.2 °C relative to places that stayed the same, and places that lost green warmed by
  about 1.7 °C, each measured against its own 1 km neighbourhood. Each +0.1 NDVI goes with 0.8 °C cooling, holding
  building change fixed. The result holds when comparing only points that started alike, on 2 km blocks, and with
  a rank-based check that is insensitive to differences between Landsat sensors.
- **RQ2. A park only cooled where it added green.** Freedom Park, a former jail, cooled relative to its neighbourhood
  (−1.32 °C, p = 0.052; NDVI +0.138). JP Park and Kadugodi Tree Park, whose greenness barely changed, show no effect;
  the tree parks were set up inside existing forest or plantation land. JB Kaval Tree Park is excluded because the
  outline used covers only 2.2 ha of the 84-acre park.
- **RQ3. BBMP parks are concentrated in the cooler wards.** The coolest quarter of wards has 7.2 parks per ward on
  average, the hottest quarter 1.5 (20 of its 50 wards have none). Wards hold similar populations (~40,000 in 2011),
  so this is a fair comparison. Holding population and area fixed, each +1 °C of ward LST goes with about 41% fewer
  parks. Parks do not follow population. This is not circular: parks are too small to cool a ward (median 1.5% of
  ward area even at 1 ha each), park-rich wards are not greener (Spearman −0.03), and the pattern holds within inner,
  middle and outer rings of the city.
- **Supporting check.** Across places at one time, each +10% green cover goes with about 0.56 °C lower LST among
  squares of equal built-up share.

## How to run

- **Notebooks (`.ipynb`)** run in Google Colab with Earth Engine; set `PROJECT_ID` to your own. Their outputs are
  already in `data/`, so they do not need to be re-run.
- **Scripts (`.py`)** run locally from the `AURA/` folder (needs pandas, numpy, scipy, matplotlib):
  `python rq1_greening_change/change_analysis.py`, `python rq3_parks_vs_heat/ward_analysis.py`,
  `python supporting_green_vs_heat/gap_analysis.py`.

## Files

| Folder / file | What it does |
|---|---|
| `heat_overview/heat_overview_2024.ipynb` | City-wide LST, NDVI, NDBI and a heat-priority index, Landsat 9, Jan–May 2024 |
| `rq1_greening_change/regreening_change.ipynb` | Samples ~25,000 points; exports NDVI, LST, 1 km neighbourhood LST and water index for 2000–04 (Landsat 5/7) and 2022–26 (Landsat 8/9), and GHSL built-up share for 2000 and 2025 → `data/bengaluru_regreening_change.csv` |
| `rq1_greening_change/change_analysis.py` | RQ1 analysis with checks → `change_results.txt`, `figures/change_lst_vs_ndvi.png`, `figures/change_map.png` |
| `rq2_park_cooling/regreening_parks.ipynb` | Before/after cooling of four parks vs a 1–2 km neighbourhood ring → `data/all_parks_comparison.csv` |
| `rq2_park_cooling/site_selection.ipynb` | City-wide NDVI–LST model and its check against the measured parks. Its candidate-site ranking is not used in the report |
| `rq3_parks_vs_heat/ward_heat.ipynb` | Exports land-only average LST and green cover for each of the 198 wards (upload `data/bbmp_wards_198.geojson` when asked) → `data/bengaluru_ward_heat.csv` |
| `rq3_parks_vs_heat/ward_analysis.py` | RQ3 analysis with checks → `ward_results.txt`, `figures/ward_parks_by_heat.png` (plus a four-group map and scatter that are not part of the report) |
| `supporting_green_vs_heat/gap_analysis_grid.ipynb` | Exports average LST, NDVI, green cover (NDVI > 0.4), water and built-up share for 500 m squares → `data/bengaluru_grid_500m_v2.csv` |
| `supporting_green_vs_heat/gap_analysis.py` | Supporting check with robustness tests → `results.txt`, `figures/heat_by_green_within_builtup.png`, `figures/hot_squares_map.png` |

| Data file | Source |
|---|---|
| `data/bengaluru_regreening_change.csv` | Output of `regreening_change.ipynb` |
| `data/all_parks_comparison.csv` | Output of `regreening_parks.ipynb` |
| `data/bengaluru_ward_heat.csv` | Output of `ward_heat.ipynb` |
| `data/bengaluru_grid_500m_v2.csv` | Output of `gap_analysis_grid.ipynb` |
| `data/bengaluru_grid_500m.csv` | First grid export, kept only as evidence of the data problems below |
| `data/grid_with_scores.csv`, `data/ward_scores.csv` | Per-square and per-ward results from the scripts |
| `data/bbmp_wards_198.geojson` | Ward outlines converted from `raw/bbmp_wards_2015_198.kml` |
| `data/raw/census2011_wardwise.csv` | OpenCity: Bengaluru Ward-wise Census Data – 2011 |
| `data/raw/bbmp_wards_2015_198.kml` | OpenCity: BBMP Ward Map – 2015 (198 wards) |
| `data/raw/bbmp_parks_map_2023.kml` | OpenCity: BBMP Parks Map (2023) |

Not included: the newer OpenCity "Map of Parks Under BBMP" (2024) KML. It contains contractor names and phone
numbers, and is used only for the completeness check below. It is on data.opencity.in under "BBMP Parks".

## Data problems found and how they were handled

- **Satellite population is wrong for south and east Bengaluru.** GHS-POP 2025 and WorldPop 2020 both put about
  2,000 people/km² in Jayanagar and Koramangala, against about 46,000 in equally built-up Malleshwaram. Census 2011
  ward population is used instead. `site_selection.ipynb` uses GHS-POP for "residents within 500 m", so those
  values are unreliable for southern and eastern sites.
- **Dynamic World land cover misses street and garden trees** (46% of squares came out 0% green), so green cover is
  measured as NDVI > 0.4.
- **A first "need score" was circular** (it included built-up share, which competes with green for the same land).
  It was withdrawn; built-up share is only used to compare like with like.
- **Ward density is almost the same as ward size** (wards were drawn to hold equal populations), so parks-per-km² and
  parks-per-person comparisons are distorted. RQ3 uses raw park counts and compares wards of similar size.
- **The 2024 BBMP parks file is unusable** (half its locations fall outside Bengaluru). The 2023 list is used; it is
  incomplete (10 of the 34 zero-park wards have parks in the 2024 list).
- **A 2 km block check in RQ1 was first done on the neighbourhood-relative measure**, which averages out to almost
  zero over a block by construction. It now uses the city-relative measure.

## Park dates (checked against news and official sources, October 2026)

Freedom Park: jail moved out ~2000, work started 2006, opened November 2008 (matches the notebook's periods).
JP Park: ~85 acres; opening reported as 2006 by one travel guide only; start year not confirmed.
JB Kaval: 84-acre tree park within a 644-acre reserved forest; opening year not found.
Kadugodi: ~22-acre tree park carved from a 711-acre plantation; set to open around April 2021; not confirmed.

## Limitations

Associations, not controlled experiments · surface temperature, not air temperature · dry season only · RQ1 combines
Landsat 5/7 and 8/9 (handled with city-relative measures and a rank check) · population from 2011 · parks counted,
not measured by size · the parks list has no dates, so RQ3 describes where existing parks are, not whether recent
investment was targeted.
