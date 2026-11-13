# Raw source files

These are the files from the original project ([dhakshargowda7-web/karnataka-rice-yield-research](https://github.com/dhakshargowda7-web/karnataka-rice-yield-research), commit `e5e8787`) that were used to assemble `data/karnataka_rice_climate_irrigation_merged_2000_2023.csv`. They are kept unmodified, except that one file was renamed (noted below).

`scripts/build_dataset.py` rebuilds the merged dataset from these files and checks that the result is identical to the committed copy.

| File | Origin in the original repo | What it supplies |
| --- | --- | --- |
| `rice_yield_final_harmonized.csv` | `Karnataka_Rice_Research_Data_and_Code.zip` → `data/rice_yield_final_harmonized(2).csv` (renamed) | `Area (Ha)`, `Production (Tonnes)`, `Yield (Kg/ha)` for 636 district-years from the ICAR-IIRR Rice APY Dashboard, already harmonised to 27 districts |
| `karnataka_chirps_2000_2023.csv` | repo root | CHIRPS rainfall (annual, May, June, May–June) for all 648 district-years, exported from Google Earth Engine |
| `karnataka_districts_27.geojson` | repo root | The 27 harmonised district polygons the CHIRPS rainfall was extracted over |
| `karnataka_27_district_centroids.csv` | repo root | District centroids (latitude/longitude) used to request NASA POWER temperature |
| `karnataka_rice_climate_2000_2023_preprocessed.csv` | `Karnataka_Rice_Research_Data_and_Code.zip` → `data/` | The base rice-climate panel. It is the only surviving copy of the NASA POWER temperature values (annual, May, June, May–June). It also holds the static district irrigation label (kept as `legacy_static_irrigation_group`) and `pmfby_policy_era`. |
| `karnataka_rice_climate_irrigation_merged_audit_2000_2023.xlsx` | repo root | The sheet **Irrigation source data** has the 81 DES district records of irrigated and unirrigated rice area for 2009-10, 2011-12 and 2017-18; 79 of them match a panel row. The sheet **Read me** documents the join rules. The other sheets are audit summaries. |

## How the merged dataset is assembled

1. Start from the 636 yield rows.
2. Add CHIRPS rainfall by district and year. The 12 district-years with no yield record (Chikkamagaluru 2000–2010 and Bagalkot 2017) are dropped.
3. Add temperature, the static irrigation label and the PMFBY flag from the base panel.
4. Join the DES irrigation records on district and the first year of the fiscal-year label (2009-10 → 2009, 2011-12 → 2011, 2017-18 → 2017). Two records have no panel row (Chikkamagaluru 2009 and Bagalkot 2017), and Bagalkot 2017-18 reports zero rice area. All other district-years are left blank; none are imputed.

## Not available

These earlier-stage sources are not in the original repository, so they could not be included:

- The original ICAR-IIRR export before district harmonisation.
- The raw NASA POWER temperature export (the centroids above are the request locations).
- The DES district report PDFs the irrigation figures were taken from. Some early values were extracted by OCR from scanned reports.

## Not used

These files from the original repository are not inputs to the merged dataset, so they are not included here:

- `irrigation_observed_years_only_dataset.csv` and `karnataka_rice_climate_irrigation_merged_2000_2023_5YEAR_AUDITED.csv` belong to a separate five-year irrigation version (2001–2002 added).
- `karnataka_irrigation_category_stability_audit.xlsx` is a check of irrigation categories across years.
