"""Rebuild data/karnataka_rice_climate_irrigation_merged_2000_2023.csv from the files in raw/.

    python scripts/build_dataset.py           # rebuild and check it equals the committed file
    python scripts/build_dataset.py --write   # also overwrite data/ with the rebuilt file

Sources (see raw/README.md):
  rice_yield_final_harmonized.csv                 area, production, yield (ICAR-IIRR, districts harmonised)
  karnataka_chirps_2000_2023.csv                  CHIRPS rainfall over the district polygons
  karnataka_rice_climate_2000_2023_preprocessed.csv
                                                  base panel; the only surviving copy of the NASA POWER
                                                  temperature values, the static irrigation label and
                                                  the PMFBY era flag
  karnataka_rice_climate_irrigation_merged_audit_2000_2023.xlsx, sheet "Irrigation source data"
                                                  DES rice irrigated/unirrigated areas, 2009-10, 2011-12, 2017-18
"""
import sys

import numpy as np
import pandas as pd

from common import DATA, ROOT

RAW = ROOT / "raw"
KEY = ["district", "year"]
RAIN = ["rainfall_annual_mm", "rainfall_may_mm", "rainfall_june_mm", "rainfall_may_june_mm"]
TEMP = ["temperature_annual_c", "temperature_may_c", "temperature_june_c", "temperature_may_june_c"]
YIELD = ["Area (Ha)", "Production (Tonnes)", "Yield (Kg/ha)"]


def build():
    yld = (pd.read_csv(RAW / "rice_yield_final_harmonized.csv")
           .rename(columns={"District": "district", "Year": "year"}))
    chirps = pd.read_csv(RAW / "karnataka_chirps_2000_2023.csv")
    chirps["year"] = chirps["year"].astype(int)
    base = pd.read_csv(RAW / "karnataka_rice_climate_2000_2023_preprocessed.csv")

    # The base panel's yield and rainfall must be the raw yield and CHIRPS values.
    check = base.merge(yld, on=KEY, suffixes=("", "_raw")).merge(chirps[KEY + RAIN], on=KEY, suffixes=("", "_raw"))
    assert len(check) == len(base) == len(yld) == 636
    for col in YIELD + RAIN:
        assert np.allclose(check[col], check[col + "_raw"]), col

    # Yield rows define the panel; CHIRPS has all 648 district-years, so 12 rows are dropped here.
    panel = (yld[["State"] + KEY + YIELD]
             .merge(chirps[KEY + RAIN], on=KEY, how="left")
             .merge(base[KEY + TEMP + ["irrigation_group", "pmfby_policy_era"]]
                    .rename(columns={"irrigation_group": "legacy_static_irrigation_group"}), on=KEY, how="left"))

    irr = pd.read_excel(RAW / "karnataka_rice_climate_irrigation_merged_audit_2000_2023.xlsx",
                        sheet_name="Irrigation source data")
    irr = irr.assign(year=irr["source_report_year"].str[:4].astype(int)).rename(columns={
        "irr_area": "rice_irrigated_area_ha", "unirr_area": "rice_unirrigated_area_ha",
        "tot_area": "rice_irrigation_report_total_area_ha", "source_report_year": "irrigation_source_year"})
    irr_cols = ["rice_irrigated_area_ha", "rice_unirrigated_area_ha", "rice_irrigation_report_total_area_ha",
                "irrigated_share_pct", "irrigation_group", "irrigation_source_year", "irrigation_data_status"]
    panel = panel.merge(irr[KEY + irr_cols], on=KEY, how="left", indicator=True)
    panel["irrigation_match_status"] = np.where(panel.pop("_merge") == "both",
                                                "Observed irrigation record matched",
                                                "No irrigation observation for this district-year")
    order = ["State"] + KEY + YIELD + RAIN + TEMP + irr_cols + [
        "irrigation_match_status", "legacy_static_irrigation_group", "pmfby_policy_era"]
    return panel[order].sort_values(KEY).reset_index(drop=True)


def main():
    rebuilt = build()
    committed = pd.read_csv(DATA).sort_values(KEY).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, committed, check_dtype=False, check_exact=False, rtol=1e-9)
    print(f"Rebuilt {len(rebuilt)} rows x {rebuilt.shape[1]} columns from raw/; identical to {DATA.relative_to(ROOT)}")
    print("Observed irrigation rows:", rebuilt.irrigation_group.notna().sum(),
          rebuilt.irrigation_group.value_counts().to_dict())
    if "--write" in sys.argv:
        rebuilt.to_csv(DATA, index=False)
        print("Wrote", DATA.relative_to(ROOT))


if __name__ == "__main__":
    main()
