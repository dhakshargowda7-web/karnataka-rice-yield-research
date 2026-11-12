# karnataka-rice-yield-research

District-level rice yield prediction in Karnataka (2000–2023) with Ridge Regression, Random Forest and XGBoost, comparing random train-test splitting with chronological walk-forward validation. The repository holds the data and code behind the paper *District-Level Rice Yield Prediction in Karnataka: Evaluating Machine Learning Models Under Different Validation Strategies*.

## Layout

```text
data/
  karnataka_rice_climate_irrigation_merged_2000_2023.csv   the merged panel used by every experiment
scripts/
  common.py            shared settings: features, model configuration, preprocessing, metrics
  run_experiments.py   runs all experiments and writes results/*.csv
  make_figures.py      draws the box plots in results/figures/ from those CSVs
results/               outputs of the two scripts (regenerated on every run)
```

## Data

`data/karnataka_rice_climate_irrigation_merged_2000_2023.csv` has 636 district-year rows for 27 harmonised districts (12 district-years are missing from the source and are not imputed). Columns:

- **Yield, area and production:** ICAR-IIRR Rice APY Dashboard (Area, Production & Yield Analytics). Ramanagara, Chikkaballapura, Yadgir and Vijayanagara are merged into Bengaluru Rural, Kolar, Kalaburagi and Ballari, and yield is recomputed as production divided by area.
- **Rainfall:** CHIRPS, extracted over district polygons (annual, May, June, May–June).
- **Temperature:** NASA POWER at district centroids (annual, May, June, May–June means).
- **Irrigation:** irrigated share of rice area from Karnataka Directorate of Economics and Statistics district reports, grouped as Rain-fed (<30%), Mixed (30–70%) or Irrigated (≥70%). It is directly observed for 79 rows (2009, 2011 and 2017). `legacy_static_irrigation_group` is an older fallback label and is **not** used as an observation.
- **`pmfby_policy_era`:** 0 for 2000–2015 and 1 for 2016–2023. It marks the policy period only, not district-level enrolment or claims.

Area and production are never used as predictors, because yield is derived from them.

## Experiments

Every experiment uses district identity (one-hot) and the eight climate variables as base predictors. Imputation, scaling and encoding are fitted on training rows only. The model settings are the same throughout:

| Model | Settings |
| --- | --- |
| Ridge Regression | `alpha=1.0` |
| Random Forest | 300 trees, `random_state=42` |
| XGBoost | 300 trees, `max_depth=3`, `learning_rate=0.05`, `random_state=42` |

1. **Random split:** 80:20, repeated over seeds 0–29.
2. **Walk-forward:** expanding window, test years 2010–2023 (376 predictions).
3. **Baselines** on the same 376 rows: district historical mean, and previous-year yield (which falls back to the historical mean when the previous year is missing).
4. **Irrigation** on the 79 observed rows: 30 paired random splits with and without `irrigation_group`, plus training on 2009 and 2011 and testing on 2017.
5. **PMFBY:** 30 paired random splits (508 train / 128 test) with and without `pmfby_policy_era`.

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_experiments.py
python scripts/make_figures.py
```

On macOS, XGBoost needs the OpenMP runtime (`brew install libomp`).

Ridge Regression and XGBoost results are deterministic. Random Forest results can differ in the third decimal place across operating systems and scikit-learn versions.
