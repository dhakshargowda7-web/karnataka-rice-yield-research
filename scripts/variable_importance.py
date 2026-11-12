"""XGBoost variable importance under each experimental condition.

    python scripts/variable_importance.py

For every train/test split used in the paper, an XGBoost pipeline is fitted on the training rows and
two importance measures are computed on the test rows:

* Permutation importance: the increase in MAE (kg/ha) when one input column is randomly shuffled
  (20 shuffles, averaged). District is shuffled as a single column, so its one-hot columns are
  treated as one variable.
* Grouped permutation importance: the same, but all variables in a group (district, the four
  rainfall variables, the four temperature variables, irrigation group, PMFBY indicator) are
  shuffled together with one shared permutation. The climate variables within each group are
  strongly correlated, so shuffling one at a time understates how much the model relies on them.
* SHAP: the mean absolute TreeSHAP contribution (kg/ha) of each input, from XGBoost's built-in
  pred_contribs. The contributions of the district one-hot columns are summed per row first.

Conditions: random 80:20 splits (30 seeds), walk-forward test years 2010-2023, irrigation with the
irrigation group (30 paired splits of the 79 observed rows, and train 2009 + 2011 / test 2017), and
PMFBY with the policy-era indicator (30 splits).
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from common import CLIMATE, RESULTS, SPLIT_SEEDS, TARGET, WALK_FORWARD_YEARS, load_data, make_pipeline

N_REPEATS = 20
GROUPS = {
    "District": ["district"],
    "Rainfall": [c for c in CLIMATE if c.startswith("rainfall")],
    "Temperature": [c for c in CLIMATE if c.startswith("temperature")],
    "Irrigation group": ["irrigation_group"],
    "PMFBY indicator": ["pmfby_policy_era"],
}
LABELS = {
    "district": "District", "irrigation_group": "Irrigation group", "pmfby_policy_era": "PMFBY indicator",
    "rainfall_annual_mm": "Annual rainfall", "rainfall_may_mm": "May rainfall", "rainfall_june_mm": "June rainfall",
    "rainfall_may_june_mm": "May–June rainfall", "temperature_annual_c": "Annual temperature",
    "temperature_may_c": "May temperature", "temperature_june_c": "June temperature",
    "temperature_may_june_c": "May–June temperature",
}


def importance(train, test, numeric, categorical, condition, split):
    cols = numeric + categorical
    pipe = make_pipeline("XGBoost", numeric, categorical)
    pipe.fit(train[cols], train[TARGET])

    perm = permutation_importance(pipe, test[cols], test[TARGET], scoring="neg_mean_absolute_error",
                                  n_repeats=N_REPEATS, random_state=0)

    # TreeSHAP contributions in the transformed (one-hot) space, summed back to input columns.
    pre, model = pipe.named_steps["preprocess"], pipe.named_steps["model"]
    Xt = pre.transform(test[cols])
    Xt = Xt.toarray() if hasattr(Xt, "toarray") else Xt
    contrib = model.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)[:, :-1]  # drop bias column
    names = pre.get_feature_names_out()
    row_contrib = {}  # per-row contribution of each input column (one-hot columns summed)
    for col in cols:
        prefix = f"num__{col}" if col in numeric else f"cat__{col}_"
        idx = [i for i, n in enumerate(names) if (n == prefix if col in numeric else n.startswith(prefix))]
        row_contrib[col] = contrib[:, idx].sum(axis=1)
    shap = {col: np.abs(v).mean() for col, v in row_contrib.items()}

    rows = [{"condition": condition, "split": split, "level": "variable", "variable": col, "n_test": len(test),
             "perm_mae_increase": perm.importances_mean[i], "mean_abs_shap": shap[col]}
            for i, col in enumerate(cols)]

    # Grouped permutation: shuffle every column of a group with the same row permutation.
    rng = np.random.default_rng(0)
    base_mae = mean_absolute_error(test[TARGET], pipe.predict(test[cols]))
    for group, members in GROUPS.items():
        members = [c for c in members if c in cols]
        if not members:
            continue
        increases = []
        for _ in range(N_REPEATS):
            shuffled = test[cols].copy()
            shuffled[members] = test[members].to_numpy()[rng.permutation(len(test))]
            increases.append(mean_absolute_error(test[TARGET], pipe.predict(shuffled)) - base_mae)
        rows.append({"condition": condition, "split": split, "level": "group", "variable": group,
                     "n_test": len(test), "perm_mae_increase": float(np.mean(increases)),
                     "mean_abs_shap": float(np.abs(sum(row_contrib[c] for c in members)).mean())})
    return rows


def main():
    df = load_data()
    base_num, base_cat = CLIMATE, ["district"]
    rows = []

    for seed in SPLIT_SEEDS:
        tr, te = train_test_split(df.index, test_size=0.20, random_state=seed)
        rows += importance(df.loc[tr], df.loc[te], base_num, base_cat, "Random split", seed)
        rows += importance(df.loc[tr], df.loc[te], base_num + ["pmfby_policy_era"], base_cat,
                           "PMFBY (with indicator)", seed)

    for year in WALK_FORWARD_YEARS:
        rows += importance(df[df.year < year], df[df.year == year], base_num, base_cat, "Walk-forward", year)

    irr = df[df.irrigation_group.notna()]
    irr_cat = base_cat + ["irrigation_group"]
    for seed in SPLIT_SEEDS:
        tr, te = train_test_split(irr.index, test_size=0.20, random_state=seed)
        rows += importance(irr.loc[tr], irr.loc[te], base_num, irr_cat, "Irrigation (with group)", seed)
    rows += importance(irr[irr.year.isin([2009, 2011])], irr[irr.year == 2017], base_num, irr_cat,
                       "Irrigation, train 2009+2011 / test 2017", 2017)

    imp = pd.DataFrame(rows)
    imp["label"] = imp.variable.map(LABELS).fillna(imp.variable)
    imp.to_csv(RESULTS / "xgboost_importance_runs.csv", index=False)

    # Walk-forward folds are weighted by test size so the mean equals the pooled MAE increase.
    def summarise(g):
        w = g.n_test
        return pd.Series({
            "perm_mae_increase_mean": np.average(g.perm_mae_increase, weights=w),
            "perm_mae_increase_q25": g.perm_mae_increase.quantile(0.25),
            "perm_mae_increase_q75": g.perm_mae_increase.quantile(0.75),
            "mean_abs_shap": np.average(g.mean_abs_shap, weights=w),
            "n_splits": len(g)})
    summary = (imp.groupby(["condition", "level", "variable", "label"]).apply(summarise, include_groups=False)
               .reset_index())
    by = summary.groupby(["condition", "level"])
    summary["perm_rank"] = by.perm_mae_increase_mean.rank(ascending=False).astype(int)
    summary["shap_rank"] = by.mean_abs_shap.rank(ascending=False).astype(int)
    summary = summary.sort_values(["condition", "level", "perm_rank"])
    summary.to_csv(RESULTS / "xgboost_importance_summary.csv", index=False)

    corr = df[CLIMATE].corr().round(2)
    corr.to_csv(RESULTS / "climate_correlations.csv")

    pd.set_option("display.width", 200)
    for (cond, level), g in summary.groupby(["condition", "level"], sort=False):
        print(f"\n== {cond} [{level}]")
        print(g[["label", "perm_mae_increase_mean", "perm_mae_increase_q25", "perm_mae_increase_q75",
                 "mean_abs_shap", "perm_rank", "shap_rank"]].round(1).to_string(index=False))
    print("\nClimate correlations\n", corr.to_string())


if __name__ == "__main__":
    main()
