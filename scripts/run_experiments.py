"""Run every experiment reported in the paper and write the results to results/.

    python scripts/run_experiments.py

Experiments
  1. Random 80:20 split, repeated over 30 seeds.
  2. Expanding-window walk-forward validation, test years 2010-2023 (376 predictions).
  3. Baselines on the same 376 walk-forward test rows: district historical mean and previous-year yield.
  4. Irrigation (79 rows with directly observed irrigation): 30 paired random splits with/without the
     irrigation group, and a temporal check (train 2009 + 2011, test 2017).
  5. PMFBY policy-era indicator: 30 paired random splits with/without the indicator.

All experiments use district identity (one-hot) and the eight climate variables as base predictors.
Year, area and production are never predictors.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (CLIMATE, MODELS, RESULTS, SPLIT_SEEDS, TARGET, WALK_FORWARD_YEARS, fit_predict,
                    load_data, metrics)

BASE_NUM, BASE_CAT = CLIMATE, ["district"]


def pooled(pred, keys):
    return (pred.groupby(keys)
            .apply(lambda g: pd.Series({"n": len(g), **metrics(g.actual, g.pred)}), include_groups=False)
            .reset_index())


def paired_change(runs, with_label, without_label):
    """Per-seed change (with minus without) and the number of runs in which each metric improved."""
    wide = runs.pivot_table(index=["seed", "model"], columns="condition", values=["R2", "MAE", "RMSE"])
    change = pd.DataFrame({m: wide[(m, with_label)] - wide[(m, without_label)]
                           for m in ["R2", "MAE", "RMSE"]}).reset_index()
    improved = change.assign(R2=change.R2 > 0, MAE=change.MAE < 0, RMSE=change.RMSE < 0)
    improved["all_three"] = improved[["R2", "MAE", "RMSE"]].all(axis=1)
    summary = change.groupby("model")[["R2", "MAE", "RMSE"]].mean().add_prefix("mean_change_")
    summary = summary.join(improved.groupby("model")[["R2", "MAE", "RMSE", "all_three"]].sum()
                           .add_prefix("runs_improved_"))
    return change, summary.reset_index()


def main():
    RESULTS.mkdir(exist_ok=True)
    df = load_data()
    print(f"rows={len(df)} districts={df.district.nunique()} years={df.year.min()}-{df.year.max()}")

    # 1. Random split -------------------------------------------------------------------------
    rows = []
    for seed in SPLIT_SEEDS:
        tr, te = train_test_split(df.index, test_size=0.20, random_state=seed)
        for m in MODELS:
            p = fit_predict(m, df.loc[tr], df.loc[te], BASE_NUM, BASE_CAT)
            rows.append({"seed": seed, "model": m, "n_test": len(te), **metrics(df.loc[te, TARGET], p)})
    random_runs = pd.DataFrame(rows)
    random_runs.to_csv(RESULTS / "random_split_runs.csv", index=False)
    random_summary = random_runs.groupby("model")[["R2", "MAE", "RMSE"]].agg(["mean", "std", "min", "max"])
    random_summary.to_csv(RESULTS / "random_split_summary.csv")

    # 2. Walk-forward -------------------------------------------------------------------------
    pred_rows, year_rows = [], []
    for year in WALK_FORWARD_YEARS:
        tr, te = df[df.year < year], df[df.year == year]
        for m in MODELS:
            p = fit_predict(m, tr, te, BASE_NUM, BASE_CAT)
            year_rows.append({"year": year, "model": m, "n_train": len(tr), "n_test": len(te),
                              **metrics(te[TARGET], p)})
            pred_rows += [{"year": year, "district": d, "model": m, "actual": a, "pred": q}
                          for d, a, q in zip(te.district, te[TARGET], p)]
    walk_pred, walk_years = pd.DataFrame(pred_rows), pd.DataFrame(year_rows)
    walk_pred.to_csv(RESULTS / "walk_forward_predictions.csv", index=False)
    walk_years.to_csv(RESULTS / "walk_forward_years.csv", index=False)
    walk_summary = pooled(walk_pred, ["model"]).merge(
        walk_years.groupby("model").R2.mean().rename("mean_yearly_R2").reset_index())
    walk_summary.to_csv(RESULTS / "walk_forward_summary.csv", index=False)

    # 3. Baselines on the same 376 test rows ------------------------------------------------------
    rows = []
    for year in WALK_FORWARD_YEARS:
        tr, te = df[df.year < year], df[df.year == year]
        hist_mean = te.district.map(tr.groupby("district")[TARGET].mean()).fillna(tr[TARGET].mean())
        previous = df[df.year == year - 1].set_index("district")[TARGET]
        # A district with no record in the previous year falls back to its historical mean.
        prev_year = te.district.map(previous).fillna(hist_mean)
        for name, p in [("District historical mean", hist_mean), ("Previous-year yield", prev_year)]:
            rows += [{"baseline": name, "year": year, "district": d, "actual": a, "pred": q}
                     for d, a, q in zip(te.district, te[TARGET], p)]
    base_pred = pd.DataFrame(rows)
    base_pred.to_csv(RESULTS / "baseline_predictions.csv", index=False)
    base_summary = pooled(base_pred, ["baseline"])
    base_summary.to_csv(RESULTS / "baseline_summary.csv", index=False)

    # 4. Irrigation -----------------------------------------------------------------------------
    irr = df[df.irrigation_group.notna()]
    conditions = [("Without irrigation", BASE_CAT), ("With irrigation", BASE_CAT + ["irrigation_group"])]
    rows = []
    for seed in SPLIT_SEEDS:
        tr, te = train_test_split(irr.index, test_size=0.20, random_state=seed)
        for m in MODELS:
            for cond, cat in conditions:
                p = fit_predict(m, irr.loc[tr], irr.loc[te], BASE_NUM, cat)
                rows.append({"seed": seed, "model": m, "condition": cond, "n_test": len(te),
                             **metrics(irr.loc[te, TARGET], p)})
    irr_runs = pd.DataFrame(rows)
    irr_runs.to_csv(RESULTS / "irrigation_random_runs.csv", index=False)
    irr_change, irr_summary = paired_change(irr_runs, "With irrigation", "Without irrigation")
    irr_change.to_csv(RESULTS / "irrigation_random_change.csv", index=False)
    irr_summary.to_csv(RESULTS / "irrigation_random_summary.csv", index=False)

    tr, te = irr[irr.year.isin([2009, 2011])], irr[irr.year == 2017]
    irr_temporal = pd.DataFrame([{"model": m, "condition": cond, "n_train": len(tr), "n_test": len(te),
                                  **metrics(te[TARGET], fit_predict(m, tr, te, BASE_NUM, cat))}
                                 for m in MODELS for cond, cat in conditions])
    irr_temporal.to_csv(RESULTS / "irrigation_temporal.csv", index=False)

    # 5. PMFBY ----------------------------------------------------------------------------------
    rows = []
    for seed in SPLIT_SEEDS:
        tr, te = train_test_split(df.index, test_size=0.20, random_state=seed)
        for m in MODELS:
            for cond, num in [("Without PMFBY", BASE_NUM), ("With PMFBY", BASE_NUM + ["pmfby_policy_era"])]:
                p = fit_predict(m, df.loc[tr], df.loc[te], num, BASE_CAT)
                rows.append({"seed": seed, "model": m, "condition": cond, "n_train": len(tr),
                             "n_test": len(te), **metrics(df.loc[te, TARGET], p)})
    pm_runs = pd.DataFrame(rows)
    pm_runs.to_csv(RESULTS / "pmfby_runs.csv", index=False)
    pm_means = pm_runs.groupby(["model", "condition"])[["R2", "MAE", "RMSE"]].mean().reset_index()
    pm_change, pm_summary = paired_change(pm_runs, "With PMFBY", "Without PMFBY")
    pm_means.to_csv(RESULTS / "pmfby_summary.csv", index=False)
    pm_summary.to_csv(RESULTS / "pmfby_change_summary.csv", index=False)

    # Report ------------------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\n1. Random split, 30 seeds\n", random_summary.round(3))
    print("\n2. Walk-forward 2010-2023\n", walk_summary.round(3).to_string(index=False))
    print("\n3. Baselines\n", base_summary.round(3).to_string(index=False))
    print(f"\n4a. Irrigation paired random splits (rows={len(irr)}, "
          f"groups={irr.irrigation_group.value_counts().to_dict()})\n", irr_summary.round(4).to_string(index=False))
    print("\n4b. Irrigation temporal\n", irr_temporal.round(4).to_string(index=False))
    print("\n5. PMFBY\n", pm_means.round(3).to_string(index=False))
    print(pm_summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
