"""Shared settings and helpers for the Karnataka rice-yield experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "karnataka_rice_climate_irrigation_merged_2000_2023.csv"
RESULTS = ROOT / "results"

TARGET = "Yield (Kg/ha)"
CLIMATE = [
    "rainfall_annual_mm", "rainfall_may_mm", "rainfall_june_mm", "rainfall_may_june_mm",
    "temperature_annual_c", "temperature_may_c", "temperature_june_c", "temperature_may_june_c",
]
MODELS = ["Ridge", "Random Forest", "XGBoost"]
MODEL_SEED = 42           # random_state for Random Forest and XGBoost
SPLIT_SEEDS = range(30)   # one random 80:20 split per seed
WALK_FORWARD_YEARS = range(2010, 2024)


def make_estimator(name):
    """The same model settings are used in every experiment."""
    if name == "Ridge":
        return Ridge(alpha=1.0)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=300, random_state=MODEL_SEED, n_jobs=-1)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                            objective="reg:squarederror", random_state=MODEL_SEED, n_jobs=-1)
    raise ValueError(name)


def make_pipeline(model_name, numeric, categorical):
    """Imputation, scaling and one-hot encoding are fitted on the training rows only."""
    transformers = [("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                      ("scale", StandardScaler())]), numeric)]
    if categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                             categorical))
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", make_estimator(model_name))])


def fit_predict(model_name, train, test, numeric, categorical):
    pipe = make_pipeline(model_name, numeric, categorical)
    pipe.fit(train[numeric + categorical], train[TARGET])
    return pipe.predict(test[numeric + categorical])


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"R2": r2_score(actual, predicted),
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted)))}


def load_data():
    df = pd.read_csv(DATA)
    df["year"] = df["year"].astype(int)
    return df
