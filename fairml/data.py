"""Dataset loading, cleaning, splitting and preprocessing.

Dataset: UCI "Adult" (a.k.a. Census Income) - free, public, ~48k rows.
Task:    predict whether a person earns more than 50K USD/year.
Why:     it is the classic benchmark for algorithmic-fairness work because it
         contains a protected attribute (sex) and a well-documented gender gap.

Loading order (first one that works wins):
  1. data/adult.csv                       (cache of a previous download)
  2. data/adult.data                      (manual download from UCI, see HOW_TO_RUN.md)
  3. OpenML via scikit-learn              (automatic download, needs internet)
  4. UCI direct URL                       (automatic download, needs internet)
Use --synthetic to run the whole pipeline offline with generated data.
"""
from __future__ import annotations

import logging
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fairml import config as cfg

logger = logging.getLogger(__name__)

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
UCI_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income",
]


# ------------------------------------------------------------------ cleaning
def _clean(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Normalise column names/values and return FEATURES + TARGET only."""
    df = df.copy()
    df.columns = [c.strip().replace("-", "_") for c in df.columns]
    if target_col != cfg.TARGET:
        df = df.rename(columns={target_col: cfg.TARGET})

    # categorical columns: str, strip, '?' / NaN -> 'Unknown'
    for col in cfg.CATEGORICAL_FEATURES:
        s = df[col].astype(object).where(df[col].notna(), "Unknown")
        s = s.astype(str).str.strip().replace({"?": "Unknown", "nan": "Unknown", "": "Unknown"})
        df[col] = s

    for col in cfg.NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    target = df[cfg.TARGET].astype(str).str.strip().str.rstrip(".")
    df[cfg.TARGET] = (target == cfg.POSITIVE_LABEL).astype(int)

    df = df[cfg.FEATURES + [cfg.TARGET]].dropna().reset_index(drop=True)
    return df


# ------------------------------------------------------------------ loading
def _download_openml() -> pd.DataFrame:
    from sklearn.datasets import fetch_openml

    bunch = fetch_openml("adult", version=2, as_frame=True)
    raw = bunch.frame
    return _clean(raw, target_col="class")


def _download_uci() -> pd.DataFrame:
    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = cfg.DATA_DIR / "adult.data"
    logger.info("Downloading %s", UCI_URL)
    urllib.request.urlretrieve(UCI_URL, path)  # noqa: S310  # nosec B310 (fixed https URL)
    return _read_uci_file(path)


def _read_uci_file(path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=UCI_COLUMNS, skipinitialspace=True, na_values=["?"])
    return _clean(raw, target_col="income")


def load_adult() -> pd.DataFrame:
    cache = cfg.DATA_DIR / "adult.csv"
    if cache.exists():
        logger.info("Loading cached dataset %s", cache)
        return pd.read_csv(cache)

    manual = cfg.DATA_DIR / "adult.data"
    df = None
    if manual.exists():
        logger.info("Loading manually downloaded file %s", manual)
        df = _read_uci_file(manual)
    else:
        errors = []
        for name, fn in (("OpenML", _download_openml), ("UCI", _download_uci)):
            try:
                logger.info("Trying to download Adult dataset from %s ...", name)
                df = fn()
                break
            except Exception as exc:  # noqa: BLE001 - we want to try the next source
                errors.append(f"{name}: {exc}")
                logger.warning("%s download failed: %s", name, exc)
        if df is None:
            raise RuntimeError(
                "Could not download the Adult dataset automatically.\n  "
                + "\n  ".join(errors)
                + f"\nFix: download {UCI_URL} in your browser and save it as "
                f"{manual}  (or run with --synthetic to test offline)."
            )

    cfg.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    logger.info("Cached cleaned dataset to %s (%d rows)", cache, len(df))
    return df


def make_synthetic(n: int = 6000, seed: int = cfg.SEED) -> pd.DataFrame:
    """Generate Adult-like data with a built-in gender gap (for offline runs/tests)."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["Male", "Female"], size=n, p=[0.67, 0.33])
    age = np.clip(rng.normal(39, 13, n), 17, 90).round().astype(int)
    edu = np.clip(rng.normal(10, 2.5, n), 1, 16).round().astype(int)
    hours = np.clip(rng.normal(40, 11, n), 1, 99).round().astype(int)
    cap_gain = np.where(rng.random(n) < 0.08, rng.integers(500, 20000, n), 0)
    cap_loss = np.where(rng.random(n) < 0.04, rng.integers(200, 3000, n), 0)
    workclass = rng.choice(["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov", "Unknown"],
                           size=n, p=[0.70, 0.08, 0.07, 0.05, 0.04, 0.06])
    marital = rng.choice(["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"],
                         size=n, p=[0.46, 0.33, 0.14, 0.03, 0.04])
    occupation = rng.choice(["Prof-specialty", "Craft-repair", "Exec-managerial", "Adm-clerical", "Sales",
                             "Other-service", "Machine-op-inspct", "Tech-support", "Unknown"], size=n)
    relationship = np.where(marital == "Married-civ-spouse",
                            np.where(sex == "Male", "Husband", "Wife"),
                            rng.choice(["Not-in-family", "Own-child", "Unmarried"], size=n))
    race = rng.choice(["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"],
                      size=n, p=[0.85, 0.09, 0.03, 0.01, 0.02])
    country = rng.choice(["United-States", "Mexico", "India", "Philippines", "Germany", "Unknown"],
                         size=n, p=[0.89, 0.02, 0.02, 0.02, 0.01, 0.04])

    logit = (-8.0 + 0.04 * age + 0.33 * edu + 0.025 * hours
             + 0.00012 * cap_gain + 0.9 * (sex == "Male") + 1.1 * (marital == "Married-civ-spouse")
             + 0.5 * (occupation == "Exec-managerial") + 0.4 * (occupation == "Prof-specialty"))
    prob = 1 / (1 + np.exp(-logit))
    income = (rng.random(n) < prob).astype(int)

    return pd.DataFrame({
        "age": age, "education_num": edu, "capital_gain": cap_gain, "capital_loss": cap_loss,
        "hours_per_week": hours, "workclass": workclass, "marital_status": marital,
        "occupation": occupation, "relationship": relationship, "race": race, "sex": sex,
        "native_country": country, cfg.TARGET: income,
    })[cfg.FEATURES + [cfg.TARGET]]


# ------------------------------------------------------------------ split / preprocess
def split_data(df: pd.DataFrame, seed: int = cfg.SEED):
    """70 / 10 / 20 stratified split -> (train, val, test)."""
    strat = df[cfg.TARGET].astype(str) + "_" + df[cfg.PROTECTED_ATTRIBUTE]
    train_val, test = train_test_split(df, test_size=0.20, random_state=seed, stratify=strat)
    strat_tv = train_val[cfg.TARGET].astype(str) + "_" + train_val[cfg.PROTECTED_ATTRIBUTE]
    train, val = train_test_split(train_val, test_size=0.125, random_state=seed, stratify=strat_tv)
    return (d.reset_index(drop=True) for d in (train, val, test))


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), cfg.NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cfg.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def feature_baselines(train_df: pd.DataFrame) -> dict:
    """'Typical person' values (median / mode) used by the /explain endpoint."""
    base = {c: float(train_df[c].median()) for c in cfg.NUMERIC_FEATURES}
    base.update({c: str(train_df[c].mode().iloc[0]) for c in cfg.CATEGORICAL_FEATURES})
    return base
