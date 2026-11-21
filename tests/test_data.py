import numpy as np

from fairml import config as cfg
from fairml import data


def test_synthetic_schema_and_balance():
    df = data.make_synthetic(2000)
    assert list(df.columns) == cfg.FEATURES + [cfg.TARGET]
    assert df.isna().sum().sum() == 0
    assert set(df[cfg.TARGET].unique()) == {0, 1}
    # the synthetic data has a built-in gender gap
    rate = df.groupby("sex")[cfg.TARGET].mean()
    assert rate["Male"] > rate["Female"]


def test_split_sizes_and_no_overlap():
    df = data.make_synthetic(2000)
    tr, va, te = data.split_data(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert abs(len(te) / len(df) - 0.20) < 0.02
    assert len(tr) > len(va)


def test_preprocessor_handles_unseen_category():
    df = data.make_synthetic(500)
    pre = data.build_preprocessor().fit(df[cfg.FEATURES])
    row = df.head(1).copy()
    row["occupation"] = "Totally-New-Job"  # unseen at fit time
    X = pre.transform(row[cfg.FEATURES])
    assert np.isfinite(X).all()
    assert X.shape[1] == pre.transform(df.head(1)[cfg.FEATURES]).shape[1]


def test_clean_maps_missing_and_target():
    import pandas as pd

    raw = pd.DataFrame({
        "age": [30], "workclass": ["?"], "fnlwgt": [1], "education": ["x"], "education-num": [9],
        "marital-status": ["Never-married"], "occupation": [None], "relationship": ["Husband"],
        "race": ["White"], "sex": ["Male"], "capital-gain": [0], "capital-loss": [0],
        "hours-per-week": [40], "native-country": ["United-States"], "class": [">50K."],
    })
    out = data._clean(raw, target_col="class")
    assert out.loc[0, "workclass"] == "Unknown"
    assert out.loc[0, "occupation"] == "Unknown"
    assert out.loc[0, cfg.TARGET] == 1
