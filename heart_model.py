"""Dataset loading and reproducible model pipeline."""

from functools import lru_cache

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = [name for name in FEATURES if name not in NUMERIC]


@lru_cache(maxsize=1)
def get_model():
    """Fetch UCI dataset 45, preprocess within CV-safe pipeline, and fit a classifier."""
    dataset = fetch_ucirepo(id=45)
    x = dataset.data.features.copy()
    y = dataset.data.targets.iloc[:, 0]
    x.columns = [str(column).strip().lower() for column in x.columns]
    x = x[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    valid = y.notna()
    x, y = x.loc[valid], (y.loc[valid] > 0).astype(int)

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ]
    )
    model = Pipeline([("preprocess", preprocessing), ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))])
    model.fit(x, y)
    return model
