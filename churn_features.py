"""Column definitions and preprocessing shared by the training notebook and the Streamlit app.

Keeping this in one place matters because the app has to rebuild exactly the frame
the pipelines were fitted on, and a CSV round-trip silently loses the dtypes.
"""

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"

# 'Area code' looks numeric but only ever holds 408, 415 or 510. It is a region
# label, so averaging or scaling it would be meaningless - encode it instead.
CATEGORICAL_COLS = ["State", "Area code"]

# Two-valued already; expanding these into dummies would just add a redundant column.
BINARY_COLS = ["International plan", "Voice mail plan"]

NUMERIC_COLS = [
    "Account length",
    "Number vmail messages",
    "Total day minutes",
    "Total day calls",
    "Total day charge",
    "Total eve minutes",
    "Total eve calls",
    "Total eve charge",
    "Total night minutes",
    "Total night calls",
    "Total night charge",
    "Total intl minutes",
    "Total intl calls",
    "Total intl charge",
    "Customer service calls",
]

FEATURE_COLS = CATEGORICAL_COLS + BINARY_COLS + NUMERIC_COLS

_TRUTHY = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}


def _to_flag(series):
    """Collapse a yes/no, true/false or 0/1 column to plain integers."""
    if is_bool_dtype(series):
        return series.astype(int)
    if is_numeric_dtype(series):
        return series.astype(int)

    mapped = series.astype(str).str.strip().str.lower().map(_TRUTHY)
    if mapped.isna().any():
        unexpected = sorted(set(series[mapped.isna()].astype(str)))[:5]
        raise ValueError(f"{series.name!r} has values that are not yes/no: {unexpected}")
    return mapped.astype(int)


def coerce_schema(frame):
    """Put a freshly read frame back into the dtypes the pipelines were fitted on.

    Safe to call twice - the app runs it on uploaded CSVs where pandas has already
    guessed 'Area code' back into an int64 and the plan columns back into strings.
    """
    restored = frame.copy()
    restored["State"] = restored["State"].astype(str).str.strip()
    restored["Area code"] = restored["Area code"].astype(str).str.strip()
    for col in BINARY_COLS:
        restored[col] = _to_flag(restored[col])
    return restored


def split_target(frame):
    """Return (features, labels) with churn as 1 and retention as 0."""
    prepared = coerce_schema(frame)
    return prepared[FEATURE_COLS], _to_flag(prepared[TARGET])


def missing_columns(frame):
    """Feature columns the app needs but an uploaded file does not have."""
    return [col for col in FEATURE_COLS if col not in frame.columns]


def build_preprocessor():
    """One ColumnTransformer, reused by every model so the comparison stays fair."""
    return ColumnTransformer(
        transformers=[
            ("scaled", StandardScaler(), NUMERIC_COLS),
            # handle_unknown keeps the app alive if an uploaded file is missing a state.
            # Dense output because GaussianNB cannot take a sparse matrix.
            ("encoded", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
            ("flags", "passthrough", BINARY_COLS),
        ],
        sparse_threshold=0,
    )


def load_raw(data_dir="data"):
    """Both shipped CSVs stacked back into the full 3,333-row dataset."""
    parts = [
        pd.read_csv(f"{data_dir}/churn-bigml-train.csv"),
        pd.read_csv(f"{data_dir}/churn-bigml-test.csv"),
    ]
    return pd.concat(parts, ignore_index=True)


# Display name -> file stem under model/. The notebook writes these, the app reads
# them back, so the ordering here is also the order shown in the app's dropdown.
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "kNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest (Ensemble)": "random_forest",
    "Gradient Boosting (Ensemble)": "gradient_boosting",
}
