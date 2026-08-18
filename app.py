"""Streamlit front-end for comparing the churn classifiers trained in model/churn_modelling.ipynb."""

import io
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_features import (
    FEATURE_COLS,
    MODEL_FILES,
    TARGET,
    coerce_schema,
    missing_columns,
)

MODEL_DIR = Path(__file__).parent / "model"

st.set_page_config(page_title="Telecom Churn Model Comparison", page_icon=None, layout="wide")


@st.cache_resource
def load_pipelines():
    """Unpickle every saved pipeline once per server process rather than once per rerun."""
    available = {}
    for name, stem in MODEL_FILES.items():
        path = MODEL_DIR / f"{stem}.pkl"
        if path.exists():
            available[name] = joblib.load(path)
    return available


@st.cache_data
def read_upload(payload):
    """Parse the uploaded CSV. Cached on the raw bytes so re-runs do not re-parse."""
    return pd.read_csv(io.BytesIO(payload))


def evaluate(actual, predicted, churn_probability):
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, churn_probability),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }


def churn_flags(series):
    """Uploaded label column may arrive as bool, as 'True'/'False' text, or as 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype.kind in "iuf":
        return series.astype(int)
    return series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "yes": 1, "no": 0})


pipelines = load_pipelines()

st.title("Telecom Churn - Model Comparison")
st.write(
    "Five classifiers from the assignment brief plus gradient boosting, all trained on the "
    "same 80/20 stratified split of the BigML telecom churn dataset. Upload the held-out "
    "test file to score them side by side."
)

if not pipelines:
    st.error(f"No .pkl files found in {MODEL_DIR}. Run model/churn_modelling.ipynb first.")
    st.stop()

st.sidebar.header("Controls")
compare_all = st.sidebar.checkbox("Compare all models", value=False)
chosen = st.sidebar.selectbox("Model", list(pipelines), disabled=compare_all)
threshold = st.sidebar.slider(
    "Churn decision threshold", 0.05, 0.95, 0.50, 0.05,
    help="A customer is flagged as churning when the predicted probability reaches this "
         "value. Lowering it catches more churners at the cost of more false alarms. "
         "AUC is threshold-independent and will not move.",
)
top_n = st.sidebar.slider("Customers in the risk list", 5, 50, 10, 5)
st.sidebar.caption(f"{len(pipelines)} pipelines loaded from model/")

uploaded = st.file_uploader("Test data (CSV)", type="csv")

if uploaded is None:
    st.info(
        "Upload **test_data.csv** from the repository root - that is the 667-customer "
        "held-out split written by the notebook, with its true `Churn` labels intact."
    )
    st.stop()

raw = read_upload(uploaded.getvalue())

absent = missing_columns(raw)
if absent:
    st.error("The uploaded file is missing these required columns: " + ", ".join(absent))
    st.stop()

# A CSV round-trip drops the dtypes the pipelines were fitted on - 'Area code' comes back as
# an integer and the plan columns as free text - so the frame is rebuilt before predicting.
features = coerce_schema(raw)[FEATURE_COLS]
has_labels = TARGET in raw.columns
actual = churn_flags(raw[TARGET]) if has_labels else None

profile = st.columns(3)
profile[0].metric("Customers", f"{len(raw):,}")
if has_labels:
    profile[1].metric("Churned", f"{int(actual.sum()):,}")
    profile[2].metric("Churn rate", f"{actual.mean():.1%}")
else:
    profile[1].metric("Churned", "n/a")
    profile[2].metric("Churn rate", "n/a")
    st.warning(
        f"No `{TARGET}` column in this file, so the metrics and confusion matrix are skipped. "
        "The risk ranking below still works."
    )

st.divider()

if has_labels and compare_all:
    st.subheader("All models on this file")
    table = {}
    for name, pipeline in pipelines.items():
        probability = pipeline.predict_proba(features)[:, 1]
        table[name] = evaluate(actual, (probability >= threshold).astype(int), probability)

    scores = pd.DataFrame(table).T.round(4).sort_values("F1", ascending=False)
    st.dataframe(scores, width="stretch")

    leader = scores.index[0]
    st.success(
        f"Best F1 at a {threshold:.2f} threshold: **{leader}** "
        f"(F1 {scores.loc[leader, 'F1']:.4f}, MCC {scores.loc[leader, 'MCC']:.4f})"
    )
    st.caption(
        "Precision and recall move as the threshold slider changes; AUC does not, because it "
        "is computed across every threshold at once."
    )

elif has_labels:
    probability = pipelines[chosen].predict_proba(features)[:, 1]
    predicted = (probability >= threshold).astype(int)
    scores = evaluate(actual, predicted, probability)

    st.subheader(f"{chosen} - evaluation metrics")
    cards = st.columns(6)
    for card, (label, value) in zip(cards, scores.items()):
        card.metric(label, f"{value:.4f}")

    left, right = st.columns([1, 1.3])

    with left:
        st.markdown("**Confusion matrix**")
        matrix = confusion_matrix(actual, predicted)
        figure, axes = plt.subplots(figsize=(4, 3.2))
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes,
            xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"],
        )
        axes.set_xlabel("Predicted")
        axes.set_ylabel("Actual")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        missed = int(matrix[1, 0])
        false_alarms = int(matrix[0, 1])
        st.caption(
            f"{missed} churners missed, {false_alarms} loyal customers wrongly flagged. "
            "Missing a churner usually costs more than a wasted retention offer, so the "
            "bottom-left cell is the one to push down with the threshold slider."
        )

    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            actual, predicted, target_names=["Retained", "Churned"], zero_division=0
        )
        st.code(report, language="text")

else:
    probability = pipelines[chosen].predict_proba(features)[:, 1]
    predicted = (probability >= threshold).astype(int)

st.divider()

st.subheader(f"Highest churn risk - top {top_n}")
model_for_risk = chosen if not compare_all else list(pipelines)[0]
risk_probability = pipelines[model_for_risk].predict_proba(features)[:, 1]

context_cols = [
    col for col in ["State", "Account length", "International plan",
                    "Customer service calls", "Total day minutes"]
    if col in raw.columns
]
risk = raw[context_cols].copy()
risk["Churn probability"] = risk_probability.round(3)
if has_labels:
    risk["Actually churned"] = raw[TARGET].values

st.dataframe(
    risk.sort_values("Churn probability", ascending=False).head(top_n),
    width="stretch",
)
st.caption(f"Ranked by {model_for_risk}. This is the list a retention team would work through.")
