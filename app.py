import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix,
                             classification_report)

st.set_page_config(page_title="Shopper Intent Classifier", page_icon="🛒", layout="wide")

TARGET = "Revenue"
LABELS = ["No purchase", "Purchase"]
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def to_int(series):
    # handles True/False booleans or "TRUE"/"FALSE" text
    if series.dtype == object:
        series = series.astype(str).str.strip().str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0})
    return series.astype(int)


def score(model, X, y):
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return pred, {
        "Accuracy": accuracy_score(y, pred),
        "AUC": roc_auc_score(y, proba),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
    }


# ---------------- Sidebar ----------------
st.sidebar.title("Controls")
choice = st.sidebar.selectbox("Choose a model", list(MODEL_FILES))
upload = st.sidebar.file_uploader("Upload test CSV (features + Revenue)", type="csv")
use_sample = False
if os.path.exists("test_data.csv"):
    use_sample = st.sidebar.checkbox("Use the bundled test_data.csv instead")
st.sidebar.caption("The CSV must contain every input feature and the Revenue column. "
                   "Models are pre-trained and are not retrained on your upload.")

# ---------------- Header ----------------
st.title("🛒 Online Shopper Purchase-Intent Classifier")
st.write("Pick a model, upload a test set, and see how well it predicts whether a "
         "browsing session ends in a purchase.")

if upload is not None:
    data = pd.read_csv(upload)
elif use_sample:
    data = pd.read_csv("test_data.csv")
else:
    st.info("Upload a CSV in the sidebar (or tick the bundled test data) to begin.")
    st.stop()

# ---------------- Validate ----------------
model = load_model(MODEL_FILES[choice])
required = list(model.feature_names_in_)
missing = [c for c in required + [TARGET] if c not in data.columns]
if missing:
    st.error(f"Your CSV is missing these columns: {missing}")
    st.stop()

try:
    X = data[required].copy()
    if "Weekend" in X.columns:
        X["Weekend"] = to_int(X["Weekend"])
    y = to_int(data[TARGET])
except Exception:
    st.error("Could not read Revenue/Weekend as 0/1 or True/False. Please check your file.")
    st.stop()

st.caption(f"{len(data):,} rows loaded · {y.mean():.1%} positive class")

pred, metrics = score(model, X, y)

tab1, tab2, tab3 = st.tabs(["Selected model", "All models compared", "Preview data"])

with tab1:
    st.subheader(choice)
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.3f}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=LABELS, yticklabels=LABELS, cbar=False, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(y, pred, target_names=LABELS,
                                       output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3))

with tab2:
    st.subheader("Same test data, every model")
    rows = []
    for name, path in MODEL_FILES.items():
        _, m = score(load_model(path), X, y)
        rows.append({"Model": name, **m})
    table = pd.DataFrame(rows).set_index("Model")
    st.dataframe(table.round(4))
    st.bar_chart(table[["Accuracy", "F1", "MCC"]])
    st.success(f"Highest MCC on this data: {table['MCC'].idxmax()}")

with tab3:
    st.dataframe(data.head(50))
