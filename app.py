import os
import json
from typing import Optional
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Tourism Purchase Predictor", layout="centered")
st.title("Tourism Purchase Predictor")

# Configuration via hardcoded defaults or sidebar
HF_TOKEN = os.getenv("HF_TOKEN", None)  # optional, only needed for private repos

# Hard-coded defaults (override via sidebar)
MODEL_REPO = "critical12/tourism-purchase-predictor-rf"
DATASET_REPO = "critical12/tourism-dataset"

with st.sidebar:
    st.header("Configuration")
    MODEL_REPO = st.text_input("Model repo (owner/name)", value=MODEL_REPO, placeholder="owner/model-repo")
    DATASET_REPO = st.text_input("Dataset repo (owner/name)", value=DATASET_REPO, placeholder="owner/dataset-repo")

@st.cache_resource(show_spinner=True)
def load_model(repo_id: str, token: Optional[str]):
    model_path = hf_hub_download(repo_id=repo_id, filename="best_model.joblib", token=token)
    model = joblib.load(model_path)
    metrics = None
    try:
        metrics_path = hf_hub_download(repo_id=repo_id, filename="metrics.json", token=token)
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception:
        pass
    return model, metrics

@st.cache_data(show_spinner=True)
def load_dataset_schema(repo_id: str, token: Optional[str]):
    if not repo_id:
        return None
    try:
        csv_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="data/tourism.csv", token=token)
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    # Clean to infer schema and defaults
    df.columns = [c.strip() for c in df.columns]
    for drop_col in ("CustomerID", "ProdTaken"):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.drop_duplicates()
    miss = df.isna().mean()
    df = df.drop(columns=miss[miss > 0.95].index.tolist())
    nunique = df.nunique(dropna=True)
    df = df.drop(columns=nunique[nunique <= 1].index.tolist())
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]
    defaults = {}
    for c in num_cols:
        try:
            defaults[c] = float(df[c].median())
        except Exception:
            defaults[c] = 0.0
    choices = {}
    for c in cat_cols:
        vals = df[c].dropna().unique().tolist()
        if len(vals) > 50:
            vals = vals[:50]
        choices[c] = vals
        defaults[c] = vals[0] if vals else ""
    return {"num_cols": num_cols, "cat_cols": cat_cols, "defaults": defaults, "choices": choices}

# Guard if no repo provided
if not MODEL_REPO:
    st.info("Set the Model repo in the sidebar (owner/name) to get started.")
    st.stop()

# Load model
try:
    model, trained_metrics = load_model(MODEL_REPO, HF_TOKEN)
except Exception as e:
    st.error(f"Failed to load model from {MODEL_REPO}: {e}")
    st.stop()

# Try to get raw feature names from pipeline
raw_num_cols, raw_cat_cols = [], []
try:
    prep = getattr(model, 'named_steps', {}).get('prep')
    if prep is not None and hasattr(prep, 'transformers_'):
        for name, transformer, cols in prep.transformers_:
            if name == 'num':
                raw_num_cols = list(cols)
            elif name == 'cat':
                raw_cat_cols = list(cols)
except Exception:
    pass

schema = load_dataset_schema(DATASET_REPO, HF_TOKEN) if DATASET_REPO else None

# UI for inputs
st.subheader("Enter Customer Details")
with st.form("predict_form", clear_on_submit=False):
    values = {}
    if raw_num_cols or raw_cat_cols:
        num_cols = raw_num_cols
        cat_cols = raw_cat_cols
    elif schema:
        num_cols = schema["num_cols"]
        cat_cols = schema["cat_cols"]
    else:
        st.warning("Could not infer feature schema. Set DATASET_REPO or ensure model includes preprocessing.")
        num_cols, cat_cols = [], []

    for col in num_cols:
        default = float(schema["defaults"].get(col, 0.0)) if schema else 0.0
        values[col] = st.number_input(col, value=default)

    for col in cat_cols:
        if schema and schema["choices"].get(col):
            values[col] = st.selectbox(col, options=schema["choices"][col], index=0)
        else:
            values[col] = st.text_input(col, value=str(schema["defaults"].get(col, "")) if schema else "")

    submitted = st.form_submit_button("Predict")

# Build DataFrame and predict
if submitted:
    if not values:
        st.error("No inputs provided.")
        st.stop()
    input_df = pd.DataFrame([values])
    st.write("Input DataFrame:")
    st.dataframe(input_df)

    csv = input_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download inputs as CSV", data=csv, file_name="inputs.csv", mime="text/csv")

    try:
        y_pred = model.predict(input_df)
        proba = None
        if hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(input_df)[:, 1]
            except Exception:
                proba = None
        st.success(f"Prediction (ProdTaken): {int(y_pred[0])}")
        if proba is not None:
            st.info(f"Purchase probability: {float(proba[0]):.3f}")
    except Exception as e:
        st.error(f"Failed to predict with the provided inputs: {e}")

if trained_metrics:
    st.subheader("Trained Model Metrics (Test)")
    st.json(trained_metrics)
