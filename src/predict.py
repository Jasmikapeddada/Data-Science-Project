"""
Unified inference: preprocess → segment → churn score.
Used by Streamlit app and batch scripts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.churn_model import assign_risk_band, load_churn_config
from src.preprocess import (
    MODELS_DIR,
    add_features,
    clean_dataframe,
    get_feature_matrix,
    load_feature_config,
)
from src.segmentation import get_segmentation_matrix, load_segmentation_config
from src.schema import apply_column_mapping, is_telco_compatible, validate_for_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "monitoring" / "logs"


@lru_cache(maxsize=1)
def load_models() -> dict:
    """Load all joblib artifacts (cached for Streamlit)."""
    return {
        "preprocessor": joblib.load(MODELS_DIR / "preprocess_pipeline.joblib"),
        "churn": joblib.load(MODELS_DIR / "churn_model.joblib"),
        "kmeans": joblib.load(MODELS_DIR / "kmeans_model.joblib"),
        "seg_scaler": joblib.load(MODELS_DIR / "segmentation_scaler.joblib"),
        "seg_meta": _load_seg_meta(),
    }


def _load_seg_meta() -> dict:
    import json

    path = MODELS_DIR / "segmentation_meta.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"segment_names": {}}


def prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean + engineer features on raw or partial uploads."""
    if "ServiceCount" in df.columns and "Churn_numeric" in df.columns:
        return df
    if is_telco_compatible(df) or "tenure" in df.columns:
        out = clean_dataframe(df)
        return add_features(out)
    raise ValueError(
        "Uploaded data is missing required columns. Use column mapping in the app sidebar."
    )


def score_dataframe(
    df: pd.DataFrame,
    column_mapping: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """
    Full scoring pipeline on a dataframe (canonical or mapped columns).
    Returns customerID, segment, churn_probability, risk_band, etc.
    """
    if column_mapping:
        df = apply_column_mapping(df, column_mapping)

    ok, missing = validate_for_pipeline(df)
    if not ok and not is_telco_compatible(df):
        raise ValueError(f"Missing required columns: {missing}")

    models = load_models()
    feat_config = load_feature_config()
    churn_config = load_churn_config()
    seg_config = load_segmentation_config()

    if "customerID" not in df.columns:
        df = df.copy()
        df["customerID"] = [f"CUST_{i:05d}" for i in range(len(df))]

    clean = prepare_input_dataframe(df)

    # Segmentation
    X_seg, _ = get_segmentation_matrix(clean, seg_config)
    seg_labels = models["kmeans"].predict(models["seg_scaler"].transform(X_seg))
    clean["Segment"] = seg_labels.astype(int)
    name_map = models["seg_meta"].get("segment_names", {})
    clean["segment_name"] = clean["Segment"].map(
        {int(k): v for k, v in name_map.items()}
    ).fillna("Segment " + clean["Segment"].astype(str))

    # Churn
    X, _, _, _ = get_feature_matrix(clean, feat_config)
    X_enc = models["preprocessor"].transform(X)
    probs = models["churn"].predict_proba(X_enc)[:, 1]

    clean["churn_probability"] = probs
    clean["churn_predicted"] = (probs >= 0.5).astype(int)
    clean["risk_band"] = [
        assign_risk_band(p, churn_config["risk_thresholds"]) for p in probs
    ]

    if feat_config["target_column"] in df.columns:
        clean[feat_config["target_column"]] = df[feat_config["target_column"]].values
    if feat_config["target_numeric"] not in clean.columns and feat_config["target_column"] in clean.columns:
        clean[feat_config["target_numeric"]] = (clean[feat_config["target_column"]] == "Yes").astype(int)

    return clean


def score_default_telco() -> pd.DataFrame:
    """Score bundled clean telco dataset."""
    from src.load_data import load_clean

    return score_dataframe(load_clean())


def log_scoring_run(
    scored: pd.DataFrame,
    source: str,
    n_uploaded: int | None = None,
) -> Path:
    """Append session summary to monitoring log (no raw PII)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "prediction_runs.csv"

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "rows_scored": len(scored),
        "rows_uploaded": n_uploaded or len(scored),
        "mean_churn_probability": round(scored["churn_probability"].mean(), 4),
        "high_risk_pct": round((scored["risk_band"] == "High").mean() * 100, 2),
        "segment_0_pct": round((scored["Segment"] == 0).mean() * 100, 2)
        if "Segment" in scored.columns
        else None,
    }
    new_row = pd.DataFrame([row])
    if log_file.exists():
        pd.concat([pd.read_csv(log_file), new_row], ignore_index=True).to_csv(
            log_file, index=False
        )
    else:
        new_row.to_csv(log_file, index=False)
    return log_file
