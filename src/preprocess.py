"""
Cleaning, feature engineering, and sklearn preprocessing for Telco churn.

Used by: training scripts, notebooks, and (later) Streamlit upload pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.load_data import load_raw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "feature_config.json"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

SERVICE_COLS = [
    "PhoneService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def load_feature_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Impute issues found in Day 1 EDA; keep human-readable columns."""
    out = df.copy()

    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    missing_tc = out["TotalCharges"].isna()
    if missing_tc.any():
        # New customers (tenure=0): total billed ≈ first month charge
        out.loc[missing_tc, "TotalCharges"] = out.loc[missing_tc, "MonthlyCharges"]

    out["Churn_numeric"] = (out["Churn"] == "Yes").astype(int)

    if out["customerID"].duplicated().any():
        out = out.drop_duplicates(subset=["customerID"], keep="first")

    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineered features for segmentation and churn modeling."""
    out = df.copy()

    out["HasInternet"] = (out["InternetService"] != "No").astype(int)
    out["IsMonthToMonth"] = (out["Contract"] == "Month-to-month").astype(int)
    out["ServiceCount"] = (out[SERVICE_COLS] == "Yes").sum(axis=1).astype(int)
    out["AvgChargePerTenure"] = out["TotalCharges"] / np.maximum(out["tenure"], 1)

    return out


def build_clean_dataset(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full Day 2 cleaning + feature engineering pipeline on raw input."""
    if df is None:
        df = load_raw()
    return add_features(clean_dataframe(df))


def build_sklearn_preprocessor(config: dict | None = None) -> ColumnTransformer:
    config = config or load_feature_config()
    numeric = config["numeric_features"]
    categorical = config["categorical_features"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def get_feature_matrix(
    df: pd.DataFrame,
    config: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Extract X (raw columns) and y before encoding."""
    config = config or load_feature_config()
    feature_cols = config["numeric_features"] + config["categorical_features"]
    target_col = config["target_numeric"]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, config["numeric_features"], config["categorical_features"]


def fit_transform_train_test(
    df: pd.DataFrame | None = None,
    config: dict | None = None,
) -> dict:
    """
    Build clean data, stratified split, fit preprocessor on train, transform both.

    Returns dict with arrays, preprocessors, and paths for downstream steps.
    """
    config = config or load_feature_config()
    clean = build_clean_dataset(df)

    X, y, _, _ = get_feature_matrix(clean, config)
    id_series = clean[config["id_column"]]

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X,
        y,
        id_series,
        test_size=config["test_size"],
        random_state=config["random_state"],
        stratify=y,
    )

    preprocessor = build_sklearn_preprocessor(config)
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()

    return {
        "clean_df": clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "id_train": id_train,
        "id_test": id_test,
        "X_train_encoded": X_train_enc,
        "X_test_encoded": X_test_enc,
        "feature_names_encoded": feature_names,
        "preprocessor": preprocessor,
        "config": config,
    }


def save_artifacts(result: dict, processed_dir: Path | None = None) -> dict[str, Path]:
    """Persist CSVs and fitted preprocessor for Day 3+."""
    processed_dir = processed_dir or PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    clean_path = processed_dir / "telco_clean.csv"
    result["clean_df"].to_csv(clean_path, index=False)
    paths["telco_clean"] = clean_path

    # Train/test splits (pre-encoding, for inspection)
    train_df = result["X_train"].copy()
    train_df[result["config"]["id_column"]] = result["id_train"].values
    train_df[result["config"]["target_numeric"]] = result["y_train"].values
    test_df = result["X_test"].copy()
    test_df[result["config"]["id_column"]] = result["id_test"].values
    test_df[result["config"]["target_numeric"]] = result["y_test"].values

    train_path = processed_dir / "telco_train.csv"
    test_path = processed_dir / "telco_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    paths["telco_train"] = train_path
    paths["telco_test"] = test_path

    # Encoded matrices as CSV (wide) for notebook inspection
    enc_train = pd.DataFrame(
        result["X_train_encoded"],
        columns=result["feature_names_encoded"],
    )
    enc_train[result["config"]["target_numeric"]] = result["y_train"].values
    enc_test = pd.DataFrame(
        result["X_test_encoded"],
        columns=result["feature_names_encoded"],
    )
    enc_test[result["config"]["target_numeric"]] = result["y_test"].values

    enc_train_path = processed_dir / "telco_train_encoded.csv"
    enc_test_path = processed_dir / "telco_test_encoded.csv"
    enc_train.to_csv(enc_train_path, index=False)
    enc_test.to_csv(enc_test_path, index=False)
    paths["telco_train_encoded"] = enc_train_path
    paths["telco_test_encoded"] = enc_test_path

    preprocessor_path = MODELS_DIR / "preprocess_pipeline.joblib"
    joblib.dump(result["preprocessor"], preprocessor_path)
    paths["preprocessor"] = preprocessor_path

    config_out = MODELS_DIR / "feature_config.json"
    with open(config_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                **result["config"],
                "encoded_feature_names": result["feature_names_encoded"],
                "n_train": len(result["y_train"]),
                "n_test": len(result["y_test"]),
            },
            f,
            indent=2,
        )
    paths["feature_config"] = config_out

    return paths


def run_pipeline(df: pd.DataFrame | None = None) -> tuple[dict, dict[str, Path]]:
    """End-to-end Day 2: clean → split → encode → save."""
    result = fit_transform_train_test(df)
    paths = save_artifacts(result)
    return result, paths
