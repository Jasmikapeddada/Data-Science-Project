"""
Churn classification models (Day 4): Logistic Regression + Random Forest.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score

from src.load_data import load_clean
from src.preprocess import (
    MODELS_DIR,
    PROCESSED_DIR,
    get_feature_matrix,
    load_feature_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHURN_CONFIG_PATH = PROJECT_ROOT / "config" / "churn_config.json"
FIGURES_DIR = PROCESSED_DIR / "figures"


def load_churn_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else CHURN_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_train_test_matrices(
    config: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Load encoded train/test from Day 2 processed CSVs."""
    config = config or load_churn_config()
    target = config["target_column"]

    train_df = pd.read_csv(PROCESSED_DIR / "telco_train_encoded.csv")
    test_df = pd.read_csv(PROCESSED_DIR / "telco_test_encoded.csv")

    feature_cols = [c for c in train_df.columns if c != target]
    X_train = train_df[feature_cols].values
    y_train = train_df[target].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target].values
    return X_train, X_test, y_train, y_test, feature_cols


def evaluate_model(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    top_pct: float = 0.2,
) -> dict:
    """Classification metrics + business KPI (churners in top risk %)."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    n_top = max(1, int(len(y_test) * top_pct))
    top_idx = np.argsort(y_prob)[-n_top:]
    metrics["churners_in_top_pct"] = float(y_test[top_idx].sum() / max(y_test.sum(), 1))
    metrics["top_pct_used"] = top_pct
    return metrics


def train_logistic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> LogisticRegression:
    lr_cfg = config["logistic_regression"]
    model = LogisticRegression(
        max_iter=lr_cfg["max_iter"],
        class_weight=lr_cfg["class_weight"],
        random_state=config["random_state"],
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict,
) -> GridSearchCV:
    rf_cfg = config["random_forest"]
    base = RandomForestClassifier(random_state=config["random_state"])
    grid = GridSearchCV(
        base,
        rf_cfg["param_grid"],
        cv=config["cv_folds"],
        scoring="recall",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    return grid


def assign_risk_band(prob: float, thresholds: dict) -> str:
    if prob >= thresholds["high"]:
        return "High"
    if prob >= thresholds["medium"]:
        return "Medium"
    return "Low"


def score_full_dataset(
    model,
    feature_cols: list[str],
    config: dict,
) -> pd.DataFrame:
    """Score all customers; merge segments if available."""
    clean = load_clean()
    preprocessor = joblib.load(MODELS_DIR / "preprocess_pipeline.joblib")
    feat_config = load_feature_config()

    X, _, _, _ = get_feature_matrix(clean, feat_config)
    X_enc = preprocessor.transform(X)
    probs = model.predict_proba(X_enc)[:, 1]

    scored = clean[[feat_config["id_column"], "Churn", feat_config["target_numeric"]]].copy()
    scored["churn_probability"] = probs
    scored["churn_predicted"] = (probs >= 0.5).astype(int)
    scored["risk_band"] = [
        assign_risk_band(p, config["risk_thresholds"]) for p in probs
    ]

    seg_path = PROCESSED_DIR / "telco_segmented.csv"
    if seg_path.exists():
        seg = pd.read_csv(seg_path)[
            [feat_config["id_column"], "Segment", "segment_name"]
        ]
        scored = scored.merge(seg, on=feat_config["id_column"], how="left")

    return scored


def plot_confusion_matrix(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Stay", "Churn"],
        yticklabels=["Stay", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — churn models")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    out_path: Path,
    top_n: int = 15,
) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="#3498db",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features — Random Forest")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def select_best_model(metrics_lr: dict, metrics_rf: dict) -> str:
    """Prefer higher recall, then F1 (retention use case)."""
    if metrics_rf["recall"] >= metrics_lr["recall"]:
        return "random_forest"
    return "logistic_regression"


def run_churn_training() -> dict:
    """Train both models, evaluate, pick champion."""
    config = load_churn_config()
    X_train, X_test, y_train, y_test, feature_cols = load_train_test_matrices(config)

    lr_model = train_logistic(X_train, y_train, config)
    rf_grid = train_random_forest(X_train, y_train, config)
    rf_model = rf_grid.best_estimator_

    metrics_lr = evaluate_model(
        "logistic_regression",
        lr_model,
        X_test,
        y_test,
        config["test_size_metric_top_pct"],
    )
    metrics_rf = evaluate_model(
        "random_forest",
        rf_model,
        X_test,
        y_test,
        config["test_size_metric_top_pct"],
    )

    # CV recall on train for reporting
    metrics_lr["cv_recall_mean"] = float(
        cross_val_score(
            LogisticRegression(
                max_iter=config["logistic_regression"]["max_iter"],
                class_weight=config["logistic_regression"]["class_weight"],
                random_state=config["random_state"],
            ),
            X_train,
            y_train,
            cv=config["cv_folds"],
            scoring="recall",
        ).mean()
    )

    champion = select_best_model(metrics_lr, metrics_rf)
    champion_model = rf_model if champion == "random_forest" else lr_model

    return {
        "config": config,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_cols": feature_cols,
        "logistic_model": lr_model,
        "rf_model": rf_model,
        "rf_grid": rf_grid,
        "metrics_lr": metrics_lr,
        "metrics_rf": metrics_rf,
        "champion_name": champion,
        "champion_model": champion_model,
        "classification_report_rf": classification_report(y_test, rf_model.predict(X_test)),
        "classification_report_lr": classification_report(y_test, lr_model.predict(X_test)),
    }


def save_churn_artifacts(result: dict) -> dict[str, Path]:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    joblib.dump(result["logistic_model"], MODELS_DIR / "logistic_churn_model.joblib")
    paths["logistic_model"] = MODELS_DIR / "logistic_churn_model.joblib"

    joblib.dump(result["rf_model"], MODELS_DIR / "random_forest_churn_model.joblib")
    paths["rf_model"] = MODELS_DIR / "random_forest_churn_model.joblib"

    joblib.dump(result["champion_model"], MODELS_DIR / "churn_model.joblib")
    paths["churn_model"] = MODELS_DIR / "churn_model.joblib"

    metrics = {
        "champion": result["champion_name"],
        "logistic_regression": result["metrics_lr"],
        "random_forest": result["metrics_rf"],
        "rf_best_params": result["rf_grid"].best_params_,
    }
    metrics_path = MODELS_DIR / "churn_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    paths["churn_metrics"] = metrics_path

    scored = score_full_dataset(
        result["champion_model"],
        result["feature_cols"],
        result["config"],
    )
    scored_path = PROCESSED_DIR / "telco_scored.csv"
    scored.to_csv(scored_path, index=False)
    paths["telco_scored"] = scored_path

    config = result["config"]
    X_test, y_test = result["X_test"], result["y_test"]

    cm_lr = FIGURES_DIR / "confusion_matrix_logistic.png"
    plot_confusion_matrix(
        y_test,
        result["logistic_model"].predict(X_test),
        "Logistic Regression",
        cm_lr,
    )
    paths["cm_logistic"] = cm_lr

    cm_rf = FIGURES_DIR / "confusion_matrix_random_forest.png"
    plot_confusion_matrix(
        y_test,
        result["rf_model"].predict(X_test),
        "Random Forest",
        cm_rf,
    )
    paths["cm_rf"] = cm_rf

    roc_path = FIGURES_DIR / "roc_curves.png"
    plot_roc_curves(
        {
            "Logistic Regression": result["logistic_model"],
            "Random Forest": result["rf_model"],
        },
        X_test,
        y_test,
        roc_path,
    )
    paths["roc_curves"] = roc_path

    fi_path = FIGURES_DIR / "feature_importance_rf.png"
    plot_feature_importance(
        result["rf_model"],
        result["feature_cols"],
        fi_path,
    )
    paths["feature_importance"] = fi_path

    return paths
