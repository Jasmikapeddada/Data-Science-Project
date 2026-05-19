"""
Customer segmentation with K-Means (Day 3).

Fits on scaled behavioral numeric features for interpretable cluster profiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.load_data import load_clean
from src.preprocess import MODELS_DIR, PROCESSED_DIR

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEG_CONFIG_PATH = PROJECT_ROOT / "config" / "segmentation_config.json"
FIGURES_DIR = PROCESSED_DIR / "figures"


def load_segmentation_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else SEG_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_segmentation_matrix(
    df: pd.DataFrame, config: dict | None = None
) -> tuple[np.ndarray, list[str]]:
    config = config or load_segmentation_config()
    features = config["segmentation_features"]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing segmentation features: {missing}")
    return df[features].values, features


def evaluate_k_range(
    X_scaled: np.ndarray,
    k_min: int = 2,
    k_max: int = 8,
    random_state: int = 42,
    n_init: int = 10,
) -> pd.DataFrame:
    """Elbow (inertia) and silhouette for each K."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = km.fit_predict(X_scaled)
        rows.append(
            {
                "k": k,
                "inertia": km.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows)


def select_best_k(metrics: pd.DataFrame) -> int:
    """Pick K with highest silhouette score."""
    return int(metrics.loc[metrics["silhouette"].idxmax(), "k"])


def label_segment(row: pd.Series) -> str:
    """Business-friendly segment name from cluster profile means."""
    tenure = row.get("tenure", 0)
    churn = row.get("Churn_numeric", 0) * 100
    m2m = row.get("IsMonthToMonth", 0) * 100 if "IsMonthToMonth" in row else 0
    charges = row.get("MonthlyCharges", 0)

    if tenure >= 48 and churn < 15:
        return "Loyal Long-Tenure"
    if tenure < 12 and churn >= 40:
        return "At-Risk New Customers"
    if m2m >= 50 and churn >= 30:
        return "Month-to-Month Volatile"
    if charges >= 70 and churn < 25:
        return "High-Spend Stable"
    if tenure < 24:
        return "Growing Accounts"
    return "Standard Mixed"


def build_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str = "Segment",
    config: dict | None = None,
) -> pd.DataFrame:
    config = config or load_segmentation_config()
    profile_cols = [c for c in config["profile_columns"] if c in df.columns]
    if "IsMonthToMonth" not in profile_cols and "IsMonthToMonth" in df.columns:
        profile_cols.append("IsMonthToMonth")

    numeric_cols = [c for c in profile_cols if pd.api.types.is_numeric_dtype(df[c])]

    agg = df.groupby(cluster_col).agg(
        customers=("customerID", "count"),
        **{f"avg_{c}": (c, "mean") for c in numeric_cols},
    )
    agg["churn_rate_pct"] = df.groupby(cluster_col)["Churn_numeric"].mean() * 100

    # Top contract & internet per cluster
    top_contract = df.groupby(cluster_col)["Contract"].agg(
        lambda s: s.mode().iloc[0] if len(s) else ""
    )
    top_internet = df.groupby(cluster_col)["InternetService"].agg(
        lambda s: s.mode().iloc[0] if len(s) else ""
    )
    agg["top_contract"] = top_contract
    agg["top_internet"] = top_internet
    agg = agg.reset_index()

    # Segment names
    rename_map = {f"avg_{c}": c for c in numeric_cols}
    agg = agg.rename(columns=rename_map)
    profile_rows = []
    for _, r in agg.iterrows():
        profile_rows.append(label_segment(r))
    agg["segment_name"] = profile_rows
    return agg


def plot_k_selection(metrics: pd.DataFrame, best_k: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(metrics["k"], metrics["inertia"], marker="o", color="#3498db")
    axes[0].axvline(best_k, color="#e74c3c", linestyle="--", label=f"Best K={best_k}")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow method")
    axes[0].legend()

    axes[1].plot(metrics["k"], metrics["silhouette"], marker="o", color="#2ecc71")
    axes[1].axvline(best_k, color="#e74c3c", linestyle="--", label=f"Best K={best_k}")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette score")
    axes[1].set_title("Silhouette by K")
    axes[1].legend()

    plt.tight_layout()
    path = out_dir / "k_selection.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca_clusters(
    X_scaled: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.5,
        s=12,
        edgecolors="none",
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.set_title("Customer segments (PCA view)")
    plt.colorbar(scatter, ax=ax, label="Segment")
    plt.tight_layout()
    path = out_dir / "segments_pca.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_segment_churn(profiles: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = [
        f"Seg {int(r['Segment'])}\n{r['segment_name']}"
        for _, r in profiles.iterrows()
    ]
    colors = sns.color_palette("RdYlGn_r", n_colors=len(profiles))
    ax.bar(x_labels, profiles["churn_rate_pct"], color=colors)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel("Segment")
    ax.set_title("Churn rate by customer segment")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = out_dir / "segment_churn_rates.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def run_segmentation(df: pd.DataFrame | None = None) -> dict:
    """Full Day 3 pipeline: K selection → KMeans → profiles → plots."""
    config = load_segmentation_config()
    df = df if df is not None else load_clean()

    X, feature_names = get_segmentation_matrix(df, config)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    metrics = evaluate_k_range(
        X_scaled,
        k_min=config["k_min"],
        k_max=config["k_max"],
        random_state=config["random_state"],
        n_init=config["n_init"],
    )
    best_k = select_best_k(metrics)

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=config["random_state"],
        n_init=config["n_init"],
    )
    labels = kmeans.fit_predict(X_scaled)
    final_silhouette = silhouette_score(X_scaled, labels)

    segmented = df.copy()
    segmented["Segment"] = labels
    segmented["Segment"] = segmented["Segment"].astype(int)

    profiles = build_cluster_profiles(segmented, config=config)
    # Map segment id → name on customer rows
    name_map = dict(zip(profiles["Segment"], profiles["segment_name"]))
    segmented["segment_name"] = segmented["Segment"].map(name_map)

    return {
        "df": segmented,
        "profiles": profiles,
        "metrics": metrics,
        "best_k": best_k,
        "silhouette": final_silhouette,
        "kmeans": kmeans,
        "scaler": scaler,
        "feature_names": feature_names,
        "X_scaled": X_scaled,
        "config": config,
    }


def save_segmentation_artifacts(result: dict) -> dict[str, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    seg_path = PROCESSED_DIR / "telco_segmented.csv"
    result["df"].to_csv(seg_path, index=False)
    paths["telco_segmented"] = seg_path

    profiles_path = PROCESSED_DIR / "cluster_profiles.csv"
    result["profiles"].to_csv(profiles_path, index=False)
    paths["cluster_profiles"] = profiles_path

    metrics_path = PROCESSED_DIR / "k_selection_metrics.csv"
    result["metrics"].to_csv(metrics_path, index=False)
    paths["k_selection_metrics"] = metrics_path

    joblib.dump(result["kmeans"], MODELS_DIR / "kmeans_model.joblib")
    paths["kmeans_model"] = MODELS_DIR / "kmeans_model.joblib"

    joblib.dump(result["scaler"], MODELS_DIR / "segmentation_scaler.joblib")
    paths["segmentation_scaler"] = MODELS_DIR / "segmentation_scaler.joblib"

    seg_meta = {
        "best_k": result["best_k"],
        "silhouette": float(result["silhouette"]),
        "features": result["feature_names"],
        "segment_names": dict(
            zip(
                result["profiles"]["Segment"].astype(int),
                result["profiles"]["segment_name"],
            )
        ),
    }
    meta_path = MODELS_DIR / "segmentation_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(seg_meta, f, indent=2)
    paths["segmentation_meta"] = meta_path

    paths["k_selection_plot"] = plot_k_selection(
        result["metrics"], result["best_k"], FIGURES_DIR
    )
    paths["pca_plot"] = plot_pca_clusters(
        result["X_scaled"], result["df"]["Segment"].values, FIGURES_DIR
    )
    paths["churn_by_segment_plot"] = plot_segment_churn(
        result["profiles"], FIGURES_DIR
    )

    return paths
