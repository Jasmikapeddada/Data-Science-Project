"""Load and lightly clean Telco churn dataset for EDA and modeling."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load CSV from data/raw/. Downloads are handled by scripts/download_data.py."""
    path = Path(path) if path else DEFAULT_RAW_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python scripts/download_data.py"
        )
    df = pd.read_csv(path)
    return df


def load_for_eda(path: Path | str | None = None) -> pd.DataFrame:
    """
    Load dataset with EDA-friendly types.
    Does not drop rows; full cleaning is Day 2 pipeline.
    """
    df = load_raw(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn_numeric"] = (df["Churn"] == "Yes").astype(int)
    return df


def load_clean(path: Path | str | None = None) -> pd.DataFrame:
    """Load Day 2 cleaned dataset from data/processed/telco_clean.csv."""
    path = Path(path) if path else PROJECT_ROOT / "data" / "processed" / "telco_clean.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {path}. Run: python scripts/run_preprocess.py"
        )
    return pd.read_csv(path)
