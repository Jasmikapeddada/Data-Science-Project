"""
Persist acceptable uploaded churn CSVs and expose them alongside the default Telco dataset.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.schema import apply_column_mapping, detect_column_mapping, validate_for_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
REGISTRY_PATH = UPLOADS_DIR / "registry.json"

DEFAULT_DATASET_ID = "telco_default"
DEFAULT_LABEL = "Telco Customer Churn (default)"


def _ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_PATH.exists():
        REGISTRY_PATH.write_text(
            json.dumps({"datasets": []}, indent=2),
            encoding="utf-8",
        )


def _read_registry() -> dict[str, Any]:
    _ensure_dirs()
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _write_registry(data: dict[str, Any]) -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _slugify(name: str) -> str:
    base = Path(name).stem
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", base).strip("_").lower()
    return slug[:60] or "upload"


def list_dataset_options() -> list[dict[str, Any]]:
    """All selectable datasets: default first, then saved uploads (newest last)."""
    options = [
        {
            "id": DEFAULT_DATASET_ID,
            "label": DEFAULT_LABEL,
            "kind": "default",
            "path": None,
            "column_mapping": None,
        }
    ]
    reg = _read_registry()
    for entry in reg.get("datasets", []):
        options.append(
            {
                "id": entry["id"],
                "label": entry["label"],
                "kind": "upload",
                "path": entry["saved_path"],
                "column_mapping": entry.get("column_mapping"),
            }
        )
    return options


def get_option_by_label(label: str) -> dict[str, Any] | None:
    for opt in list_dataset_options():
        if opt["label"] == label:
            return opt
    return None


def mapped_dataframe(
    df: pd.DataFrame,
    column_mapping: dict[str, str | None] | None,
) -> pd.DataFrame:
    if column_mapping:
        return apply_column_mapping(df.copy(), column_mapping)
    return df.copy()


def assess_upload(
    df: pd.DataFrame,
    column_mapping: dict[str, str | None] | None = None,
) -> tuple[bool, str, list[str]]:
    """
    Check if upload fits the telco churn model (schema + successful scoring).
    Returns (acceptable, message, missing_columns).
    """
    if df.empty:
        return False, "File is empty.", []

    if len(df) < 10:
        return False, "Need at least 10 customer rows.", []

    mapped = mapped_dataframe(df, column_mapping)
    ok, missing = validate_for_pipeline(mapped)
    if not ok:
        return (
            False,
            "Not compatible with this churn model (missing telco-style customer fields).",
            missing,
        )

    # Prove the full ML pipeline runs on this file
    try:
        from src.predict import score_dataframe

        score_dataframe(df, column_mapping=column_mapping)
    except Exception as exc:
        return False, f"Pipeline failed on this file: {exc}", missing

    return True, "Dataset is compatible and was scored successfully.", []


def save_acceptable_upload(
    df: pd.DataFrame,
    original_filename: str,
    column_mapping: dict[str, str | None] | None,
    display_name: str | None = None,
) -> dict[str, Any]:
    """
    Save canonical-mapped CSV under data/uploads/ and register for the app selector.
    """
    acceptable, message, missing = assess_upload(df, column_mapping)
    if not acceptable:
        raise ValueError(f"{message} Missing: {missing}" if missing else message)

    mapped = mapped_dataframe(df, column_mapping)
    slug = _slugify(original_filename)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dataset_id = f"upload_{slug}_{ts}"
    saved_path = UPLOADS_DIR / f"{dataset_id}.csv"
    mapped.to_csv(saved_path, index=False)

    label = display_name or Path(original_filename).stem.replace("_", " ").title()
    entry = {
        "id": dataset_id,
        "label": label,
        "original_filename": original_filename,
        "saved_path": str(saved_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "column_mapping": column_mapping or {},
        "row_count": len(mapped),
        "uploaded_at": datetime.now(timezone.utc).isoformat(),
    }

    reg = _read_registry()
    reg["datasets"] = [d for d in reg.get("datasets", []) if d.get("original_filename") != original_filename]
    reg["datasets"].append(entry)
    _write_registry(reg)
    return entry


def load_dataset_raw(option: dict[str, Any]) -> pd.DataFrame:
    """Load raw CSV for a registry option (uploads are stored canonical-mapped)."""
    if option["kind"] == "default":
        from src.load_data import load_clean

        return load_clean()

    path = PROJECT_ROOT / option["path"]
    if not path.exists():
        raise FileNotFoundError(f"Saved dataset not found: {path}")
    return pd.read_csv(path)


def delete_upload(dataset_id: str) -> bool:
    """Remove a saved upload from registry and disk."""
    reg = _read_registry()
    found = None
    for d in reg.get("datasets", []):
        if d["id"] == dataset_id:
            found = d
            break
    if not found:
        return False
    path = PROJECT_ROOT / found["saved_path"]
    if path.exists():
        path.unlink()
    reg["datasets"] = [d for d in reg["datasets"] if d["id"] != dataset_id]
    _write_registry(reg)
    return True
